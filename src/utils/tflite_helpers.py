#!/usr/bin/env python3
"""
TFLite Helpers - Utilities for optimized TensorFlow Lite usage in face recognition system.
This module provides optimized methods for TFLite model loading, inference, quantization,
and delegate usage to improve performance on various hardware platforms.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import multiprocessing

# Configure logging
logger = logging.getLogger("TFLite_Helpers")

# Global flag to track if TFLite runtime is available
TFLITE_RUNTIME_AVAILABLE = False
TFLITE_MODULE = None

# Import tflite runtime (required for deployment)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME_AVAILABLE = True
    TFLITE_MODULE = tflite
    logger.info("Using tflite_runtime for inference (optimized for deployment)")
except ImportError:
    logger.error("TFLite Runtime not available. Please install tflite_runtime package.")


def get_optimal_threads() -> int:
    """
    Calculate optimal number of threads for TFLite inference.
    
    Returns:
        int: Optimal number of threads
    """
    # Use all available cores except one to avoid system slowdown
    return max(1, multiprocessing.cpu_count() - 1)


def load_tflite_model(model_path: str, 
                     num_threads: Optional[int] = None,
                     use_xnnpack: bool = True) -> Optional[Any]:
    """
    Load a TFLite model with optimized settings.
    
    Args:
        model_path: Path to the TFLite model file
        num_threads: Number of threads to use (None = automatic)
        use_xnnpack: Whether to try using XNNPACK delegate for CPU acceleration
        
    Returns:
        TFLite interpreter or None if loading failed
    """
    if not TFLITE_RUNTIME_AVAILABLE:
        logger.error("TFLite runtime is not available")
        return None
        
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        # Determine optimal thread count if not specified
        if num_threads is None:
            num_threads = get_optimal_threads()
            
        logger.info(f"Loading TFLite model from {model_path} with {num_threads} threads")
        
        # Define options
        interpreter = None
        
        # Try to use delegates for hardware acceleration
        # 1. First try: XNNPACK delegate for CPU optimization
        if use_xnnpack:
            try:
                # For tflite_runtime, we need to check if load_delegate is available
                if hasattr(TFLITE_MODULE, 'Interpreter'):
                    interpreter = TFLITE_MODULE.Interpreter(
                        model_path=model_path,
                        num_threads=num_threads
                    )
                    logger.info("Created TFLite interpreter with multi-threading support")
            except Exception as e:
                logger.warning(f"Failed to create optimized interpreter: {e}")
        
        # If first attempt failed, try standard interpreter
        if interpreter is None:
            interpreter = TFLITE_MODULE.Interpreter(model_path=model_path, num_threads=num_threads)
            logger.info("Created standard TFLite interpreter")
            
        # Allocate tensors
        interpreter.allocate_tensors()
        
        return interpreter
    
    except Exception as e:
        logger.error(f"Failed to load TFLite model: {e}")
        return None


def get_input_details(interpreter) -> List[Dict[str, Any]]:
    """
    Get input tensor details from a loaded interpreter.
    
    Args:
        interpreter: TFLite interpreter
        
    Returns:
        List of input tensor details
    """
    if interpreter is None:
        logger.error("Cannot get input details: interpreter is None")
        return []
        
    try:
        return interpreter.get_input_details()
    except Exception as e:
        logger.error(f"Failed to get input details: {e}")
        return []


def get_output_details(interpreter) -> List[Dict[str, Any]]:
    """
    Get output tensor details from a loaded interpreter.
    
    Args:
        interpreter: TFLite interpreter
        
    Returns:
        List of output tensor details
    """
    if interpreter is None:
        logger.error("Cannot get output details: interpreter is None")
        return []
        
    try:
        return interpreter.get_output_details()
    except Exception as e:
        logger.error(f"Failed to get output details: {e}")
        return []


def run_inference(interpreter, 
                 input_data: np.ndarray, 
                 input_index: int = 0,
                 output_index: int = 0,
                 profile: bool = False) -> Tuple[Optional[np.ndarray], float]:
    """
    Run inference with the given interpreter and input data.
    
    Args:
        interpreter: TFLite interpreter
        input_data: Input data as numpy array
        input_index: Index of the input tensor
        output_index: Index of the output tensor
        profile: Whether to profile and log inference time
        
    Returns:
        Tuple of (output tensor data, inference time in ms)
    """
    if interpreter is None:
        logger.error("Cannot run inference: interpreter is None")
        return None, 0.0
        
    try:
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Validate indices
        if input_index >= len(input_details):
            logger.error(f"Invalid input index: {input_index}, max is {len(input_details)-1}")
            return None, 0.0
            
        if output_index >= len(output_details):
            logger.error(f"Invalid output index: {output_index}, max is {len(output_details)-1}")
            return None, 0.0
        
        # Get shape and type info
        input_shape = input_details[input_index]['shape']
        input_dtype = input_details[input_index]['dtype']
        
        # Check and prepare input
        if input_data.shape != input_shape and input_data.shape != tuple(input_shape):
            # Try to reshape if the total size matches
            if input_data.size == np.prod(input_shape):
                logger.warning(f"Reshaping input from {input_data.shape} to {input_shape}")
                input_data = input_data.reshape(input_shape)
            else:
                logger.error(f"Input shape mismatch: got {input_data.shape}, expected {input_shape}")
                return None, 0.0
        
        # Handle quantization if needed
        is_quantized = input_details[input_index].get('quantization', (0, 0)) != (0, 0)
        if is_quantized:
            scale, zero_point = input_details[input_index]['quantization']
            if scale != 0:
                logger.debug("Quantizing input data")
                input_data = input_data / scale + zero_point
                input_data = input_data.astype(input_dtype)
        
        # Set input tensor
        interpreter.set_tensor(input_details[input_index]['index'], input_data)
        
        # Run inference
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # to ms
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[output_index]['index'])
        
        # Handle dequantization if needed
        is_quantized = output_details[output_index].get('quantization', (0, 0)) != (0, 0)
        if is_quantized:
            scale, zero_point = output_details[output_index]['quantization']
            if scale != 0:
                logger.debug("Dequantizing output data")
                output_data = (output_data.astype(np.float32) - zero_point) * scale
        
        if profile:
            logger.info(f"TFLite inference time: {inference_time:.2f}ms")
            
        return output_data, inference_time
    
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        return None, 0.0


class OptimizedTFLiteModel:
    """
    A wrapper class for TFLite models with optimized inference capabilities.
    This class provides a more user-friendly interface for TFLite model usage.
    """
    
    def __init__(self, model_path: str, num_threads: Optional[int] = None, verbose: bool = False):
        """
        Initialize the optimized TFLite model.
        
        Args:
            model_path: Path to the TFLite model file
            num_threads: Number of threads to use for inference (None = auto)
            verbose: Whether to print verbose logs
        """
        self.model_path = model_path
        self.num_threads = num_threads if num_threads is not None else get_optimal_threads()
        self.verbose = verbose
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.output_shape = None
        
        # Load the model
        self._load_model()
        
    def _load_model(self):
        """Load the TFLite model with optimal settings."""
        try:
            self.interpreter = load_tflite_model(
                self.model_path,
                num_threads=self.num_threads
            )
            
            if self.interpreter:
                self.input_details = get_input_details(self.interpreter)
                self.output_details = get_output_details(self.interpreter)
                
                if self.input_details and self.output_details:
                    self.input_shape = self.input_details[0]['shape']
                    self.output_shape = self.output_details[0]['shape']
                    
                    if self.verbose:
                        logger.info(f"Model loaded from {self.model_path}")
                        logger.info(f"Input shape: {self.input_shape}")
                        logger.info(f"Output shape: {self.output_shape}")
                        logger.info(f"Using {self.num_threads} threads for inference")
                else:
                    logger.error("Failed to get model input/output details")
            else:
                logger.error(f"Failed to load model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error initializing TFLite model: {e}")
    
    def predict(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference on the input data.
        
        Args:
            input_data: Input data as numpy array
            
        Returns:
            Output data as numpy array or None if inference failed
        """
        if self.interpreter is None:
            logger.error("Model not loaded")
            return None
        
        # Run inference
        output, inf_time = run_inference(
            self.interpreter,
            input_data,
            profile=self.verbose
        )
        
        if self.verbose and output is not None:
            logger.info(f"Inference completed in {inf_time:.2f}ms")
            
        return output


def get_tflite_metadata(model_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a TFLite model file if available.
    
    Args:
        model_path: Path to the TFLite model file
        
    Returns:
        Dictionary with metadata information
    """
    metadata = {'available': False}
    
    # Check if we can access TFLite metadata
    try:
        # First try with tflite_runtime
        if TFLITE_RUNTIME_AVAILABLE:
            # Load the model file
            with open(model_path, 'rb') as f:
                model_data = f.read()
                
            # Try to extract basic information
            interpreter = TFLITE_MODULE.Interpreter(model_content=model_data)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            metadata = {
                'available': True,
                'num_inputs': len(input_details),
                'num_outputs': len(output_details),
                'inputs': [],
                'outputs': []
            }
            
            # Extract input details
            for i, input_detail in enumerate(input_details):
                metadata['inputs'].append({
                    'name': input_detail.get('name', f'input_{i}'),
                    'shape': input_detail['shape'].tolist(),
                    'dtype': str(input_detail['dtype'])
                })
                
            # Extract output details
            for i, output_detail in enumerate(output_details):
                metadata['outputs'].append({
                    'name': output_detail.get('name', f'output_{i}'),
                    'shape': output_detail['shape'].tolist(),
                    'dtype': str(output_detail['dtype'])
                })
            
            return metadata
            
    except Exception as e:
        logger.error(f"Failed to extract TFLite model metadata: {e}")
        
    return metadata