#!/usr/bin/env python3
"""
Test script to demonstrate TFLite optimization and profiling utilities.
This script loads a FaceNet model with our optimized TFLite helpers and 
performs benchmarks to show the performance improvements.
"""

import argparse
import logging
import time
import numpy as np
import os
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TFLite_Test")

# Add parent directory to path so we can import utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Import our TFLite optimization utilities
from src.utils.tflite_helpers import (
    OptimizedTFLiteModel,
    load_tflite_model,
    run_inference,
    get_tflite_metadata,
    get_optimal_threads
)
from src.utils.ml_config import optimize_ml_environment
from src.config import get_path
from src.utils.image import preprocess_image

def load_test_image(image_path, target_size=(160, 160)):
    """
    Load and preprocess a test image for FaceNet input.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (height, width)
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image from {image_path}")
            return None
            
        # Preprocess image (resize, normalize)
        image = cv2.resize(image, target_size)
        image = preprocess_image(image)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None


def benchmark_tflite_model(model_path, image_path, num_warmup_runs=3, num_benchmark_runs=10):
    """
    Benchmark a TFLite model with our optimized helpers.
    
    Args:
        model_path: Path to the TFLite model file
        image_path: Path to a test image
        num_warmup_runs: Number of warmup runs
        num_benchmark_runs: Number of runs for benchmarking
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking TFLite model: {model_path}")
    
    # Get model metadata
    metadata = get_tflite_metadata(model_path)
    logger.info(f"Model metadata: {metadata}")
    
    # Load test image
    image = load_test_image(image_path)
    if image is None:
        return {'error': 'Failed to load test image'}
    
    # Create optimized model using our helper class
    optimized_model = OptimizedTFLiteModel(model_path, verbose=True)
    
    # Warmup runs
    logger.info(f"Performing {num_warmup_runs} warmup runs...")
    for i in range(num_warmup_runs):
        _ = optimized_model.predict(image)
    
    # Benchmark runs
    logger.info(f"Running {num_benchmark_runs} benchmark iterations...")
    inference_times = []
    
    for i in range(num_benchmark_runs):
        start_time = time.time()
        output = optimized_model.predict(image)
        inference_time = (time.time() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        logger.info(f"Run {i+1}: {inference_time:.2f}ms")
    
    # Compute statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    
    results = {
        'average_ms': avg_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'samples': num_benchmark_runs,
        'threads': get_optimal_threads()
    }
    
    logger.info(f"Benchmark results: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")
    
    # Also print output shape for verification
    logger.info(f"Output shape: {output.shape if output is not None else 'None'}")
    
    return results


def compare_optimized_vs_standard(model_path, image_path):
    """
    Compare the performance between standard TFLite and our optimized implementation.
    
    Args:
        model_path: Path to the TFLite model file
        image_path: Path to a test image
    """
    logger.info("Comparing standard vs optimized TFLite implementation")
    
    # Load test image
    image = load_test_image(image_path)
    if image is None:
        logger.error("Failed to load test image")
        return
    
    # 1. Standard TFLite implementation
    try:
        import tflite_runtime.interpreter as tflite
        logger.info("Using standard TFLite runtime implementation...")
        
        # Load model
        start_time = time.time()
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        loading_time_standard = (time.time() - start_time) * 1000  # ms
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Run inference (10 times)
        standard_times = []
        for i in range(10):
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], image)
            
            # Measure inference time
            start_time = time.time()
            interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # ms
            standard_times.append(inference_time)
            
        avg_standard = sum(standard_times) / len(standard_times)
        logger.info(f"Standard TFLite: avg={avg_standard:.2f}ms, loading={loading_time_standard:.2f}ms")
    
    except Exception as e:
        logger.error(f"Error in standard implementation: {e}")
        avg_standard = None
    
    # 2. Our optimized implementation
    try:
        # Load model with our helper
        start_time = time.time()
        optimized_model = OptimizedTFLiteModel(model_path, verbose=False)
        loading_time_optimized = (time.time() - start_time) * 1000  # ms
        
        # Run inference (10 times)
        optimized_times = []
        for i in range(10):
            # Measure inference time
            start_time = time.time()
            output = optimized_model.predict(image)
            inference_time = (time.time() - start_time) * 1000  # ms
            optimized_times.append(inference_time)
            
        avg_optimized = sum(optimized_times) / len(optimized_times)
        logger.info(f"Optimized TFLite: avg={avg_optimized:.2f}ms, loading={loading_time_optimized:.2f}ms")
    
    except Exception as e:
        logger.error(f"Error in optimized implementation: {e}")
        avg_optimized = None
    
    # Compare results
    if avg_standard and avg_optimized:
        improvement = ((avg_standard - avg_optimized) / avg_standard) * 100
        logger.info(f"Performance improvement: {improvement:.1f}%")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test TFLite optimization utilities')
    parser.add_argument('--model', type=str, help='Path to TFLite model file')
    parser.add_argument('--image', type=str, help='Path to test image file')
    parser.add_argument('--compare', action='store_true', help='Compare standard vs optimized implementation')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    # Apply ML optimizations
    logger.info("Optimizing ML environment...")
    optimize_ml_environment()
    
    # Use default paths if not specified
    model_path = args.model or get_path("recognition.facenet.model_path")
    # Use one of the test images if available
    if not args.image:
        data_dir = os.path.join(os.path.dirname(__file__), "../../data")
        test_images = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
        if test_images:
            image_path = os.path.join(data_dir, test_images[0])
        else:
            logger.error("No test images found and --image not specified")
            return
    else:
        image_path = args.image
    
    # Run requested tests
    if args.compare:
        compare_optimized_vs_standard(model_path, image_path)
    
    if args.benchmark or not args.compare:
        benchmark_tflite_model(model_path, image_path)


if __name__ == "__main__":
    main()
