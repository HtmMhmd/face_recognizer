#!/usr/bin/env python3
"""
Performance testing script for TFLite models.
This script benchmarks TFLite models using the optimized tflite-runtime package.
"""

import argparse
import logging
import time
import numpy as np
import os
import cv2
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TFLite_Performance")

# Add parent directory to path so we can import utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Import TFLite optimization utilities
from src.utils.tflite_helpers import (
    OptimizedTFLiteModel,
    load_tflite_model,
    run_inference,
    get_optimal_threads,
    get_tflite_metadata
)
from src.utils.ml_config import optimize_ml_environment, configure_tflite_runtime
from src.config import get_path
from src.utils.image import preprocess_image

def load_sample_image(image_path: str, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    Load and preprocess a sample image for model input.
    
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
            # Generate a random image as fallback
            return np.random.random((1, *target_size, 3)).astype(np.float32)
            
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Preprocess (normalize pixel values)
        image = preprocess_image(image)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading test image: {e}")
        # Generate a random image as fallback
        return np.random.random((1, *target_size, 3)).astype(np.float32)

def benchmark_tflite_model(model_path: str, input_data: np.ndarray, 
                         num_runs: int = 100, num_warmup_runs: int = 5,
                         num_threads_list: List[int] = None) -> Dict:
    """
    Benchmark a TFLite model with different thread configurations.
    
    Args:
        model_path: Path to the TFLite model
        input_data: Input data for the model
        num_runs: Number of inference runs to benchmark
        num_warmup_runs: Number of warmup runs before benchmarking
        num_threads_list: List of thread counts to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    if num_threads_list is None:
        optimal_threads = get_optimal_threads()
        num_threads_list = [1, optimal_threads, optimal_threads * 2]
    
    # Print model information
    logger.info(f"Model: {os.path.basename(model_path)}")
    
    # Get model metadata
    metadata = get_tflite_metadata(model_path)
    if metadata['available']:
        logger.info(f"Model has {metadata['num_inputs']} inputs and {metadata['num_outputs']} outputs")
        for i, input_info in enumerate(metadata['inputs']):
            logger.info(f"Input {i}: shape={input_info['shape']}, dtype={input_info['dtype']}")
        for i, output_info in enumerate(metadata['outputs']):
            logger.info(f"Output {i}: shape={output_info['shape']}, dtype={output_info['dtype']}")
    
    # Benchmark with different thread configurations
    for num_threads in num_threads_list:
        logger.info(f"\nBenchmarking with {num_threads} threads:")
        
        # Load model with specified number of threads
        interpreter = load_tflite_model(model_path, num_threads=num_threads)
        
        if interpreter is None:
            logger.error("Failed to load interpreter")
            continue
        
        # Warm up
        logger.info(f"Warming up with {num_warmup_runs} runs...")
        for _ in range(num_warmup_runs):
            _, _ = run_inference(interpreter, input_data)
        
        # Benchmark
        logger.info(f"Running {num_runs} benchmarking iterations...")
        inference_times = []
        
        for i in range(num_runs):
            _, inference_time = run_inference(interpreter, input_data)
            inference_times.append(inference_time)
            
            # Print progress
            if (i+1) % 10 == 0:
                logger.info(f"Completed {i+1}/{num_runs} runs")
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        p90_time = np.percentile(inference_times, 90)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)
        
        # Store results
        results[num_threads] = {
            'avg_ms': avg_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'p90_ms': p90_time,
            'p95_ms': p95_time,
            'p99_ms': p99_time
        }
        
        # Print results
        logger.info(f"Results with {num_threads} threads:")
        logger.info(f"  Average inference time: {avg_time:.2f}ms")
        logger.info(f"  Min inference time: {min_time:.2f}ms")
        logger.info(f"  Max inference time: {max_time:.2f}ms")
        logger.info(f"  p90 inference time: {p90_time:.2f}ms")
        logger.info(f"  p95 inference time: {p95_time:.2f}ms")
        logger.info(f"  p99 inference time: {p99_time:.2f}ms")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark TFLite models')
    parser.add_argument('--model', type=str, default=None,
                      help='Path to the TFLite model file')
    parser.add_argument('--image', type=str, default=None,
                      help='Path to a sample image file')
    parser.add_argument('--num-runs', type=int, default=100,
                      help='Number of inference runs for benchmarking')
    parser.add_argument('--input-shape', type=str, default='160,160',
                      help='Input shape (height,width) for the model')
    args = parser.parse_args()
    
    # Optimize ML environment before running benchmarks
    logger.info("Optimizing ML environment")
    optimize_ml_environment()
    
    # Determine model path
    model_path = args.model
    if model_path is None:
        model_path = get_path("recognition.facenet.model_path")
        logger.info(f"Using default FaceNet model: {model_path}")
    
    # Load sample image or create a random one
    input_shape = tuple(map(int, args.input_shape.split(',')))
    if args.image:
        logger.info(f"Loading sample image from {args.image}")
        input_data = load_sample_image(args.image, target_size=input_shape)
    else:
        logger.info(f"Creating random input with shape {input_shape}")
        input_data = np.random.random((1, *input_shape, 3)).astype(np.float32)
    
    # Run benchmarks
    logger.info(f"Starting benchmark with {args.num_runs} runs")
    results = benchmark_tflite_model(
        model_path=model_path,
        input_data=input_data,
        num_runs=args.num_runs
    )
    
    # Print summary
    logger.info("\n===== BENCHMARK SUMMARY =====")
    for num_threads, thread_results in results.items():
        logger.info(f"Threads: {num_threads}, Avg time: {thread_results['avg_ms']:.2f}ms, " +
                   f"Min: {thread_results['min_ms']:.2f}ms, Max: {thread_results['max_ms']:.2f}ms")
    
    # Recommend optimal thread configuration
    optimal_threads = min(results.keys(), key=lambda t: results[t]['avg_ms'])
    logger.info(f"\nRecommended thread configuration: {optimal_threads} threads")
    logger.info(f"This configuration achieved an average inference time of {results[optimal_threads]['avg_ms']:.2f}ms")

if __name__ == "__main__":
    main()
