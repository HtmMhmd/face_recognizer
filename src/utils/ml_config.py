import os
import logging
import platform
import multiprocessing
import numpy as np

# Configure logging
logger = logging.getLogger("ML_Config")

def configure_tflite_runtime():
    """
    Configure TFLite Runtime to suppress warnings and optimize performance.
    This function is designed for tflite-runtime which is a lightweight
    package without full TensorFlow functionality, ideal for deployment.
    
    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    try:
        # Configure environment variables for TFLite
        os.environ['TFLITE_DISABLE_GPU'] = '0'  # Try to use GPU if available
        
        # Import TFLite Runtime
        import tflite_runtime.interpreter as tflite
        
        # Log TFLite availability and configuration
        logger.info("TFLite Runtime configuration complete")
        
        # Check for delegate support
        has_delegates = hasattr(tflite, 'load_delegate')
        if has_delegates:
            logger.info("TFLite delegate support is available")
        
        # Try to get version info
        try:
            version = str(getattr(tflite, '__version__', 'unknown'))
            logger.info(f"TFLite Runtime version: {version}")
        except:
            pass
            
        return True
    
    except ImportError:
        logger.warning("TFLite Runtime not found, skipping configuration")
        return False
    except Exception as e:
        logger.error(f"Error configuring TFLite Runtime: {e}")
        return False

def configure_tensorflow():
    """
    This function is no longer used as we're exclusively using TFLite runtime.
    It's kept as a stub for backwards compatibility.
    
    Returns:
        bool: Always returns False as TensorFlow is not supported
    """
    logger.warning("Full TensorFlow is not supported in this optimized version")
    logger.warning("Using tflite-runtime exclusively for better performance and reduced resource usage")
    return False


def configure_opencv():
    """
    Configure OpenCV for optimal performance.
    
    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    try:
        import cv2
        
        # Set OpenCV thread optimization
        cv2.setNumThreads(min(multiprocessing.cpu_count(), 4))
        
        # Check if OpenCV was built with optimization
        has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if has_cuda:
            logger.info("OpenCV CUDA support is available")
        else:
            logger.info("OpenCV running on CPU only")
        
        # Set optimized flags
        cv2.useOptimized()
        
        logger.info(f"OpenCV {cv2.__version__} configuration complete")
        return True
    
    except ImportError:
        logger.warning("OpenCV not found, skipping configuration")
        return False
    except Exception as e:
        logger.error(f"Error configuring OpenCV: {e}")
        return False

def configure_mediapipe():
    """
    Configure MediaPipe for optimal performance.
    
    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    try:
        # Set environment variables before importing
        os.environ['GLOG_minloglevel'] = '2'  # 0=info, 1=warning, 2=error, 3=fatal
        
        import mediapipe as mp
        
        logger.info(f"MediaPipe {mp.__version__} configuration complete")
        return True
    
    except ImportError:
        logger.warning("MediaPipe not found, skipping configuration")
        return False
    except Exception as e:
        logger.error(f"Error configuring MediaPipe: {e}")
        return False

def configure_numpy():
    """
    Configure NumPy for optimal performance.
    
    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    try:
        # Disable numpy warnings
        np.seterr(all='ignore')
        
        # Check for MKL/OpenBLAS
        if np.__config__.get_info('openblas_info') or np.__config__.get_info('blas_mkl_info'):
            logger.info("NumPy is using optimized linear algebra libraries")
        else:
            logger.warning("NumPy is not using optimized linear algebra libraries")
        
        logger.info(f"NumPy {np.__version__} configuration complete")
        return True
    
    except Exception as e:
        logger.error(f"Error configuring NumPy: {e}")
        return False

def configure_onnx():
    """
    Configure ONNX Runtime for optimal performance.
    
    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    try:
        import onnxruntime as ort
        
        # Configure session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Check available providers
        providers = ort.get_available_providers()
        
        # Log available execution providers
        logger.info(f"Available ONNX Runtime providers: {providers}")
        
        # Optimize threading
        num_cores = multiprocessing.cpu_count()
        session_options.intra_op_num_threads = max(1, num_cores - 2)
        session_options.inter_op_num_threads = max(1, num_cores // 2)
        
        logger.info(f"ONNX Runtime {ort.__version__} configuration complete")
        return True
    
    except ImportError:
        logger.warning("ONNX Runtime not found, skipping configuration")
        return False
    except Exception as e:
        logger.error(f"Error configuring ONNX Runtime: {e}")
        return False
    
def configure_system_info():
    """
    Log system information that's relevant for ML optimization.
    """
    try:
        system = platform.system()
        release = platform.release()
        machine = platform.machine()
        processor = platform.processor()
        cores = multiprocessing.cpu_count()
        
        logger.info(f"System: {system} {release} {machine}")
        logger.info(f"Processor: {processor} with {cores} cores")
        
        # Check for available memory
        try:
            import psutil
            mem_info = psutil.virtual_memory()
            logger.info(f"Available memory: {mem_info.available / (1024**3):.2f} GB of {mem_info.total / (1024**3):.2f} GB")
        except ImportError:
            pass
        
    except Exception as e:
        logger.error(f"Error collecting system info: {e}")

def optimize_ml_environment():
    """
    Configure the complete ML environment for optimal performance.
    This function calls all individual optimization functions.
    
    Returns:
        dict: Dictionary with status of each configuration
    """
    logger.info("Starting ML environment optimization")
    
    # Log system information
    configure_system_info()
    
    # Configure all components - TFLite runtime only, no full TensorFlow
    results = {
        "tflite_runtime": configure_tflite_runtime(),
        "opencv": configure_opencv(),
        "mediapipe": configure_mediapipe(),
        "numpy": configure_numpy(),
        "onnx": configure_onnx()
    }
    
    # Log overall status
    successful = sum(1 for status in results.values() if status)
    logger.info(f"ML environment optimization complete: {successful}/{len(results)} components configured")
    
    return results