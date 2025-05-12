import numpy as np
from src.utils.image import preprocess_image
import time
import logging
from src.config import recognition_settings, get_path
from src.utils.tflite_helpers import (
    OptimizedTFLiteModel,
    get_optimal_threads,
    load_tflite_model,
    get_input_details,
    get_output_details,
    run_inference
)

# Configure logging
logger = logging.getLogger("FaceNet_TFLite")

class FaceNetTFLiteHandler:
    def __init__(self, model_path=None, verbose=False):
        # Use configuration values for model path
        self.model_path = model_path or get_path("recognition.facenet.model_path")
        self.verbose = verbose
        
        # Use optimized TFLite helpers to load the model
        self.interpreter = load_tflite_model(
            model_path=self.model_path,
            num_threads=get_optimal_threads(),
            use_xnnpack=True
        )
        
        if self.interpreter is None:
            raise RuntimeError(f"Failed to load TFLite model from {self.model_path}")
            
        # Get input and output details
        self.input_details = get_input_details(self.interpreter)
        self.output_details = get_output_details(self.interpreter)
        
        # Use configuration values for input/output shapes if available
        self.input_shape = self.input_details[0]['shape'][1:3]  # (height, width)
        self.output_shape = self.output_details[0]['shape'][1]  # 512
        
        if self.verbose:
            logger.info(f"Loaded FaceNet model from {self.model_path}")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output shape: {self.output_shape}")
            logger.info(f"Using {get_optimal_threads()} threads for inference")

    def forward(self, image: np.ndarray) -> np.ndarray:
        # Use optimized inference function
        output_data, inference_time = run_inference(
            self.interpreter,
            image,
            input_index=0,
            output_index=0,
            profile=self.verbose
        )
        
        if self.verbose:
            logger.info(f"Embedding Model Inference Time: {inference_time:.2f} ms")
            
        return output_data[0] if output_data is not None else None