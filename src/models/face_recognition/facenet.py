import tflite_runtime.interpreter as tflite
import numpy as np
from src.utils.image import preprocess_image
import time
from src.config import recognition_settings, get_path

class FaceNetTFLiteHandler:
    def __init__(self, model_path=None, verbose=False):
        # Use configuration values for model path
        self.model_path = model_path or get_path("recognition.facenet.model_path")
        self.verbose = verbose
        
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Use configuration values for input/output shapes if available
        self.input_shape = self.input_details[0]['shape'][1:3]  # (height, width)
        self.output_shape = self.output_details[0]['shape'][1]  # 512
        
        if self.verbose:
            print(f"Loaded FaceNet model from {self.model_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Output shape: {self.output_shape}")

    def forward(self, image: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        
        start_time = time.time()
        self.interpreter.invoke()
        end_time = time.time()
        
        inference_time = end_time - start_time
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        if self.verbose:
            print(f"Embedding Model Inference Time: {inference_time*1000:.2f} ms")
        return output_data[0]