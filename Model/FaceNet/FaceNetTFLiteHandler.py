import tflite_runtime.interpreter as tflite
import numpy as np
from ImageUtilis.image_utilis import preprocess_image
import time

class FaceNetTFLiteHandler:
    def __init__(self, model_path: str = 'Model/FaceNet/facenet.tflite', verbose: bool = False):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]  # (height, width)
        self.output_shape = self.output_details[0]['shape'][1]  # 512
        self.verbose = verbose

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