import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from ImageUtilis.image_utilis import preprocess_image

class FaceNetTFLiteHandler:
    def __init__(self, model_path: str = 'Model/FaceNet/facenet.tflite'):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]  # (height, width)
        self.output_shape = self.output_details[0]['shape'][1]  # 512

    def forward(self, image: np.ndarray) -> np.ndarray:
        input_tensor = preprocess_image(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0]