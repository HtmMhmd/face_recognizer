import cv2
import numpy as np
from Model.Detection.YoloV8OnnxRuntime.YOLOv8 import YOLOv8
from Model.FaceDetection import FaceDetector

class Yolov8OnnxRuntimeDetector(FaceDetector):
    """
    Face detection using a YOLOv8 model.
    """
    def __init__(self, model_path="Model/Detection/YoloV8OnnxRuntime/model.onnx", verbose=False):  # Adjust path as needed
        super().__init__()
        try:
            # Initialize YOLOv8 object detector
            self.model = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.5, verbose= verbose)
        except Exception as e:
            raise ImportError(f"Error loading YOLOv8 model from {model_path}: {e}")


    def detect_faces(self, image):
        results = self.model(image)
        return results
    
    def draw_detections(self, image):
        return self.model.draw_detections(image)
