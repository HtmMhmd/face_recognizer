import cv2
import ultralytics
import numpy as np
from Model.FaceDetection import FaceDetector

class YOLOv8Detector(FaceDetector):
    """
    Face detection using a YOLOv8 model.
    """
    def __init__(self, model_path="Model/Detection/YoloDetector/model.pt"):  # Adjust path as needed
        super().__init__()
        try:
            self.model = ultralytics.YOLO(model_path)
        except Exception as e:
            raise ImportError(f"Error loading YOLOv8 model from {model_path}: {e}")


    def detect_faces(self, image):
        results = self.model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        self.boxes = boxes
        return self.boxes


