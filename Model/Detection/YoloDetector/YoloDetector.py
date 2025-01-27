import cv2
import ultralytics
import numpy as np
from Model.FaceDetection import FaceDetector
from Model.Detection.YoloDetectionResult import YoloDetectionResult
from Model.Detection.detection_utilis import draw_detections

class Yolov8Detector(FaceDetector):
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
        
        results = self.model(image, conf = 0.6, iou = 0.5)
        print(results[0].boxes.conf.cpu())
        # boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        self.results = YoloDetectionResult()
        if len(results[0].boxes.conf.cpu()) > 0:           
            for result in results:
                # print(result.boxes.xyxy.cpu().numpy())
                # print(result.boxes.conf.cpu().numpy())
                # print(result.boxes.cls.cpu().numpy())
                self.results.add(
                    box     =    np.uint16(result.boxes.xyxy.cpu().numpy()[0]),
                    score   =    result.boxes.conf.cpu().numpy()[0],
                    class_id=0
                )
        return self.results

    def draw_detections(self, image):
        return draw_detections(image, self.results.boxes, self.results.scores, self.results.class_ids)