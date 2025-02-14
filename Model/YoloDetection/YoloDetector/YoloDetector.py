import ultralytics
import numpy as np
from Model.FaceDetection import FaceDetector
from Model.detection_utilis import draw_detections, get_cropped_faces
from Model.DetectionFaces import DetectionFaces

class Yolov8Detector(FaceDetector):
    """
    Face detection using a YOLOv8 model.
    """
    def __init__(self, model_path="Model/Detection/YoloDetector/model.pt", detection_faces=None):  # Adjust path as needed
        super().__init__()
        try:
            self.model = ultralytics.YO(model_path)
        except Exception as e:
            raise ImportError(f"Error loading YOLOv8 model from {model_path}: {e}")
        self.detection_faces = detection_faces if detection_faces is not None else DetectionFaces()

    def detect_faces(self, image):
        """
        Detects faces in the input image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            DetectionFaces: The detection results containing bounding boxes, scores, class IDs, and cropped faces.
        """
        self.detection_faces.reset()  # Reset the detection faces object
        results = self.model(image, conf=0.6, iou=0.5)
        
        if len(results[0].boxes.conf.cpu()) > 0:
            cropped_faces = get_cropped_faces(image, results[0].boxes.xyxy.cpu().numpy())
            for i, result in enumerate(results):
                self.detection_faces.add(
                    box=np.uint16(result.boxes.xyxy.cpu().numpy()[0]),
                    score=result.boxes.conf.cpu().numpy()[0],
                    class_id=0,
                    cropped_face=cropped_faces[i]
                )
        return self.detection_faces

    def draw_detections(self, image, detection_faces):
        """
        Draws bounding boxes on the detected faces in the image.

        Args:
            image (np.ndarray): The input image.
            detection_faces (DetectionFaces): The detection results containing bounding boxes.

        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        return draw_detections(image, self.detection_faces.boxes, self.detection_faces.scores, self.detection_faces.class_ids)