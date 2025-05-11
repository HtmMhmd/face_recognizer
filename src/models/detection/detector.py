import time
import numpy as np
from typing import Optional, List, Tuple

from src.models.detection.detection_faces import DetectionFaces
from src.models.detection.mediapipe_detector import MediapipeFaceDetector
from src.models.detection.yolo_detector import Yolov8Detector, Yolov8OnnxRuntimeDetector
from src.config import detection_settings
from src.models.detection.mediapipe_landmarker import FaceMeshDetector

class Detector:
    """
    A class that provides a unified interface for different face detection models.
    
    Attributes:
        detector_type: Type of detector to use (mediapipe, yolov8, yolov8_onnx)
        detection_faces: DetectionFaces object to store detection results
        verbose: Whether to print debugging information
        detector: The selected detector implementation
    """
    
    def __init__(self, detector_type=None, detection_faces=None, verbose=False):
        """
        Initialize the detector with the specified configuration.
        
        Args:
            detector_type: Type of detector to use (mediapipe, yolov8, yolov8_onnx)
            detection_faces: Optional DetectionFaces object to store detection results
            verbose: Whether to print debugging information
        """
        # Get default model type from config if not specified
        self.detector_type = detector_type or detection_settings.get("default_model", "mediapipe")
        self.detection_faces = detection_faces or DetectionFaces()
        self.verbose = verbose
        
        # Validate detector type
        valid_types = ["mediapipe", "yolov8", "yolov8_onnx", "landmark"]
        if self.detector_type not in valid_types:
            raise ValueError(f"Invalid detector type: {self.detector_type}. Must be one of {valid_types}")
        
        # Initialize the appropriate detector
        if self.detector_type == "mediapipe":
            confidence = detection_settings.get("mediapipe", {}).get("min_detection_confidence", 0.5)
            self.detector = MediapipeFaceDetector(min_detection_conf=confidence, verbose=verbose)
        
        elif self.detector_type == "yolov8":
            model_path = detection_settings.get("yolov8", {}).get("model_path", "models/yolov8n-face.pt") 
            conf_threshold = detection_settings.get("yolov8", {}).get("confidence_threshold", 0.5)
            iou_threshold = detection_settings.get("yolov8", {}).get("iou_threshold", 0.45)
            
            self.detector = Yolov8Detector(
                model_path=model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                verbose=verbose
            )
        
        elif self.detector_type == "yolov8_onnx":
            model_path = detection_settings.get("yolov8_onnx", {}).get("model_path", "models/yolov8n_face.onnx")
            conf_threshold = detection_settings.get("yolov8_onnx", {}).get("confidence_threshold", 0.5)
            iou_threshold = detection_settings.get("yolov8_onnx", {}).get("iou_threshold", 0.45)
            input_shape = tuple(detection_settings.get("yolov8_onnx", {}).get("input_shape", [640, 640]))
            
            self.detector = Yolov8OnnxRuntimeDetector(
                model_path=model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                input_shape=input_shape,
                verbose=verbose
            )
        
        elif self.detector_type == "landmark":         
            self.detector = None  # No face detection, only landmarking
        
        if self.verbose:
            print(f"Initialized {self.detector_type} detector")
        max_faces = detection_settings.get("face_mesh", {}).get("max_faces", 1)
        confidence = detection_settings.get("face_mesh", {}).get("min_detection_confidence", 0.5)
        self.landmarker = FaceMeshDetector(max_faces=max_faces, min_detection_conf=confidence, verbose=verbose)

    
    def detect(self, image) -> DetectionFaces:
        """
        Detect faces in the given image.
        
        Args:
            image: The input image
            
        Returns:
            DetectionFaces containing the detection results
        """
        if image is None:
            raise ValueError("Input image is None")
            
        start_time = time.time()
        detection_faces = self.detector.detect_faces(image)
        end_time = time.time()
        
        if self.verbose:
            inference_time = end_time - start_time
            print(f"{self.detector_type} face detection took {inference_time*1000:.2f} ms")
            print(f"Detected {len(detection_faces.boxes)} faces")
        
        # Store the results
        self.detection_faces = detection_faces
        return detection_faces
    
    def landmark(self, image):
        """
        Detect facial landmarks in the given image.
        
        Args:
            image: The input image
            
        Returns:
            The detected landmarks
        """
        return self.landmarker.landmark(image)
    
    def get_eye_mouth_keypoints(self):
        """
        Get the eye and mouth keypoints from the detected landmarks.
        
        Returns:
            Dictionary containing eye and mouth keypoints
        """
        return self.landmarker.get_eye_mouth_keypoints()
    
    def draw_detections(self, image):
        """
        Draw the detected bounding boxes on the image.
        
        Args:
            image: The input image
            
        Returns:
            The image with drawn detections
        """
        return self.detector.draw_detections(image)
    def draw_landmarks(self, image):
        """
        Draw the detected landmarks on the image.
        Args:
            image: The input image
        Returns:
            The image with drawn landmarks
        """
        return self.landmarker.draw_landmarks(image)
    def get_detections(self):
        """
        Get the detected faces.
        
        Returns:
            The detected faces
        """
        return self.detection_faces
