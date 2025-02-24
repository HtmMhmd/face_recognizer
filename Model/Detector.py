from Model.YoloDetection.YoloV8OnnxRuntime.Yolov8OnnxRuntimeDetector import Yolov8OnnxRuntimeDetector
# from Model.YoloDetection.YoloDetector.YoloDetector import Yolov8Detector
from Model.MediapipeDetection.MediapipeFaceDetector import MediapipeFaceDetector
from Model.MediapipeDetection.MediapipeFaceLandmarker import FaceMeshDetector
from Model.DetectionFaces import DetectionFaces

class Detector:
    def __init__(self, detector_type, min_detection_conf=0.5, verbose=False, detection_faces=None):
        """
        Initializes the Detector with the specified type and configuration.

        Args:
            detector_type (str): The type of detector to use ('yolov8_onnx', 'yolov8', 'mediapipe').
            model_path (str): The path to the model file (required for YOLO detectors).
            min_detection_conf (float): Minimum confidence value for detection. Defaults to 0.5.
            verbose (bool): Enables verbose output for debugging. Defaults to False.
        """
        self.detector_type = detector_type

        self.detection_faces = detection_faces if detection_faces is not None else DetectionFaces()
        
        if detector_type == 'yolov8_onnx':
            self.detector = Yolov8OnnxRuntimeDetector( verbose=verbose, detection_faces = self.detection_faces)
        elif detector_type == 'yolov8':
            self.detector = Yolov8Detector(detection_faces = self.detection_faces, verbose=verbose)
        elif detector_type == 'mediapipe':
            self.detector = MediapipeFaceDetector(min_detection_conf, verbose=verbose, detection_faces = self.detection_faces)
        else:
            raise ValueError("Invalid detector type. Choose from 'yolov8_onnx', 'yolov8', or 'mediapipe'.")

        self.landmarker = FaceMeshDetector(max_faces=1, min_detection_conf=min_detection_conf, verbose=verbose)


    def detect(self, image):
        """
        Detects faces in the input image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            DetectionFaces: The detection results containing bounding boxes, scores, class IDs, and cropped faces.
        """
        self.detection_faces.reset()  # Reset the detection faces object
        self.detector.detect_faces(image)
        return self.detection_faces

    def draw_detections(self, image):
        """
        Draws bounding boxes on the detected faces in the image.

        Args:
            image (np.ndarray): The input image.
            detection_faces (DetectionFaces): The detection results containing bounding boxes.

        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        return self.detector.draw_detections(image)

    def landmark(self, image):
        """
        Applies landmark detection on the input image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            The landmarks detected by MediaPipe.
        """
        return self.landmarker.landmark(image)
    
    def get_eye_mouth_keypoints(self):

        return self.landmarker.get_eye_mouth_keypoints()

    def draw_landmarks(self, image):
        """
        Draws landmarks on the detected faces in the image.

        Args:
            image (np.ndarray): The input image.
            landmarks: The landmarks detected by MediaPipe.

        Returns:
            np.ndarray: The image with landmarks drawn.
        """
        return self.landmarker.draw_landmarks(image)
