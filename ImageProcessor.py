import cv2
import numpy as np
from Verify.Verify import FaceVerifier
from Align.Align import FaceAligner
from ImageUtilis.image_utilis import *
# from Model.Detection.YoloDetector.YoloDetector import YOLOv8Detector
from Model.Detection.detection_utilis import get_cropped_faces
from Model.FaceNet.FaceNetTFLiteHandler import FaceNetTFLiteHandler
from Model.Detection.OpencvDetector import OpencvDetector
from Model.Detection.YoloV8OnnxRuntime.Yolov8OnnxRuntimeDetector import Yolov8OnnxRuntimeDetector
from Model.Embedding import *
# from Model.FaceNet.Facenet import *


class ImageProcessor:
    def __init__(self, use_yolo: bool = False, verbose: bool = False):
        self.use_yolo = use_yolo
        self.facenet_handler = FaceNetTFLiteHandler(verbose = verbose)
        if self.use_yolo:
            self.detection_model = Yolov8OnnxRuntimeDetector(verbose = verbose)
        else:
            self.detection_model = OpencvDetector()
        self.Embeddings = EmbeddingContainer()

    def apply_detection_model(self, image: np.ndarray):
        results = self.detection_model.detect_faces(image)
        # Check if results is not empty
        image_cropped = get_cropped_faces(image, results)
        zip_results = zip(results, image_cropped)
        return zip_results

    def apply_opencv_face(self, image: np.ndarray):
        results = self.opencv_model.detect_faces(image)
        # Check if results is not empty
        image_cropped = get_cropped_faces(image, results)
        zip_results = zip(results, image_cropped)
        return zip_results

    def process_image(self, image):

        if isinstance(image, str) and os.path.isfile(image):
            image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image}")
        
        if self.use_yolo:
            images = self.apply_detection_model(image)
        else:
            images = self.apply_opencv_face(image)

        for bbox, face in images:
            face = preprocess_image(face)
            embedding = self.facenet_handler.forward(face.astype(np.float32))
            self.Embeddings.add(bbox.boxes[0], embedding)
        return self.Embeddings
