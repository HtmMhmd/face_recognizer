import cv2
import numpy as np
from typing import List, Tuple

from Verify.Verify import FaceVerifier
from Align.Align import FaceAligner
from ImageUtilis.image_utilis import *

# from Model.Detection.YoloDetector.YoloDetector import YOLOv8Detector
from Model.Detection.detection_utilis import get_cropped_faces
from Model.FaceNet.FaceNetTFLiteHandler import FaceNetTFLiteHandler
from Model.Detection.OpencvDetector import OpencvDetector
from Model.Detection.YoloV8OnnxRuntime.Yolov8OnnxRuntimeDetector import Yolov8OnnxRuntimeDetector
from Model.Landmark.Landmarker import FaceMeshDetector
from Model.Landmark.utilis import draw_landmarks
from Model.Detection.detection_utilis import draw_detections
from Model.Embedding import *
# from Model.FaceNet.Facenet import *
# from Model.FaceNet.Facenet import FaceNetTFLiteClient

from UsersDatabaseHandeler.UsersDatabaseHandeler import EmbeddingCSVHandler

class ImageProcessor:
    def __init__(self, use_yolo: bool = False, verbose: bool = False):
        """
        Initializes the ImageProcessor with a face detection model and an embedding container.

        Args:
            use_yolo (bool): Indicates whether to use the YOLOv8 model for face detection.
            verbose (bool): Enables verbose output for debugging.

        """
        # Use YOLOv8 for detection if specified, otherwise use OpenCV
        self.use_yolo = use_yolo
        self.facenet_handler = FaceNetTFLiteHandler(verbose=True)
        self.face_mesh_detector = FaceMeshDetector()

        # Initialize the detection model based on the use_yolo flag
        if self.use_yolo:
            self.detection_model = Yolov8OnnxRuntimeDetector(verbose=verbose)
        else:
            self.detection_model = OpencvDetector()



    def apply_detection_model(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Applies the face detection model to the given image.

        Args:
            image: The input image.

        Returns:
            A list of tuples containing the bounding box coordinates and the cropped face image.
        """
        results = self.detection_model.detect_faces(image)
        # Check if results is not empty
        if len(results) == 0:
            return []
        image_cropped = get_cropped_faces(image, results)
        zip_results = list(zip(results, image_cropped))
        return zip_results

    def apply_opencv_face(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Applies the OpenCV face detection model to the given image.

        Args:
            image: The input image.

        Returns:
            A list of tuples containing the bounding box coordinates and the cropped face image.
        """
        results = self.detection_model.detect_faces(image)
        # Check if results is not empty
        if len(results) == 0:
            return []
        image_cropped = get_cropped_faces(image, results)
        zip_results = list(zip(results, image_cropped))
        return zip_results

    def process_image(self, image):
        """
        Processes an image to detect faces and extract embeddings.

        Args:
            image: The input image, either as a file path or a NumPy array.

        Returns:
            An EmbeddingContainer containing the face embeddings.
        """
        # Initialize the embedding container to store face embeddings
        self.Embeddings = EmbeddingContainer()
        # Load the image if a file path is provided
        if isinstance(image, str) and os.path.isfile(image):
            image = cv2.imread(image)

        # Raise an error if the image could not be loaded
        if image is None:
            raise FileNotFoundError(f"Image not found at {image}")
        
        # Apply the appropriate face detection model based on the use_yolo flag
        if self.use_yolo:
            images = self.apply_detection_model(image)
        else:
            images = self.apply_opencv_face(image)

        # Iterate over detected faces and extract embeddings
        for bbox, face in images:
            # Display the image

            # cv2.imshow('Image', face)
            # # Wait for the 'q' key to be pressed
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # # Close all OpenCV windows
            # cv2.destroyAllWindows()

            # Preprocess the cropped face image
            face = preprocess_image(face)
            # Get the embedding from the FaceNet model
            embedding = self.facenet_handler.forward(face.astype(np.float32))
            # Add the bounding box and embedding to the container
            self.Embeddings.add(bbox.boxes[0], embedding)

        return self.Embeddings

    def detect_landmarks(self, image):
        """
        Detects facial landmarks in the given image.

        Args:
            image: The input image.

        Returns:
            The image with landmarks drawn on it.
        """
        landmarks = self.face_mesh_detector.get_landmarks(image)
        return landmarks
    
    def verify_faces(self, image) -> List[dict]:
        """
        Verifies faces in an image against the user database.

        Args:
            image (np.ndarray): The input image.

        Returns:
            List[dict]: A list of dictionaries containing bounding boxes, user names, and verification results.
        """
        database_handeler = EmbeddingCSVHandler()

        # Load the image if a file path is provided
        if isinstance(image, str) and os.path.isfile(image):
            image = cv2.imread(image)

        # Raise an error if the image could not be loaded
        if image is None:
            raise FileNotFoundError(f"Image not found at {image}")

        faces = self.process_image(image)
        face_verifier = FaceVerifier()

        results = []
        for face in faces:

            embedding = face['embedding']
            for i in range(len(database_handeler)):
                db_embedding, user_name = database_handeler.read_embedding(i)
                verification_result = face_verifier.verify_faces(embedding, db_embedding, verbose=False)
                results.append({
                    'bbox': face['bbox'],
                    'user_name': user_name,
                    'verification_result': verification_result
                })
        return results