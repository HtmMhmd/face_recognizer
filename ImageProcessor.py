import numpy as np
import cv2
from typing import List, Tuple

from Verify.Verify import FaceVerifier
from Align.Align import FaceAligner
from ImageUtilis.image_utilis import preprocess_image

# from UsersDatabaseHandeler.UsersDatabaseHandeler import EmbeddingCSVHandler
from Model.Detector import Detector
from Model.DetectionEmbedding import DetectionEmbedding
from Model.FaceNet.FaceNetTFLiteHandler import FaceNetTFLiteHandler
from Model.DetectionFaces import DetectionFaces
from database.db_handler import FaceDatabase
from Model.detection_utilis import draw_user_names_on_bboxes

class ImageProcessor:
    def __init__(self, model_architecture='mediapipe', verbose=False, detection_embedding=None):
        """
        Initializes the ImageProcessor with the specified configuration.

        Args:
            model_architecture (str): The model architecture to use for detection ('yoloonnx', 'mediapipe', or 'ediapipe'). Defaults to 'yoloonnx'.
            verbose (bool): Enables verbose output for debugging. Defaults to False.
            detection_embedding (DetectionEmbedding): Optional DetectionEmbedding object to store detection results and embeddings.
        """
        self.verbose = verbose
        self.detection_embedding = detection_embedding if detection_embedding is not None else DetectionEmbedding(DetectionFaces(), [])
        
        self.detector = Detector(detector_type=model_architecture, detection_faces=self.detection_embedding.detection_faces, verbose=verbose)
        self.facenet = FaceNetTFLiteHandler(verbose=verbose)
        self.detection_embedding = DetectionEmbedding()

    def process_image(self, image):
        """
        Processes the input image to detect faces and extract embeddings.

        Args:
            image (np.ndarray): The input image.

        Returns:
            DetectionEmbedding: The detection results containing bounding boxes, scores, class IDs, cropped faces, and embeddings.
        """
        detection_faces = self.detector.detect(image)
        
        embeddings = []
        for cropped_face in detection_faces.cropped_faces:
            embedding = self.facenet.forward(preprocess_image(cropped_face))
            embeddings.append(embedding)
        self.detection_embedding.assign(detection_faces, embeddings)

        return self.detection_embedding

    def detect_landmarks(self, image):
        """
        Detects facial landmarks in the given image.

        Args:
            image: The input image.

        Returns:
            The image with landmarks drawn on it.
        """
        self.landmarks = self.detector.landmark(image)
        return self.landmarks
    
    def verify_faces(self) -> List[dict]:
        """
        Verifies faces in an image against the user database.

        Returns:
            List[dict]: A list of dictionaries containing bounding boxes, user names, and verification results.
        """
        database_handler = FaceDatabase()
        face_verifier = FaceVerifier()

        results = []
        for bbox, embedding in zip(self.detection_embedding.detection_faces.boxes, self.detection_embedding.embeddings):
            # Get all embeddings from database
            all_embeddings = database_handler.get_all_embeddings()
            for user_name, db_embedding in all_embeddings.items():
                print(f"Verifying face with user {user_name}")
                verification_result = face_verifier.verify_faces(embedding, db_embedding, verbose=self.verbose)
                if verification_result['cosine']['verified'] and \
                   verification_result['euclidean']['verified'] and \
                   verification_result['euclidean_l2']['verified']:
                    results.append({
                        'bbox': bbox,
                        'user_name': user_name,
                        'verification_result': verification_result
                    })
        return results

    def align_faces(self, image):
        """
        Aligns faces in an image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            The aligned image.
        """
        face_aligner = FaceAligner()

        return face_aligner.align_faces(image)
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
    
    def draw_landmarks(self, image): 
        """
        Draws landmarks on the detected faces in the image.

        Args:
            image (np.ndarray): The input image.
            landmarks: The landmarks detected by MediaPipe.

        Returns:
            np.ndarray: The image with landmarks drawn.
        """
        return self.detector.draw_landmarks(image)
    
    def draw_user_names(self, image, results):
        """
        Draws user names on the bounding boxes of detected faces.

        Args:
            image (np.ndarray): The input image.
            results (List[dict]): The results of face verification.

        Returns:
            np.ndarray: The image with user names drawn on the bounding boxes.
        """
        return draw_user_names_on_bboxes(image, results)
