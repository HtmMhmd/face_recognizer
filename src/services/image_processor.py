from typing import List, Tuple
import numpy as np

class ImageProcessor:
    def __init__(self, model_architecture='mediapipe', verbose=False, detection_embedding=None):
        """
        Initializes the ImageProcessor with the specified configuration.

        Args:
            model_architecture (str): The model architecture to use for detection ('yolov8_onnx', 'yolov8', 'mediapipe'). 
                                      Defaults to 'mediapipe'.
            verbose (bool): Enables verbose output for debugging. Defaults to False.
            detection_embedding (DetectionEmbedding): Optional DetectionEmbedding object to store detection results and embeddings.
        """
        from src.models.detection import Detector, DetectionEmbedding, DetectionFaces
        from src.models.face_recognition import FaceNetTFLiteHandler
        
        self.verbose = verbose
        self.detection_embedding = detection_embedding if detection_embedding is not None else DetectionEmbedding(DetectionFaces(), [])
        
        self.detector = Detector(detector_type=model_architecture, detection_faces=self.detection_embedding.detection_faces, verbose=verbose)
        self.facenet = FaceNetTFLiteHandler(verbose=verbose)
        self.detection_embedding = DetectionEmbedding()
        self.landmarks = None

    def process_image(self, image):
        """
        Processes the input image to detect faces and extract embeddings.

        Args:
            image (np.ndarray): The input image.

        Returns:
            DetectionEmbedding: The detection results containing bounding boxes, scores, class IDs, cropped faces, and embeddings.
        """
        from src.utils.image import preprocess_image
        
        detection_faces = self.detector.detect(image)
        
        embeddings = []
        for i, cropped_face in enumerate(detection_faces.cropped_faces):
            try:
                if cropped_face is None or cropped_face.size == 0:
                    print(f"Warning: Invalid cropped face at index {i}")
                    continue
                    
                embedding = self.facenet.forward(preprocess_image(cropped_face))
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing face {i}: {str(e)}")
                # Skip this face and continue with others
                continue
                
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
    
    def get_eye_mouth_keypoints(self):
        """
        Gets eye and mouth keypoints for drowsiness detection.

        Returns:
            Dictionary containing eye and mouth keypoints.
        """
        return self.detector.get_eye_mouth_keypoints()

    def verify_faces(self) -> List[dict]:
        """
        Verifies faces in an image against the user database.

        Returns:
            List[dict]: A list of dictionaries containing bounding boxes, user names, and verification results.
        """
        from src.database.face_db import FaceDatabase
        from src.core.verification import FaceVerifier
        
        database_handler = FaceDatabase()
        face_verifier = FaceVerifier()

        results = []
        for bbox, embedding in zip(self.detection_embedding.detection_faces.boxes, self.detection_embedding.embeddings):
            # Get all embeddings from database
            all_embeddings = database_handler.get_all_embeddings()
            for user_name, db_data in all_embeddings.items():
                db_embedding = db_data['embedding']
                verification_result = face_verifier.verify_faces(embedding, db_embedding, verbose=self.verbose)
                if verification_result['cosine']['verified'] and \
                   verification_result['euclidean']['verified'] and \
                   verification_result['euclidean_l2']['verified']:
                    # Update last_login field
                    database_handler.update_last_login(user_name)
                    results.append({
                        'bbox': bbox,
                        'user_name': user_name,
                        'verification_result': verification_result
                    })
                    if self.verbose:
                        print(f"Face verified as user: {user_name}")
        return results

    def align_faces(self, image):
        """
        Aligns faces in an image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            The aligned image.
        """
        from src.core.alignment import FaceAligner
        face_aligner = FaceAligner()
        return face_aligner.align_faces(image)

    def draw_detections(self, image):
        """
        Draws bounding boxes on the detected faces in the image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        return self.detector.draw_detections(image)
    
    def draw_landmarks(self, image): 
        """
        Draws landmarks on the detected faces in the image.

        Args:
            image (np.ndarray): The input image.

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
        from src.utils.image.drawing import draw_user_names_on_bboxes
        return draw_user_names_on_bboxes(image, results)