import cv2
import mediapipe as mp
from typing import Tuple, Union, Dict, List

class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        """
        Initializes the FaceMeshDetector with specified configuration.

        Args:
            max_faces (int): The maximum number of faces to detect. Defaults to 1.
            min_detection_conf (float): Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful. Defaults to 0.5.
            min_tracking_conf (float): Minimum confidence value ([0.0, 1.0]) for the tracking to be considered successful. Defaults to 0.5.
        """

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )

    def get_landmarks(self, image):
        
        """
        Processes the input image to detect facial landmarks.

        Args:
            image (np.ndarray): The input image in which facial landmarks are to be detected.

        Returns:
            A mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList object containing
            the detected facial landmarks.
        """

        
        print("Image shape: ", image.shape)
        results = self.face_mesh.process(image)
        return results

    def get_eye_mouth_keypoints(self, face_landmarks, image_shape) -> Dict[str, List[Tuple[int, int]]]:
        """
        Extracts the keypoints of the left eye, right eye, and mouth from the facial landmarks.

        Args:
            face_landmarks (mp.solutions.face_mesh.FaceMesh.LandmarkList):
                The facial landmarks detected by the FaceMesh model.
            image_shape (Tuple[int, int, int]):
                The shape of the image in which the facial landmarks were detected.

        Returns:
            A dictionary with the following keys: "left_eye", "right_eye", and "mouth".
            The values are lists of tuples, each tuple containing the x and y coordinates of
            the corresponding keypoints.
        """
        eye_mouth_keypoints = {
            "left_eye": [],
            "right_eye": [],
            "mouth": []
        }
        h, w = image_shape

        LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  
        RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  
        MOUTH_INDICES = [61, 291, 39, 181, 17, 405]

        for idx, landmark in enumerate(face_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            if idx in LEFT_EYE_INDICES:
                eye_mouth_keypoints["left_eye"].append((cx, cy))
            if idx in RIGHT_EYE_INDICES:
                eye_mouth_keypoints["right_eye"].append((cx, cy))
            if idx in MOUTH_INDICES:
                eye_mouth_keypoints["mouth"].append((cx, cy))
        return eye_mouth_keypoints
