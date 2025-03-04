import mediapipe as mp
import time
from typing import Tuple, Dict, List
from Model.MediapipeDetection.mediapipe_utilis import draw_landmarks

class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5, verbose=False):
        
        """
        Initializes the FaceMeshDetector with specified configuration.

        Args:
            max_faces (int): The maximum number of faces to detect. Defaults to 1.
            min_detection_conf (float): Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful. Defaults to 0.5.
            min_tracking_conf (float): Minimum confidence value ([0.0, 1.0]) for the tracking to be considered successful. Defaults to 0.5.
            verbose (bool): Enables verbose output for debugging. Defaults to False.
        """
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.verbose = verbose
        self.landmarks = None
        self.image_shape = []

    def landmark(self, image):
        """
        Processes the input image to detect facial landmarks.
        """
        self.image_shape = image.shape
        results = self.face_mesh.process(image)

        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0]  # Store detected face
        else:
            print("⚠️ Warning: No landmarks detected, keeping previous landmarks.")
            # Do not set self.landmarks to None, retain last known landmarks

        return self.landmarks



    def get_eye_mouth_keypoints(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Extracts the keypoints of the left eye, right eye, and mouth from the facial landmarks.
        """
        eye_mouth_keypoints = {
            "left_eye": [],
            "right_eye": [],
            "mouth": []
        }

        if self.landmarks is None:  # If no face detected, return empty keypoints
            print("No face landmarks detected!")
            return eye_mouth_keypoints

        h, w, _ = self.image_shape

        LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        MOUTH_INDICES = [61, 291, 39, 181, 17, 405]

        for idx in range(len(self.landmarks.landmark)):  # Iterate over valid landmarks
            landmark = self.landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            if idx in LEFT_EYE_INDICES:
                eye_mouth_keypoints["left_eye"].append((cx, cy))
            elif idx in RIGHT_EYE_INDICES:
                eye_mouth_keypoints["right_eye"].append((cx, cy))
            elif idx in MOUTH_INDICES:
                eye_mouth_keypoints["mouth"].append((cx, cy))

        return eye_mouth_keypoints


    def draw_landmarks(self, image):
        """
        Draws the full face mesh and highlights eye landmarks.
        """
        if self.landmarks is None:
            print("No landmarks detected.")
            return image  

        return draw_landmarks(image, self.landmarks)  # Pass the full face landmarks
