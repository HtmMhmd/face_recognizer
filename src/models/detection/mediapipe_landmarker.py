import mediapipe as mp
import time
from typing import Tuple, Dict, List
import logging
def draw_landmarks(image, landmarks):
    """
    Draws facial landmarks on the image.
    
    Args:
        image: Input image
        landmarks: Mediapipe facial landmarks
        
    Returns:
        Image with landmarks drawn
    """
    import cv2
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    if landmarks is None:
        return image
        
    image_copy = image.copy()
    
    # Draw the face mesh
    mp_drawing.draw_landmarks(
        image=image_copy,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
    )
    
    # Draw eyes and eyebrows
    mp_drawing.draw_landmarks(
        image=image_copy,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )
    
    return image_copy

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
            logging.warning("⚠️ Warning: No landmarks detected, keeping previous landmarks.")
            # Do not set self.landmarks to None, retain last known landmarks

        return self.landmarks



    def get_eye_mouth_keypoints(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Extracts the keypoints of the left eye, right eye, and mouth from the facial landmarks.
        """
        eye_mouth_keypoints = {
            "left_eye": [],
            "right_eye": [],
            "mouth": [],
            "nose": []
        }

        if self.landmarks is None:  # If no face detected, return empty keypoints
            logging.warning("No face landmarks detected!")
            return eye_mouth_keypoints

        h, w = self.image_shape[0], self.image_shape[1]

        LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        MOUTH_INDICES = [61, 291, 39, 181, 17, 405]
        NOSE_INDICES = [1, 4, 5, 6, 197, 198]  # Central nose landmarks

        for idx in range(len(self.landmarks.landmark)):  # Iterate over valid landmarks
            landmark = self.landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            if idx in LEFT_EYE_INDICES:
                eye_mouth_keypoints["left_eye"].append((cx, cy))
            elif idx in RIGHT_EYE_INDICES:
                eye_mouth_keypoints["right_eye"].append((cx, cy))
            elif idx in MOUTH_INDICES:
                eye_mouth_keypoints["mouth"].append((cx, cy))
            elif idx in NOSE_INDICES:
                eye_mouth_keypoints["nose"].append((cx, cy))

        return eye_mouth_keypoints


    def draw_landmarks(self, image):
        """
        Draws the full face mesh and highlights eye landmarks.
        """
        if self.landmarks is None:
            logging.warning("No landmarks detected.")
            return image  

        return draw_landmarks(image, self.landmarks)  # Pass the full face landmarks

    def calculate_gaze_direction(self):
        """
        Calculates the distance between the nose and both eyes to determine gaze direction.
        
        Returns:
            dict: Contains gaze direction info and eye-nose distances
                'direction': 'left', 'right', 'center', or 'unknown'
                'left_eye_nose_dist': distance from left eye to nose
                'right_eye_nose_dist': distance from right eye to nose
                'ratio': ratio between the two distances
        """
        import numpy as np
        import math
        
        keypoints = self.get_eye_mouth_keypoints()
        
        if not keypoints["left_eye"] or not keypoints["right_eye"] or not keypoints["nose"]:
            return {"direction": "unknown", "left_eye_nose_dist": 0, "right_eye_nose_dist": 0, "ratio": 1.0}
        
        # Calculate center points for each feature
        left_eye_center = np.mean(keypoints["left_eye"], axis=0)
        right_eye_center = np.mean(keypoints["right_eye"], axis=0)
        nose_center = np.mean(keypoints["nose"], axis=0)
        
        # Calculate Euclidean distances
        left_eye_nose_dist = math.sqrt((left_eye_center[0] - nose_center[0])**2 + 
                                     (left_eye_center[1] - nose_center[1])**2)
        right_eye_nose_dist = math.sqrt((right_eye_center[0] - nose_center[0])**2 + 
                                      (right_eye_center[1] - nose_center[1])**2)
        
        # Calculate ratio between distances (normalize)
        total_distance = left_eye_nose_dist + right_eye_nose_dist
        if total_distance > 0:
            left_ratio = left_eye_nose_dist / total_distance
            right_ratio = right_eye_nose_dist / total_distance
        else:
            left_ratio = right_ratio = 0.5
        
        # Determine gaze direction based on the difference in distances
        # When looking left, distance to right eye increases
        # When looking right, distance to left eye increases
        threshold = 0.08  # Threshold to determine if looking left/right or center
        ratio = left_eye_nose_dist / max(right_eye_nose_dist, 0.1)  # Prevent division by zero
        
        if ratio > 1 + threshold:
            direction = "right"  # Looking right (nose closer to left eye)
        elif ratio < 1 - threshold:
            direction = "left"   # Looking left (nose closer to right eye)
        else:
            direction = "center" # Looking straight ahead
        logging.debug(f"Left Eye-Nose Distance: {left_eye_nose_dist}, Right Eye-Nose Distance: {right_eye_nose_dist}, Ratio: {ratio}, Direction: {direction}")
        # Return gaze direction and distances
        return {
            "direction": direction,
            "left_eye_nose_dist": float(left_eye_nose_dist),
            "right_eye_nose_dist": float(right_eye_nose_dist),
            "ratio": float(ratio)
        }
