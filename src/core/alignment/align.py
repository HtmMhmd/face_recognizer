import numpy as np
import cv2
from typing import Tuple
from src.core.alignment.eye_detect import EyeDetector
from src.config import alignment_settings

class FaceAligner:
    def __init__(self):
        self.eye_detector = EyeDetector()
        # Get minimum alignment degree from config
        self.min_degree = alignment_settings.get("min_alignment_degree", 5.0)

    def align_face(self, img: np.ndarray, min_degree: float = None) -> np.ndarray:
        # Use configured min_degree if not provided
        if min_degree is None:
            min_degree = self.min_degree
            
        eye_region = self.eye_detector.detect_eyes(img)
        if eye_region is None:
            return None

        left_eye_center = eye_region.left_eye
        right_eye_center = eye_region.right_eye

        # Calculate the angle between the eyes
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = 180- np.degrees(np.arctan2(dy, dx))
        if angle < min_degree:
            return None

        # Calculate the center between the two eyes
        eyes_center = (int((left_eye_center[0] + right_eye_center[0]) // 2),
                       int((left_eye_center[1] + right_eye_center[1]) // 2))

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

        # Apply the affine transformation
        aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

        return aligned_img