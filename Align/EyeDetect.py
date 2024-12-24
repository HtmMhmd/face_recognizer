import cv2
from typing import Tuple
import numpy as np

class FacialAreaRegion:
    def __init__(self, left_eye: Tuple[int, int], right_eye: Tuple[int, int]):
        self.left_eye = left_eye
        self.right_eye = right_eye

class EyeDetector:
    def __init__(self):
        self.eye_cascade = self.__build_cascade()

    def __build_cascade(self) -> cv2.CascadeClassifier:

        return  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


    def detect_eyes(self, img: np.ndarray) -> FacialAreaRegion:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 10)

        eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

        # ----------------------------------------------------------------
        if len(eyes) >= 2:
            # decide left and right eye

            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                right_eye = eye_1
                left_eye = eye_2
            else:
                right_eye = eye_2
                left_eye = eye_1

            # -----------------------
            # find center of eyes
            left_eye = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2)),
            )
            right_eye = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2)),
            )

            return FacialAreaRegion(left_eye, right_eye)
        return None