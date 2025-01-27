import numpy as np
from Model.FaceDetection import FaceDetector
import cv2

class OpenCVHaarCascadeDetector(FaceDetector):
    """
    Face detection using OpenCV's Haar cascade classifier.
    """
    def __init__(self, cascade_path="faces.xml"):
        super().__init__()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Could not load Haar cascade classifier from {cascade_path}")

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        self.boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in faces])
        return self.boxes

