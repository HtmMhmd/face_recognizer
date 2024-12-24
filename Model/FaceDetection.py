from abc import ABC, abstractmethod
import numpy as np

class FaceDetector(ABC):
    """
    Abstract base class for face detection.
    """
    def __init__(self):
        self.boxes = np.zeros((0,4)) # Initialize bounding boxes to an empty array

    @abstractmethod
    def detect_faces(self, image):
        """
        Detects faces in an image.

        Args:
            image: The input image (as a NumPy array).

        Returns:
            A NumPy array of bounding boxes (x_min, y_min, x_max, y_max).  Returns an empty array if no faces are detected.
        """
        pass

