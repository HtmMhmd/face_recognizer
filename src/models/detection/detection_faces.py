from src.models.detection.detection_result import DetectionResult

class DetectionFaces(DetectionResult):
    def __init__(self):
        """
        Initializes an empty DetectionFaces object containing a DetectionResult object and a list of cropped face images.
        """
        super().__init__()
        self.cropped_faces = []

    def add(self, box, score, class_id, cropped_face):
        """
        Adds a detected face's bounding box, score, class ID, and cropped face image to the DetectionFaces object.

        Args:
            box (tuple): The bounding box of the detected face.
            score (float): The confidence score of the detected face.
            class_id (int): The class ID of the detected face.
            cropped_face (np.ndarray): The cropped face image.
        """
        super().add(box, score, class_id)
        self.cropped_faces.append(cropped_face)

    def __getitem__(self, key):
        """
        Accesses the DetectionResult and cropped face image by index or key.

        Args:
            key (int or str): The index of the detected face or the key for accessing attributes.

        Returns:
            tuple or list: A tuple containing the DetectionResult object and the cropped face image at the specified index, or a list of attributes.
        """
        if isinstance(key, int):
            return super().__getitem__(key), self.cropped_faces[key]
        elif isinstance(key, str):
            if key == "cropped_faces":
                return self.cropped_faces
            else:
                return super().__getitem__(key)
        else:
            raise TypeError("Index must be an integer or string")

    def reset(self):
        """
        Resets the DetectionFaces object, clearing all detected faces and cropped face images.
        """
        super().reset()
        self.cropped_faces = []

    def filter_largest_face(self):
        """
        Filters the detection results to keep only the largest face.
        
        Returns:
            self: The filtered detection results with only the largest face
        """
        import logging
        logger = logging.getLogger("Detection_Faces")
        
        if not self.boxes or len(self.boxes) == 0:
            logger.info("No faces detected to filter")
            return self
        
        # Find the largest bounding box by area
        largest_idx = -1
        largest_area = 0
        
        for i, bbox in enumerate(self.boxes):
            # Calculate box area (width * height)
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            
            if area > largest_area:
                largest_area = area
                largest_idx = i
        
        if largest_idx >= 0:
            logger.info(f"Filtering to keep only the largest face (index {largest_idx}) with area {largest_area}")
            
            # Keep only the largest face
            self.boxes = [self.boxes[largest_idx]] if largest_idx < len(self.boxes) else []
            self.scores = [self.scores[largest_idx]] if largest_idx < len(self.scores) else []
            # if largest_idx < len(self.class_id) :
            #     self.class_id = [self.class_id[largest_idx]]
            # else :
            #     pass
            self.cropped_faces = [self.cropped_faces[largest_idx]] if largest_idx < len(self.cropped_faces) else []
        
        return self
