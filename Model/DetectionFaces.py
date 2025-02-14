from Model.DetectionResult import DetectionResult

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
