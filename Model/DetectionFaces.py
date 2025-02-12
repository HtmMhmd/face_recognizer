from Model.DetectionResult import DetectionResult

class DetectionFaces:
    def __init__(self):
        """
        Initializes an empty DetectionFaces object containing a DetectionResult object and a list of cropped face images.
        """
        self.detection_result = DetectionResult()
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
        self.detection_result.add(box, score, class_id)
        self.cropped_faces.append(cropped_face)

    def __getitem__(self, index):
        """
        Accesses the DetectionResult and cropped face image by index.

        Args:
            index (int): The index of the detected face.

        Returns:
            tuple: A tuple containing the DetectionResult object and the cropped face image at the specified index.
        """
        if isinstance(index, int):
            return self.detection_result[index], self.cropped_faces[index]
        else:
            raise TypeError("Index must be an integer")

    def reset(self):
        """
        Resets the DetectionFaces object, clearing all detected faces and cropped face images.
        """
        self.detection_result.reset()
        self.cropped_faces = []

    @property
    def n_faces(self):
        """
        Returns the number of detected faces.

        Returns:
            int: The number of detected faces.
        """
        return self.detection_result.n_faces
