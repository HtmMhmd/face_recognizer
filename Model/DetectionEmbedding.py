from Model.DetectionFaces import DetectionFaces

class DetectionEmbedding:
    def __init__(self, detection_faces=None, embeddings=None):
        """
        Initializes a DetectionEmbedding object containing DetectionFaces and a list of embeddings.

        Args:
            detection_faces (DetectionFaces): The DetectionFaces object containing detected faces.
            embeddings (list): A list of embeddings corresponding to the detected faces.
        """
        
        self.detection_faces = detection_faces if detection_faces is not None else DetectionFaces()
        self.embeddings = embeddings if embeddings is not None else []

    def assign(self, detection_faces, embeddings):

        # self.reset()

        # if len(self.detection_faces) != len(self.embeddings):
        #     raise ValueError("The number of embeddings must match the number of detected faces.")
        
        self.detection_faces = detection_faces
        self.embeddings = embeddings
    
    def add(self, box, score, class_id, cropped_face, embedding):
        """
        Adds a detected face's bounding box, score, class ID, cropped face image, and embedding to the DetectionEmbedding object.

        Args:
            box (tuple): The bounding box of the detected face.
            score (float): The confidence score of the detected face.
            class_id (int): The class ID of the detected face.
            cropped_face (np.ndarray): The cropped face image.
            embedding (np.ndarray): The embedding of the detected face.
        """
        self.detection_faces.add(box, score, class_id, cropped_face)
        self.embeddings.append(embedding)

    def __getitem__(self, key):
        """
        Accesses the DetectionFaces and embedding by index or key.

        Args:
            key (int or str): The index of the detected face or the key for accessing attributes.

        Returns:
            tuple or list: A tuple containing the DetectionFaces object and the embedding at the specified index, or a list of attributes.
        """
        if isinstance(key, int):
            return DetectionEmbedding(self.detection_faces[key], self.embeddings[key])
        elif isinstance(key, str):
            if key == "embeddings":
                return self.embeddings
            else:
                return self.detection_faces[key]
        else:
            raise TypeError("Index must be an integer or string")
        
    def __len__(self):
        """
        Returns the number of detected faces.

        Returns:
            int: The number of detected faces.
        """
        return len(self.embeddings)

    def reset(self):
        """
        Resets the DetectionEmbedding object, clearing all detected faces, cropped face images, and embeddings.
        """
        self.detection_faces.reset()
        self.embeddings = []
