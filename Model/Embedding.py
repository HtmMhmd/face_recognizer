class Embedding:
    def __init__(self, bbox, embedding):
        self.bbox = bbox
        self.embedding = embedding

    def __getitem__(self, key):
        if key == "bbox":
            return self.bbox
        elif key == "embedding":
            return self.embedding
        else:
            raise KeyError(f"Invalid key: {key}")

class EmbeddingContainer:
    def __init__(self):
        """
        Initializes an empty EmbeddingContainer contains a list of bbox and embedding objects.
        """
        self.embeddings = []  # List of Embedding objects

    def add(self, bbox, embedding):
        self.embeddings.append(Embedding(bbox, embedding))

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.embeddings[index]
        elif isinstance(index, str):
            if index == "bbox":
                return [embedding.bbox for embedding in self.embeddings]
            elif index == "embedding":
                return [embedding.embedding for embedding in self.embeddings]
            else:
                raise KeyError(f"Invalid key: {index}")
        else:
            raise TypeError("Index must be an integer or a string")

    def __len__(self):
        return len(self.embeddings)