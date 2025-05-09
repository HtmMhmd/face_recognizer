import requests
import numpy as np
import base64
import os

class DatabaseClient:
    def __init__(self):
        self.base_url = os.getenv('DB_API_URL', 'http://localhost:5000')

    def add_user(self, username: str, embedding: np.ndarray):
        embedding_bytes = embedding.tobytes()
        embedding_b64 = base64.b64encode(embedding_bytes).decode()
        
        response = requests.post(f"{self.base_url}/user", 
                               json={"username": username, "embedding": embedding_b64})
        return response.json()

    def get_user(self, username: str) -> np.ndarray:
        response = requests.get(f"{self.base_url}/user/{username}")
        if response.status_code == 200:
            embedding_b64 = response.json()['embedding']
            embedding_bytes = base64.b64decode(embedding_b64)
            return np.frombuffer(embedding_bytes, dtype=np.float64).reshape(512, 1)
        return None

    def get_all_embeddings(self) -> dict:
        response = requests.get(f"{self.base_url}/users")
        embeddings = {}
        for username, emb_b64 in response.json().items():
            embedding_bytes = base64.b64decode(emb_b64)
            embeddings[username] = np.frombuffer(embedding_bytes, dtype=np.float64).reshape(256, 1)
        return embeddings
