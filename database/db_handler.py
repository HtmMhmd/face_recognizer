import sqlite3
import numpy as np
import os

class FaceDatabase:
    def __init__(self, db_path="/data/face_embeddings.db"):
        self.db_path = db_path
        self.db_dir = os.path.dirname(self.db_path)
        os.makedirs(self.db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        embedding BLOB NOT NULL
                    )
                ''')
                conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}")
            print(f"Current permissions: {os.stat(self.db_dir).st_mode}")
            raise

    def add_user(self, username: str, embedding: np.ndarray):
        embedding_blob = embedding
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('INSERT OR REPLACE INTO users (username, embedding) VALUES (?, ?)',
                       (username, embedding_blob))
            conn.commit()

    def get_user(self, username: str) -> np.ndarray:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('SELECT embedding FROM users WHERE username = ?', (username,)).fetchone()
            if result:
                return np.frombuffer(result[0], dtype=np.float32)#.reshape(512, 1)
            raise ValueError(f"User '{username}' not found")

    def delete_user(self, username: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM users WHERE username = ?', (username,))
            conn.commit()
            return cursor.rowcount > 0
        
    def get_all_embeddings(self) -> dict:
        """Retrieves all user embeddings from the database.

        Returns:
            dict: A dictionary where keys are usernames and values are NumPy arrays representing their embeddings.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('SELECT username, embedding FROM users').fetchall()
                return {
                    username: self.get_user(username)
                    for username, _ in results
                }
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise

# Example usage:
if __name__ == '__main__':
    db = FaceDatabase()  # Adjust size as needed

    # Add a user
    embedding = np.array(np.random.rand(512, 1), dtype='float32')
    db.add_user('test_user', embedding)
    emb = (db.get_all_embeddings())
    for user_name, db_embedding in emb.items():
        print(user_name)
        print(db_embedding.shape)
        print(db_embedding)


    # Get the user's embedding
    retrieved_embedding = db.get_user('test_user')
    print(embedding == retrieved_embedding)
    if retrieved_embedding is not None:
        print(f"Retrieved embedding shape: {retrieved_embedding.shape}")
    else:
        print("User not found")
    # Delete the user
    # db.delete_user('hatem')
    db.delete_user('test_user')

