import sqlite3
import numpy as np
import os
from datetime import datetime

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
                        embedding BLOB NOT NULL,
                        date_added TEXT NOT NULL,
                        last_login TEXT NOT NULL
                    )
                ''')
                conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}")
            print(f"Current permissions: {os.stat(self.db_dir).st_mode}")
            raise

    def add_user(self, username: str, embedding: np.ndarray):
        date_added = datetime.now().strftime("%c")
        last_login = date_added
        embedding_blob = embedding
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('INSERT OR REPLACE INTO users (username, embedding, date_added, last_login) VALUES (?, ?, ?, ?)',
                       (username, embedding_blob, date_added, last_login))
            conn.commit()

    def get_user(self, username: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('SELECT embedding, date_added, last_login FROM users WHERE username = ?', (username,)).fetchone()
            if result:
                return {
                    "embedding": np.frombuffer(result[0], dtype=np.float32),
                    "date_added": result[1],
                    "last_login": result[2]
                }
            raise ValueError(f"User '{username}' not found")

    def update_last_login(self, username: str):
        last_login = datetime.now().strftime("%c")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE users SET last_login = ? WHERE username = ?', (last_login, username))
            conn.commit()

    def delete_user(self, username: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM users WHERE username = ?', (username,))
            conn.commit()
            return cursor.rowcount > 0
        
    def get_all_embeddings(self) -> dict:
        """Retrieves all user embeddings from the database.

        Returns:
            dict: A dictionary where keys are usernames and values are dictionaries containing their embeddings, date_added, and last_login.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('SELECT username, embedding, date_added, last_login FROM users').fetchall()
                return {
                    username: {
                        "embedding": np.frombuffer(embedding, dtype=np.float32),
                        "date_added": date_added,
                        "last_login": last_login
                    }
                    for username, embedding, date_added, last_login in results
                }
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise

    def get_all_users(self) -> list:
        """
        Get a list of all usernames in the database.
        
        Returns:
            list: List of usernames
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('SELECT username FROM users').fetchall()
            # Flatten the result list of tuples into a simple list of usernames
            return [username[0] for username in result]

# Example usage:
if __name__ == '__main__':
    db = FaceDatabase()  # Adjust size as needed

    # Add a user
    embedding = np.array(np.random.rand(512, 1), dtype='float32')
    db.add_user('test_user', embedding)
    emb = (db.get_all_embeddings())
    for user_name, db_embedding in emb.items():
        print(user_name)
        print(db_embedding['embedding'].shape)
        print(db_embedding)


    # Get the user's embedding
    retrieved_embedding = db.get_user('test_user')
    print(embedding == retrieved_embedding['embedding'])
    if retrieved_embedding is not None:
        print(f"Retrieved embedding shape: {retrieved_embedding['embedding'].shape}")
    else:
        print("User not found")
    # Delete the user
    # db.delete_user('hatem')
    # db.delete_user('test_user')

