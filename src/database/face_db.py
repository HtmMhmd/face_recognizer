from .handlers.db_handler import UserDatabase

class FaceDatabase:
    """
    A wrapper class for face database operations.
    """
    def __init__(self):
        self.db_handler = UserDatabase()

    def get_all_embeddings(self):
        """Get all user embeddings from the database."""
        return self.db_handler.get_all_embeddings()

    def add_or_update_user(self, username, embedding):
        """Add or update a user in the database."""
        return self.db_handler.add_or_update_user(username, embedding)

    def delete_user(self, username):
        """Delete a user from the database."""
        return self.db_handler.delete_user(username)

    def update_last_login(self, username):
        """Update the last login time for a user."""
        return self.db_handler.update_last_login(username)