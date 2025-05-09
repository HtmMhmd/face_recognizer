import csv
import os
from typing import List, Tuple
from datetime import datetime
from src.config import database_settings, get_path

class CSVFileChecker:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.check_and_create_csv()

    def check_and_create_csv(self):
        if not os.path.isfile(self.csv_file):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = [f'embedding_{i}' for i in range(512)] + ['user_name', 'last_login']
                writer.writerow(header)

class EmbeddingCSVHandler:
    def __init__(self, csv_file: str = None):
        # Use database path from configuration
        self.csv_file = csv_file or get_path("database.user_embeddings_file")
        CSVFileChecker(self.csv_file)  # Ensure the CSV file exists and is properly formatted
        self.file = None
        self.open_file('r')

    def open_file(self, mode: str):
        self.file = open(self.csv_file, mode=mode, newline='')

    def close_file(self):
        if self.file:
            self.file.close()
            self.file = None

    def write_embedding(self, embedding: List[float], user_name: str):
        if not isinstance(embedding, list) or len(embedding) != 512:
            raise ValueError("Embedding must be a list of 512 floats")
        if not isinstance(user_name, str):
            raise ValueError("User name must be a string")
        
        # Ensure the file is open for writing
        self.close_file()
        if self.file is None:
            self.open_file('a')
        writer = csv.writer(self.file)
        writer.writerow(embedding + [user_name])
        self.close_file()

    def read_embedding(self, index: int) -> Tuple[List[float], str]:
        index += 1  # Add 1 for the header row

        if self.file is None:
            self.open_file('r')
        reader = csv.reader(self.file)
        for i, row in enumerate(reader):
            if i == index:
                embedding = list(map(float, row[:-1]))
                user_name = row[-1]
                self.close_file()
                return embedding, user_name
        self.close_file()
        raise IndexError("Index out of range")

    def __len__(self) -> int:
        if self.file is None:
            self.open_file('r')
        reader = csv.reader(self.file)
        length = sum(1 for row in reader) - 1  # Subtract 1 for the header row
        self.close_file()
        return length

class UsersDatabaseHandeler:
    def __init__(self, csv_file: str = "data/users_embeddings.csv"):
        """
        Initialize the UsersDatabaseHandeler with a CSV file.
        
        Args:
            csv_file: Path to the CSV file containing user embeddings
        """
        self.csv_handler = EmbeddingCSVHandler(csv_file)
        
    def get_all_embeddings(self):
        """
        Get all embeddings from the database.
        
        Returns:
            dict: A dictionary mapping user names to their embeddings and metadata
        """
        embeddings_dict = {}
        
        self.csv_handler.close_file()
        self.csv_handler.open_file('r')
        
        try:
            with open(self.csv_handler.csv_file, 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 513:  # Ensure we have at least embedding + username
                        embedding = list(map(float, row[:512]))
                        user_name = row[512]
                        
                        # Get last login time if it exists
                        last_login = row[513] if len(row) > 513 else "Never"
                        
                        embeddings_dict[user_name] = {
                            'embedding': embedding,
                            'last_login': last_login
                        }
        except Exception as e:
            print(f"Error reading embeddings: {str(e)}")
            
        return embeddings_dict
    
    def add_or_update_user(self, user_name, embedding):
        """
        Add or update a user in the database.
        
        Args:
            user_name: Name of the user
            embedding: Face embedding for the user
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert embedding to list if it's a numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
                
            # Get all current embeddings
            embeddings_dict = self.get_all_embeddings()
            
            # Remove the user if they already exist (to update them)
            if user_name in embeddings_dict:
                self._delete_user_from_csv(user_name)
            
            # Write the user with current timestamp
            with open(self.csv_handler.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow(embedding + [user_name, current_time])
                
            return True
            
        except Exception as e:
            print(f"Error adding/updating user: {str(e)}")
            return False
    
    def delete_user(self, user_name):
        """
        Delete a user from the database.
        
        Args:
            user_name: Name of the user to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self._delete_user_from_csv(user_name)
        
    def update_last_login(self, user_name):
        """
        Update the last login time for a user.
        
        Args:
            user_name: Name of the user
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current embeddings
            embeddings_dict = self.get_all_embeddings()
            
            if user_name in embeddings_dict:
                # Get current embedding
                embedding = embeddings_dict[user_name]['embedding']
                
                # Update the user with new timestamp
                return self.add_or_update_user(user_name, embedding)
            else:
                print(f"User {user_name} not found")
                return False
        except Exception as e:
            print(f"Error updating last login: {str(e)}")
            return False
            
    def _delete_user_from_csv(self, user_name):
        """
        Delete a user from the CSV file.
        
        Args:
            user_name: Name of the user to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            temp_file = self.csv_handler.csv_file + ".temp"
            deleted = False
            
            # Read the entire file and write all rows except the one to delete
            with open(self.csv_handler.csv_file, 'r', newline='') as file_in:
                with open(temp_file, 'w', newline='') as file_out:
                    reader = csv.reader(file_in)
                    writer = csv.writer(file_out)
                    
                    # Write header
                    header = next(reader)
                    writer.writerow(header)
                    
                    # Write all rows except the one with matching user_name
                    for row in reader:
                        if len(row) >= 513 and row[512] != user_name:
                            writer.writerow(row)
                        elif len(row) >= 513 and row[512] == user_name:
                            deleted = True
            
            # Replace the original file with the temp file
            os.replace(temp_file, self.csv_handler.csv_file)
            
            return deleted
        except Exception as e:
            print(f"Error deleting user: {str(e)}")
            return False