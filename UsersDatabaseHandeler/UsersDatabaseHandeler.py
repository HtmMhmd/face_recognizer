import csv
import os
from typing import List, Tuple

class CSVFileChecker:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.check_and_create_csv()

    def check_and_create_csv(self):
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = [f'embedding_{i}' for i in range(512)] + ['user_name']
                writer.writerow(header)

class EmbeddingCSVHandler:
    def __init__(self, csv_file: str = "UsersDatabaseHandeler/users_embeddings.csv"):
        self.csv_file = csv_file
        CSVFileChecker(csv_file)  # Ensure the CSV file exists and is properly formatted
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

# Example usage
if __name__ == "__main__":
    handler = EmbeddingCSVHandler("embeddings.csv")

    # Write an embedding and user name to the CSV file
    embedding = [0.1] * 512  # Example embedding
    user_name = "John Doe"
    handler.write_embedding(embedding, user_name)

    # Read an embedding and user name from the CSV file by index
    index = 0
    embedding, user_name = handler.read_embedding(index)
    print(f"Embedding: {embedding}")
    print(f"User Name: {user_name}")

    # Get the total length of the CSV file
    total_length = len(handler)
    print(f"Total Length: {total_length}")