import cv2
import argparse
import os
import random
import numpy as np
from ImageProcessor import ImageProcessor
from database import FaceDatabase

def add_user(image_path: str, user_name: str):
    """
    Add a new user to the face recognition database.
    
    Args:
        image_path (str): Path to the user's face image
        user_name (str): Name of the user to add
    
    Returns:
        bool: True if user was added successfully, False otherwise
    """
    # Validate input
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False
        
    try:
        image_processor = ImageProcessor(verbose=True)
        database_handler = FaceDatabase()

        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Failed to read image")
            return False

        print("Detecting faces and generating embeddings...")
        embeddings = image_processor.process_image(image)
        if embeddings is None or len(embeddings.embeddings) == 0:
            print("No faces detected in the image. Please use an image with a clear face.")
            return False
            
        if len(embeddings.embeddings) > 1:
            print(f"Warning: Multiple faces ({len(embeddings.embeddings)}) detected in the image.")
            print("Using the first detected face for the user profile.")
        
        print(f"Adding user '{user_name}' to database...")
        database_handler.add_user(user_name, np.array(embeddings.embeddings[0], dtype='float32'))
        print(f"✅ Success! User '{user_name}' added to the face recognition database.")
        return True
        
    except Exception as e:
        print(f"Error adding user: {str(e)}")
        return False

def verify_faces_in_image(self, image_path: str):
    """
    Verifies faces in a specified image path against the user database.

    Args:
        image_path (str): The path to the input image.

    Returns:
        List[dict]: A list of dictionaries containing bounding boxes, user names, and verification results.
    """
    image_processor = ImageProcessor(verbose=True)

    if image_path is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    result = image_processor.verify_faces(image_path)

    return result

def get_random_image_path(folder_path: str) -> str:
    """
    Gets a random image path from the specified folder.

    Args:
        folder_path (str): The path to the folder containing images.

    Returns:
        str: The path to a random image file.
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]
    if not image_files:
        raise FileNotFoundError(f"No image files found in folder {folder_path}")
    return os.path.join(folder_path, random.choice(image_files))

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new user to the face recognition system")
    parser.add_argument("--username", "-u", type=str, help="Username to add to the database", required=True)
    parser.add_argument("--image", "-i", type=str, help="Path to the user's face image", required=True)
    
    args = parser.parse_args()
    
    # Add the user
    add_user(args.image, args.username)