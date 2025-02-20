import cv2
from UsersDatabaseHandeler.UsersDatabaseHandeler import EmbeddingCSVHandler
from ImageProcessor import ImageProcessor
import os
import random

import numpy as np
from database.db_handler import FaceDatabase

def add_user(image_path: str, user_name: str):
    image_processor = ImageProcessor(verbose=True)
    database_handler = FaceDatabase()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    embeddings = image_processor.process_image(image)
    if len(embeddings) == 0:
        print("No faces detected")
        return    
    
    database_handler.add_user(user_name,np.array(embeddings['embeddings'][0], dtype='float32'))
    print(f"User {user_name} added to the database")

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

    image_path3 = "hatem.png"  # Provide the path to your test image
    user_name = "hatem"  # Provide the user name
    add_user(image_path3, user_name)

    # image_processor = ImageProcessor(use_yolo=True, verbose=False)
    # print('Negative Test Case')
    # image_path = get_random_image_path("gallery_faces")  # Provide the path to your test image
    # results = image_processor.verify_faces(image_path)
    # print(f"Results for image: {image_path}")
    # for result in results:
    #     print(f"Bounding Box: {result['bbox']}")
    #     print(f"User Name: {result['user_name']}")
    #     print(f"Verification Result: {result['verification_result']}")
    #     print("-------------------------------------")

    # image_processor2 = ImageProcessor(use_yolo=True, verbose=False)
    # print('Positive Test Case')
    # image_path2 = 'gallery_faces/gallery_13_2.jpg'
    # results = image_processor2.verify_faces(image_path2)
    # print(f"Results for image: {image_path2}")
    # for result in results:
    #     print(f"Bounding Box: {result['bbox']}")
    #     print(f"User Name: {result['user_name']}")
    #     print(f"Verification Result: {result['verification_result']}")
    #     print("-------------------------------------")


