#!/usr/bin/env python3
"""
Script to read images from a user's profile folder and add them to the face recognition database.
This script processes all the face images from the specified user folder and adds the profile
to the database with the folder name as the user ID.
"""

import os
import cv2
import argparse
import logging
import numpy as np
from src.models.face_recognition import FaceNetTFLiteHandler
from src.services.image_processor import ImageProcessor
from src.database.face_db import FaceDatabase
from src.utils.image import preprocess_image
from src.database.handlers.db_handler import UserDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Add_Profile_To_Database")

def add_profile_to_database(profile_folder, detector_type='mediapipe', verbose=False):
    """
    Read all images from a profile folder and add them to the face recognition database.
    
    Args:
        profile_folder (str): Path to the profile folder containing face images
        detector_type (str): Face detection model to use
        verbose (bool): Enable verbose output
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if profile folder exists
    if not os.path.exists(profile_folder):
        logger.error(f"Profile folder not found: {profile_folder}")
        return False
    
    # Get the user ID from the folder name
    user_id = os.path.basename(profile_folder)
    logger.info(f"Processing profile for user: {user_id}")
    
    # Use writable path in Docker volume
    db_path = '/data/face_embeddings.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Use the custom database path
    database_handler = FaceDatabase()
    database_handler.db_handler = UserDatabase(db_path=db_path)
    
    # Initialize FaceNet for direct embedding extraction if needed
    facenet = FaceNetTFLiteHandler(verbose=verbose)
    
    # Initialize image processor with proper error handling
    image_processor = None

    try:
        # First attempt with image processor
        image_processor = ImageProcessor(model_architecture=detector_type, verbose=verbose)
        use_processor = True
    except Exception as e:
        logger.error(f"Error initializing image processor: {e}")
        logger.info("Falling back to direct image processing")
        use_processor = False
    
    # Find all image files in the profile folder
    image_files = []
    for filename in os.listdir(profile_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(profile_folder, filename))
    
    if not image_files:
        logger.error(f"No image files found in profile folder: {profile_folder}")
        return False
    
    logger.info(f"Found {len(image_files)} image files")
    
    # Process each image to extract face embeddings
    embeddings = []
    for image_file in image_files:
        logger.info(f"Processing image: {image_file}")
        
        # Load the image
        image = cv2.imread(image_file)
        if image is None:
            logger.warning(f"Failed to read image: {image_file}")
            continue
        
        embedding = None

        try:
            detection_embedding = image_processor.process_image(image)
            print(f"Detection result: {detection_embedding}")
            if detection_embedding is not None and len(detection_embedding.embeddings) > 0:
                # Add the embedding from the detected face
                embedding = detection_embedding.embeddings[0]
                embeddings.append(embedding)
                logger.info(f"Successfully extracted embedding from detected face: {image_file}")
            else:
                logger.warning(f"No faces detected in image with processor: {image_file}")
                embedding = None
        except Exception as e:
            logger.warning(f"Error using processor: {e}")
            embedding = None
            logger.info("Falling back to direct embedding extraction")
            
    # Check if we have any valid embeddings
    if len(embeddings) == 0:
        logger.error("Failed to extract any valid embeddings")
        return False

    # Convert to numpy array for averaging
    embeddings_array = np.array(embeddings, dtype=np.float32)
    logger.info(f"Number of valid embeddings extracted: {len(embeddings)}")

    # Calculate the average embedding
    if len(embeddings) > 1:
        logger.info(f"Averaging {len(embeddings)} embeddings")
        final_embedding = np.mean(embeddings_array, axis=0)
        # final_embedding = embeddings_array[1]

    else:
        final_embedding = embeddings_array[0]
    logger.info(f"Final embedding shape: {final_embedding.shape}")

    # Check if the user already exists
    all_embeddings = database_handler.get_all_embeddings()
    if user_id in all_embeddings:
        logger.warning(f"User '{user_id}' already exists in the database, updating...")

    # Add or update the user in the database
    database_handler.add_or_update_user(user_id, final_embedding)
    logger.info(f"Profile for user '{user_id}' added/updated successfully")


def main():
    parser = argparse.ArgumentParser(description="Add profile to face recognition database")
    parser.add_argument("--profile", type=str, default="profiles/Hatem",
                      help="Path to the profile folder containing face images")
    parser.add_argument("--detector", type=str, default="mediapipe", 
                      choices=["mediapipe", "yolov8", "yolov8_onnx"],
                      help="Face detector model to use")
    parser.add_argument("--verbose", action="store_true", 
                      help="Enable verbose output")
    args = parser.parse_args()
    
    add_profile_to_database(args.profile, args.detector, args.verbose)

if __name__ == "__main__":
    main()
