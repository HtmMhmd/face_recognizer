#!/usr/bin/env python3

import argparse
import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def add_user(username, image_path=None, detector_type='mediapipe', capture_live=False, camera_index=0, verbose=False):
    """
    Add a new user to the face recognition database.
    
    Args:
        username (str): Username to add to the database
        image_path (str): Path to image file containing the user's face
        detector_type (str): Face detection model to use
        capture_live (bool): Whether to capture from camera instead of loading image
        camera_index (int): Camera index to use if capturing live
        verbose (bool): Enable verbose output
    """
    from src.services.image_processor import ImageProcessor
    from src.database.face_db import FaceDatabase
    
    print(f"[INFO] Adding user '{username}' to the database")
    
    # Initialize the image processor
    image_processor = ImageProcessor(model_architecture=detector_type, verbose=verbose)
    database_handler = FaceDatabase()
    
    # Get the face image either from file or camera
    if capture_live:
        print("[INFO] Capturing image from camera. Press SPACE to capture or ESC to cancel.")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {camera_index}")
            return False
            
        captured_image = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
                
            # Display instruction text
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press SPACE to capture or ESC to cancel", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Draw face detection on preview
            embeddings = image_processor.process_image(frame)
            if embeddings and len(embeddings.embeddings) > 0:
                display_frame = image_processor.draw_detections(display_frame)
                cv2.putText(display_frame, "Face detected", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No face detected", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Capture Face", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("[INFO] Capture cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
            elif key == 32:  # SPACE key
                if embeddings and len(embeddings.embeddings) > 0:
                    captured_image = frame.copy()
                    print("[INFO] Image captured successfully")
                    break
                else:
                    print("[WARNING] No face detected. Please position your face in the camera view.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_image is None:
            print("[ERROR] Failed to capture image")
            return False
            
        image = captured_image
    else:
        # Load image from file
        if not image_path or not os.path.exists(image_path):
            print(f"[ERROR] Input image not found: {image_path}")
            return False
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Failed to read image: {image_path}")
            return False
    
    # Process the image to extract face embeddings
    print("[INFO] Processing face...")
    embeddings = image_processor.process_image(image)
    
    if embeddings is None or len(embeddings.embeddings) == 0:
        print("[ERROR] No faces detected in the image")
        return False
        
    if len(embeddings.embeddings) > 1:
        print("[WARNING] Multiple faces detected. Using the first face.")
    
    # Add the user to the database
    embedding = embeddings.embeddings[0]
    
    # Check if username already exists
    all_embeddings = database_handler.get_all_embeddings()
    if username in all_embeddings:
        choice = input(f"User '{username}' already exists. Do you want to overwrite? (y/n): ")
        if choice.lower() != 'y':
            print("[INFO] Operation cancelled")
            return False
    
    # Add or update the user
    success = database_handler.add_or_update_user(username, embedding)
    
    if success:
        print(f"[INFO] User '{username}' successfully added to the database")
        return True
    else:
        print("[ERROR] Failed to add user to the database")
        return False

def delete_user(username, verbose=False):
    """
    Delete a user from the face recognition database.
    
    Args:
        username (str): Username to delete from the database
        verbose (bool): Enable verbose output
    """
    from src.database.face_db import FaceDatabase
    
    print(f"[INFO] Deleting user '{username}' from the database")
    
    database_handler = FaceDatabase()
    
    # Check if username exists
    all_embeddings = database_handler.get_all_embeddings()
    if username not in all_embeddings:
        print(f"[ERROR] User '{username}' not found in the database")
        return False
    
    # Confirm deletion
    choice = input(f"Are you sure you want to delete user '{username}'? (y/n): ")
    if choice.lower() != 'y':
        print("[INFO] Operation cancelled")
        return False
    
    # Delete the user
    success = database_handler.delete_user(username)
    
    if success:
        print(f"[INFO] User '{username}' successfully deleted from the database")
        return True
    else:
        print("[ERROR] Failed to delete user from the database")
        return False

def list_users(verbose=False):
    """
    List all users in the face recognition database.
    
    Args:
        verbose (bool): Enable verbose output to show last login times
    """
    from src.database.face_db import FaceDatabase
    
    print("[INFO] Listing all users in the database")
    
    database_handler = FaceDatabase()
    
    # Get all embeddings
    all_embeddings = database_handler.get_all_embeddings()
    
    if not all_embeddings:
        print("[INFO] No users found in the database")
        return
    
    print("\n--- Registered Users ---")
    for i, (username, data) in enumerate(all_embeddings.items(), 1):
        last_login = data.get('last_login', 'Never')
        if verbose:
            print(f"{i}. {username} (Last seen: {last_login})")
        else:
            print(f"{i}. {username}")
    print("----------------------")

def main():
    """
    Main function to parse command-line arguments and execute the appropriate action.
    """
    parser = argparse.ArgumentParser(description="Face Recognition User Management")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Add user parser
    add_parser = subparsers.add_parser("add", help="Add a new user")
    add_parser.add_argument("username", help="Username to add")
    add_parser.add_argument("--image", help="Path to image file containing the user's face")
    add_parser.add_argument("--capture", action="store_true", help="Capture image from camera")
    add_parser.add_argument("--detector", choices=["mediapipe", "yolov8", "yolov8_onnx"], 
                          default="mediapipe", help="Face detector to use")
    add_parser.add_argument("--camera", type=int, default=0, help="Camera index to use if capturing")
    
    # Delete user parser
    delete_parser = subparsers.add_parser("delete", help="Delete an existing user")
    delete_parser.add_argument("username", help="Username to delete")
    
    # List users parser
    list_parser = subparsers.add_parser("list", help="List all users")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    
    # Global arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        return
    
    if args.action == "add":
        if not args.image and not args.capture:
            print("[ERROR] Either --image or --capture is required")
            return
        add_user(args.username, args.image, args.detector, args.capture, args.camera, args.verbose)
    elif args.action == "delete":
        delete_user(args.username, args.verbose)
    elif args.action == "list":
        list_users(args.verbose)

if __name__ == "__main__":
    main()