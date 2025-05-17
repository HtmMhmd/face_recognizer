#!/usr/bin/env python3
"""
Test script for the add_user direction detection feature.
This script provides a simplified interface to test the add_user direction
detection capabilities of the face recognition system.
"""

import cv2
import time
import argparse
import logging
import numpy as np
import zmq
import base64
import os
import shutil
from src.services.image_processor import ImageProcessor
from src.config.settings import Settings
import json

parser = argparse.ArgumentParser(description="ADD_USER_SCRIPT")
parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
parser.add_argument("--detector", type=str, default="mediapipe",
                    choices=["mediapipe"], help="Face detector model to use")
parser.add_argument("--display", action="store_true", default=True,
                    help="Display the video feed with add_user information")
parser.add_argument("--zmq-server", action="store_false", help="Run as ZMQ add_user server")
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ADD_USER_SCRIPT")


def detect_face_direction(image_processor, frame, direction):
    detection_result = image_processor.process_image(frame, filter_largest=True)
    if detection_result is None:
        return False
    image_processor.detect_landmarks(frame)
    add_user_info = image_processor.get_add_user_direction(detection_result)
    if not add_user_info or add_user_info.get('direction') == 'unknown':
        return False
    mapping = {"front": "center", "left": "left", "right": "right"}
    return add_user_info['direction'] == mapping.get(direction, "")


def publish_and_wait_zmq(socket_pub, socket_rep, cap, image_processor, view_name, capture_command, save_path):
    logger.info(f"[{view_name}] Waiting for '{capture_command}'...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        # Process the image and get detection result
        detection_result = image_processor.process_image(frame, filter_largest=True)
        detected = False
        display_image = frame.copy()
        
        # Check if face is detected and aligned in correct direction
        if detection_result is not None and len(detection_result.detection_faces.boxes) > 0:
            image_processor.detect_landmarks(frame)
            add_user_info = image_processor.get_gaze_direction(detection_result)
            if add_user_info and add_user_info.get('direction') != 'unknown':
                mapping = {"front": "center", "left": "left", "right": "right"}
                detected = add_user_info['direction'] == mapping.get(view_name, "")
            
            # Get the cropped face from detection result for sending
            if detected and len(detection_result.detection_faces.cropped_faces) > 0:
                # Use the cropped face for display
                cropped_face = detection_result.detection_faces.cropped_faces[0]
                
                # Add colored border to indicate alignment status
                color = (0, 255, 0)  # Green border for correct alignment
                bordered_face = cv2.copyMakeBorder(cropped_face, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=color)
                
                # Draw landmarks on the image for better visual feedback
                display_image = bordered_face
            else:
                # If not detected properly or no cropped face, use full frame with red border
                color = (0, 0, 255)  # Red border for incorrect alignment
                display_image = cv2.copyMakeBorder(frame, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=color)
                display_image = image_processor.draw_landmarks(display_image)
        else:
            # No face detected at all
            color = (255, 0, 255)  # Red + BLue border
            display_image = cv2.copyMakeBorder(frame, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=color)

        # Encode and send the image
        _, buffer = cv2.imencode('.jpg', display_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoded = base64.b64encode(buffer).decode('utf-8')
        socket_pub.send_multipart([b"capture", encoded.encode('utf-8')])

        try:
            if socket_rep.poll(timeout=1):
                command = socket_rep.recv_string()
                logger.info(f"Received: {command}")
                if command == capture_command:
                    if detected:
                        cv2.imwrite(save_path, frame)  # Still save the full frame (not cropped)
                        logger.info(f"{capture_command} saved at {save_path}")
                        socket_rep.send_string(f"{capture_command}_ack")
                        return
                    else:
                        logger.warning("Face not aligned correctly!")
                        socket_rep.send_string(f"{capture_command}_nack")
        except zmq.Again:
            continue


def zmq_add_user_server():
    global args
    context = zmq.Context()
    socket_rep = context.socket(zmq.REP)
    socket_rep.bind("tcp://*:5555")
    logger.info("REP socket bound at tcp://*:5555")

    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind("tcp://*:5544")
    logger.info("PUB socket bound at tcp://*:5544")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Camera failed to open.")
        return

    # Create the main profiles directory
    os.makedirs("profiles", exist_ok=True)
    image_processor = ImageProcessor(model_architecture=args.detector, verbose=True)

    try:
        while True:
            logger.info("Waiting for 'start_add_profile' command...")
            command = socket_rep.recv_string()
            if command == "start_add_profile":
                # Acknowledge the start command
                socket_rep.send_string("start_add_profile_ack")
                binding_username = 'binding_user'
                binding_user_dir = os.path.join("profiles", binding_username)
                os.makedirs(binding_user_dir, exist_ok=True)
                logger.info(f"Created directory for binding user: {binding_user_dir}")

                # Start capturing the profile images
                logger.info(f"Starting profile capture for user: {binding_username}")
                for view, cmd, filename in [
                    ("front", "capture_front", "front.jpg"),
                    ("left", "capture_left", "left.jpg"),
                    ("right", "capture_right", "right.jpg")
                ]:
                    
                    # Save images to the user-specific directory
                    save_path = os.path.join(binding_user_dir, filename)
                    publish_and_wait_zmq(socket_pub, socket_rep, cap, image_processor, view, cmd, save_path)
                
                # Wait for user credentials as JSON
                logger.info("Waiting for user credentials...")
                json_data = socket_rep.recv_string()
                
                try:
                    user_data = json.loads(json_data)
                    
                    username = user_data.get('username', '')
                    password = user_data.get('password', '')
                    
                    if not username:
                        logger.error("Empty username in JSON data")
                        socket_rep.send_string("invalid_username")
                        continue
                        
                    if not password:
                        logger.error("Empty password in JSON data")
                        socket_rep.send_string("invalid_password")
                        continue
                        
                    logger.info(f"Received credentials for user: {username}")
                    socket_rep.send_string("user_metadata_ack")
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON data received")
                    socket_rep.send_string("user_metadata_nack")
                    continue
                except Exception as e:
                    logger.error(f"Error processing credentials: {str(e)}")
                    socket_rep.send_string("user_metadata_nack")
                    continue
                
                
                logger.info(f"Received username: {username} and password: {password}")
                # Create user-specific directory
                user_dir = os.path.join("profiles", username)

                try:
                    os.makedirs(user_dir, exist_ok=True)
                    logger.info(f"Created directory for user: {user_dir}")
                except OSError as e:
                    logger.error(f"Failed to create directory {user_dir}: {e}")
                    socket_rep.send_string("directory_creation_failed")
                    continue

                # Move the captured images to the user-specific directory
                for filename in ["front.jpg", "left.jpg", "right.jpg"]:
                    src_path = os.path.join(binding_user_dir, filename)
                    dst_path = os.path.join(user_dir, filename)
                    shutil.copy(src_path, dst_path)

            #     logger.info(f"Profile capture complete for user: {username}")
            #     socket_rep.send_string("profile_complete")
            # else:
            #     logger.warning(f"Unknown command: {command}")
            #     socket_rep.send_string("unknown_command")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        socket_rep.close()
        socket_pub.close()
        context.term()
        logger.info("Clean shutdown.")


def main():

    if args.zmq_server:
        zmq_add_user_server()


if __name__ == "__main__":
    main()
