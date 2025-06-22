#!/usr/bin/env python3
"""
Unified service automation script that provides ZMQ interfaces for:
- User registration (add_user)
- User authentication
- Drowsiness detection

This script can run in different modes to provide different services.
"""

import cv2
import time
import argparse
import logging
import numpy as np
import zmq
import base64
import os
import json
import shutil
from src.services.image_processor import ImageProcessor
from src.database.face_db import FaceDatabase
from src.database.handlers.db_handler import UserDatabase
from src.core.drowsiness.detection import DrowsinessDetector
from src.utils.camera.camera_handler import CameraHandler
from src.config.settings import Settings

# Configure argument parser
parser = argparse.ArgumentParser(description="Service Automation")
parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
parser.add_argument("--detector", type=str, default="mediapipe",
                    choices=["mediapipe", "yolov8", "yolov8_onnx"], 
                    help="Face detector model to use")
parser.add_argument("--mode", type=str, required=True,
                    choices=["add_user", "authentication", "drowsiness"],
                    help="Service mode: add_user, authentication, or drowsiness")
parser.add_argument("--handler", type=str, choices=["threaded", "regular"], default="regular",
                    help="Camera handler type: threaded or regular")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO if args.verbose else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Service_Automation")

# Database path
DB_PATH = '/data/face_embeddings.db'

def init_camera():
    """Initialize the camera based on handler type."""
    try:
        if args.handler == "threaded":
            logger.info("Using threaded camera handler")
            camera = CameraHandler(args.camera)
            camera_opened = True
        else:
            logger.info("Using regular OpenCV camera")
            camera = cv2.VideoCapture(args.camera)
            camera_opened = camera.isOpened()
            
        if not camera_opened:
            logger.error("Failed to open camera")
            return None
            
        return camera
    except Exception as e:
        logger.error(f"Camera initialization error: {e}")
        return None

def read_frame(camera):
    """Read a frame from the camera considering handler type."""
    if isinstance(camera, CameraHandler):
        timestamp, frame = camera.read()
        return frame is not None, frame
    else:
        return camera.read()

def encode_frame(frame, quality=70):
    """Encode a frame as base64 string for ZMQ transmission."""
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')

def run_authentication_service():
    """Run the authentication service with ZMQ interface."""
    context = zmq.Context()
    socket_rep = context.socket(zmq.REP)
    socket_rep.bind("tcp://*:5555")
    logger.info("Authentication service bound at tcp://*:5555")
    
    # Initialize camera
    camera = init_camera()
    if camera is None:
        return
    
    # Initialize image processor and database
    image_processor = ImageProcessor(model_architecture=args.detector, verbose=args.verbose)
    database = FaceDatabase()
    database.db_handler = UserDatabase(db_path=DB_PATH)
    
    try:
        while True:
            # Wait for authentication request
            message = socket_rep.recv_string()
            
            if message == "authenticate":
                # Read frame
                ret, frame = read_frame(camera)
                if not ret:
                    socket_rep.send_string(json.dumps({
                        "status": "error",
                        "message": "Could not capture image"
                    }))
                    continue
                
                # Process image with the image processor to get detection and embeddings
                detection_result = image_processor.process_image(frame, filter_largest=True)
                
                # Check if detection was successful and has embeddings
                if detection_result is None or not hasattr(detection_result, 'embeddings') or len(detection_result.embeddings) == 0:
                    socket_rep.send_string(json.dumps({
                        "status": "error",
                        "message": "No face detected"
                    }))
                    continue
                
                # Get the embedding of the detected face
                embedding = detection_result.embeddings[0]
                
                # Use the built-in verify_faces method of image_processor for verification
                verify_results = image_processor.verify_faces()
                
                if verify_results and len(verify_results) > 0:
                    # Extract verification details
                    user_info = verify_results[0]
                    user_name = user_info.get('user_name', "Unknown")
                    verification_details = user_info.get('verification_result', {})
                    
                    # Use cosine similarity for confidence score
                    if 'cosine' in verification_details:
                        confidence = verification_details['cosine'].get('confidence', 0)
                        verified = verification_details['cosine'].get('verified', False)
                    else:
                        confidence = 0
                        verified = False
                    
                    if verified:
                        # Successfully verified
                        socket_rep.send_string(json.dumps({
                            "status": "success",
                            "user": user_name,
                            "confidence": float(confidence)
                        }))
                        logger.info(f"Authentication successful for user: {user_name}")
                    else:
                        # Verification failed
                        socket_rep.send_string(json.dumps({
                            "status": "failed",
                            "message": "Authentication failed",
                            "confidence": float(confidence)
                        }))
                        logger.info(f"Authentication failed for detected face")
                else:
                    # No matching user found
                    socket_rep.send_string(json.dumps({
                        "status": "failed",
                        "message": "User not recognized"
                    }))
                    logger.info("No matching user found in database")
            
            elif message == "status":
                socket_rep.send_string("running")
            else:
                socket_rep.send_string(json.dumps({
                    "status": "error",
                    "message": "Unknown command"
                }))
    
    except KeyboardInterrupt:
        logger.info("Authentication service stopped")
    finally:
        try:
            camera.release()
        except:
            pass
        socket_rep.close()
        context.term()

def run_drowsiness_service():
    """Run the drowsiness detection service with ZMQ interface."""
    context = zmq.Context()
    socket_rep = context.socket(zmq.REP)
    socket_rep.bind("tcp://*:5555")
    logger.info("Drowsiness detection service bound at tcp://*:5555")
    
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind("tcp://*:5544")
    logger.info("PUB socket bound at tcp://*:5544")
    
    # Initialize camera
    camera = init_camera()
    if camera is None:
        return
    
    # Initialize image processor and drowsiness detector
    image_processor = ImageProcessor(model_architecture=args.detector, verbose=args.verbose)
    drowsiness_detector = DrowsinessDetector()
    
    # State variables
    monitoring = False
    alert_count = 0
    
    try:
        while True:
            # Check for commands
            if socket_rep.poll(timeout=1):
                command = socket_rep.recv_string()
                
                if command == "start_monitoring":
                    monitoring = True
                    socket_rep.send_string("monitoring_started")
                    logger.info("Drowsiness monitoring started")
                
                elif command == "stop_monitoring":
                    monitoring = False
                    socket_rep.send_string("monitoring_stopped")
                    logger.info("Drowsiness monitoring stopped")
                
                elif command == "status":
                    socket_rep.send_string("running" if monitoring else "idle")
                
                else:
                    socket_rep.send_string("unknown_command")
            
            if not monitoring:
                time.sleep(0.1)
                continue
            
            # Read frame
            ret, frame = read_frame(camera)
            if not ret:
                time.sleep(0.1)
                continue
            
            # Process for drowsiness detection - get both detection result and display frame
            detection_result = image_processor.process_image(frame, filter_largest=True)
            display_frame = frame.copy()
            
            if detection_result is not None and hasattr(detection_result.detection_faces, 'boxes') and len(detection_result.detection_faces.boxes) > 0:
                # Draw detections on the display frame
                display_frame = image_processor.draw_detections(display_frame)
                
                # Detect landmarks and get eye/mouth points
                landmarks = image_processor.detect_landmarks(frame)
                if landmarks is not None:
                    # Draw landmarks on display frame
                    display_frame = image_processor.draw_landmarks(display_frame)
                    
                    # Get eye and mouth points for drowsiness detection
                    eye_mouth = image_processor.get_eye_mouth_keypoints()
                    
                    # Process with drowsiness detector if eye points are available
                    if eye_mouth and ('left_eye' in eye_mouth and 'right_eye' in eye_mouth):
                        # Process frame with drowsiness detector
                        display_frame = drowsiness_detector.process_frame(display_frame, eye_mouth)
                        
                        # Send drowsiness alerts if needed
                        if drowsiness_detector.is_drowsy():
                            alert_count += 1
                            if alert_count >= 3:  # Send alert after multiple consecutive detections
                                alert_level = drowsiness_detector.get_drowsiness_level()
                                logger.info(f"Drowsiness detected! Level: {alert_level}")
                                socket_pub.send_multipart([
                                    b"drowsiness_alert", 
                                    json.dumps({
                                        "status": "alert",
                                        "message": "Drowsiness detected",
                                        "level": alert_level
                                    }).encode('utf-8')
                                ])
                        else:
                            alert_count = 0
            
            # Send processed frame
            encoded = encode_frame(display_frame)
            socket_pub.send_multipart([b"frame", encoded.encode('utf-8')])
            
            # Sleep to reduce CPU usage
            time.sleep(0.03)
    
    except KeyboardInterrupt:
        logger.info("Drowsiness service stopped")
    finally:
        try:
            camera.release()
        except:
            pass
        socket_rep.close()
        socket_pub.close()
        context.term()

def main():
    """Main entry point for the service."""
    if args.mode == "add_user":
        run_add_user_service()
    elif args.mode == "authentication":
        run_authentication_service()
    elif args.mode == "drowsiness":
        run_drowsiness_service()
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
