#!/usr/bin/env python3
"""
Test script for the gaze direction detection feature.
This script provides a simplified interface to test the gaze direction
detection capabilities of the face recognition system.
"""

import cv2
import time
import argparse
import logging
import numpy as np
from src.services.image_processor import ImageProcessor
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Gaze_Detection_Test")

def main():
    """Run a standalone test of the gaze direction detection feature."""
    parser = argparse.ArgumentParser(description="Gaze Direction Detection Test")
    parser.add_argument("--camera", type=int, default=0,
                      help="Camera index to use")
    parser.add_argument("--detector", type=str, default="mediapipe",
                      choices=["mediapipe"],  # Only mediapipe supported for gaze detection
                      help="Face detector model to use")
    parser.add_argument("--display", action="store_true", default=True,
                      help="Display the video feed with gaze information")
    
    args = parser.parse_args()
    
    # Initialize image processor with the specified detector
    logger.info(f"Initializing image processor with {args.detector} detector")
    image_processor = ImageProcessor(model_architecture=args.detector, verbose=True)
    
    # Open the camera
    logger.info(f"Opening camera {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Starting gaze detection test. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Process the image with face detection
            start_time = time.time()
            detection_result = image_processor.process_image(frame, filter_largest=True)
            process_time = time.time() - start_time
            
            if detection_result is not None:
                # Detect facial landmarks for gaze detection
                image_processor.detect_landmarks(frame)
                
                # Get gaze direction 
                gaze_info = image_processor.get_gaze_direction(detection_result)
                
                if gaze_info and gaze_info.get('direction') != 'unknown':
                    direction = gaze_info['direction']
                    left_dist = gaze_info.get('left_eye_nose_dist', 0)
                    right_dist = gaze_info.get('right_eye_nose_dist', 0)
                    ratio = gaze_info.get('ratio', 0)
                    
                    logger.info(f"Gaze Direction: {direction.upper()} " 
                               f"(L: {left_dist:.1f}, R: {right_dist:.1f}, Ratio: {ratio:.2f})")
                    
                    # Draw information on frame
                    if args.display:
                        # Draw detection boxes
                        for box in detection_result.detection_faces.boxes:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw landmarks if available
                        frame = image_processor.draw_landmarks(frame)
                        
                        # Draw gaze direction text
                        cv2.putText(frame, f"Looking: {direction.upper()}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, f"L: {left_dist:.1f}, R: {right_dist:.1f}", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Ratio: {ratio:.2f}", (10, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"FPS: {1/max(process_time, 0.001):.1f}", (10, 130), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            # Display the frame
            if args.display:
                cv2.imshow('Gaze Detection Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    break
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in gaze detection test: {e}")
    finally:
        # Release resources
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("Gaze detection test completed")

if __name__ == "__main__":
    main()
