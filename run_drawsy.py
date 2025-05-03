#!/usr/bin/env python3
import cv2
import argparse
import threading
import os

from CameraUtilis.CameraHandler import CameraHandler
from SingletonImageProcessor import SingletonImageProcessor
from Model.MediapipeDetection.MediapipeFaceLandmarker import FaceMeshDetector
from drowsiness.EAR import DrowsinessDetector
from shared_state import SharedState

output_frame = None
lock = threading.Lock()

# Initialize shared state
shared_state = SharedState(namespace="drowsiness")

# Function to process camera feed using cv2.VideoCapture
def process_camera_feed(face_mesh, drowsiness_detector=None, show_gui=False):
    global output_frame, shared_state
    # Get singleton image processor
    image_processor = SingletonImageProcessor.get_instance(verbose=True)
    
    cap = cv2.VideoCapture(0)  # Use the appropriate camera index
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize the frame
            frame = cv2.resize(frame, (480, 360))
            
            # Process the frame with face mesh
            landmarks = face_mesh.landmark(frame)
            frame_with_landmarks = face_mesh.draw_landmarks(frame)
            
            # Get eye and mouth keypoints for drowsiness detection
            eye_mouth = face_mesh.get_eye_mouth_keypoints()
            
            # If landmarks found, perform drowsiness detection
            if eye_mouth:
                result = drowsiness_detector.process_frame(frame_with_landmarks, eye_mouth)
                ear_value = drowsiness_detector.get_current_ear()
                mar_value = drowsiness_detector.get_current_mar()
                display_frame = result
                
                # Store detection results in shared state
                shared_state.set_value("ear_value", ear_value)
                shared_state.set_value("mar_value", mar_value)
                shared_state.set_value("is_drowsy", drowsiness_detector.is_drowsy())
                shared_state.set_value("is_yawning", drowsiness_detector.is_yawning())
            else:
                display_frame = frame_with_landmarks
                print("No face detected")
                shared_state.set_value("face_detected", False)

            # Only show GUI if explicitly enabled
            if show_gui:
                cv2.imshow("Drowsiness Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if show_gui:
            cv2.destroyAllWindows()

# Function to process camera feed using CameraHandler
def process_camera_handler(face_mesh, drowsiness_detector=None, show_gui=False):
    camera = CameraHandler(0)  # Initialize CameraHandler
    try:
        while True:
            timestamp, frame = camera.read()
            if frame is not None:
                # Resize the frame
                frame = cv2.resize(frame, (480, 360))
                
                # Process the frame with face mesh
                landmarks = face_mesh.landmark(frame)
                frame_with_landmarks = face_mesh.draw_landmarks(frame)
                
                # Get eye and mouth keypoints for drowsiness detection
                eye_mouth = face_mesh.get_eye_mouth_keypoints()
                
                # If landmarks found, perform drowsiness detection
                if eye_mouth:
                    result = drowsiness_detector.process_frame(frame_with_landmarks, eye_mouth)
                    ear_value = drowsiness_detector.get_current_ear()
                    mar_value = drowsiness_detector.get_current_mar()
                    display_frame = result
                else:
                    display_frame = frame_with_landmarks
                    print("No face detected")

                # Only show GUI if explicitly enabled
                if show_gui:
                    cv2.imshow("Drowsiness Detection", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        camera.release()
        if show_gui:
            cv2.destroyAllWindows()

# Main function
def main(run_on_camera=True, use_camera_handler=False, 
         enable_drowsiness=True, show_gui=True, verbose=False):
    # Initialize face mesh detector
    face_mesh = FaceMeshDetector(max_faces=1, min_detection_conf=0.5, verbose=verbose)
    
    # Initialize drowsiness detector
    drowsiness_detector = DrowsinessDetector() if enable_drowsiness else None

    if run_on_camera:
        if use_camera_handler:
            process_camera_handler(face_mesh, drowsiness_detector, show_gui)
        else:
            process_camera_feed(face_mesh, drowsiness_detector, show_gui)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drowsiness detection using face mesh")
    parser.add_argument("-roc", "--run_on_camera", action='store_true', default=True, 
                       help="Set to True to run on camera feed (default: True)")
    parser.add_argument("-ch", "--use_camera_handler", action='store_true', default=False, 
                       help="Set to True to use CameraHandler, False to use cv2.VideoCapture")
    parser.add_argument("-ed", "--enable_drowsiness", action='store_true', default=True, 
                       help="Enable drowsiness detection (default: True)")
    parser.add_argument("-ng", "--no_gui", action='store_true', default=False, 
                       help="Disable GUI display (no window display)")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    main(args.run_on_camera, 
         args.use_camera_handler, 
         args.enable_drowsiness, 
         not args.no_gui,
         args.verbose)
