#!/usr/bin/env python3
import cv2
import argparse
import threading

from CameraUtilis.CameraHandler import CameraHandler
from Model.Detector import Detector
from drowsiness.EAR import DrowsinessDetector

output_frame = None
lock = threading.Lock()

# Function to process camera feed using cv2.VideoCapture
def process_camera_feed(detector, drowsiness_detector=None, show_gui=False):
    global output_frame
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
            
            # Process the frame with detection
            detection_result = detector.detect(frame)
            frame_with_detections = detector.draw_detections(frame)
            
            # Run landmark detection on the frame
            landmarks = detector.landmark(frame)
            frame_with_landmarks = detector.draw_landmarks(frame_with_detections)
            
            # Get eye and mouth keypoints for drowsiness detection
            eye_mouth = detector.get_eye_mouth_keypoints()
            
            # If landmarks found, perform drowsiness detection
            if eye_mouth:
                result = drowsiness_detector.process_frame(frame_with_landmarks, eye_mouth)
                ear_value = drowsiness_detector.get_current_ear()
                mar_value = drowsiness_detector.get_current_mar()
                display_frame = result
                
                if drowsiness_detector.is_drowsy():
                    print(f"ALERT: Drowsiness detected! EAR: {ear_value:.2f}")
                if drowsiness_detector.is_yawning():
                    print(f"ALERT: Yawning detected! MAR: {mar_value:.2f}")
            else:
                display_frame = frame_with_landmarks
                print("No face detected")

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
def process_camera_handler(detector, drowsiness_detector=None, show_gui=False):
    camera = CameraHandler(0)  # Initialize CameraHandler
    try:
        while True:
            timestamp, frame = camera.read()
            if frame is not None:
                # Resize the frame
                frame = cv2.resize(frame, (480, 360))
                
                # Process the frame with detection
                detection_result = detector.detect(frame)
                frame_with_detections = detector.draw_detections(frame)
                
                # Run landmark detection on the frame
                landmarks = detector.landmark(frame)
                frame_with_landmarks = detector.draw_landmarks(frame_with_detections)
                
                # Get eye and mouth keypoints for drowsiness detection
                eye_mouth = detector.get_eye_mouth_keypoints()
                
                # If landmarks found, perform drowsiness detection
                if eye_mouth:
                    result = drowsiness_detector.process_frame(frame_with_landmarks, eye_mouth)
                    ear_value = drowsiness_detector.get_current_ear()
                    mar_value = drowsiness_detector.get_current_mar()
                    display_frame = result
                    
                    if drowsiness_detector.is_drowsy():
                        print(f"ALERT: Drowsiness detected! EAR: {ear_value:.2f}")
                    if drowsiness_detector.is_yawning():
                        print(f"ALERT: Yawning detected! MAR: {mar_value:.2f}")
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

# Function to process image and save the result (matching main.py structure)
def process_image_and_save(detector, drowsiness_detector, image_path, output_path, show_gui=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Process the frame with detection
    detection_result = detector.detect(image)
    frame_with_detections = detector.draw_detections(image)
    
    # Run landmark detection on the frame
    landmarks = detector.landmark(image)
    frame_with_landmarks = detector.draw_landmarks(frame_with_detections)
    
    # Get eye and mouth keypoints for drowsiness detection
    eye_mouth = detector.get_eye_mouth_keypoints()
    
    # If landmarks found, perform drowsiness detection
    if eye_mouth:
        result = drowsiness_detector.process_frame(frame_with_landmarks, eye_mouth)
        ear_value = drowsiness_detector.get_current_ear()
        mar_value = drowsiness_detector.get_current_mar()
        
        print(f"Analysis results - EAR: {ear_value:.2f}, MAR: {mar_value:.2f}")
        print(f"Drowsy: {drowsiness_detector.is_drowsy()}, Yawning: {drowsiness_detector.is_yawning()}")
        
        # Display the image if GUI is enabled
        if show_gui:
            cv2.imshow("Drowsiness Analysis", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save the result
        cv2.imwrite(output_path, result)
        print(f"Processed image saved to {output_path}")
    else:
        print("No face detected in the image")

# Main function (matching main.py structure)
def main(run_on_camera=True, use_camera_handler=False, image_path=None, output_path=None,
         detector_type='landmark', enable_drowsiness=True, show_gui=True, verbose=False):
    
    # Initialize detector with the specified type
    detector = Detector(detector_type=detector_type, min_detection_conf=0.5, verbose=verbose)
    
    # Initialize drowsiness detector
    drowsiness_detector = DrowsinessDetector() if enable_drowsiness else None

    if run_on_camera:
        print("Starting drowsiness detection...")
        print(f"Using {'CameraHandler' if use_camera_handler else 'OpenCV VideoCapture'}")
        print(f"Detector type: {detector_type}")
        print("Press 'q' to quit")
        
        if use_camera_handler:
            process_camera_handler(detector, drowsiness_detector, show_gui)
        else:
            process_camera_feed(detector, drowsiness_detector, show_gui)
    else:
        if image_path is None or output_path is None:
            raise ValueError("Image path and output path must be provided when run_on_camera is False")
        process_image_and_save(detector, drowsiness_detector, image_path, output_path, show_gui)

if __name__ == "__main__":
    # Command-line argument parsing (matching main.py structure)
    parser = argparse.ArgumentParser(description="Drowsiness detection using face detection")
    parser.add_argument("-roc", "--run_on_camera", action='store_true', default=False, 
                       help="Set to True to run on camera feed, False to run on an image")
    parser.add_argument("-ch", "--use_camera_handler", action='store_true', default=False, 
                       help="Set to True to use CameraHandler, False to use cv2.VideoCapture")
    parser.add_argument('-ip', "--image_path", type=str, default=None, 
                       help="Provide the path to your test image")
    parser.add_argument('-op', "--output_path", type=str, default=None, 
                       help="Provide the path to save the processed image")
    parser.add_argument('-dt', "--detector_type", type=str, default='mediapipe', 
                       help="Type of detector to use ('yolov8_onnx', 'yolov8', 'mediapipe')")
    parser.add_argument('-ed', "--enable_drowsiness", action='store_true', default=True, 
                       help="Enable drowsiness detection")
    parser.add_argument('-gui', "--show_gui", action='store_true', default=False, 
                       help="Enable GUI display (imshow windows)")
    parser.add_argument('-v', "--verbose", action='store_true', default=False, 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    main(args.run_on_camera, 
         args.use_camera_handler, 
         args.image_path,
         args.output_path,
         args.detector_type,
         args.enable_drowsiness, 
         args.show_gui,
         args.verbose)
