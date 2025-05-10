import cv2
import time
import json
from datetime import datetime

from src.services.image_processor import ImageProcessor
from src.core.drowsiness.detection import DrowsinessDetector
from src.utils.camera.camera_handler import CameraHandler

def run_camera_feed(detector_type='mediapipe', enable_drowsiness=False, show_gui=False, 
                   output_json=None, camera_index=0, verbose=False):
    """
    Run the face recognition system on a live camera feed.
    
    Args:
        detector_type (str): Type of detector to use ('mediapipe', 'yolov8', 'yolov8_onnx')
        enable_drowsiness (bool): Whether to enable drowsiness detection
        show_gui (bool): Whether to show GUI display (requires access to display)
        output_json (str): Path to save JSON output of face recognition results
        camera_index (int): Camera index to use (default: 0)
        verbose (bool): Enable verbose output
    """
    print(f"[INFO] Starting face recognition with {detector_type} detector")
    
    # Initialize image processor
    image_processor = ImageProcessor(model_architecture=detector_type, verbose=verbose)
    
    # Initialize drowsiness detector if enabled
    drowsiness_detector = DrowsinessDetector() if enable_drowsiness else None
    
    # Open video capture
    print(f"[INFO] Opening camera {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}")
        return
    
    try:
        # Dictionary to store recognition results
        recognition_results = {}
        
        print("[INFO] Processing video stream. Press 'q' to quit.")
        while True:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            # Resize the frame for faster processing
            frame = cv2.resize(frame, (480, 360))
            
            # Process the frame
            embeddings = image_processor.process_image(frame)
            
            # Check if faces were detected
            if embeddings is None or len(embeddings.embeddings) == 0:
                print("[INFO] No faces detected")
                if show_gui:
                    cv2.imshow("Face Recognition", frame)
                continue
            
            # Clone the frame for drawing
            display_frame = frame.copy()
            
            # Draw detections
            display_frame = image_processor.draw_detections(display_frame)
            
            # Detect landmarks
            landmarks = image_processor.detect_landmarks(frame)
            
            # Draw landmarks
            display_frame = image_processor.draw_landmarks(display_frame)
            
            # Verify faces against the database
            verify_results = image_processor.verify_faces()
            
            # Draw user names
            display_frame = image_processor.draw_user_names(display_frame, verify_results)
            
            # Get eye and mouth keypoints for drowsiness detection
            eye_mouth = image_processor.get_eye_mouth_keypoints()
            
            # Process drowsiness detection if enabled
            if drowsiness_detector and eye_mouth:
                display_frame = drowsiness_detector.process_frame(display_frame, eye_mouth)
            
            # Print recognition results
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[RESULTS] {timestamp}")
            
            if verify_results:
                for result in verify_results:
                    user_name = result.get('user_name', 'Unknown')
                    verified = result.get('verification_result', False)
                    confidence = result.get('confidence', 0)
                    print(f"  User: {user_name}, Verified: {verified}")
                    
                    # Store the latest result for each user
                    recognition_results[user_name] = {
                        'timestamp': timestamp,
                        'verified': verified
                    }
            else:
                print("  No users recognized")
            
            # Show the frame with detections if GUI is enabled
            if show_gui:
                cv2.imshow("Face Recognition", display_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Short delay to reduce CPU usage
            time.sleep(0.01)
            
    finally:
        # Clean up
        cap.release()
        if show_gui:
            cv2.destroyAllWindows()
        
        # Save results to JSON if requested
        if output_json and recognition_results:
            try:
                with open(output_json, 'w') as f:
                    json.dump(recognition_results, f, indent=4)
                print(f"[INFO] Results saved to {output_json}")
            except Exception as e:
                print(f"[ERROR] Failed to save results: {e}")
        
        print("[INFO] Face recognition stopped")

def run_with_camera_handler(detector_type='mediapipe', enable_drowsiness=False, show_gui=False, 
                          output_json=None, camera_index=0, verbose=False):
    """
    Alternative implementation using the threaded CameraHandler for potentially better performance.
    """
    print(f"[INFO] Starting face recognition with {detector_type} detector using CameraHandler")
    
    # Initialize image processor
    image_processor = ImageProcessor(model_architecture=detector_type, verbose=verbose)
    
    # Initialize drowsiness detector if enabled
    drowsiness_detector = DrowsinessDetector() if enable_drowsiness else None
    
    # Initialize camera handler
    camera = CameraHandler(camera_index)
    
    try:
        recognition_results = {}
        
        while True:
            timestamp, frame = camera.read()
            if frame is None:
                continue
                
            # Resize the frame
            frame = cv2.resize(frame, (480, 360))

            # Process the frame
            embeddings = image_processor.process_image(frame)
            
            # Add landmark detection and drowsiness detection functionality
            landmarks = image_processor.detect_landmarks(frame)
            if landmarks:
                display_frame = image_processor.draw_landmarks(frame)

            # Process drowsiness detection if enabled
            if drowsiness_detector and landmarks:
                display_frame = drowsiness_detector.process_frame(display_frame, landmarks)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.release()
        if show_gui:
            cv2.destroyAllWindows()
            
        # Save results to JSON if requested
        if output_json and recognition_results:
            try:
                with open(output_json, 'w') as f:
                    json.dump(recognition_results, f, indent=4)
                print(f"[INFO] Results saved to {output_json}")
            except Exception as e:
                print(f"[ERROR] Failed to save results: {e}")
                
        print("[INFO] Face recognition stopped")