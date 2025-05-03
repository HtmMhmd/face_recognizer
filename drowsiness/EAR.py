import cv2
import numpy as np
import subprocess
import os
import threading
import time
from collections import deque
from scipy.spatial import distance

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, mar_threshold=0.5, 
                 drowsy_time_threshold=2.0, ear_frames=20, 
                 text_pos=(10, 30), verbose=False):
        """
        Initialize the drowsiness detector with thresholds and tracking variables.
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold below which eyes are considered closed
            mar_threshold: Mouth Aspect Ratio threshold above which mouth is considered open
            drowsy_time_threshold: Time threshold (in seconds) for considering drowsiness
            ear_frames: Number of frames to keep in history for EAR
            text_pos: Position to display status text on frame
            verbose: Whether to print detailed information
        """
        # Thresholds
        self.EAR_THRESHOLD = ear_threshold
        self.MAR_THRESHOLD = mar_threshold
        self.DROWSY_TIME_THRESHOLD = drowsy_time_threshold
        
        # Tracking variables
        self.eye_closed_start_time = None
        self.current_ear = 1.0  # Default to open eyes
        self.current_mar = 0.0  # Default to closed mouth
        
        # History tracking
        self.ear_history = deque(maxlen=ear_frames)
        self.drowsy_status = False
        self.yawning_status = False
        
        # Display settings
        self.TEXT_POS = text_pos
        self.verbose = verbose

    def calculate_ear(self, eye_points):
        """Calculate the Eye Aspect Ratio (EAR) for given eye points."""
        if not eye_points or len(eye_points) < 6:
            return 1.0  # Default to open eyes if points missing
            
        # Calculate vertical distances
        v1 = distance.euclidean(eye_points[1], eye_points[5])
        v2 = distance.euclidean(eye_points[2], eye_points[4])
        
        # Calculate horizontal distance
        h = distance.euclidean(eye_points[0], eye_points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 1.0
        
        return ear

    def calculate_mar(self, mouth_points):
        """Calculate the Mouth Aspect Ratio (MAR) for given mouth points."""
        if not mouth_points or len(mouth_points) < 6:
            return 0.0  # Default to closed mouth if points missing
            
        # Calculate vertical distance
        v = distance.euclidean(mouth_points[2], mouth_points[5])
        
        # Calculate horizontal distances
        h1 = distance.euclidean(mouth_points[0], mouth_points[3])
        h2 = distance.euclidean(mouth_points[1], mouth_points[4])
        
        # Calculate MAR
        mar = v / ((h1 + h2) / 2.0) if h1 + h2 > 0 else 0.0
        
        return mar

    def process_frame(self, frame, eye_mouth_keypoints):
        """
        Process a frame for drowsiness detection.
        
        Args:
            frame: The video frame to process
            eye_mouth_keypoints: Dictionary with keys 'left_eye', 'right_eye', 'mouth' 
                                 containing lists of (x,y) coordinate tuples
                                 
        Returns:
            The frame with drowsiness indicators drawn
        """
        # Make a copy of the frame to avoid modifying the original
        result_frame = frame.copy()
        
        # Check if we have valid keypoints
        if not eye_mouth_keypoints:
            if self.verbose:
                print("No valid keypoints provided")
            return result_frame
            
        left_eye = eye_mouth_keypoints.get('left_eye', [])
        right_eye = eye_mouth_keypoints.get('right_eye', [])
        mouth = eye_mouth_keypoints.get('mouth', [])
        
        # Calculate EAR
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average the EAR of both eyes
        self.current_ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(self.current_ear)
        
        # Calculate MAR
        self.current_mar = self.calculate_mar(mouth)
        
        # Check for drowsiness
        if self.current_ear < self.EAR_THRESHOLD:
            # Eyes are closed
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = time.time()
            
            # Check if eyes have been closed for long enough
            if self.eye_closed_start_time and time.time() - self.eye_closed_start_time >= self.DROWSY_TIME_THRESHOLD:
                self.drowsy_status = True
                cv2.putText(result_frame, "DROWSINESS ALERT!", self.TEXT_POS, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(result_frame, "Eyes Closed", self.TEXT_POS, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Eyes are open
            self.eye_closed_start_time = None
            self.drowsy_status = False
            cv2.putText(result_frame, "Eyes Open", self.TEXT_POS, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Check for yawning
        if self.current_mar > self.MAR_THRESHOLD:
            self.yawning_status = True
            cv2.putText(result_frame, f"Yawning (MAR: {self.current_mar:.2f})", 
                       (self.TEXT_POS[0], self.TEXT_POS[1] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.yawning_status = False
            cv2.putText(result_frame, f"MAR: {self.current_mar:.2f}", 
                       (self.TEXT_POS[0], self.TEXT_POS[1] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the EAR value
        cv2.putText(result_frame, f"EAR: {self.current_ear:.2f}", 
                   (self.TEXT_POS[0], self.TEXT_POS[1] + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                   
        return result_frame

    def is_drowsy(self) -> bool:
        """
        Return True if the person is detected as drowsy, False otherwise.
        Uses both current EAR and historical EAR values to determine drowsiness.
        """
        # Check if current EAR indicates drowsiness
        return self.drowsy_status

    def is_yawning(self) -> bool:
        """
        Return True if the person is detected as yawning, False otherwise.
        Uses the mouth aspect ratio to determine if person is yawning.
        """
        # Return True if MAR exceeds threshold
        return self.yawning_status
        
    def get_current_ear(self):
        """Get the current Eye Aspect Ratio."""
        return self.current_ear
        
    def get_current_mar(self):
        """Get the current Mouth Aspect Ratio."""
        return self.current_mar

if __name__ == "__main__":
    # Create a single camera instance
    camera = cv2.VideoCapture(0)
    detector = DrowsinessDetector()
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        # Simulate keypoints for testing
        keypoints = {
            "left_eye": [(30, 40), (35, 45), (40, 50), (45, 55), (50, 60), (55, 65)],
            "right_eye": [(60, 70), (65, 75), (70, 80), (75, 85), (80, 90), (85, 95)],
            "mouth": [(100, 110), (105, 115), (110, 120), (115, 125), (120, 130), (125, 135)]
        }
        processed_frame = detector.process_frame(frame, keypoints)
        cv2.imshow("Drowsiness and Yawning Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()