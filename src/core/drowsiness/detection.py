import cv2
import numpy as np
import time
from collections import deque
from src.config import drowsiness_settings

class DrowsinessDetector:
    def __init__(self):
        # Use configuration values for thresholds
        self.ear_threshold = drowsiness_settings.get("ear_threshold", 0.25)
        self.mar_threshold = drowsiness_settings.get("mar_threshold", 0.5)
        self.drowsy_time_threshold = drowsiness_settings.get("drowsy_time_threshold", 2.0)
        
        # Number of frames to keep EAR history
        self.ear_frames = drowsiness_settings.get("ear_frames", 20)
        
        # Position to display status text
        self.text_position = tuple(drowsiness_settings.get("text_position", [10, 30]))
        
        # Internal state variables
        self.ear_history = deque(maxlen=self.ear_frames)
        self.drowsy_start_time = None
        self.status = "Alert"
        self.status_color = (0, 255, 0)
    
    def calculate_ear(self, eye_points):
        """Calculate eye aspect ratio for a single eye"""
        # Vertical eye landmarks (p1-p5, p2-p4)
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal eye landmarks
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_points):
        """Calculate mouth aspect ratio"""
        # Vertical mouth landmarks
        A = np.linalg.norm(mouth_points[1] - mouth_points[7])
        B = np.linalg.norm(mouth_points[2] - mouth_points[6])
        C = np.linalg.norm(mouth_points[3] - mouth_points[5])
        
        # Horizontal mouth landmarks
        D = np.linalg.norm(mouth_points[0] - mouth_points[4])
        
        # Calculate MAR
        mar = (A + B + C) / (3.0 * D)
        return mar
    
    def process_frame(self, frame, landmarks):
        """
        Process a frame to detect drowsiness
        
        Args:
            frame: Video frame to process
            landmarks: Dict containing 'left_eye', 'right_eye', and 'mouth' keypoints
            
        Returns:
            frame: Processed frame with drowsiness info
        """
        # Extract eye and mouth landmarks
        left_eye = landmarks.get('left_eye')
        right_eye = landmarks.get('right_eye')
        mouth = landmarks.get('mouth')
        
        if left_eye is None or right_eye is None:
            return frame
        
        # Calculate metrics
        left_ear = self.calculate_ear(np.array(left_eye))
        right_ear = self.calculate_ear(np.array(right_eye))
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)
        
        # Calculate MAR if mouth points available
        mar = self.calculate_mar(np.array(mouth)) if mouth is not None else 0
        
        # Detect drowsiness
        if ear < self.ear_threshold or (mouth is not None and mar > self.mar_threshold):
            if self.drowsy_start_time is None:
                self.drowsy_start_time = time.time()
            
            # Check if drowsy for more than threshold seconds
            if time.time() - self.drowsy_start_time > self.drowsy_time_threshold:
                self.status = "DROWSY!"
                self.status_color = (0, 0, 255)  # Red
        else:
            self.drowsy_start_time = None
            self.status = "Alert"
            self.status_color = (0, 255, 0)  # Green
        
        # Draw EAR and status on frame
        cv2.putText(frame, f"Status: {self.status}", 
                    self.text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, self.status_color, 2)
        
        cv2.putText(frame, f"EAR: {ear:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 0, 0), 2)
                    
        if mouth is not None:
            cv2.putText(frame, f"MAR: {mar:.2f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 0), 2)
        
        return frame