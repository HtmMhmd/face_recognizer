#!/usr/bin/env python3
import cv2
import time
import argparse
import numpy as np
import threading
from src.utils.zmq_utils import ZmqPublisher, ZmqSubscriber, ZmqTopics, CaptureCommands, ZmqConfig
from src.config.settings import Settings

# Load application settings
settings = Settings()

# Sample face images for simulation
SAMPLE_FACES = [
    "data/hatem.png",
    "data/abokhalil.png"
]

class FaceRecognitionSimulator:
    """Simulates the face recognition system without real cameras."""
    
    def __init__(self, capture_port=None, image_port=None, recognition_port=None, deployment_mode="local"):
        """Initialize the simulator.
        
        Args:
            capture_port (int, optional): Port for capture commands. If None, uses config value.
            image_port (int, optional): Port for publishing simulated images. If None, uses config value.
            recognition_port (int, optional): Port for receiving recognition results. If None, uses config value.
            deployment_mode (str): Deployment mode ('local' or 'docker')
        """
        # Use configuration or fall back to defaults
        self.capture_port = capture_port or ZmqConfig.get_port("capture", 5555)
        self.image_port = image_port or ZmqConfig.get_port("image", 5556)
        self.recognition_port = recognition_port or ZmqConfig.get_port("recognition", 5558)
        self.host = ZmqConfig.get_host("dashboard_service", deployment_mode)
        
        print(f"Initializing simulator with: capture_port={self.capture_port}, "
              f"image_port={self.image_port}, recognition_port={self.recognition_port}, "
              f"host={self.host}")
        
        # Initialize ZMQ subscribers
        self.command_subscriber = ZmqSubscriber(
            host=self.host,
            port=self.capture_port,
            topic_name=ZmqTopics.CAPTURE.decode('utf-8')
        )
        
        self.recognition_subscriber = ZmqSubscriber(
            host=self.host,
            port=self.recognition_port,
            topic_name=ZmqTopics.RECOGNITION_RESULT.decode('utf-8')
        )
        
        # Initialize ZMQ publishers
        self.image_publisher = ZmqPublisher(
            port=self.image_port,
            topic_name=ZmqTopics.IMAGE.decode('utf-8')
        )
        
        # Load sample faces
        self.sample_faces = []
        for face_path in SAMPLE_FACES:
            try:
                face = cv2.imread(face_path)
                if face is not None:
                    self.sample_faces.append(face)
                    print(f"Loaded sample face: {face_path}")
                else:
                    print(f"Failed to load sample face: {face_path}")
            except Exception as e:
                print(f"Error loading sample face {face_path}: {e}")
        
        # If no sample faces were loaded, create dummy faces
        if not self.sample_faces:
            print("No sample faces loaded, creating dummy faces")
            dummy1 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy1, "Dummy Face 1", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            dummy2 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy2, "Dummy Face 2", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            self.sample_faces = [dummy1, dummy2]
        
        # Current face index
        self.current_face_index = 0
        
        # Flag to control threads
        self.running = True
        
        print("Face Recognition Simulator initialized")
    
    def listen_for_commands(self):
        """Listen for capture commands and respond with images."""
        print("Listening for capture commands...")
        
        while self.running:
            # Check for capture commands
            result = self.command_subscriber.receive(timeout=100)  # 100ms timeout
            
            if result is not None:
                topic, command = result
                
                if isinstance(command, dict):
                    command_type = command.get('command', '')
                elif isinstance(command, str):
                    command_type = command
                else:
                    # Try to decode bytes
                    try:
                        command_type = command.decode('utf-8')
                    except:
                        command_type = ""
                
                print(f"Received capture command: {command_type}")
                
                # Process commands
                if command_type in [CaptureCommands.TAKE_FRONT, CaptureCommands.TAKE_LEFT, CaptureCommands.TAKE_RIGHT]:
                    # Get the next sample face
                    face = self.sample_faces[self.current_face_index]
                    self.current_face_index = (self.current_face_index + 1) % len(self.sample_faces)
                    
                    # Add command text to the image
                    face_copy = face.copy()
                    cv2.putText(face_copy, f"Command: {command_type}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add timestamp
                    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(face_copy, timestamp_str, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Publish the image
                    self.image_publisher.publish_image(face_copy)
                    print(f"Published simulated image for command: {command_type}")
            
            time.sleep(0.01)
    
    def listen_for_recognition_results(self):
        """Listen for recognition results."""
        print("Listening for recognition results...")
        
        while self.running:
            # Check for recognition results
            result = self.recognition_subscriber.receive(timeout=100)  # 100ms timeout
            
            if result is not None:
                topic, recognition_result = result
                
                if isinstance(recognition_result, dict):
                    user_name = recognition_result.get('user_name', 'unknown')
                    verified = recognition_result.get('verified', False)
                    confidence = recognition_result.get('confidence', 0.0)
                    
                    print(f"Recognition Result: User={user_name}, Verified={verified}, Confidence={confidence:.2f}")
            
            time.sleep(0.01)
    
    def run(self):
        """Run the simulator."""
        # Start threads for listening
        command_thread = threading.Thread(target=self.listen_for_commands)
        command_thread.daemon = True
        command_thread.start()
        
        recognition_thread = threading.Thread(target=self.listen_for_recognition_results)
        recognition_thread.daemon = True
        recognition_thread.start()
        
        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Simulator interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        self.running = False
        time.sleep(0.5)  # Give threads time to exit
        
        self.command_subscriber.close()
        self.recognition_subscriber.close()
        self.image_publisher.close()
        print("Simulator stopped")

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Simulator")
    parser.add_argument("--capture-port", type=int, default=None,
                      help="Port for capture commands (overrides config)")
    parser.add_argument("--image-port", type=int, default=None,
                      help="Port for publishing simulated images (overrides config)")
    parser.add_argument("--recognition-port", type=int, default=None,
                      help="Port for receiving recognition results (overrides config)")
    parser.add_argument("--deployment-mode", type=str, choices=["local", "docker"], default="local",
                      help="Deployment mode (local or docker)")
    
    args = parser.parse_args()
    
    simulator = FaceRecognitionSimulator(
        capture_port=args.capture_port,
        image_port=args.image_port,
        recognition_port=args.recognition_port,
        deployment_mode=args.deployment_mode
    )
    
    simulator.run()

if __name__ == "__main__":
    main()
