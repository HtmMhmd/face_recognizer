import zmq
import subprocess
import numpy as np
import cv2
import time
import os
import logging
from src.utils.camera.camera_handler import CameraHandler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ZMQ_Automation")

# Suppress OpenCV error logs
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "ERROR"

class DockerAutomation:
    def __init__(self, port=5555, camera_index=0):
        """Initialize ZMQ server and Docker automation."""
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        logger.info(f"ZMQ server started on port {port}")
        
        # Docker commands
        self.docker_run_cmd = "docker compose up mjpg-streamer -d"
        self.docker_compose_cmd = "docker compose up -d"
        
        # Camera setup
        self.camera_index = camera_index
        self.camera = None
        self.camera_initialized = False
        self.retry_count = 0
        self.max_retries = 3
        

    def run_docker_container(self):
        """Run the Docker container and return the container ID."""
        try:
            logger.info("Starting Docker container...")
            result = subprocess.run(self.docker_run_cmd, shell=True, check=True, 
                                   text=True, capture_output=True)
            container_id = result.stdout.strip()
            logger.info(f"Docker container started with ID: {container_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker container: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def run_docker_compose(self):
        """Start services with Docker Compose."""
        try:
            logger.info("Starting Docker Compose services...")
            result = subprocess.run(self.docker_compose_cmd, shell=True, check=True, 
                                   text=True, capture_output=True)
            logger.info("Docker Compose services started successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker Compose services: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def initialize_camera(self):
        """Initialize the camera if not already initialized using CameraHandler."""
        if self.camera_initialized and self.camera is not None:
            return True
            
        try:
            # Release any existing camera
            if self.camera is not None:
                self.camera.release()
                
            # Use CameraHandler instead of cv2.VideoCapture
            self.camera = CameraHandler(self.camera_index)
            
            # Wait for camera initialization
            time.sleep(1)
            
            # Check if we can read from the camera
            timestamp, test_frame = self.camera.read()
            
            if test_frame is None:
                logger.error(f"Failed to read frame from camera (attempt {self.retry_count + 1}/{self.max_retries})")
                
                # Try a few more times before giving up
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    time.sleep(1)  # Wait a bit before retrying
                    return self.initialize_camera()
                else:
                    self.retry_count = 0
                    return False
            
            self.retry_count = 0  # Reset retry counter on success
            logger.info("Camera initialized successfully using CameraHandler")
            self.camera_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.camera = None
            self.camera_initialized = False
            return False
    
    def release_camera(self):
        """Release the camera resource."""
        if self.camera_initialized and self.camera is not None:
            try:
                self.camera.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self.camera = None
                self.camera_initialized = False
    
    def capture_camera_image(self):
        """Capture an image from the camera using CameraHandler."""
        # Try to initialize or reinitialize the camera
        if not self.initialize_camera():
            # Fall back to black image if camera initialization fails
            logger.warning("Using black image as fallback due to camera initialization failure")
            return self.create_black_image()
            
        try:
            # Read a frame from CameraHandler
            for _ in range(3):  # Try a few times to get a good frame
                timestamp, frame = self.camera.read()
                
                if frame is not None:
                    break
                time.sleep(0.1)  # Small delay between attempts
            
            if frame is None:
                logger.error("Failed to capture valid frame from camera after multiple attempts")
                return self.create_black_image()
                
            # Resize the frame
            frame = cv2.resize(frame, (640, 480))
            
            # Add timestamp to the image
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp_str, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode the image
            _, img_encoded = cv2.imencode('.jpg', frame)
            logger.info(f"Successfully captured frame from camera at {timestamp_str}")
            return img_encoded.tobytes()
            
        except Exception as e:
            logger.error(f"Error capturing from camera: {e}")
            # Camera might be in a bad state, try to release it
            self.release_camera()
            return self.create_black_image()
    
    def create_black_image(self):
        """Create a black image as fallback."""
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add text to show this is a fallback image
        cv2.putText(black_img, "Camera Unavailable", (160, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(black_img, time.strftime("%Y-%m-%d %H:%M:%S"), (160, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                   
        _, img_encoded = cv2.imencode('.jpg', black_img)
        return img_encoded.tobytes()
    
    def run(self):
        """Main loop to handle incoming ZMQ messages."""
        logger.info("Waiting for messages...")
        
        while True:
            try:
                # Receive message
                message = self.socket.recv_string()
                logger.info(f"Received message: {message}")
                
                # Process based on message content
                if message.lower() == "init":
                    # Handle 'init' command - start Docker container and reply with '1'
                    success = self.run_docker_container()
                    if success:
                        self.socket.send_string("init_ack")
                        logger.info("Sent response: 1")
                    else:
                        self.socket.send_string("init_nack")
                        logger.info("Sent response: 0 (Docker container failed to start)")
                
                elif message.lower() == "capture":
                    # Handle 'capture' command - capture camera image without starting Docker
                    camera_image = self.capture_camera_image()
                    self.socket.send(camera_image)
                    logger.info("Sent camera image response")
                    
                elif message.lower() == "capture_with_docker":
                    # Handle command to start Docker and then capture
                    success = self.run_docker_compose()
                    camera_image = self.capture_camera_image()
                    self.socket.send(camera_image)
                    logger.info("Sent camera image response after Docker startup")
                
                elif message.lower().startswith("camera:"):
                    # Change camera index - format: "camera:1" for index 1
                    try:
                        new_index = int(message.split(':')[1])
                        # Release existing camera if any
                        self.release_camera()
                        self.camera_index = new_index
                        logger.info(f"Camera index changed to {new_index}")
                        self.socket.send_string(f"Camera index set to {new_index}")
                    except (IndexError, ValueError) as e:
                        logger.error(f"Invalid camera index specification: {e}")
                        self.socket.send_string("Error: Invalid camera index format. Use 'camera:N'")
                
                elif message.lower() == "restart_camera":
                    # Force camera reinitializing
                    self.release_camera()
                    success = self.initialize_camera()
                    if success:
                        self.socket.send_string("Camera restarted successfully")
                    else:
                        self.socket.send_string("Failed to restart camera")
                
                else:
                    # Handle unknown commands
                    self.socket.send_string(f"Unknown command: {message}")
                    logger.warning(f"Received unknown command: {message}")
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                try:
                    self.socket.send_string(f"Error: {str(e)}")
                except zmq.error.ZMQError:
                    logger.error("Failed to send error response")
                    
    def cleanup(self):
        """Clean up resources."""
        self.release_camera()
        self.socket.close()
        self.context.term()
        logger.info("ZMQ resources cleaned up")


if __name__ == "__main__":
    automation = DockerAutomation()
    try:
        automation.run()
    except KeyboardInterrupt:
        logger.info("Shutting down ZMQ automation server...")
    finally:
        automation.cleanup()