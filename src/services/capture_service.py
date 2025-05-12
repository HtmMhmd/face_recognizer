#!/usr/bin/env python3
import cv2
import time
import os
import argparse
import logging
from src.utils.camera.camera_handler import CameraHandler
from src.utils.zmq_utils import ZmqSubscriber, ZmqPublisher, ZmqTopics, CaptureCommands, ZmqConfig
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Capture_Service")

# Suppress OpenCV error logs
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "ERROR"

class CaptureService:
    """Service responsible for capturing images from camera."""
    
    def __init__(self, capture_port=None, image_port=None, camera_index=None, host=None, 
                 deployment_mode="local"):
        """Initialize the capture service.
        
        Args:
            capture_port (int, optional): Port to subscribe for capture commands. If None, uses config value.
            image_port (int, optional): Port to publish captured images. If None, uses config value.
            camera_index (int, optional): Camera index to use. If None, uses config value.
            host (str, optional): Host to connect to for subscribing. If None, uses config value.
            deployment_mode (str): Deployment mode ('local' or 'docker')
        """
        # Load application settings
        self.settings = Settings()
        
        # Use configuration or fall back to defaults
        self.camera_index = camera_index if camera_index is not None else self.settings.camera.get("default_index", 0)
        self.capture_port = capture_port or ZmqConfig.get_port("capture", 5555)
        self.image_port = image_port or ZmqConfig.get_port("image", 5556)
        self.zmq_host = host or ZmqConfig.get_host("dashboard_service", deployment_mode)
        
        self.camera = None
        self.camera_initialized = False
        
        # Initialize ZMQ subscriber for capture commands
        self.command_subscriber = ZmqSubscriber(
            host=self.zmq_host, 
            port=self.capture_port, 
            topic_name=ZmqTopics.CAPTURE.decode('utf-8')
        )
        
        # Initialize ZMQ publisher for captured images
        self.image_publisher = ZmqPublisher(
            port=self.image_port,
            topic_name=ZmqTopics.IMAGE.decode('utf-8')
        )
        
        logger.info(f"Capture Service initialized with camera {self.camera_index}")
        logger.info(f"ZMQ Host: {self.zmq_host}, Ports - Capture: {self.capture_port}, Image: {self.image_port}")
        
        logger.info(f"Capture Service initialized with camera {camera_index}")
    
    def initialize_camera(self):
        """Initialize the camera if not already initialized."""
        if self.camera_initialized and self.camera is not None:
            return True
            
        try:
            # Release any existing camera
            if self.camera is not None:
                self.camera.release()
                
            # Use CameraHandler for camera access
            logger.info(f"Initializing camera {self.camera_index}")
            self.camera = CameraHandler(self.camera_index)
            
            # Wait for camera initialization
            time.sleep(1)
            
            # Check if we can read from the camera
            timestamp, test_frame = self.camera.read()
            
            if test_frame is None:
                logger.error("Failed to read frame from camera")
                return False
                
            logger.info("Camera initialized successfully")
            self.camera_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.camera = None
            self.camera_initialized = False
            return False
    
    def capture_image(self):
        """Capture an image from the camera.
        
        Returns:
            numpy.ndarray: Captured image or None if failed
        """
        if not self.camera_initialized:
            if not self.initialize_camera():
                logger.error("Failed to initialize camera for capture")
                return None
        
        try:
            timestamp, frame = self.camera.read()
            
            if frame is None:
                logger.error("Failed to capture frame")
                return None
                
            # Resize the frame
            frame = cv2.resize(frame, (640, 480))
            
            # Add timestamp to the image
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp_str, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            logger.info(f"Successfully captured frame at {timestamp_str}")
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None
    
    def release_camera(self):
        """Release camera resources."""
        if self.camera_initialized and self.camera is not None:
            try:
                self.camera.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self.camera = None
                self.camera_initialized = False
    
    def run(self):
        """Run the capture service main loop."""
        logger.info("Capture Service starting...")
        
        try:
            while True:
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
                    
                    logger.info(f"Received capture command: {command_type}")
                    
                    # Initialize camera if needed
                    if not self.camera_initialized:
                        self.initialize_camera()
                    
                    # Process commands
                    if command_type == CaptureCommands.TAKE_FRONT or \
                       command_type == CaptureCommands.TAKE_LEFT or \
                       command_type == CaptureCommands.TAKE_RIGHT:
                        
                        # Capture image
                        image = self.capture_image()
                        
                        if image is not None:
                            # Publish the captured image
                            self.image_publisher.publish_image(image)
                            logger.info(f"Published captured image for command: {command_type}")
                        else:
                            logger.error(f"Failed to capture image for command: {command_type}")
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                
        except KeyboardInterrupt:
            logger.info("Capture Service interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.release_camera()
        self.command_subscriber.close()
        self.image_publisher.close()
        logger.info("Capture Service stopped")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Recognition Capture Service")
    parser.add_argument("--capture-port", type=int, default=None,
                      help="Port to subscribe for capture commands (overrides config)")
    parser.add_argument("--image-port", type=int, default=None,
                      help="Port to publish captured images (overrides config)")
    parser.add_argument("--camera", type=int, default=None,
                      help="Camera index to use (overrides config)")
    parser.add_argument("--host", type=str, default=None,
                      help="Host to connect to for subscribing (overrides config)")
    parser.add_argument("--deployment-mode", type=str, choices=["local", "docker"], default="local",
                      help="Deployment mode (local or docker)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    service = CaptureService(
        capture_port=args.capture_port,
        image_port=args.image_port,
        camera_index=args.camera,
        host=args.host,
        deployment_mode=args.deployment_mode
    )
    
    service.run()
