#!/usr/bin/env python3
import cv2
import time
import argparse
import logging
import numpy as np
from src.utils.zmq_utils import ZmqSubscriber, ZmqPublisher, ZmqTopics, ZmqConfig
from src.services.image_processor import ImageProcessor
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Image_Processing_Service")

class ImageProcessingService:
    """Service responsible for processing images to detect and crop faces."""
    
    def __init__(self, image_port=None, cropped_face_port=None, host=None, 
                 detector_type=None, deployment_mode="local"):
        """Initialize the image processing service.
        
        Args:
            image_port (int, optional): Port to subscribe for images. If None, uses config value.
            cropped_face_port (int, optional): Port to publish cropped faces. If None, uses config value.
            host (str, optional): Host to connect to for subscribing. If None, uses config value.
            detector_type (str, optional): Type of detector to use ('mediapipe', 'yolov8', 'yolov8_onnx'). If None, uses config value.
            deployment_mode (str): Deployment mode ('local' or 'docker')
        """
        # Load application settings
        self.settings = Settings()
        
        # Use configuration or fall back to defaults
        self.image_port = image_port or ZmqConfig.get_port("image", 5556)
        self.cropped_face_port = cropped_face_port or ZmqConfig.get_port("cropped_face", 5557)
        self.zmq_host = host or ZmqConfig.get_host("capture_service", deployment_mode)
        self.detector_type = detector_type or self.settings.detection.get("default_model", "mediapipe")
        
        # Initialize ZMQ subscriber for images
        self.image_subscriber = ZmqSubscriber(
            host=self.zmq_host,
            port=self.image_port,
            topic_name=ZmqTopics.IMAGE.decode('utf-8')
        )
        
        # Initialize ZMQ publisher for cropped faces
        self.cropped_face_publisher = ZmqPublisher(
            port=self.cropped_face_port,
            topic_name=ZmqTopics.CROPPED_FACE.decode('utf-8')
        )
        
        # Initialize image processor
        self.image_processor = ImageProcessor(model_architecture=self.detector_type, verbose=False)
        
        logger.info(f"Image Processing Service initialized with {self.detector_type} detector")
        logger.info(f"ZMQ Host: {self.zmq_host}, Ports - Image: {self.image_port}, Cropped Face: {self.cropped_face_port}")
    
    def process_image(self, image):
        """Process the image to detect and crop faces.
        
        Args:
            image (numpy.ndarray): Input image to process
            
        Returns:
            dict: Processing results with cropped faces and metadata
        """
        try:
            # Process the image to detect faces and extract embeddings
            # Use filter_largest=True to only keep the largest face if multiple faces are detected
            detection_embedding = self.image_processor.process_image(image, filter_largest=True)
            
            if detection_embedding is None or not detection_embedding.detection_faces.boxes:
                logger.info("No faces detected in the image")
                return None
                
            # Get cropped faces
            cropped_faces = detection_embedding.detection_faces.cropped_faces
            if not cropped_faces:
                logger.info("No cropped faces available")
                return None
                
            # Get bounding boxes
            bboxes = detection_embedding.detection_faces.boxes
            
            # Create result dictionary
            result = {
                'timestamp': time.time(),
                'num_faces': len(cropped_faces),
                'faces': []
            }
            
            # Process each face
            for i, (cropped_face, bbox) in enumerate(zip(cropped_faces, bboxes)):
                if cropped_face is None or cropped_face.size == 0:
                    continue
                    
                # Encode the cropped face
                _, face_encoded = cv2.imencode('.jpg', cropped_face)
                
                # Add face info to result
                result['faces'].append({
                    'face_id': i,
                    'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox,
                    'face_data': face_encoded.tobytes()
                })
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
    
    def run(self):
        """Run the image processing service main loop."""
        logger.info("Image Processing Service starting...")
        
        try:
            while True:
                # Receive image
                image = self.image_subscriber.receive_image(timeout=1000)  # 1s timeout
                
                if image is not None:
                    logger.info("Received image for processing")
                    
                    # Process the image
                    result = self.process_image(image)
                    
                    if result is not None and result['num_faces'] > 0:
                        # Publish each face
                        for face_info in result['faces']:
                            face_result = {
                                'timestamp': result['timestamp'],
                                'face_id': face_info['face_id'],
                                'bbox': face_info['bbox'],
                                'face_data': face_info['face_data']
                            }
                            self.cropped_face_publisher.publish(face_result)
                            logger.info(f"Published cropped face {face_info['face_id']}")
                    else:
                        logger.info("No faces to publish")
                
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                
        except KeyboardInterrupt:
            logger.info("Image Processing Service interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.image_subscriber.close()
        self.cropped_face_publisher.close()
        logger.info("Image Processing Service stopped")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Recognition Image Processing Service")
    parser.add_argument("--image-port", type=int, default=None,
                      help="Port to subscribe for images (overrides config)")
    parser.add_argument("--cropped-face-port", type=int, default=None,
                      help="Port to publish cropped faces (overrides config)")
    parser.add_argument("--host", type=str, default=None,
                      help="Host to connect to for subscribing (overrides config)")
    parser.add_argument("--detector", type=str, default=None,
                      choices=["mediapipe", "yolov8", "yolov8_onnx"],
                      help="Face detector model to use (overrides config)")
    parser.add_argument("--deployment-mode", type=str, choices=["local", "docker"], default="local",
                      help="Deployment mode (local or docker)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    service = ImageProcessingService(
        image_port=args.image_port,
        cropped_face_port=args.cropped_face_port,
        host=args.host,
        detector_type=args.detector,
        deployment_mode=args.deployment_mode
    )
    
    service.run()
