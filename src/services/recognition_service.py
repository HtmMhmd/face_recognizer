#!/usr/bin/env python3
import cv2
import time
import argparse
import logging
import numpy as np
import json
import os
from src.utils.zmq_utils import ZmqSubscriber, ZmqPublisher, ZmqTopics, ZmqConfig
from src.services.image_processor import ImageProcessor
from src.database.face_db import FaceDatabase
from src.database.api.db_client import DatabaseClient
from src.core.verification import FaceVerifier
from src.utils.image import preprocess_image
from src.models.face_recognition import FaceNetTFLiteHandler
from src.config.settings import Settings
from src.utils.ml_config import optimize_ml_environment, configure_tflite_runtime
from src.utils.tflite_helpers import OptimizedTFLiteModel

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Recognition_Service")

# Apply ML optimizations before model loading - prioritize TFLite runtime
logger.info("Optimizing ML environment for face recognition")
optimize_ml_environment()

class RecognitionService:
    """Service responsible for face recognition and user verification."""
    
    def __init__(self, cropped_face_port=None, recognition_port=None, add_user_port=None, 
                add_user_response_port=None, host=None, deployment_mode="local"):
        """Initialize the recognition service.
        
        Args:
            cropped_face_port (int, optional): Port to subscribe for cropped faces. If None, uses config value.
            recognition_port (int, optional): Port to publish recognition results. If None, uses config value.
            add_user_port (int, optional): Port to subscribe for add user requests. If None, uses config value.
            add_user_response_port (int, optional): Port to publish add user responses. If None, uses config value.
            host (str, optional): Host to connect to for subscribing. If None, uses config value.
            deployment_mode (str): Deployment mode ('local' or 'docker')
        """
        # Load application settings
        self.settings = Settings()
        
        # Use configuration or fall back to defaults
        self.cropped_face_port = cropped_face_port or ZmqConfig.get_port("cropped_face", 5557)
        self.recognition_port = recognition_port or ZmqConfig.get_port("recognition", 5558)
        self.add_user_port = add_user_port or ZmqConfig.get_port("add_user", 5559)
        self.add_user_response_port = add_user_response_port or ZmqConfig.get_port("add_user_response", 5560)
        self.zmq_host = host or ZmqConfig.get_host("image_processing_service", deployment_mode)
        
        # Initialize ZMQ subscribers
        self.face_subscriber = ZmqSubscriber(
            host=self.zmq_host,
            port=self.cropped_face_port,
            topic_name=ZmqTopics.CROPPED_FACE.decode('utf-8')
        )
        
        self.add_user_subscriber = ZmqSubscriber(
            host=self.zmq_host,
            port=self.add_user_port,
            topic_name=ZmqTopics.ADD_USER_REQUEST.decode('utf-8')
        )
        
        # Initialize ZMQ publishers
        self.recognition_publisher = ZmqPublisher(
            port=self.recognition_port,
            topic_name=ZmqTopics.RECOGNITION_RESULT.decode('utf-8')
        )
        
        self.add_user_response_publisher = ZmqPublisher(
            port=self.add_user_response_port,
            topic_name=ZmqTopics.ADD_USER_RESPONSE.decode('utf-8')
        )
        
        logger.info(f"Recognition Service initialized with deployment mode: {deployment_mode}")
        logger.info(f"ZMQ Host: {self.zmq_host}")
        logger.info(f"ZMQ Ports - Cropped Face: {self.cropped_face_port}, Recognition: {self.recognition_port}")
        logger.info(f"ZMQ Ports - Add User: {self.add_user_port}, Add User Response: {self.add_user_response_port}")
        
        # Initialize face embedder and verifier
        self.facenet = FaceNetTFLiteHandler(verbose=False)
        self.face_verifier = FaceVerifier()
        
        # Initialize database handler
        # Use the ZMQ-based DatabaseClient instead of direct FaceDatabase access
        self.database = DatabaseClient(deployment_mode=deployment_mode)
        
        logger.info("Recognition Service initialized")
    
    def recognize_face(self, face_data):
        """Recognize a face from face data.
        
        Args:
            face_data (dict): Face data containing encoded image and metadata
            
        Returns:
            dict: Recognition results
        """
        try:
            # Extract face image from the message
            if isinstance(face_data.get('face_data'), bytes):
                # Convert bytes to numpy array
                img_array = np.frombuffer(face_data['face_data'], dtype=np.uint8)
                # Decode the image
                face_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                logger.error("Invalid face data format")
                return None
            
            # Preprocess the face image
            preprocessed_face = preprocess_image(face_image)
            
            # Get the embedding
            embedding = self.facenet.forward(preprocessed_face)
            
            # Get all embeddings from database
            all_embeddings = self.database.get_all_embeddings()
            
            # Check against all users in database
            best_match = None
            highest_confidence = 0.0
            
            for user_name, db_data in all_embeddings.items():
                db_embedding = db_data['embedding']
                
                # Verify against this user's embedding
                verification_result = self.face_verifier.verify_faces(embedding, db_embedding, verbose=False)
                
                # Check if verified with all metrics
                if (verification_result['cosine']['verified'] and 
                    verification_result['euclidean']['verified'] and 
                    verification_result['euclidean_l2']['verified']):
                    
                    # Calculate average confidence
                    confidence = (
                        verification_result['cosine']['score'] +
                        verification_result['euclidean']['score'] +
                        verification_result['euclidean_l2']['score']
                    ) / 3.0
                    
                    # Update if this is the best match so far
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_match = {
                            'user_name': user_name,
                            'verified': True,
                            'confidence': confidence,
                            'verification_details': verification_result
                        }
            
            if best_match:
                # Update last login time
                self.database.update_last_login(best_match['user_name'])
                return best_match
            else:
                return {
                    'user_name': 'unknown',
                    'verified': False,
                    'confidence': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None
    
    def add_user(self, user_data):
        """Add a new user to the database.
        
        Args:
            user_data (dict): User data containing faces and user info
            
        Returns:
            dict: Response with success status and message
        """
        try:
            # Extract user info
            user_name = user_data.get('user_name')
            password = user_data.get('password', '')
            face_images = user_data.get('face_images', [])
            
            if not user_name or not face_images:
                return {
                    'success': False,
                    'message': 'Missing required user data (name or face images)'
                }
                
            # Process each face image to get embeddings
            embeddings = []
            for face_image_data in face_images:
                try:
                    # Convert bytes to numpy array
                    img_array = np.frombuffer(face_image_data, dtype=np.uint8)
                    # Decode the image
                    face_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    # Preprocess the face image
                    preprocessed_face = preprocess_image(face_image)
                    
                    # Get the embedding
                    embedding = self.facenet.forward(preprocessed_face)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error processing face image: {e}")
            
            if not embeddings:
                return {
                    'success': False,
                    'message': 'Failed to extract embeddings from face images'
                }
                
            # Average the embeddings if multiple
            if len(embeddings) > 1:
                final_embedding = np.mean(embeddings, axis=0)
            else:
                final_embedding = embeddings[0]
                
            # Add user to database
            result = self.database.add_user(
                username=user_name,
                password=password,
                embedding=final_embedding
            )
            
            if result:
                return {
                    'success': True,
                    'message': f'User {user_name} added successfully'
                }
            else:
                return {
                    'success': False, 
                    'message': f'Failed to add user {user_name}'
                }
                
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def run(self):
        """Run the recognition service main loop."""
        logger.info("Recognition Service starting...")
        
        try:
            while True:
                # Check for cropped faces
                face_result = self.face_subscriber.receive(timeout=100)  # 100ms timeout
                
                if face_result is not None:
                    topic, face_data = face_result
                    
                    logger.info("Received cropped face for recognition")
                    
                    # Recognize the face
                    recognition_result = self.recognize_face(face_data)
                    
                    if recognition_result is not None:
                        # Add original face data for reference
                        if isinstance(face_data, dict):
                            recognition_result['face_id'] = face_data.get('face_id')
                            recognition_result['bbox'] = face_data.get('bbox')
                            recognition_result['timestamp'] = face_data.get('timestamp', time.time())
                        
                        # Publish the recognition result
                        self.recognition_publisher.publish(recognition_result)
                        logger.info(f"Published recognition result: {recognition_result['user_name']}")
                
                # Check for add user requests
                add_user_result = self.add_user_subscriber.receive(timeout=100)  # 100ms timeout
                
                if add_user_result is not None:
                    topic, user_data = add_user_result
                    
                    logger.info(f"Received add user request for: {user_data.get('user_name', 'Unknown')}")
                    
                    # Add the user to the database
                    response = self.add_user(user_data)
                    
                    # Publish the response
                    self.add_user_response_publisher.publish(response)
                    logger.info(f"Published add user response: {response}")
                
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                
        except KeyboardInterrupt:
            logger.info("Recognition Service interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.face_subscriber.close()
        self.add_user_subscriber.close()
        self.recognition_publisher.close()
        self.add_user_response_publisher.close()
        
        # Close database client connection
        self.database.close()
        
        logger.info("Recognition Service stopped")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Recognition Service")
    parser.add_argument("--cropped-face-port", type=int, default=None,
                      help="Port to subscribe for cropped faces (overrides config)")
    parser.add_argument("--recognition-port", type=int, default=None,
                      help="Port to publish recognition results (overrides config)")
    parser.add_argument("--add-user-port", type=int, default=None,
                      help="Port to subscribe for add user requests (overrides config)")
    parser.add_argument("--add-user-response-port", type=int, default=None,
                      help="Port to publish add user responses (overrides config)")
    parser.add_argument("--host", type=str, default=None,
                      help="Host to connect to for subscribing (overrides config)")
    parser.add_argument("--deployment-mode", type=str, choices=["local", "docker"], default="local",
                      help="Deployment mode (local or docker)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    service = RecognitionService(
        cropped_face_port=args.cropped_face_port,
        recognition_port=args.recognition_port,
        add_user_port=args.add_user_port,
        add_user_response_port=args.add_user_response_port,
        host=args.host,
        deployment_mode=args.deployment_mode
    )
    
    service.run()
