#!/usr/bin/env python3
import time
import argparse
import logging
import json
import numpy as np
from src.utils.zmq_utils import ZmqSubscriber, ZmqPublisher, ZmqTopics, ZmqConfig
from src.database.face_db import FaceDatabase
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Database_Service")

# Define database command topics
class DatabaseCommands:
    ADD_USER = b'add_user'
    GET_USER = b'get_user'
    DELETE_USER = b'delete_user'
    GET_ALL_USERS = b'get_all_users'
    UPDATE_LAST_LOGIN = b'update_last_login'

class DatabaseService:
    """Service responsible for database operations through ZMQ."""
    
    def __init__(self, db_request_port=None, db_response_port=None, host=None, deployment_mode="local"):
        """Initialize the database service.
        
        Args:
            db_request_port (int, optional): Port to subscribe for database requests. If None, uses config value.
            db_response_port (int, optional): Port to publish database responses. If None, uses config value.
            host (str, optional): Host to connect to for subscribing. If None, uses config value.
            deployment_mode (str): Deployment mode ('local' or 'docker')
        """
        # Load application settings
        self.settings = Settings()
        
        # Use configuration or fall back to defaults
        self.db_request_port = db_request_port or ZmqConfig.get_port("db_request", 5561)
        self.db_response_port = db_response_port or ZmqConfig.get_port("db_response", 5562)
        self.zmq_host = host or ZmqConfig.get_host("recognition_service", deployment_mode)
        
        # Initialize ZMQ subscriber for database requests
        self.db_request_subscriber = ZmqSubscriber(
            host=self.zmq_host,
            port=self.db_request_port,
            topic_name=ZmqTopics.DB_REQUEST.decode('utf-8')
        )
        
        # Initialize ZMQ publisher for database responses
        self.db_response_publisher = ZmqPublisher(
            port=self.db_response_port,
            topic_name=ZmqTopics.DB_RESPONSE.decode('utf-8')
        )
        
        # Initialize database handler
        self.database = FaceDatabase()
        
        logger.info(f"Database Service initialized with deployment mode: {deployment_mode}")
        logger.info(f"ZMQ Host: {self.zmq_host}")
        logger.info(f"ZMQ Ports - DB Request: {self.db_request_port}, DB Response: {self.db_response_port}")
    
    def process_request(self, request_data):
        """Process a database request.
        
        Args:
            request_data (dict): The request data containing command and parameters
            
        Returns:
            dict: Response with results or error message
        """
        try:
            command = request_data.get('command', '')
            request_id = request_data.get('request_id', '')
            
            logger.info(f"Processing database request: {command} (ID: {request_id})")
            
            if command == DatabaseCommands.ADD_USER.decode('utf-8'):
                # Add a user to the database
                username = request_data.get('username', '')
                embedding = np.frombuffer(request_data.get('embedding', b''), dtype=np.float32)
                password = request_data.get('password', '')
                
                result = self.database.add_user(username, embedding, password)
                return {
                    'success': result,
                    'request_id': request_id,
                    'message': f"User {username} {'added successfully' if result else 'could not be added'}"
                }
                
            elif command == DatabaseCommands.GET_USER.decode('utf-8'):
                # Get a user from the database
                username = request_data.get('username', '')
                
                try:
                    user_data = self.database.get_user(username)
                    # Convert numpy array to bytes for transmission
                    user_data['embedding'] = user_data['embedding'].tobytes()
                    return {
                        'success': True,
                        'request_id': request_id,
                        'user_data': user_data
                    }
                except ValueError as e:
                    return {
                        'success': False,
                        'request_id': request_id,
                        'message': str(e)
                    }
                
            elif command == DatabaseCommands.DELETE_USER.decode('utf-8'):
                # Delete a user from the database
                username = request_data.get('username', '')
                
                result = self.database.delete_user(username)
                return {
                    'success': result,
                    'request_id': request_id,
                    'message': f"User {username} {'deleted successfully' if result else 'not found'}"
                }
                
            elif command == DatabaseCommands.GET_ALL_USERS.decode('utf-8'):
                # Get all users from the database
                user_embeddings = self.database.get_all_embeddings()
                
                # Convert numpy arrays to bytes for transmission
                serialized_embeddings = {}
                for username, data in user_embeddings.items():
                    serialized_embeddings[username] = {
                        'embedding': data['embedding'].tobytes(),
                        'date_added': data.get('date_added', ''),
                        'last_login': data.get('last_login', '')
                    }
                
                return {
                    'success': True,
                    'request_id': request_id,
                    'users': serialized_embeddings
                }
                
            elif command == DatabaseCommands.UPDATE_LAST_LOGIN.decode('utf-8'):
                # Update last login time for a user
                username = request_data.get('username', '')
                
                self.database.update_last_login(username)
                return {
                    'success': True,
                    'request_id': request_id,
                    'message': f"Last login updated for user {username}"
                }
                
            else:
                return {
                    'success': False,
                    'request_id': request_id,
                    'message': f"Unknown command: {command}"
                }
                
        except Exception as e:
            logger.error(f"Error processing database request: {e}")
            return {
                'success': False,
                'request_id': request_id if 'request_id' in request_data else '',
                'message': f"Error: {str(e)}"
            }
    
    def run(self):
        """Run the database service main loop."""
        logger.info("Database Service starting...")
        
        try:
            while True:
                # Check for database requests
                request_result = self.db_request_subscriber.receive(timeout=1000)  # 1s timeout
                
                if request_result is not None:
                    topic, request_data = request_result
                    
                    logger.info(f"Received database request: {request_data}")
                    
                    # Process the request
                    response = self.process_request(request_data)
                    
                    # Publish the response
                    self.db_response_publisher.publish(response)
                    logger.info(f"Published database response: {response.get('success', False)}")
                
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                
        except KeyboardInterrupt:
            logger.info("Database Service interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.db_request_subscriber.close()
        self.db_response_publisher.close()
        logger.info("Database Service stopped")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Recognition Database Service")
    parser.add_argument("--db-request-port", type=int, default=None,
                      help="Port to subscribe for database requests (overrides config)")
    parser.add_argument("--db-response-port", type=int, default=None,
                      help="Port to publish database responses (overrides config)")
    parser.add_argument("--host", type=str, default=None,
                      help="Host to connect to for subscribing (overrides config)")
    parser.add_argument("--deployment-mode", type=str, choices=["local", "docker"], default="local",
                      help="Deployment mode (local or docker)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    service = DatabaseService(
        db_request_port=args.db_request_port,
        db_response_port=args.db_response_port,
        host=args.host,
        deployment_mode=args.deployment_mode
    )
    
    service.run()
