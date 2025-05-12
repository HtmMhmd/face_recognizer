import time
import logging
import uuid
import numpy as np
import os
from src.utils.zmq_utils import ZmqPublisher, ZmqSubscriber, ZmqTopics, ZmqConfig
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Database_Client")

# Fallback to HTTP client if needed
try:
    import requests
    import base64
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class DatabaseClient:
    """ZMQ client for database operations."""
    
    def __init__(self, db_request_port=None, db_response_port=None, host=None, deployment_mode="local", timeout=5000):
        """Initialize the database client.
        
        Args:
            db_request_port (int, optional): Port to publish database requests. If None, uses config value.
            db_response_port (int, optional): Port to subscribe for database responses. If None, uses config value.
            host (str, optional): Host to connect to for responses. If None, uses config value.
            deployment_mode (str): Deployment mode ('local' or 'docker')
            timeout (int): Timeout for database operations in milliseconds
        """
        # Check if we should use HTTP or ZMQ
        self.use_http = os.getenv('USE_HTTP_DB', 'false').lower() in ('true', '1', 't')
        
        # HTTP client initialization
        if self.use_http:
            if not REQUESTS_AVAILABLE:
                raise ImportError("Requests library not available but USE_HTTP_DB is enabled")
            self.base_url = os.getenv('DB_API_URL', 'http://localhost:5000')
            logger.info(f"Using HTTP database client with base URL: {self.base_url}")
            return
            
        # Load application settings for ZMQ
        self.settings = Settings()
        
        # Use configuration or fall back to defaults
        self.db_request_port = db_request_port or ZmqConfig.get_port("db_request", 5561)
        self.db_response_port = db_response_port or ZmqConfig.get_port("db_response", 5562)
        self.zmq_host = host or ZmqConfig.get_host("database_service", deployment_mode)
        self.timeout = timeout
        
        # Initialize ZMQ publisher for database requests
        self.db_request_publisher = ZmqPublisher(
            port=self.db_request_port,
            topic_name=ZmqTopics.DB_REQUEST.decode('utf-8')
        )
        
        # Initialize ZMQ subscriber for database responses
        self.db_response_subscriber = ZmqSubscriber(
            host=self.zmq_host,
            port=self.db_response_port,
            topic_name=ZmqTopics.DB_RESPONSE.decode('utf-8')
        )
        
        logger.info(f"Database Client initialized with deployment mode: {deployment_mode}")
        logger.info(f"ZMQ Host: {self.zmq_host}")
        logger.info(f"ZMQ Ports - DB Request: {self.db_request_port}, DB Response: {self.db_response_port}")
    
    def send_request(self, command, **kwargs):
        """Send a request to the database service.
        
        Args:
            command (str): The command to send
            **kwargs: Additional parameters for the command
            
        Returns:
            dict: The response from the database service or None if timeout/error
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Prepare the request
        request = {
            'command': command,
            'request_id': request_id,
            **kwargs
        }
        
        # Publish the request
        self.db_request_publisher.publish(request)
        
        # Wait for the response
        start_time = time.time()
        while (time.time() - start_time) * 1000 < self.timeout:
            result = self.db_response_subscriber.receive(timeout=100)  # 100ms timeout
            
            if result is not None:
                topic, response = result
                
                # Check if this is the response we're waiting for
                if isinstance(response, dict) and response.get('request_id') == request_id:
                    return response
            
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
        
        logger.error(f"Timeout waiting for response to request {request_id}")
        return None
        
    # HTTP Client Methods
    def _http_add_user(self, username, embedding, **kwargs):
        embedding_bytes = embedding.tobytes()
        embedding_b64 = base64.b64encode(embedding_bytes).decode()
        
        response = requests.post(f"{self.base_url}/user", 
                               json={"username": username, "embedding": embedding_b64})
        return response.json()
    
    def _http_get_user(self, username):
        response = requests.get(f"{self.base_url}/user/{username}")
        if response.status_code == 200:
            embedding_b64 = response.json()['embedding']
            embedding_bytes = base64.b64decode(embedding_b64)
            user_data = response.json()
            user_data['embedding'] = np.frombuffer(embedding_bytes, dtype=np.float32)
            return user_data
        return None
    
    def _http_get_all_embeddings(self):
        response = requests.get(f"{self.base_url}/users")
        embeddings = {}
        for username, user_data in response.json().items():
            embedding_b64 = user_data['embedding']
            embedding_bytes = base64.b64decode(embedding_b64)
            embeddings[username] = {
                'embedding': np.frombuffer(embedding_bytes, dtype=np.float32),
                'date_added': user_data.get('date_added', ''),
                'last_login': user_data.get('last_login', '')
            }
        return embeddings
    
    # ZMQ Client Methods
    def add_user(self, username, embedding, password=""):
        """Add a user to the database."""
        if self.use_http:
            return self._http_add_user(username, embedding, password=password)
            
        response = self.send_request(
            'add_user',
            username=username,
            embedding=embedding.tobytes(),
            password=password
        )
        
        if response is None:
            return False
            
        return response.get('success', False)
    
    def get_user(self, username):
        """Get a user from the database."""
        if self.use_http:
            return self._http_get_user(username)
            
        response = self.send_request('get_user', username=username)
        
        if response is None or not response.get('success', False):
            return None
            
        user_data = response.get('user_data', {})
        
        # Convert embedding bytes back to numpy array
        if 'embedding' in user_data:
            user_data['embedding'] = np.frombuffer(user_data['embedding'], dtype=np.float32)
            
        return user_data
    
    def get_all_embeddings(self):
        """Get all users and their embeddings from the database."""
        if self.use_http:
            return self._http_get_all_embeddings()
            
        response = self.send_request('get_all_users')
        
        if response is None or not response.get('success', False):
            return {}
            
        users = response.get('users', {})
        
        # Convert embedding bytes back to numpy arrays
        for username, user_data in users.items():
            if 'embedding' in user_data:
                user_data['embedding'] = np.frombuffer(user_data['embedding'], dtype=np.float32)
        
        return users

    def delete_user(self, username):
        """Delete a user from the database.
        
        Args:
            username (str): Username to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.use_http:
            try:
                response = requests.delete(f"{self.base_url}/user/{username}")
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Error deleting user: {e}")
                return False
                
        response = self.send_request('delete_user', username=username)
        
        if response is None:
            return False
            
        return response.get('success', False)
    
    def update_last_login(self, username):
        """Update the last login timestamp for a user.
        
        Args:
            username (str): Username to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.use_http:
            try:
                response = requests.put(f"{self.base_url}/user/{username}/login")
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Error updating last login: {e}")
                return False
                
        response = self.send_request('update_last_login', username=username)
        
        if response is None:
            return False
            
        return response.get('success', False)
    
    def close(self):
        """Close the ZMQ connections."""
        if not self.use_http:
            self.db_request_publisher.close()
            self.db_response_subscriber.close()
            logger.info("Database Client closed")
