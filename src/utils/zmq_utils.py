import zmq
import json
import pickle
import numpy as np
import cv2
import logging
import time
from threading import Thread
from src.config.settings import Settings

# Load application settings
settings = Settings()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ZmqPublisher:
    """A class for publishing messages to ZMQ topics."""
    
    def __init__(self, port, topic_name):
        """Initialize ZMQ publisher.
        
        Args:
            port (int): Port number to bind to
            topic_name (str): The topic name for messages
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.topic_name = topic_name
        self.logger = logging.getLogger(f"ZMQ_PUB_{topic_name}")
        self.logger.info(f"Publisher for topic '{topic_name}' initialized on port {port}")
        # Small delay to allow connection setup
        time.sleep(0.2)
    
    def publish(self, message):
        """Publish a message to the topic.
        
        Args:
            message: The message to publish (can be string, dict, or binary data)
        """
        try:
            if isinstance(message, dict):
                # Convert dict to JSON string
                message_data = json.dumps(message).encode('utf-8')
            elif isinstance(message, str):
                # Encode string to bytes
                message_data = message.encode('utf-8')
            else:
                # Assume binary data (like image)
                message_data = message
                
            # Send topic name followed by message
            self.socket.send_multipart([self.topic_name.encode('utf-8'), message_data])
            self.logger.debug(f"Published message to topic '{self.topic_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Error publishing to topic '{self.topic_name}': {e}")
            return False
    
    def publish_image(self, image):
        """Publish OpenCV image.
        
        Args:
            image (numpy.ndarray): OpenCV image
        """
        try:
            _, img_encoded = cv2.imencode('.jpg', image)
            self.publish(img_encoded.tobytes())
            return True
        except Exception as e:
            self.logger.error(f"Error publishing image: {e}")
            return False
            
    def close(self):
        """Close the ZMQ socket and context."""
        self.socket.close()
        self.context.term()
        self.logger.info(f"Publisher for topic '{self.topic_name}' closed")

class ZmqSubscriber:
    """A class for subscribing to ZMQ topics."""
    
    def __init__(self, host, port, topic_name):
        """Initialize ZMQ subscriber.
        
        Args:
            host (str): Host to connect to
            port (int): Port number to connect to
            topic_name (str): The topic to subscribe to
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, topic_name.encode('utf-8'))
        self.topic_name = topic_name
        self.logger = logging.getLogger(f"ZMQ_SUB_{topic_name}")
        self.logger.info(f"Subscriber for topic '{topic_name}' connected to {host}:{port}")
        
    def receive(self, timeout=None):
        """Receive a message from the topic.
        
        Args:
            timeout (int, optional): Timeout in milliseconds
            
        Returns:
            tuple: (topic_name, message_data) or None if timeout
        """
        try:
            if timeout is not None:
                if self.socket.poll(timeout) == 0:
                    return None
                    
            topic, message = self.socket.recv_multipart()
            topic_str = topic.decode('utf-8')
            
            # Try to decode as JSON, if fails return raw bytes
            try:
                return topic_str, json.loads(message.decode('utf-8'))
            except:
                return topic_str, message
                
        except Exception as e:
            self.logger.error(f"Error receiving from topic '{self.topic_name}': {e}")
            return None
    
    def receive_image(self, timeout=None):
        """Receive an image from the topic.
        
        Args:
            timeout (int, optional): Timeout in milliseconds
            
        Returns:
            numpy.ndarray: OpenCV image or None if timeout/error
        """
        result = self.receive(timeout)
        if result is None:
            return None
            
        topic, message = result
        
        try:
            # Convert bytes to numpy array
            img_array = np.frombuffer(message, dtype=np.uint8)
            # Decode the image
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            self.logger.error(f"Error decoding image: {e}")
            return None
            
    def close(self):
        """Close the ZMQ socket and context."""
        self.socket.close()
        self.context.term()
        self.logger.info(f"Subscriber for topic '{self.topic_name}' closed")

# ZMQ Topic definitions
class ZmqTopics:
    """Constants for ZMQ topics used in the system."""
    # Use configuration values or fall back to defaults if not defined
    CAPTURE = settings.zmq.get("topics", {}).get("capture", "Capture").encode('utf-8')
    IMAGE = settings.zmq.get("topics", {}).get("image", "Image").encode('utf-8')
    CROPPED_FACE = settings.zmq.get("topics", {}).get("cropped_face", "CroppedFace").encode('utf-8')
    RECOGNITION_RESULT = settings.zmq.get("topics", {}).get("recognition_result", "RecognitionResult").encode('utf-8')
    ADD_USER_REQUEST = settings.zmq.get("topics", {}).get("add_user_request", "AddUserRequest").encode('utf-8')
    ADD_USER_RESPONSE = settings.zmq.get("topics", {}).get("add_user_response", "AddUserResponse").encode('utf-8')
    VERIFY_USER_REQUEST = settings.zmq.get("topics", {}).get("verify_user_request", "VerifyUserRequest").encode('utf-8')
    VERIFY_USER_RESPONSE = settings.zmq.get("topics", {}).get("verify_user_response", "VerifyUserResponse").encode('utf-8')
    DB_REQUEST = settings.zmq.get("topics", {}).get("db_request", "DBRequest").encode('utf-8')
    DB_RESPONSE = settings.zmq.get("topics", {}).get("db_response", "DBResponse").encode('utf-8')
    STATUS_UPDATE = settings.zmq.get("topics", {}).get("status_update", "StatusUpdate").encode('utf-8')

# Command definitions
class CaptureCommands:
    """Constants for capture commands."""
    TAKE_FRONT = settings.zmq.get("commands", {}).get("take_front", "take_front")
    TAKE_LEFT = settings.zmq.get("commands", {}).get("take_left", "take_left")
    TAKE_RIGHT = settings.zmq.get("commands", {}).get("take_right", "take_right")

# Port and host configuration
class ZmqConfig:
    """Port and host configuration for ZMQ services."""
    
    @staticmethod
    def get_port(port_name, default=None):
        """Get a port number from configuration.
        
        Args:
            port_name: Port name (e.g., 'capture', 'image')
            default: Default value if not in config
            
        Returns:
            int: Port number
        """
        port = settings.zmq.get("ports", {}).get(port_name, default)
        return port
    
    @staticmethod
    def get_host(service_name, deployment_mode="local"):
        """Get a host name for a service based on deployment mode.
        
        Args:
            service_name: Service name (e.g., 'capture_service')
            deployment_mode: Deployment mode ('local' or 'docker')
            
        Returns:
            str: Host name
        """
        if deployment_mode == "local":
            return settings.zmq.get("hosts", {}).get("local", "localhost")
        elif deployment_mode == "docker":
            docker_hosts = settings.zmq.get("hosts", {}).get("docker", {})
            return docker_hosts.get(service_name, service_name)
        else:
            return "localhost"
