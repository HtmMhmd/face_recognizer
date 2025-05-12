#!/usr/bin/env python3
import cv2
import time
import argparse
import logging
import numpy as np
import base64
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

from src.utils.zmq_utils import ZmqPublisher, ZmqSubscriber, ZmqTopics, CaptureCommands, ZmqConfig
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Dashboard_Service")

# Global variables to store latest data
latest_cropped_faces = {}  # face_id -> face_image
latest_recognition_results = {}  # face_id -> recognition_result
latest_add_user_response = None

class DashboardService:
    """Service responsible for the dashboard UI and ZMQ communication."""
    
    def __init__(self, capture_port=None, cropped_face_port=None, recognition_port=None,
                add_user_port=None, add_user_response_port=None, host=None,
                dashboard_host=None, dashboard_port=None, deployment_mode="local"):
        """Initialize the dashboard service.
        
        Args:
            capture_port (int, optional): Port to publish capture commands. If None, uses config value.
            cropped_face_port (int, optional): Port to subscribe for cropped faces. If None, uses config value.
            recognition_port (int, optional): Port to subscribe for recognition results. If None, uses config value.
            add_user_port (int, optional): Port to publish add user requests. If None, uses config value.
            add_user_response_port (int, optional): Port to subscribe for add user responses. If None, uses config value.
            host (str, optional): Host to connect for ZMQ sockets. If None, uses config value.
            dashboard_host (str, optional): Host to bind Flask server. If None, uses config value.
            dashboard_port (int, optional): Port to bind Flask server. If None, uses config value.
            deployment_mode (str): Deployment mode ('local' or 'docker')
        """
        # Load application settings
        self.settings = Settings()
        
        # Use configuration or fall back to provided values or defaults
        self.capture_port = capture_port or ZmqConfig.get_port("capture", 5555)
        self.cropped_face_port = cropped_face_port or ZmqConfig.get_port("cropped_face", 5557)
        self.recognition_port = recognition_port or ZmqConfig.get_port("recognition", 5558)
        self.add_user_port = add_user_port or ZmqConfig.get_port("add_user", 5559)
        self.add_user_response_port = add_user_response_port or ZmqConfig.get_port("add_user_response", 5560)
        self.zmq_host = host or ZmqConfig.get_host("recognition_service", deployment_mode)
        self.dashboard_host = dashboard_host or self.settings.api.get("host", "0.0.0.0")
        self.dashboard_port = dashboard_port or self.settings.api.get("port", 8080)
        
        logger.info(f"Dashboard Service initialized with deployment mode: {deployment_mode}")
        logger.info(f"ZMQ Host: {self.zmq_host}")
        logger.info(f"Dashboard Host: {self.dashboard_host}, Port: {self.dashboard_port}")
        logger.info(f"ZMQ Ports - Capture: {self.capture_port}, Recognition: {self.recognition_port}")
        
        # Initialize ZMQ publishers
        self.capture_publisher = ZmqPublisher(
            port=self.capture_port,
            topic_name=ZmqTopics.CAPTURE.decode('utf-8')
        )
        
        self.add_user_publisher = ZmqPublisher(
            port=self.add_user_port,
            topic_name=ZmqTopics.ADD_USER_REQUEST.decode('utf-8')
        )
        
        # Initialize ZMQ subscribers
        self.cropped_face_subscriber = ZmqSubscriber(
            host=self.zmq_host,
            port=self.cropped_face_port,
            topic_name=ZmqTopics.CROPPED_FACE.decode('utf-8')
        )
        
        self.recognition_subscriber = ZmqSubscriber(
            host=self.zmq_host,
            port=self.recognition_port,
            topic_name=ZmqTopics.RECOGNITION_RESULT.decode('utf-8')
        )
        
        self.add_user_response_subscriber = ZmqSubscriber(
            host=self.zmq_host,
            port=self.add_user_response_port,
            topic_name=ZmqTopics.ADD_USER_RESPONSE.decode('utf-8')
        )
        
        # Dashboard Flask app
        self.app = Flask(__name__, template_folder='templates')
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        
        # Set up routes
        self._setup_routes()
        
        # Start ZMQ message receiver thread
        self.running = True
        self.receiver_thread = threading.Thread(target=self._receive_messages)
        self.receiver_thread.daemon = True  # Thread will exit when main thread exits
        
        logger.info("Dashboard Service initialized")
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            """Render the dashboard homepage."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/capture/<direction>')
        def capture(direction):
            """Capture an image."""
            if direction == 'front':
                command = CaptureCommands.TAKE_FRONT
            elif direction == 'left':
                command = CaptureCommands.TAKE_LEFT
            elif direction == 'right':
                command = CaptureCommands.TAKE_RIGHT
            else:
                return jsonify({'status': 'error', 'message': 'Invalid direction'})
            
            # Publish capture command
            self.capture_publisher.publish(command)
            logger.info(f"Published capture command: {command}")
            
            return jsonify({'status': 'success', 'message': f'Capture {direction} command sent'})
        
        @self.app.route('/api/add_user', methods=['POST'])
        def add_user():
            """Add a new user."""
            global latest_add_user_response
            
            # Reset the latest response
            latest_add_user_response = None
            
            # Get request data
            user_name = request.form.get('user_name')
            password = request.form.get('password', '')
            
            if not user_name:
                return jsonify({'status': 'error', 'message': 'User name is required'})
            
            # Get face images from the request
            face_images = []
            for key in request.files:
                file = request.files[key]
                if file:
                    # Read image data
                    image_bytes = file.read()
                    face_images.append(image_bytes)
            
            if not face_images:
                return jsonify({'status': 'error', 'message': 'No face images provided'})
            
            # Prepare user data
            user_data = {
                'user_name': user_name,
                'password': password,
                'face_images': face_images
            }
            
            # Publish add user request
            self.add_user_publisher.publish(user_data)
            logger.info(f"Published add user request for: {user_name}")
            
            # Wait for response with timeout
            timeout = time.time() + 10  # 10 seconds timeout
            while latest_add_user_response is None and time.time() < timeout:
                time.sleep(0.1)
            
            if latest_add_user_response:
                return jsonify({
                    'status': 'success' if latest_add_user_response.get('success', False) else 'error',
                    'message': latest_add_user_response.get('message', 'Unknown error')
                })
            else:
                return jsonify({'status': 'error', 'message': 'Timeout waiting for response'})
        
        @self.app.route('/api/latest_faces')
        def get_latest_faces():
            """Get the latest cropped faces."""
            global latest_cropped_faces
            
            result = {}
            for face_id, face_data in latest_cropped_faces.items():
                # Convert the face image to base64
                if isinstance(face_data, np.ndarray):
                    _, buffer = cv2.imencode('.jpg', face_data)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    result[face_id] = img_str
            
            return jsonify(result)
        
        @self.app.route('/api/latest_results')
        def get_latest_results():
            """Get the latest recognition results."""
            global latest_recognition_results
            
            return jsonify(latest_recognition_results)
    
    def _receive_messages(self):
        """Continuously receive messages from ZMQ subscribers."""
        global latest_cropped_faces, latest_recognition_results, latest_add_user_response
        
        while self.running:
            try:
                # Check for cropped faces
                face_result = self.cropped_face_subscriber.receive(timeout=100)  # 100ms timeout
                
                if face_result is not None:
                    topic, face_data = face_result
                    
                    # Extract face image
                    if isinstance(face_data, dict) and 'face_data' in face_data and 'face_id' in face_data:
                        # Convert bytes to numpy array
                        img_array = np.frombuffer(face_data['face_data'], dtype=np.uint8)
                        # Decode the image
                        face_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        # Store in global dict
                        latest_cropped_faces[str(face_data['face_id'])] = face_image
                        logger.debug(f"Received cropped face {face_data['face_id']}")
                
                # Check for recognition results
                recognition_result = self.recognition_subscriber.receive(timeout=100)  # 100ms timeout
                
                if recognition_result is not None:
                    topic, result = recognition_result
                    
                    if isinstance(result, dict) and 'face_id' in result:
                        # Store in global dict
                        face_id = str(result['face_id'])
                        latest_recognition_results[face_id] = {
                            'user_name': result.get('user_name', 'unknown'),
                            'verified': result.get('verified', False),
                            'confidence': result.get('confidence', 0.0),
                            'timestamp': result.get('timestamp', time.time()),
                            'date_time': datetime.fromtimestamp(
                                result.get('timestamp', time.time())
                            ).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        logger.info(f"Received recognition result for face {face_id}: {result.get('user_name', 'unknown')}")
                
                # Check for add user responses
                add_user_response = self.add_user_response_subscriber.receive(timeout=100)  # 100ms timeout
                
                if add_user_response is not None:
                    topic, response = add_user_response
                    
                    # Store latest response
                    latest_add_user_response = response
                    logger.info(f"Received add user response: {response}")
                
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                
            except Exception as e:
                logger.error(f"Error receiving messages: {e}")
                time.sleep(0.1)  # Sleep a bit longer on error
    
    def run(self):
        """Run the dashboard service."""
        logger.info("Starting ZMQ message receiver thread...")
        self.receiver_thread.start()
        
        logger.info(f"Starting Dashboard web server on {self.dashboard_host}:{self.dashboard_port}...")
        self.app.run(host=self.dashboard_host, port=self.dashboard_port, debug=False)
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.running = False
        
        if self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0)
            
        self.capture_publisher.close()
        self.add_user_publisher.close()
        self.cropped_face_subscriber.close()
        self.recognition_subscriber.close()
        self.add_user_response_subscriber.close()
        logger.info("Dashboard Service stopped")

def parse_args():
    """Parse command line arguments."""
    # Load settings to get default values
    settings = Settings()
    
    parser = argparse.ArgumentParser(description="Face Recognition Dashboard Service")
    parser.add_argument("--capture-port", type=int, default=None,
                      help="Port to publish capture commands (overrides config)")
    parser.add_argument("--cropped-face-port", type=int, default=None,
                      help="Port to subscribe for cropped faces (overrides config)")
    parser.add_argument("--recognition-port", type=int, default=None,
                      help="Port to subscribe for recognition results (overrides config)")
    parser.add_argument("--add-user-port", type=int, default=None,
                      help="Port to publish add user requests (overrides config)")
    parser.add_argument("--add-user-response-port", type=int, default=None,
                      help="Port to subscribe for add user responses (overrides config)")
    parser.add_argument("--host", type=str, default=None,
                      help="Host to connect to for ZMQ (overrides config)")
    parser.add_argument("--dashboard-host", type=str, default=None,
                      help="Host to bind Flask server (overrides config)")
    parser.add_argument("--dashboard-port", type=int, default=None,
                      help="Port to bind Flask server (overrides config)")
    parser.add_argument("--deployment-mode", type=str, choices=["local", "docker"], default="local",
                      help="Deployment mode (local or docker)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    service = DashboardService(
        capture_port=args.capture_port,
        cropped_face_port=args.cropped_face_port,
        recognition_port=args.recognition_port,
        add_user_port=args.add_user_port,
        add_user_response_port=args.add_user_response_port,
        host=args.host,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        deployment_mode=args.deployment_mode
    )
    
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Dashboard Service interrupted by user")
    finally:
        service.cleanup()
