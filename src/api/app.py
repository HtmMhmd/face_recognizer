from flask import Flask, render_template, Response, jsonify
import cv2
import time
from threading import Thread
import os
from src.config import api_settings, app_settings

# Create Flask application
app = Flask(__name__, 
    template_folder=api_settings.get("templates_dir", "src/api/templates"),
    static_folder=api_settings.get("static_dir", "src/api/static")
)

# Configure Flask app from settings
app.config['DEBUG'] = api_settings.get("debug", False)
app.config['SECRET_KEY'] = api_settings.get("session_secret", "default-secret-key")

class VideoCamera:
    def __init__(self, camera_service):
        self.camera_service = camera_service
        self.is_running = True
        
    def __del__(self):
        self.is_running = False
    
    def get_frame(self):
        while self.is_running:
            frame = self.camera_service.get_current_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n\r\n')
            time.sleep(0.01)

# Global variable to hold the camera service instance
camera_service = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera_service
    return Response(VideoCamera(camera_service).get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detection_results')
def detection_results():
    global camera_service
    results = camera_service.get_detection_results()
    return jsonify(results)

def start_api_server(camera_svc, host=None, port=None):
    """
    Start the Flask API server for the face recognition system.
    
    Args:
        camera_svc: Camera service for video feed
        host: Host address to listen on (default from config)
        port: Port to listen on (default from config)
    """
    global camera_service
    camera_service = camera_svc
    
    # Use values from config if not provided
    host = host or api_settings.get("host", "0.0.0.0")
    port = port or api_settings.get("port", 8000)
    
    app.run(host=host, port=port, threaded=True)

if __name__ == "__main__":
    # This is for direct testing of the API module
    from src.services.camera_service import CameraService
    
    # Default camera index from config
    camera_idx = camera_settings.get("default_index", 0)
    camera_service = CameraService(camera_idx)
    
    # Start API server with config settings
    host = api_settings.get("host", "0.0.0.0")
    port = api_settings.get("port", 8000)
    
    app.run(host=host, port=port, debug=api_settings.get("debug", True), threaded=True)
