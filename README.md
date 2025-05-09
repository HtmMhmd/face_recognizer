# Face Recognition System

A comprehensive facial recognition system that can detect, align, and recognize faces using deep learning techniques. The system is containerized using Docker for easy deployment across different environments.

## 📋 Features

- **Face Detection**: Uses multiple face detection models (MediaPipe, YOLOv8)
- **Face Alignment**: Automatically aligns detected faces for better recognition accuracy
- **Face Recognition**: FaceNet model for generating face embeddings and recognition
- **Drowsiness Detection**: Optional eye-aspect ratio based drowsiness detection
- **User Management**: Complete system to add, delete, and manage user face embeddings
- **Web Interface**: Simple web interface for monitoring and verification
- **API Endpoints**: RESTful API for integration with other systems
- **Docker Support**: Containerized application for easy deployment

## 🏗️ Project Structure

The project is organized into a clean, modular structure:

```
face_recognizer/
├── src/                   # Main source code directory
│   ├── api/               # API endpoints and web interfaces
│   ├── core/              # Core functionality
│   │   ├── alignment/     # Face alignment algorithms
│   │   ├── detection/     # Face detection implementations
│   │   ├── drowsiness/    # Drowsiness detection algorithms
│   │   ├── verification/  # Face embedding verification
│   ├── database/          # Database implementations
│   ├── models/            # ML models for face detection/recognition
│   ├── services/          # High-level services combining components
│   ├── utils/             # Utility functions and helpers
│   │   ├── camera/        # Camera handling utilities
│   │   ├── image/         # Image processing utilities
│   └── config/            # Configuration files
├── models/                # Model weights and files
├── run.py                 # Main entry point
├── user_management.py     # CLI for managing users
├── setup.py               # Package installation script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
└── docker-compose.yaml    # Multi-container orchestration
```

## 🚀 Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face_recognizer.git
   cd face_recognizer
   ```

2. Start the containers:
   ```bash
   docker-compose up --build
   ```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face_recognizer.git
   cd face_recognizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Run the application:
   ```bash
   python run.py --mode api
   ```

## 🎮 Usage

### Running the Application

The system supports three modes of operation:

1. **Camera Mode** (Live video processing):
   ```bash
   python run.py --mode camera [options]
   ```

2. **Image Mode** (Single image processing):
   ```bash
   python run.py --mode image --image-path /path/to/image.jpg [options]
   ```

3. **API Mode** (Web interface and REST endpoints):
   ```bash
   python run.py --mode api [options]
   ```

### Command Line Options

- `--detector` - Detector type: `mediapipe`, `yolov8`, `yolov8_onnx` (default: `mediapipe`)
- `--camera` - Camera index (default: 0)
- `--enable-drowsiness` - Enable drowsiness detection
- `--gui` - Show graphical interface
- `--output` - Save results to JSON file (camera mode)
- `--port` - API server port (API mode, default: 8000)
- `--verbose` - Enable verbose output

### User Management

The system includes a user management CLI to add, delete and list users:

1. **Add a new user** (with camera capture):
   ```bash
   python user_management.py add username --capture
   ```

2. **Add a user from an image**:
   ```bash
   python user_management.py add username --image /path/to/image.jpg
   ```

3. **List all users**:
   ```bash
   python user_management.py list
   ```

4. **Delete a user**:
   ```bash
   python user_management.py delete username
   ```

### Web Interface

When running in API mode, access the web interface at:
- http://localhost:8000

## 🛠️ Docker Deployment

### Full Stack Deployment

```bash
docker-compose up
```

### Individual Services

The system consists of several services:

1. **face_recognizer**: Main recognition service with web interface
2. **drowsiness_detector**: Dedicated drowsiness detection service
3. **db_service**: Database service for storing user embeddings
4. **mjpg-streamer**: Camera streaming service

Each can be started individually:

```bash
docker-compose up face_recognizer
```

## 🧪 Development

For development, install the package in development mode:

```bash
pip install -e .
```

This allows you to modify the code and see changes immediately without reinstalling.

## 📄 License

This project includes components with their respective licenses. The OpenCV Haar cascade classifier is used under the Intel License Agreement for Open Source Computer Vision Library.