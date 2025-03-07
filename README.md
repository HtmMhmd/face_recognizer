# Face Recognition System

This repository contains a comprehensive facial recognition system that can detect, align, and recognize faces using deep learning techniques. The system is containerized using Docker for easy deployment across different environments.

## Features

- **Face Detection**: Uses OpenCV's Haar cascade classifier for efficient face detection
- **Face Alignment**: Automatically aligns detected faces for better recognition accuracy
- **Face Recognition**: Employs FaceNet model for generating face embeddings and recognition
- **User Management**: Stores and manages user face embeddings in a database
- **Web Interface**: Provides a simple web interface for monitoring and verification
- **API Endpoints**: RESTful API for integration with other systems
- **Docker Support**: Containerized application for easy deployment

## System Architecture

The system consists of two main services:

1. **Face Recognizer Service**:
   - Handles face detection, alignment, and recognition
   - Provides web interface and API endpoints
   - Communicates with the database service

2. **Database Service**:
   - Manages user data and embeddings
   - Exposes API for data operations

## Installation

### Prerequisites

- Docker and Docker Compose
- Webcam (for live detection)
- Git

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd face_recognizer
   ```

2. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

### Manual Setup (without Docker)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API:
   ```bash
   python api.py
   ```

## Usage

### Web Interface

Access the web interface by navigating to:
- http://localhost:8000

Available pages:
- Video Feed: http://localhost:8000/video_feed
- Verification Results: http://localhost:8000/verify_results

### Adding a New User

Use the add_user.py script to add a new user to the system:

```bash
python add_user.py
```

### Using the API

The system exposes several API endpoints:

- `GET /video_feed`: Streams the webcam with face detection overlay
- `GET /verify_results`: Shows verification results

## Docker Deployment

### Raspberry Pi Deployment

```bash
sudo docker run -it --rm --device=/dev/video0 --network=host --ipc=host face_recognizer_container
```

### PC Deployment

```bash
xhost +local:docker
sudo docker run -it --rm --env=DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --device=/dev/video0:/dev/video0 --network=host --ipc=host -v $(pwd):/workspace/src/my_mtcnn_node:ro face_recognizer_container
```

### Using Docker Compose

```bash
docker-compose up
```

To access a running container:
```bash
docker exec -it face_recognizer_container_face_recognizer_1 bash
```

## Project Structure

```
├── __init__.py
├── add_user.py             # Script to add new users
├── api.py                  # Main API and web server
├── CameraUtilis/           # Camera handling utilities
├── database/               # Database related files
├── docker-compose.yaml     # Docker Compose configuration
├── Dockerfile              # Main service Dockerfile
├── Dockerfile.db           # Database service Dockerfile
├── ImageUtilis/            # Image processing utilities
├── Model/                  # ML models
│   ├── OpencvDetector/     # OpenCV face detection files
│   └── ...
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
└── UsersDatabaseHandeler/  # User database management
```

## Model Information

The system uses a cascade classifier for face detection, located in faces.xml. For face recognition, it employs a TFLite model to generate 512-dimensional face embeddings.

## Database Structure

User embeddings are stored in a CSV format with 512 embedding values plus the username. The database handling is managed by the `EmbeddingCSVHandler` class.

## License

This project includes components with their respective licenses. The OpenCV Haar cascade classifier is used under the Intel License Agreement for Open Source Computer Vision Library.