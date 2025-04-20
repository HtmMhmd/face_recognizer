# Face Recognition System

<img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9+-blue.svg">
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
<img alt="Docker" src="https://img.shields.io/badge/Docker-Supported-blue.svg">
<img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Powered-green.svg">

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
   git clone https://github.com/HtmMhmd/face_recognizer.git
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

### API Usage Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│                 │     │                  │     │               │
│  Video Feed     │────►│  Face Detection  │────►│ Face Alignment│
│                 │     │                  │     │               │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│                 │     │                  │     │               │
│  Verification   │◄────│  Face Comparison │◄────│ Face Embedding│
│  Result         │     │                  │     │ Generation    │
└─────────────────┘     └────────┬─────────┘     └───────▲───────┘
                                 │                       │
                                 ▼                       │
                        ┌──────────────────┐     ┌───────────────┐
                        │                  │     │               │
                        │  User Database   │─────│  User Data    │
                        │  Lookup          │     │  Storage      │
                        └──────────────────┘     └───────────────┘
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CAMERA_INDEX` | Camera device index to use | `0` |
| `DETECTION_CONFIDENCE` | Minimum confidence threshold for face detection | `0.8` |
| `DATABASE_PATH` | Path to user database | `./database/users.csv` |
| `API_PORT` | Port for the web interface and API | `8000` |
| `DEBUG_MODE` | Enable debug logging | `false` |

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--camera` | Camera index to use (overrides environment variable) |
| `--confidence` | Detection confidence threshold |
| `--database` | Path to user database file |
| `--port` | Port for the web API server |
| `--debug` | Enable debug mode |
| `--no-ui` | Run without web interface |

## 🔍 Detailed System Explanation

The Face Recognition System implements a complete pipeline for detecting and recognizing faces:

1. **Video Input Layer**
   - Captures video frames from camera or file input
   - Pre-processes frames for detection (resizing, color conversion)
   - Manages frame rate and buffering

2. **Detection Layer**
   - Employs OpenCV Haar cascade classifier for face detection
   - Identifies face regions within each frame
   - Filters detections based on confidence threshold
   - Outputs bounding box coordinates

3. **Alignment Layer**
   - Detects facial landmarks (eyes, nose, mouth)
   - Calculates optimal rotation to align face
   - Performs geometric transformation

4. **Recognition Layer**
   - Preprocesses aligned face images
   - Generates 512-dimensional face embeddings
   - Compares embeddings with stored user database
   - Determines identity with confidence score

### Key Components

#### Face Detector

The `FaceDetector` class handles the detection of faces in input frames:

```python
def detect_faces(self, frame):
    """
    Detect faces in a frame using Haar cascade classifier.
    
    Args:
        frame: Input image frame
        
    Returns:
        List of face bounding boxes (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = self.face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces
```

#### Face Aligner

The alignment module positions faces consistently for better recognition:

```python
def align_face(self, frame, face_bbox):
    """
    Align a detected face based on eye positions.
    
    Args:
        frame: Input frame containing the face
        face_bbox: Bounding box of the detected face (x, y, w, h)
        
    Returns:
        Aligned face image
    """
    landmarks = self.landmark_detector.detect_landmarks(frame, face_bbox)
    if landmarks is None:
        return self._crop_face(frame, face_bbox)
    
    left_eye, right_eye = landmarks['left_eye'], landmarks['right_eye']
    return self._align_by_eyes(frame, face_bbox, left_eye, right_eye)
```

#### Embedding Generator

The system uses a FaceNet model to generate embeddings:

```python
def generate_embedding(self, face_img):
    """
    Generate a 512-dimensional face embedding.
    
    Args:
        face_img: Preprocessed face image
        
    Returns:
        512-dimensional embedding vector
    """
    # Ensure face image is properly sized
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype(np.float32) / 255.0
    
    # Generate embedding
    embedding = self.model.predict(np.expand_dims(face_img, axis=0))[0]
    return embedding / np.linalg.norm(embedding)
```

#### Database Handler

User embeddings are stored and retrieved efficiently:

```python
def add_user(self, username, embedding):
    """
    Add a new user with their face embedding to the database.
    
    Args:
        username: Unique identifier for the user
        embedding: 512-dimensional face embedding
        
    Returns:
        Boolean indicating success
    """
    if self.user_exists(username):
        return False
    
    # Prepare data for insertion
    user_data = [username] + list(embedding)
    
    # Add to database
    with open(self.db_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(user_data)
    
    return True
```

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

## Root Files
- [__init__.py](__init__.py): Python package initialization file
- [add_user.py](add_user.py): Script to add new users to the face recognition database
- [api.py](api.py): Main web server and REST API implementation
- [main.py](main.py): Main application entry point
- [docker-compose.yaml](docker-compose.yaml): Configuration for running the entire system with Docker Compose
- [Dockerfile](Dockerfile): Container configuration for the main face recognition service
- [Dockerfile.db](Dockerfile.db): Container configuration for the database service
- [requirements.txt](requirements.txt): Lists all Python dependencies

## Directories
- [CameraUtilis](CameraUtilis/): Utilities for camera handling and video stream processing
- [database](database/): Database implementations and storage logic
- [ImageUtilis](ImageUtilis/): Image processing functions for preprocessing face images
- [Model](Model/): Contains all machine learning models
  - [OpencvDetector](Model/OpencvDetector/): OpenCV Haar cascade classifier (faces.xml) used for face detection
- [templates](templates/): HTML templates for the web interface (includes [index.html](templates/index.html))
- [UsersDatabaseHandeler](UsersDatabaseHandeler/): Logic for managing user embeddings in the database
- [Align](Align/): Algorithms for aligning detected faces to improve recognition accuracy
- [Landmark](Landmark/): Facial landmark detection for identifying key facial features
- [Verify](Verify/): Verification logic for comparing face embeddings
- [drowsiness](drowsiness/): Likely contains drowsiness detection algorithms

## Additional Components
- __pycache__: Contains compiled Python bytecode
- .vscode: Visual Studio Code configuration settings
- .gitignore: Specifies files to be ignored by Git

The system uses OpenCV's Haar cascade classifier for face detection and a FaceNet model that generates 512-dimensional embeddings for face recognition. It's deployed as two Docker containers: one for face recognition and one for the database service.

## Model Information

The system uses a cascade classifier for face detection, located in faces.xml. For face recognition, it employs a TFLite model to generate 512-dimensional face embeddings.

## Database Structure

User embeddings are stored in a CSV format with 512 embedding values plus the username. The database handling is managed by the `EmbeddingCSVHandler` class.

SQLlite database 

## License

This project includes components with their respective licenses. The OpenCV Haar cascade classifier is used under the Intel License Agreement for Open Source Computer Vision Library.