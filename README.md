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


## Model Information

The system uses OpenCV's Haar cascade classifier for face detection and a FaceNet model that generates 512-dimensional embeddings for face recognition. It's deployed as two Docker containers: one for face recognition and one for the database service.

## Database Structure

User embeddings are stored in a CSV format with 512 embedding values plus the username. The database handling is managed by the `EmbeddingCSVHandler` class.

SQLlite database 

---
## DVC Tutorial for Managing TFLite Models

### What is DVC?

[DVC (Data Version Control)](https://dvc.org/) is an open-source version control system for machine learning projects. It helps track changes to large files like models and datasets without storing them directly in Git.

### Setting Up DVC for .tflite Models

1. Install DVC:
   ```bash
   pip install dvc
   pip install dvc-gdrive
   ```

2. Initialize DVC in your repository:
   ```bash
   dvc init
   git add .dvc .dvcignore
   git commit -m "Initialize DVC"
   ```

### Configuring Google Drive Remote Storage

1. **Enable Google Drive API** (required to fix the 403 error):
   - Visit the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API" and enable it
   - **Wait a few minutes** for the changes to take effect

2. Set up authentication:
   
   **STEP 1: Using OAuth (Interactive)**
   ```bash
   dvc remote add -d myremote gdrive://your-folder-id
   ```

   **STEP 2: Using Service Account (Non-interactive, recommended for automation)**
   - Create a service account in Google Cloud Console
   - Go to "IAM & Admin" > "Service Accounts" > "Create Service Account"
   - Provide a name and grant necessary permissions
   - Create a key (JSON format) and download it

   **STEP 3: Configure DVC to use the service account:**
   ```bash
   dvc remote modify myremote gdrive_use_service_account true

   dvc remote modify myremote gdrive_service_account_json_file_path path/to/credentials.json
   ```

3. Add additional configuration if needed:
   ```bash
   # To bypass confirmation prompts for downloading potentially malicious files
   dvc remote modify myremote gdrive_acknowledge_abuse true
   
   # Commit your DVC configuration
   git add .dvc/config
   git commit -m "Configure DVC remote storage"
   ```

### Tracking TFLite Models with DVC

1. Add your .tflite model to DVC:
   ```bash
   dvc add Model/path/to/your/model.tflite
   ```

2. Stop Git from tracking the model file:
   ```bash
   git rm -r --cached 'Model/path/to/your/model.tflite'
   ```

3. Commit the DVC file to Git:
   ```bash
   git add Model/path/to/your/model.tflite.dvc
   git commit -m "Add model with DVC tracking"
   ```

4. Push the model to remote storage:
   ```bash
   dvc push
   ```

### Troubleshooting DVC with Google Drive

1. **Error 403: Google Drive API not enabled**
   - Follow the link in the error message to enable the Google Drive API
   - Wait a few minutes for the change to take effect
   - Try again with `dvc push`

2. **Authentication issues with service account**
   - Ensure the service account has access to the Google Drive folder
   - Share the folder with the service account email address
   - Make sure the JSON key file path is correct

3. **Permission issues**
   - Verify the folder ID is correct
   - Ensure your account or service account has write access to the folder
   - Try creating a new folder specifically for DVC storage

4. **Alternative approach: Use a different remote**
   If Google Drive continues to cause issues, consider alternatives:
   ```bash
   # Local remote
   dvc remote add -d localremote /path/to/local/storage
   
   # AWS S3
   pip install dvc[s3]
   dvc remote add -d s3remote s3://bucket/path
   
   # Azure Blob Storage
   pip install dvc[azure]
   dvc remote add -d azremote azure://container/path
   ```

### Working with Model Versions

1. Update a model:
   ```bash
   # Replace the model file with a new version
   cp /path/to/new/model.tflite models/face_recognition.tflite
   
   # Track the changes
   dvc add models/face_recognition.tflite
   git add models/face_recognition.tflite.dvc
   git commit -m "Update face recognition model"
   dvc push
   ```

2. Switch between model versions:
   ```bash
   # Checkout a specific Git commit
   git checkout <commit-hash>
   
   # Pull the corresponding model version
   dvc pull
   ```

3. Create a model tag:
   ```bash
   git tag -a model-v1.0 -m "Model version 1.0"
   git push origin model-v1.0
   ```

### Best Practices

1. Always run `dvc push` after adding or updating models
2. Use meaningful commit messages for model changes
3. Consider tagging important model versions
4. Add model metrics to track performance changes

### Using DVC in CI/CD

For automated workflows, use these commands in your CI scripts:

```bash
# Pull the latest models
dvc pull

# Run your tests/deployment
python your_script.py
```

For more information, visit the [DVC documentation](https://dvc.org/doc).

## CI/CD with GitHub Actions

This project uses GitHub Actions to automatically build and push Docker images for multiple architectures (x86_64 and ARM/Raspberry Pi) to Docker Hub.

### Workflow Overview

The GitHub Actions workflow:
- Builds Docker images for both x86_64 (AMD64) and ARM architectures (Raspberry Pi 4)
- Tags the images with version numbers and architecture identifiers
- Pushes the images to a private Docker Hub repository
- Creates multi-architecture manifests for easy deployment

### Setup Instructions

To set up the CI/CD pipeline:

1. **Create Docker Hub Repository**:
   - Create a private repository on Docker Hub for your images

2. **Configure GitHub Secrets**:
   In your GitHub repository, go to Settings > Secrets and add:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token (create in Docker Hub account settings)

3. **Trigger Workflow**:
   - Automatically triggered on:
     - Pushes to the main branch
     - Creating version tags (e.g., v1.0.0)
   - Can be manually triggered from the Actions tab

### Image Tagging Strategy

The workflow creates several tags:
- `latest-amd64`: Latest x86_64 build
- `latest-arm`: Latest ARM build (Raspberry Pi)
- `v1.0.0-amd64`: Version-specific x86_64 build
- `v1.0.0-arm`: Version-specific ARM build
- `latest`: Multi-architecture image
- `v1.0.0`: Version-specific multi-architecture image

### Using the Built Images

To use the appropriate image for your architecture:

For x86_64 systems:
```bash
docker pull yourusername/face_recognizer:latest-amd64
# or specific version
docker pull yourusername/face_recognizer:v1.0.0-amd64
```

For Raspberry Pi:
```bash
docker pull yourusername/face_recognizer:latest-arm
# or specific version
docker pull yourusername/face_recognizer:v1.0.0-arm
```

Using multi-architecture image (automatically selects correct architecture):
```bash
docker pull yourusername/face_recognizer:latest
```

### Creating a New Release

To create a new release and trigger the workflow:

1. Create and push a new tag:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

2. GitHub Actions will automatically build and tag the Docker images with this version

### Workflow File

The workflow is defined in `.github/workflows/docker-build.yml`. You can view and modify it to customize the build process.

## License

This project includes components with their respective licenses. The OpenCV Haar cascade classifier is used under the Intel License Agreement for Open Source Computer Vision Library.