# Enhanced Face Recognition System - Usage Guide

This document explains how to use the enhanced face recognition system with the newly added features:
- Face filtering (focus on largest face)
- Gaze direction detection
- ZMQ communication

## Run Scripts

We've created several scripts to make it easier to run and test the system:

### 1. Main Run Script (`run.sh`)

This is a versatile shell script that provides a simple interface for running the system:

```bash
# Make the script executable
chmod +x run.sh

# Show help
./run.sh --help

# Run full system with default settings
./run.sh run

# Run with specific options
./run.sh run --detector=mediapipe --no-filter --no-gaze

# Test just the gaze detection
./run.sh test-gaze --camera=0

# Test the database service
./run.sh test-db

# Run with Docker
./run.sh docker
```

### 2. Python Run Script (`run_face_recognition_system.py`)

This Python script provides more control over system parameters:

```bash
# Make executable
chmod +x run_face_recognition_system.py

# Run with default settings
./run_face_recognition_system.py 

# Run with specific options
./run_face_recognition_system.py --docker --detector mediapipe --disable-face-filtering --disable-gaze-detection

# Show help
./run_face_recognition_system.py --help
```

### 3. Gaze Detection Test (`test_gaze_detection.py`)

This script focuses specifically on testing the gaze direction detection feature:

```bash
# Make executable
chmod +x test_gaze_detection.py

# Run with default camera
./test_gaze_detection.py

# Run with specified camera
./test_gaze_detection.py --camera 1

# Show help
./test_gaze_detection.py --help
```

## Docker Deployment

We've also created an enhanced Docker Compose file that includes the new features:

```bash
# Start the system with enhanced features
docker-compose -f docker-compose-enhanced.yaml up
```

## Features and Parameters

### Face Filtering

The system can now automatically filter multiple faces to focus on the largest face (typically the closest person to the camera). This is enabled by default but can be disabled:

- In `run.sh`: Use `--no-filter` option
- In Python scripts: Use `--disable-face-filtering` option
- In Docker: Set environment variable `FILTER_LARGEST_FACE=false`

### Gaze Direction Detection

The system can now detect which way a person is looking (left, right, center) by measuring the distances between the nose and eyes:

- In `run.sh`: Use `--no-gaze` option to disable
- In Python scripts: Use `--disable-gaze-detection` option
- In Docker: Set environment variable `ENABLE_GAZE_DETECTION=false`

### Detector Selection

The system supports multiple face detectors:

- `mediapipe` (recommended for gaze detection)
- `yolov8` 
- `yolov8_onnx`

You can select the detector using the `--detector` option in the run scripts.

## Example Use Cases

### Basic Face Recognition

```bash
./run.sh run
```

### Face Recognition with YOLOv8

```bash
./run.sh run --detector=yolov8
```

### Testing Gaze Detection

```bash
./run.sh test-gaze
```

### Face Recognition without Face Filtering

```bash
./run.sh run --no-filter
```

### Docker Deployment

```bash
./run.sh docker
# or
docker-compose -f docker-compose-enhanced.yaml up
```
