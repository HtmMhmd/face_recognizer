#!/bin/bash
# Run script for the Face Recognition System with enhanced features

# Function to display usage information
show_usage() {
  echo "Face Recognition System Runner"
  echo "Usage: $0 [command] [options]"
  echo ""
  echo "Commands:"
  echo "  run             Run the complete face recognition system"
  echo "  test-gaze       Test just the gaze detection functionality" 
  echo "  test-db         Test the database service"
  echo "  docker          Run the system using Docker"
  echo "  docker-enhanced Run the system using Docker with enhanced features"
  echo ""
  echo "Options for 'run':"
  echo "  --detector=TYPE    Set detector type (mediapipe, yolov8, yolov8_onnx)"
  echo "  --no-filter        Disable largest face filtering"
  echo "  --no-gaze          Disable gaze detection"
  echo ""
  echo "Options for 'test-gaze':"
  echo "  --camera=ID        Set camera ID (default: 0)"
  echo ""
  echo "Examples:"
  echo "  $0 run                      # Run the system with default settings"
  echo "  $0 run --detector=mediapipe # Run with mediapipe detector"
  echo "  $0 run --no-filter         # Run without face filtering"
  echo "  $0 test-gaze               # Test gaze detection" 
  echo "  $0 docker                  # Run with Docker"
}

# Default values
DETECTOR="mediapipe"
FACE_FILTER=""
GAZE_DETECTION=""
CAMERA="0"

# Parse command
if [ $# -lt 1 ]; then
  show_usage
  exit 1
fi

COMMAND=$1
shift

# Parse options
while [ $# -gt 0 ]; do
  case "$1" in
    --detector=*)
      DETECTOR="${1#*=}"
      ;;
    --no-filter)
      FACE_FILTER="--disable-face-filtering"
      ;;
    --no-gaze)
      GAZE_DETECTION="--disable-gaze-detection"
      ;;
    --camera=*)
      CAMERA="${1#*=}"
      ;;
    --help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
  shift
done

# Execute the requested command
case "$COMMAND" in
  run)
    echo "Starting Face Recognition System..."
    python3 run_face_recognition_system.py --detector "$DETECTOR" $FACE_FILTER $GAZE_DETECTION
    ;;
  test-gaze)
    echo "Starting Gaze Detection Test..."
    python3 test_gaze_detection.py --camera "$CAMERA" --detector "$DETECTOR"
    ;;
  test-db)
    echo "Starting Database Service Test..."
    python3 db_service_test.py
    ;;
  docker)
    echo "Starting Face Recognition System with Docker..."
    docker-compose -f docker-compose-zmq.yaml up
    ;;
  docker-enhanced)
    echo "Starting Enhanced Face Recognition System with Docker..."
    docker-compose -f docker-compose-enhanced.yaml up
    ;;
  *)
    echo "Unknown command: $COMMAND"
    show_usage
    exit 1
    ;;
esac
