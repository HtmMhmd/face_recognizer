#!/usr/bin/env python3
"""
Run script for the face recognition system with all features enabled.
This script starts the complete distributed face recognition system with:
- Face filtering (focusing on largest face when multiple faces are detected)
- Gaze direction detection (determining where a person is looking)
- ZMQ communication between services

Use this script to run the complete system locally or with Docker.
"""

import argparse
import logging
from src.config.settings import Settings
from service_orchestrator import ServiceOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Face_Recognition_System")

def main():
    """Start the face recognition system with all features enabled."""
    parser = argparse.ArgumentParser(description="Face Recognition System with Enhanced Features")
    parser.add_argument("--docker", action="store_true", default=False,
                      help="Use Docker for services")
    parser.add_argument("--detector", type=str, choices=["mediapipe", "yolov8", "yolov8_onnx"],
                      default="mediapipe", help="Detector model to use (mediapipe recommended for gaze detection)")
    parser.add_argument("--disable-face-filtering", action="store_true",
                      help="Disable face filtering (process all detected faces)")
    parser.add_argument("--disable-gaze-detection", action="store_true",
                      help="Disable gaze direction detection")
    
    args = parser.parse_args()
    
    # Log startup configuration
    logger.info("Starting Face Recognition System with enhanced features:")
    logger.info(f"Deployment mode: {'Docker' if args.docker else 'Local'}")
    logger.info(f"Detector: {args.detector}")
    logger.info(f"Face filtering: {'Disabled' if args.disable_face_filtering else 'Enabled'}")
    logger.info(f"Gaze detection: {'Disabled' if args.disable_gaze_detection else 'Enabled'}")
    
    # Load settings and update with command line arguments
    settings = Settings()
    settings.detection["default_model"] = args.detector
    settings.detection["filter_largest_face"] = not args.disable_face_filtering
    settings.detection["gaze_detection"] = not args.disable_gaze_detection
    
    # Start the orchestrator with the specified deployment mode
    orchestrator = ServiceOrchestrator(use_docker=args.docker)
    
    try:
        orchestrator.start_all_services()
        logger.info(f"Face Recognition System is running.")
        logger.info(f"Dashboard UI available at http://localhost:{orchestrator.ports['dashboard']}")
        logger.info("Press Ctrl+C to stop the system")
        
        # Wait for completion (or user interrupt)
        orchestrator.wait_for_completion()
    except KeyboardInterrupt:
        logger.info("System shutdown requested by user")
    except Exception as e:
        logger.error(f"Error running face recognition system: {e}")
    finally:
        orchestrator.stop_all_services()
        logger.info("Face Recognition System has been shut down")

if __name__ == "__main__":
    main()
