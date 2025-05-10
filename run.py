#!/usr/bin/env python3
import argparse
import os
import sys
import time
import cv2
from src.services import ImageProcessor, run_with_camera_handler, run_camera_feed, process_image
from src.api.app import start_api_server
from src.config import settings, app_settings, camera_settings, detection_settings, drowsiness_settings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Recognition System")
    
    parser.add_argument("--mode", type=str, default=app_settings.get("default_mode", "camera"),
                        choices=["image", "camera", "api"],
                        help="Application mode: image, camera, or api")
    
    parser.add_argument("--image-path", type=str,
                        help="Path to the image file for image mode")
    
    parser.add_argument("--camera", type=int, default=camera_settings.get("default_index", 0),
                        help="Camera index to use")
    
    parser.add_argument("--detector", type=str, default=detection_settings.get("default_model", "mediapipe"),
                        choices=["mediapipe", "yolov8", "yolov8_onnx", "landmark"],
                        help="Face detector model to use")
    
    parser.add_argument("--enable-drowsiness", action="store_true", default=drowsiness_settings.get("enabled", False),
                        help="Enable drowsiness detection")
    
    parser.add_argument("--gui", action="store_true", default=camera_settings.get("show_gui", False),
                        help="Show GUI for camera mode")
    
    parser.add_argument("--output", type=str,
                        help="Output file for saving results")
    
    parser.add_argument("--port", type=int, default=settings.api.get("port", 8000),
                        help="Port for API server in API mode")
    
    parser.add_argument("--verbose", action="store_true", default=app_settings.get("verbose", True),
                        help="Enable verbose output")
    
    # Add handler type argument for camera mode
    parser.add_argument("--handler", type=str, choices=["threaded", "regular"], default="threaded",
                        help="Camera handler type: threaded (CameraHandler) or regular (direct OpenCV)")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == "image" and args.image_path is None:
        parser.error("--image-path is required for image mode")
    
    return args

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    if args.verbose:
        print(f"Starting Face Recognition System in {args.mode} mode")
        print(f"Using {args.detector} detector")
        if args.enable_drowsiness:
            print("Drowsiness detection is enabled")
    
    # Handle different application modes
    if args.mode == "image":
        process_image_file(args)
    elif args.mode == "camera":
        process_camera(args)
    elif args.mode == "api":
        start_api(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

def process_image_file(args):
    """Process a single image using the image service."""
    if not os.path.exists(args.image_path):
        print(f"Image file not found: {args.image_path}")
        sys.exit(1)
    
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Failed to read image: {args.image_path}")
        sys.exit(1)
    
    # Use the image service function instead of directly using ImageProcessor
    processed_image = process_image(
        image, 
        detector_type=args.detector,
        output_path=args.output,
        verbose=args.verbose
    )
    
    # Show the result
    cv2.imshow("Face Detection", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_camera(args):
    """Process video from camera using the selected camera handler."""
    if args.verbose:
        print(f"Using {args.detector} detector with {'threaded' if args.handler == 'threaded' else 'regular'} camera handler")
    
    # Handle landmark detector mode
    if args.detector == "landmark":
        if args.verbose:
            print("Landmark mode selected: Only performing landmark detection")
        run_with_camera_handler(
            detector_type=args.detector,
            enable_drowsiness=False,  # Disable drowsiness detection in landmark mode
            show_gui=args.gui,
            output_json=args.output,
            camera_index=args.camera,
            verbose=args.verbose
        )
        return

    # Choose the appropriate camera handler based on arguments
    if args.handler == "threaded":
        if args.verbose:
            print("Using threaded camera handler")
        run_with_camera_handler(
            detector_type=args.detector,
            enable_drowsiness=args.enable_drowsiness,
            show_gui=args.gui,
            output_json=args.output,
            camera_index=args.camera,
            verbose=args.verbose
        )
    else:
        if args.verbose:
            print("Using regular camera handler")
        run_camera_feed(
            detector_type=args.detector,
            enable_drowsiness=args.enable_drowsiness,
            show_gui=args.gui,
            output_json=args.output,
            camera_index=args.camera,
            verbose=args.verbose
        )

def start_api(args):
    """Start API server mode."""
    if args.verbose:
        print(f"Starting API server on port {args.port}")
    
    # Initialize camera service for API
    camera_service = ImageProcessor(
        model_architecture=args.detector,
        verbose=args.verbose
    )
    
    # Start API server
    try:
        start_api_server(camera_service, port=args.port)
    except KeyboardInterrupt:
        print("API server interrupted by user")

if __name__ == "__main__":
    main()