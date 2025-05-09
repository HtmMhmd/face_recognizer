#!/usr/bin/env python3
import argparse
import os
import sys
import time
import cv2
from src.services.image_processor import ImageProcessor
from src.services.camera_service import CameraService
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
                        choices=["mediapipe", "yolov8", "yolov8_onnx"],
                        help="Face detector model to use")
    
    parser.add_argument("--enable-drowsiness", action="store_true", default=drowsiness_settings.get("enabled", False),
                        help="Enable drowsiness detection")
    
    parser.add_argument("--gui", action="store_true", default=camera_settings.get("show_gui", False),
                        help="Show GUI for camera mode")
    
    parser.add_argument("--output", type=str,
                        help="Output file for saving results")
    
    parser.add_argument("--port", type=int, default=settings.api.get("port", 8000),
                        help="Port for API server in API mode")
    
    parser.add_argument("--verbose", action="store_true", default=app_settings.get("verbose", False),
                        help="Enable verbose output")
    
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
        process_image(args)
    elif args.mode == "camera":
        process_camera(args)
    elif args.mode == "api":
        start_api(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

def process_image(args):
    """Process a single image."""
    if not os.path.exists(args.image_path):
        print(f"Image file not found: {args.image_path}")
        sys.exit(1)
    
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Failed to read image: {args.image_path}")
        sys.exit(1)
    
    processor = ImageProcessor(model_architecture=args.detector, verbose=args.verbose)
    
    # Process the image
    results = processor.process_image(image)
    
    # Draw detection results on the image
    if results:
        image = processor.draw_detections(image)
        
        # Verify faces if any detected
        if len(results.detection_faces.boxes) > 0:
            verification_results = processor.verify_faces()
            if verification_results:
                image = processor.draw_user_names(image, verification_results)
    
    # Show the result
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save results if requested
    if args.output:
        cv2.imwrite(args.output, image)
        print(f"Results saved to {args.output}")

def process_camera(args):
    """Process video from camera."""
    # Initialize camera service
    camera_service = CameraService(
        camera_index=args.camera,
        detector_type=args.detector,
        enable_drowsiness=args.enable_drowsiness,
        show_gui=args.gui,
        save_output=args.output is not None,
        output_path=args.output,
        verbose=args.verbose
    )
    
    # Start camera service
    try:
        camera_service.start()
        while True:
            if args.gui:
                # Wait for key press to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        camera_service.stop()

def start_api(args):
    """Start API server mode."""
    if args.verbose:
        print(f"Starting API server on port {args.port}")
    
    # Initialize camera service for API
    camera_service = CameraService(
        camera_index=args.camera,
        detector_type=args.detector,
        enable_drowsiness=args.enable_drowsiness,
        show_gui=False,  # No GUI in API mode
        verbose=args.verbose
    )
    
    # Start camera service
    camera_service.start()
    
    try:
        # Start API server
        start_api_server(camera_service, port=args.port)
    except KeyboardInterrupt:
        print("API server interrupted by user")
    finally:
        camera_service.stop()

if __name__ == "__main__":
    main()