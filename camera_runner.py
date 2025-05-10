#!/usr/bin/env python3
import argparse
import sys
from src.services import run_with_camera_handler, run_camera_feed
from src.config import app_settings, camera_settings, detection_settings, drowsiness_settings

def parse_args():
    """Parse command line arguments for camera runner."""
    parser = argparse.ArgumentParser(description="Face Recognition Camera Runner")
    
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
                        help="Output file for saving results (JSON)")
    
    parser.add_argument("--verbose", action="store_true", default=app_settings.get("verbose", False),
                        help="Enable verbose output")
    
    parser.add_argument("--handler", type=str, choices=["threaded", "regular"], default="threaded",
                        help="Camera handler type: threaded (CameraHandler) or regular (direct OpenCV)")
    
    return parser.parse_args()

def main():
    """Main entry point for the camera runner application."""
    args = parse_args()
    
    print(f"Starting Face Recognition Camera Runner")
    print(f"Using {args.detector} detector with {'threaded' if args.handler == 'threaded' else 'regular'} camera handler")
    
    if args.enable_drowsiness:
        print("Drowsiness detection is enabled")
    
    try:
        # Choose the appropriate camera handler based on arguments
        if args.handler == "threaded":
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
            print("Using regular camera handler")
            run_camera_feed(
                detector_type=args.detector,
                enable_drowsiness=args.enable_drowsiness,
                show_gui=args.gui,
                output_json=args.output,
                camera_index=args.camera,
                verbose=args.verbose
            )
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running camera: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
