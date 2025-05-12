import cv2
import os
import json
import logging
from datetime import datetime

from src.services.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Image_Service")

def process_image(detector_type='mediapipe', image_path=None, output_path=None, 
                 show_gui=False, verbose=False):
    """
    Process a single image for face recognition.
    
    Args:
        detector_type (str): Type of detector to use ('mediapipe', 'yolov8', 'yolov8_onnx')
        image_path (str): Path to the input image file
        output_path (str): Path to save the output image file (optional)
        show_gui (bool): Whether to show GUI display of results
        verbose (bool): Enable verbose output
    """
    if not image_path or not os.path.exists(image_path):
        logger.error(f"Input image not found: {image_path}")
        return
    
    logger.info(f"Processing image: {image_path}")
    
    # Initialize image processor
    image_processor = ImageProcessor(model_architecture=detector_type, verbose=verbose)
    
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return
    
    # Process the image
    embeddings = image_processor.process_image(image)
    
    # Check if faces were detected
    if embeddings is None or len(embeddings.embeddings) == 0:
        print("[INFO] No faces detected in the image")
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"[INFO] Original image saved to: {output_path}")
        if show_gui:
            cv2.imshow("Face Recognition", image)
            cv2.waitKey(0)
        return
    
    # Clone the image for drawing
    display_image = image.copy()
    
    # Draw detections
    display_image = image_processor.draw_detections(display_image)
    
    # Detect landmarks
    landmarks = image_processor.detect_landmarks(image)
    
    # Draw landmarks
    display_image = image_processor.draw_landmarks(display_image)
    
    # Verify faces against the database
    verify_results = image_processor.verify_faces()
    
    # Draw user names
    display_image = image_processor.draw_user_names(display_image, verify_results)
    
    # Print recognition results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[RESULTS] {timestamp}")
    
    if verify_results:
        for result in verify_results:
            user_name = result.get('user_name', 'Unknown')
            verified = result.get('verification_result', False)
            print(f"  User: {user_name}, Verified: {verified}")
    else:
        print("  No users recognized")
    
    # Save the output image if requested
    if output_path:
        cv2.imwrite(output_path, display_image)
        print(f"[INFO] Processed image saved to: {output_path}")
    
    # Show the image with detections if GUI is enabled
    if show_gui:
        cv2.imshow("Face Recognition", display_image)
        print("[INFO] Press any key to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("[INFO] Image processing completed")