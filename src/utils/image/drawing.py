import cv2
import numpy as np

def draw_user_names_on_bboxes(image, results):
    """
    Draw user names on bounding boxes.
    
    Args:
        image (np.ndarray): Input image
        results (list): List of dictionaries with bbox, user_name, and verification_result keys
        
    Returns:
        np.ndarray: Image with user names drawn on bounding boxes
    """
    if not results:
        return image
        
    img_with_names = image.copy()
    
    for result in results:
        bbox = result.get("bbox")
        user_name = result.get("user_name", "Unknown")
        
        if bbox is None:
            continue
            
        # Draw rectangle and name
        cv2.rectangle(img_with_names, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(img_with_names, user_name, (bbox[0], bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                   
    return img_with_names