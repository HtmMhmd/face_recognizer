# from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

def resize_image(img: np.ndarray) -> np.ndarray:
    """
    Resize an image to expected size of a ml model while maintaining aspect ratio.
    Args:
        img (np.ndarray): pre-loaded image as numpy array
    Returns:
        img (np.ndarray): resized and padded input image
    """
    target_size = [160, 160]
    
    # Safe division with zero checking
    factor_0 = target_size[0] / img.shape[0] if img.shape[0] != 0 else 0
    factor_1 = target_size[1] / img.shape[1] if img.shape[1] != 0 else 0
    
    # If both factors are zero, return empty image of target size
    if factor_0 == 0 and factor_1 == 0:
        return np.zeros((target_size[0], target_size[1], img.shape[2]), dtype=img.dtype)
    
    factor = min(factor_0, factor_1) if factor_0 * factor_1 != 0 else max(factor_0, factor_1)

    # Ensure we have valid dimensions
    new_height = max(1, int(img.shape[0] * factor))
    new_width = max(1, int(img.shape[1] * factor))

    # Resize image maintaining aspect ratio
    resized = cv2.resize(img, (new_width, new_height))

    # Calculate padding
    pad_height = target_size[0] - new_height
    pad_width = target_size[1] - new_width

    # Add padding to center the image
    padded = np.pad(
        resized,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
            (0, 0),
        ),
        "constant",
    )

    return padded

# Function to preprocess the image

def preprocess_image(img) -> np.ndarray:
    image = img.copy()
    if isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("img1 must be a valid file path or a numpy array")
    
    # Ensure float32 type and proper normalization
    image = image.astype(np.float32)
    image = image / 127.5
    image = image - 1.0
    image = resize_image(image)
    image = np.expand_dims(image, axis=0)
    
    # Final type check
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    return image
