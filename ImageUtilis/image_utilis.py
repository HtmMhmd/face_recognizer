# from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

def resize_image(img: np.ndarray) -> np.ndarray:
    """
    Resize an image to expected size of a ml model with adding black pixels.
    Args:
        img (np.ndarray): pre-loaded image as numpy array
        target_size (tuple): input shape of ml model
    Returns:
        img (np.ndarray): resized input image
    """
    target_size = [160,160] 
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, (160,160))

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    # Put the base image in the middle of the padded image
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, (160,160))

    if img.max() > 1:
        # img = (img.astype(np.float32) / 255.0).astype(np.float32)
        pass
    return img



# Function to preprocess the image

def preprocess_image(image):
    if isinstance(image, str) and os.path.isfile(image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image
    else:
        raise ValueError("img1 must be a valid file path or a numpy array")

    image = image / 127.5
    image = image - 1
    # mean, std = img.mean(), img.std()
    # img = (img - mean) / std
    image = resize_image(image)
    # image = image / 255.0
    image = np.expand_dims(image, axis=0)  # to (1, 224, 224, 3)


    return image
