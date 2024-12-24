import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def plot_images(img1, img2, title1='Image 1', title2='Image 2'):
    # Check if img1 is a file path or an array
    if isinstance(img1, str) and os.path.isfile(img1):
        img1 = cv2.imread(img1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    elif isinstance(img1, np.ndarray):
        if img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("img1 must be a valid file path or a numpy array")

    # Check if img2 is a file path or an array
    if isinstance(img2, str) and os.path.isfile(img2):
        img2 = cv2.imread(img2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    elif isinstance(img2, np.ndarray):
        if img2.dtype != np.uint8:
            img2 = (img2 * 255).astype(np.uint8)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("img2 must be a valid file path or a numpy array")

    # Create a figure to display the images
    plt.figure(figsize=(10, 5))
    
    # Plot the first image
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis('off')
    
    # Plot the second image
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis('off')
    
    # Display the images
    plt.show()

# Example usage
if __name__ == '__main__':
    img1_path = 'path_to_image1.jpg'
    img2_path = 'path_to_image2.jpg'
    plot_images(img1_path, img2_path)