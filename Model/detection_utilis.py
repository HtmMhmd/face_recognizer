import cv2
import numpy as np

class_names = ['face']

rng = np.random.default_rng(3)# Create a list of colors for each class where each color is a tuple of 3 integer values
colors = rng.uniform(0, 255, size=(len(class_names), 3))

def get_cropped_faces(image, boxes):
    """
    Returns cropped face images from the input image based on detected bounding boxes.

    Args:
        image: The input image (as a NumPy array).

    Returns:
        A list of cropped face images.
    """
    cropped_faces = []
    if (boxes is not None) and (len(boxes) >= 1):
        for box in boxes:
            x_min, y_min, x_max, y_max  = box
            cropped_face = image[y_min:y_max, x_min:x_max]
            cropped_faces.append(cropped_face)
    return np.array(cropped_faces, dtype='float32')

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    """
    Draws bounding boxes and labels of detections on the input image.

    Args:
        image: The input image (as a NumPy array).
        boxes: Bounding boxes of detections (as a NumPy array).
        scores: Scores of detections (as a NumPy array).
        class_ids: Class IDs of detections (as a NumPy array).
        mask_alpha: Alpha value of the bounding box mask image (default is 0.3).

    Returns:
        An image with the bounding boxes and labels of detections drawn on it.
    """
    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = np.uint16(box)

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)


    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img

def draw_user_names_on_bboxes(image, results):
    """Draws bounding boxes and user names on an image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        results (list): A list of dictionaries, where each dictionary contains:
            - 'bbox' (tuple): A tuple of (x1, y1, x2, y2) representing the bounding box.
            - 'user_name' (str): The user name to display.
            - 'verification_result' (bool): Boolean indicating the verification result (optional, used for color).

    Returns:
        numpy.ndarray: The modified image with bounding boxes and user names.
    """

    for result in results:
        bbox = result['bbox']
        user_name = result['user_name']
        verification_result = result.get('verification_result', True)  # Default to True if not present

        x1, y1, x2, y2 = map(int, bbox)  # Convert bbox coordinates to integers

        # Determine bounding box color based on verification result
        color = (0, 255, 0) if verification_result else (0, 0, 255)  # Green for True, Red for False

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Set font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = color  # Use the same color as the bounding box
        font_thickness = 2

        # Calculate text size to avoid exceeding image boundaries and for better positioning
        text_size = cv2.getTextSize(user_name, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 10  # Position text slightly above the bounding box

        # Adjust text position if it goes out of bounds
        if text_y < 0:
            text_y = y1 + text_size[1] + 10 # Place below box if above

        # Draw the user name on the image
        cv2.putText(image, user_name, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return image
