def get_cropped_faces(image, boxes):
    """
    Returns cropped face images from the input image based on detected bounding boxes.

    Args:
        image: The input image (as a NumPy array).

    Returns:
        A list of cropped face images.
    """
    cropped_faces = []
    if boxes is not None:
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_face = image[y_min:y_max, x_min:x_max]
            cropped_faces.append(cropped_face)
    return cropped_faces
