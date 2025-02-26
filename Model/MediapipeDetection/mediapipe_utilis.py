import mediapipe as mp

def draw_landmarks(image_original, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    image = image_original.copy()

    # Check if landmarks is the full result (with multi_face_landmarks) or just a single face's landmarks
    if hasattr(landmarks, "multi_face_landmarks"):  # Full Mediapipe output
        face_landmarks_list = landmarks.multi_face_landmarks
    elif hasattr(landmarks, "landmark"):  # Single face's landmarks
        face_landmarks_list = [landmarks]
    else:
        print("Invalid landmarks format in draw_landmarks")
        return image  # Return original image if landmarks are not valid

    for face_landmarks in face_landmarks_list:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )

    return image
