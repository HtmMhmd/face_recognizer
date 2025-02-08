import mediapipe as mp
from typing import Tuple, Dict, List
# from Model.Landmark.Facedetector import MediapipeFaceDetector

class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )

    def get_landmarks(self, image):
        results = self.face_mesh.process(image)
        return results

    def get_eye_mouth_keypoints(self, face_landmarks, image_shape) -> Dict[str, List[Tuple[int, int]]]:
        eye_mouth_keypoints = {
            "left_eye": [],
            "right_eye": [],
            "mouth": []
        }
        h, w, _ = image_shape

        LEFT_EYE_INDICES = [33, 133, 160, 144, 158, 153]
        RIGHT_EYE_INDICES = [362, 263, 387, 373, 380, 374]
        MOUTH_INDICES = [61, 291, 39, 181, 17, 405]

        for idx, landmark in enumerate(face_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            if idx in LEFT_EYE_INDICES:
                eye_mouth_keypoints["left_eye"].append((cx, cy))
            if idx in RIGHT_EYE_INDICES:
                eye_mouth_keypoints["right_eye"].append((cx, cy))
            if idx in MOUTH_INDICES:
                eye_mouth_keypoints["mouth"].append((cx, cy))
        return eye_mouth_keypoints
