import cv2
import mediapipe as mp
from Model.DetectionResult import DetectionResult
from Model.FaceDetection import FaceDetector

class MediapipeFaceDetector(FaceDetector):
    def __init__(self, min_detection_conf=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_conf)

    def detect_faces(self, image):
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection_result = DetectionResult()
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw= image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                detection_result.add(
                    box=(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    score=detection.score[0],
                    class_id=0  # Assuming class_id 0 for faces
                )
        return detection_result