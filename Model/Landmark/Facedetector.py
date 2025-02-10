import cv2
import mediapipe as mp
import time
from Model.DetectionResult import DetectionResult
from Model.FaceDetection import FaceDetector

class MediapipeFaceDetector(FaceDetector):
    def __init__(self, min_detection_conf=0.5, verbose=False):
        """
        Initializes the MediapipeFaceDetector with the specified minimum detection confidence.

        Args:
            min_detection_conf (float): The minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful. Defaults to 0.5.
            verbose (bool): Enables verbose output for debugging. Defaults to False.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_conf)
        self.verbose = verbose

    def detect_faces(self, image):
        """
        Processes the input image to detect faces using the MediaPipe face detection model.

        Args:
            image (np.ndarray): The input image in which faces are to be detected.

        Returns:
            A DetectionResult object containing the detected faces.
        """
        start_time = time.time()
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inference_time = time.time() - start_time

        if self.verbose:
            print(f"MediapipeFaceDetector Inference Time: {inference_time * 1000:.2f} ms")

        detection_result = DetectionResult()
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                detection_result.add(
                    box=(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    score=detection.score[0],
                    class_id=0  # Assuming class_id 0 for faces
                )
        return detection_result