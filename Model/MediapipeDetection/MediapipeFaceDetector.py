import cv2
import mediapipe as mp
import time
from Model.FaceDetection import FaceDetector
from Model.detection_utilis import draw_detections, get_cropped_faces
from Model.DetectionFaces import DetectionFaces

class MediapipeFaceDetector(FaceDetector):
    def __init__(self, min_detection_conf=0.5, verbose=False, detection_faces=None):
        """
        Initializes the MediapipeFaceDetector with the specified minimum detection confidence.

        Args:
            min_detection_conf (float): The minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful. Defaults to 0.5.
            verbose (bool): Enables verbose output for debugging. Defaults to False.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_conf)
        self.verbose = verbose
        self.detection_faces = detection_faces if detection_faces is not None else DetectionFaces()

    def detect_faces(self, image):
        """
        Processes the input image to detect faces using the MediaPipe face detection model.

        Args:
            image (np.ndarray): The input image in which faces are to be detected.

        Returns:
            DetectionFaces: The detection results containing bounding boxes, scores, class IDs, and cropped faces.
        """
        self.detection_faces.reset()  # Reset the detection faces object
        start_time = time.time()
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inference_time = time.time() - start_time

        if self.verbose:
            print(f"MediapipeFaceDetector Inference Time: {inference_time * 1000:.2f} ms")

        if results.detections:
            for i, detection in enumerate(results.detections):
                
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                box=[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                self.detection_faces.add(
                    box= box,
                    score=detection.score[0],
                    class_id=0,  # Assuming class_id 0 for faces
                    cropped_face=get_cropped_faces(image, [box])
                )
        return self.detection_faces

    def draw_detections(self, image):
        """
        Draws bounding boxes on the detected faces in the image.

        Args:
            image (np.ndarray): The input image.
            detection_faces (DetectionFaces): The detection results containing bounding boxes.

        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        return draw_detections(image, self.detection_faces.boxes, self.detection_faces.scores, self.detection_faces.class_ids)