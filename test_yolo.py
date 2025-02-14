import cv2
# from ultralytics import YOLO
from Model.YoloDetection.YoloV8OnnxRuntime.Yolov8OnnxRuntimeDetector import Yolov8OnnxRuntimeDetector
from Model.MediapipeDetection.MediapipeFaceDetector import MediapipeFaceDetector
from Model.MediapipeDetection.MediapipeFaceLandmarker import FaceMeshDetector
from Model.detection_utilis import draw_detections
from Model.MediapipeDetection.mediapipe_utilis import draw_landmarks
# from Model.Detection.YoloDetector.YoloDetector import Yolov8Detector

# Initialize the face detector and face mesh detector
face_detector = MediapipeFaceDetector()
face_mesh_detector = FaceMeshDetector()

# Open the video capture
cap = cv2.VideoCapture(0)  # Use the appropriate camera index

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run face detection on the frame
        detection_results = face_detector.detect_faces(frame)
        
        # Check if detection results are not empty
        if detection_results.boxes:
            # Draw the detection results
            image_with_bbox = draw_detections(frame, detection_results.boxes, detection_results.scores, detection_results.class_ids)
        else:
            image_with_bbox = frame

        # Run face mesh detection on the frame
        landmarks = face_mesh_detector.get_landmarks(frame)
        draw_landmarks(image_with_bbox, landmarks)

        # Display the frame with detections and landmarks
        cv2.imshow("YOLOv8 Detection with Landmarks", cv2.flip(image_with_bbox, 1))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

