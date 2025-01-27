import cv2
# from ultralytics import YOLO
from Model.Detection.YoloV8OnnxRuntime.Yolov8OnnxRuntimeDetector import Yolov8OnnxRuntimeDetector
from Model.Detection.detection_utilis import draw_detections
# from Model.Detection.YoloDetector.YoloDetector import Yolov8Detector

# Load the YOLOv8 model
# model = YOLO("Model/Detection/YoloDetector/yolov8n-face(2).pt")  # Ensure you have the correct model file

model = Yolov8OnnxRuntimeDetector()  # Ensure you have the correct model file

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
        
        # Run YOLOv8 model on the frame
        results = model.detect_faces(frame)
        # Check if results is not empty
        if len(results)>0:
            # Render the results on the frame
            # annotated_frame = results[0]
            # print(annotated_frame.boxes)
            # print(annotated_frame.keypoints)
            for obj in results:
                image = draw_detections(frame, obj.boxes, obj.scores, obj.class_ids)
            cv2.imshow("YOLOv8 Detection", image)
        else:
            cv2.imshow("YOLOv8 Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

