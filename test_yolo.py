import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("model.pt")  # Ensure you have the correct model file

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
        results = model(frame)
        print(results)
        # Check if results is not empty
        if len(results)>0:
            # Render the results on the frame
            annotated_frame = results[0]
            print(annotated_frame.boxes.data)
            print(annotated_frame.keypoints)
            for obj in results:
                x1, y1, x2, y2 ,conf ,cls  = annotated_frame.boxes.data[0].tolist()
                cv2.rectangle(annotated_frame[0].orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(annotated_frame[0].orig_img, f"{model.names[int(cls)]} {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("YOLOv8 Detection", annotated_frame[0].orig_img)
        else:
            cv2.imshow("YOLOv8 Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

