import cv2
import numpy as np
from CameraUtilis.CameraHandler import CameraHandler  # Import CameraHandler

class HaarCascadeDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cv2_window_name = "Camera Feed with Haar Cascade Detection"  # Window Name
        cv2.namedWindow(self.cv2_window_name)  # Create the window
        self.camera_handler = CameraHandler(0)  # Initialize CameraHandler

    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))
        return faces

    def process_frame(self):
        timestamp, img = self.camera_handler.read()  # Capture frame using CameraHandler
        if img is not None:
            # Display the image before Haar Cascade processing
            cv2.imshow(self.cv2_window_name, img)
            cv2.waitKey(1)  # Update the display
            faces = self.detect_faces(img)
            if faces is not None:
                # Draw bounding boxes around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Show Image with Bounding Boxes after Haar Cascade
                cv2.imshow(self.cv2_window_name, img)
                cv2.waitKey(1)

def main():
    detector_type = "haar"  # Change to "mtcnn" to use MTCNN
    if detector_type == "mtcnn":
        pass
    elif detector_type == "haar":
        detector = HaarCascadeDetector()
    else:
        print("Invalid detector type. Choose 'mtcnn' or 'haar'.")
        return

    try:
        while True:
            detector.process_frame()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cv2.destroyAllWindows()  # Clean up OpenCV windows when exiting
        detector.camera_handler.release()

if __name__ == '__main__':
    main()