import cv2
import argparse

from CameraUtilis.CameraHandler import CameraHandler  # Import CameraHandler
from ImageProcessor import ImageProcessor
from drowsiness.EAR import DrowsinessDetector  # Import DrowsinessDetector

# Function to process camera feed using cv2.VideoCapture
def process_camera_feed(image_processor, drowsiness_detector=None):
    cap = cv2.VideoCapture(0)  # Use the appropriate camera index
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize the frame
            frame = cv2.resize(frame, (480, 360))

            embeddings = image_processor.process_image(frame)
            if len(embeddings) == 0:
                print("No faces detected")
                image_with_landmarks = frame
            else:
                for item_index in range(len(embeddings)):
                    bbox = embeddings['boxes'][item_index]
                    ff = frame.copy()

                # Run face detection on the frame
                image_with_detections = image_processor.draw_detections(ff)

                # Run landmark detection on the frame
                landmarks = image_processor.detect_landmarks(ff)
                image_with_landmarks = image_processor.draw_landmarks(image_with_detections)

                verify_results = image_processor.verify_faces()
                
                image_username = image_processor.draw_user_names(image_with_landmarks, verify_results)

            # Process drowsiness detection if enabled
            if drowsiness_detector:
                image_with_landmarks = drowsiness_detector.process_frame(image_with_landmarks)

            cv2.imshow("Detection with Landmarks", image_with_landmarks)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Function to process camera feed using CameraHandler
def process_camera_handler(image_processor, drowsiness_detector=None):
    camera = CameraHandler(0)  # Initialize CameraHandler
    try:
        while True:
            timestamp, frame = camera.read()
            if frame is not None:
                # Resize the frame
                frame = cv2.resize(frame, (480, 360))

                embeddings = image_processor.process_image(frame)
                if len(embeddings) == 0:
                    print("No faces detected")
                    continue

                for item_index in range(len(embeddings)):
                    bbox = embeddings['boxes'][item_index]
                    ff = frame.copy()

                # Run face detection on the frame
                image_with_detections = image_processor.draw_detections(ff)

                # Run landmark detection on the frame
                landmarks = image_processor.detect_landmarks(ff)
                image_withlandmarks = image_processor.draw_landmarks(image_with_detections)

                # Process drowsiness detection if enabled
                if drowsiness_detector:
                    image_with_landmarks = drowsiness_detector.process_frame(image_with_landmarks)

                cv2.imshow("Detection with Landmarks", image_with_landmarks)
                ff = None
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

# Function to process image and save the result
def process_image_and_save(image_processor, image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    embeddings = image_processor.process_image(image)
    if len(embeddings) == 0:
        print("No faces detected")
    else:
        for item in embeddings:
            bbox = item['bbox']
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        # Run face detection on the image
        detection_faces = image_processor.detect(image)
        # Run face detection on the frame
        image_with_detections = image_processor.draw_detections(image)

        # Run landmark detection on the frame
        landmarks = image_processor.detect_landmarks(image)
        image_with_landmarks = image_processor.draw_landmarks(image_with_detections)

        cv2.imwrite(output_path, image_with_landmarks)
        print(f"Processed image saved to {output_path}")

# Main function
def main(run_on_camera=True, use_camera_handler=False, image_path=None, output_path=None, detector_type='mediapipe', enable_drowsiness=False):
    image_processor = ImageProcessor(model_architecture=detector_type, verbose=True)
    drowsiness_detector = DrowsinessDetector() if enable_drowsiness else None

    if run_on_camera:
        if use_camera_handler:
            process_camera_handler(image_processor, drowsiness_detector)
        else:
            process_camera_feed(image_processor, drowsiness_detector)
    else:
        if image_path is None or output_path is None:
            raise ValueError("Image path and output path must be provided when run_on_camera is False")
        process_image_and_save(image_processor, image_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition using YOLOv8 and FaceNet")
    parser.add_argument("-roc","--run_on_camera",action='store_true',default=False, help="Set to True to run on camera feed, False to run on an image")
    parser.add_argument("-ch","--use_camera_handler", action='store_true', default=False, help="Set to True to use CameraHandler, False to use cv2.VideoCapture")
    parser.add_argument('-ip',"--image_path", type=str, default=None, help="Provide the path to your test image")
    parser.add_argument('-op',"--output_path", type=str, default=None, help="Provide the path to save the processed image")
    parser.add_argument('-dt',"--detector_type", type=str, default='mediapipe', help="Type of detector to use ('yolov8_onnx', 'yolov8', 'mediapipe')")
    parser.add_argument('-ed', "--enable_drowsiness", action='store_true', default=False, help="Enable drowsiness detection")

    args = parser.parse_args()
    main(args.run_on_camera, args.use_camera_handler, args.image_path, args.output_path, args.detector_type, args.enable_drowsiness)