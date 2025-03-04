import cv2
import argparse
from flask import Flask, Response, render_template
import threading

from CameraUtilis.CameraHandler import CameraHandler  # Import CameraHandler
from ImageProcessor import ImageProcessor
from drowsiness.EAR import DrowsinessDetector  # Import DrowsinessDetector

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

def flask_stream():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(flask_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return render_template("index.html")

def play_alarm():
    import os
    os.system("paplay drowsiness/alarm2.mp3")

# Function to process camera feed using cv2.VideoCapture
def process_camera_feed(image_processor, drowsiness_detector=None, use_flask=False):
    global output_frame
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
            # frame = cv2.resize(frame, (480, 360))

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
                # print(landmarks)

                image_with_landmarks = image_processor.draw_landmarks(image_with_detections)

                verify_results = image_processor.verify_faces()
                
                image_username = image_processor.draw_user_names(image_with_landmarks, verify_results)

                eye_mouth = image_processor.get_eye_mouth_keypoints()

            # Process drowsiness detection if enabled
                if drowsiness_detector:
                    image_with_landmarks = drowsiness_detector.process_frame(image_username, eye_mouth)

            if use_flask:
                with lock:
                    output_frame = image_with_landmarks.copy()
            else:
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

                if image_with_detections is None:
                    print("Face detection failed: No faces detected!")
                else:
                    print("Face detection successful!")

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
def main(run_on_camera=True, use_camera_handler=False, image_path=None, output_path=None, detector_type='mediapipe', enable_drowsiness=False, use_flask=False):
    image_processor = ImageProcessor(model_architecture=detector_type, verbose=False)
    drowsiness_detector = DrowsinessDetector() if enable_drowsiness else None

    if run_on_camera:
        if use_camera_handler:
            process_camera_handler(image_processor, drowsiness_detector)
        else:
            process_camera_feed(image_processor, drowsiness_detector, use_flask)
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
    parser.add_argument('-uf', "--use_flask", action='store_true', default=False, help="Use Flask for image display instead of cv2.imshow")

    args = parser.parse_args()
    if args.use_flask:
        threading.Thread(target=main, args=(args.run_on_camera, args.use_camera_handler, args.image_path, args.output_path, args.detector_type, args.enable_drowsiness, args.use_flask)).start()
        app.run(host='0.0.0.0', port=9000, debug=True, use_reloader=False)
    else:
        main(args.run_on_camera, args.use_camera_handler, args.image_path, args.output_path, args.detector_type, args.enable_drowsiness, args.use_flask)