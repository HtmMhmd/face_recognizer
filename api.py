import cv2
from flask import Flask, Response, render_template, jsonify
import threading

from ImageProcessor import ImageProcessor
from drowsiness.EAR import DrowsinessDetector  # Import DrowsinessDetector

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
verification_results = []

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

@app.route("/verify_results")
def verify_results():
    global verification_results
    with lock:
        results = verification_results.copy()
    return jsonify(results)

def process_camera_feed(image_processor, drowsiness_detector=None):
    global output_frame, verification_results
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

            embeddings = image_processor.process_image(frame)
            if embeddings is None or len(embeddings.embeddings) == 0:
                print("No faces detected")
                image_with_landmarks = frame
            else:
                for item_index in range(len(embeddings.embeddings)):
                    bbox = embeddings.detection_faces.boxes[item_index]
                    ff = frame.copy()

                # Run face detection on the frame
                image_with_detections = image_processor.draw_detections(ff)

                # Run landmark detection on the frame
                landmarks = image_processor.detect_landmarks(ff)
                
                image_with_landmarks = image_processor.draw_landmarks(image_with_detections)

                verify_results = image_processor.verify_faces()
                
                image_username = image_processor.draw_user_names(image_with_landmarks, verify_results)

                eye_mouth = image_processor.get_eye_mouth_keypoints()

                # Process drowsiness detection if enabled
                if drowsiness_detector:
                    image_with_landmarks = drowsiness_detector.process_frame(image_username, eye_mouth)

                with lock:
                    verification_results = verify_results

            with lock:
                output_frame = image_with_landmarks.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_processor = ImageProcessor(model_architecture='mediapipe', verbose=False)
    drowsiness_detector = DrowsinessDetector()  # Initialize DrowsinessDetector
    threading.Thread(target=process_camera_feed, args=(image_processor, drowsiness_detector)).start()
    app.run(host='0.0.0.0', port=9000, debug=True, use_reloader=True)
