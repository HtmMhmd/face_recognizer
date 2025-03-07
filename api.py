import cv2
from flask import Flask, Response, render_template, jsonify
import threading
import logging

from ImageProcessor import ImageProcessor
from drowsiness.EAR import DrowsinessDetector  # Import DrowsinessDetector

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
verification_results = []

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def flask_stream():
    global output_frame
    # global camera_thread
    # camera_thread.start()
    while True:
        with lock:
            if output_frame is None:
                logging.debug("Output frame is None")
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                logging.error("Failed to encode image")
                continue
        logging.debug("Streaming frame")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route("/")
def index():
    global camera_thread
    camera_thread.start()
    logging.debug("Index page requested")
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    logging.debug("Video feed requested")
    return Response(flask_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/verify_results")
def verify_results():
    global verification_results
    # global camera_thread
    # camera_thread.start()
    with lock:
        results = verification_results.copy()
    logging.debug(f"Verification results: {results}")
    return jsonify(results)

def process_camera_feed(image_processor, drowsiness_detector=None):
    global output_frame, verification_results
    logging.debug("Starting camera feed processing")
    cap = cv2.VideoCapture(0)  # Use the appropriate camera index
    if not cap.isOpened():
        logging.error("Error: Could not open video capture")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break

            # Resize the frame
            frame = cv2.resize(frame, (480, 360))

            logging.debug("Processing frame")

            image_with_landmarks = frame.copy()
            embeddings = image_processor.process_image(frame)
            if embeddings is None or len(embeddings.embeddings) == 0:
                logging.info("No faces detected")
            else:
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
                logging.debug("Updated output frame")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.debug("Starting Flask app")
    image_processor = ImageProcessor(model_architecture='mediapipe', verbose=True)
    drowsiness_detector = DrowsinessDetector()  # Initialize DrowsinessDetector

    global camera_thread
    # Start the camera feed processing in a separate thread
    camera_thread = threading.Thread(target=process_camera_feed, args=(image_processor, drowsiness_detector))
    # camera_thread.start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=9000, debug=False, use_reloader=True)
