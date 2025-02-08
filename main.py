import cv2
import argparse

from CameraUtilis.CameraHandler import CameraHandler  # Import CameraHandler
from ImageProcessor import *
from Model.Detection.detection_utilis import draw_detections
from Model.Landmark.utilis import draw_landmarks

# Function to resize the frame to a specified width while maintaining the aspect ratio
def resize_frame(frame, width=360):
    aspect_ratio = frame.shape[1] / frame.shape[0]
    height = int(width / aspect_ratio)
    return cv2.resize(frame, (width, height))

# Function to process camera feed using cv2.VideoCapture
def process_camera_feed(image_processor):
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
            frame = resize_frame(frame)

            embeddings = image_processor.process_image(frame)
            if len(embeddings) == 0:
                print("No faces detected")
                continue
            for item in embeddings:
                bbox = item['bbox']
                ff = frame.copy()
                # cv2.rectangle(ff, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                print("Bounding Box:", bbox)
                print("Embedding:", item['embedding'][0:3])
                print("-------------------------------------")

            # Run landmark detection on the frame
            landmarks = image_processor.detect_landmarks(ff)
            draw_landmarks(ff, landmarks)

            # Draw detections on the frame
            ff = draw_detections(ff, [item['bbox'] for item in embeddings], [1.0] * len(embeddings), [0] * len(embeddings))

            cv2.imshow("YOLOv8 Detection with Landmarks", ff)
            ff = None

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Function to process camera feed using CameraHandler
def process_camera_handler(image_processor):
    camera = CameraHandler(0)  # Initialize CameraHandler
    try:
        while True:
            timestamp, frame = camera.read()
            if frame is not None:
                # Resize the frame
                frame = resize_frame(frame)

                embeddings = image_processor.process_image(frame)
                if len(embeddings) == 0:
                    print("No faces detected")
                    continue
                
                for item in embeddings:
                    bbox = item['bbox']
                    ff = frame.copy()
                    # cv2.rectangle(ff, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    
                    print("Bounding Box:", bbox)
                    print("Embedding:", item['embedding'][0:3])
                    print("-------------------------------------")
            
                # Run landmark detection on the frame
                landmarks = image_processor.detect_landmarks(ff)
                draw_landmarks(ff, landmarks)

                # Draw detections on the frame
                ff = draw_detections(ff, [item['bbox'] for item in embeddings], [1.0] * len(embeddings), [0] * len(embeddings))

                cv2.imshow("YOLOv8 Detection with Landmarks", ff)
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

    # Resize the image
    image = resize_frame(image)

    embeddings = image_processor.process_image(image)
    if len(embeddings) == 0:
        print("No faces detected")
    else:
        for item in embeddings:
            bbox = item['bbox']
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        # Run landmark detection on the image
        landmarks = image_processor.detect_landmarks(image)
        image_with_landmarks = draw_landmarks(image, landmarks)

        # Draw detections on the image
        image_with_detections = draw_detections(image_with_landmarks, [item['bbox'] for item in embeddings], [1.0] * len(embeddings), [0] * len(embeddings))

        cv2.imwrite(output_path, image_with_detections)
        print(f"Processed image saved to {output_path}")

# Main function
def main(run_on_camera=True, use_camera_handler=False, image_path=None, output_path=None):
    image_processor = ImageProcessor(use_yolo=True, verbose=True)

    if run_on_camera:
        if use_camera_handler:
            process_camera_handler(image_processor)
        else:
            process_camera_feed(image_processor)
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

    args = parser.parse_args()
    main(args.run_on_camera, args.use_camera_handler, args.image_path, args.output_path)