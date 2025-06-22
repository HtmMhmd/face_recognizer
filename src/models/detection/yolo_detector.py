import cv2
import numpy as np
import time
import onnxruntime as ort
from src.models.detection.base import FaceDetector
from src.models.detection.detection_faces import DetectionFaces
from src.models.detection.utils import draw_detections, get_cropped_faces

class Yolov8OnnxRuntimeDetector(FaceDetector):
    def __init__(self, model_path="models/yolov8n_face.onnx", conf_threshold=0.5, iou_threshold=0.45, 
                 input_shape=(640, 640), verbose=False, detection_faces=None):
        """
        Initializes the Yolov8OnnxRuntimeDetector with the specified model path.

        Args:
            model_path (str): The path to the ONNX model file. Defaults to 'models/yolov8n_face.onnx'.
            conf_threshold (float): Confidence threshold for detections. Defaults to 0.5.
            iou_threshold (float): IoU threshold for NMS. Defaults to 0.45.
            input_shape (tuple): Input shape for the model. Defaults to (640, 640).
            verbose (bool): Enables verbose output for debugging. Defaults to False.
            detection_faces (DetectionFaces): Optional DetectionFaces object to store detection results.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        self.detection_faces = detection_faces if detection_faces is not None else DetectionFaces()
        
        # Initialize the ONNX Runtime session
        try:
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = self.session.get_inputs()[0].shape[2:4]  # Height, width
            
            if self.verbose:
                print(f"YOLO model loaded from {model_path}")
                print(f"Input shape: {self.input_shape}")
        except Exception as e:
            print(f"Failed to load ONNX model: {str(e)}")
            print("Creating placeholder session")
            # Create a placeholder session for testing/development
            self.session = None
            self.input_name = "input"
            self.output_names = ["output"]
            self.input_shape = (640, 640)

    def detect_faces(self, image):
        """
        Detects faces in the input image using YOLOv8.

        Args:
            image (np.ndarray): The input image.

        Returns:
            DetectionFaces: The detection results containing bounding boxes, scores, class IDs, and cropped faces.
        """
        self.detection_faces.reset()  # Reset the detection faces object
        
        if self.session is None:
            # Return empty results if session failed to initialize
            return self.detection_faces
            
        start_time = time.time()
        
        # Preprocess image
        input_image = self._preprocess(image)
        
        # Run inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_image})
            boxes = self._process_output(outputs, image.shape)
            
            # Extract faces and add to detection_faces
            for box, score, class_id in boxes:
                cropped_face = get_cropped_faces(image, [box])
                self.detection_faces.add(box, score, class_id, cropped_face)
                
            inference_time = time.time() - start_time
            
            if self.verbose:
                print(f"YOLOv8 Inference Time: {inference_time * 1000:.2f} ms")
                print(f"Detected {len(self.detection_faces)} faces")
        except Exception as e:
            if self.verbose:
                print(f"Error running YOLO inference: {str(e)}")
        
        return self.detection_faces

    def _preprocess(self, image):
        """
        Preprocesses the input image for YOLO.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The preprocessed image.
        """
        # Resize
        input_img = cv2.resize(image, self.input_shape)
        
        # Convert to RGB (from BGR)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to float32
        input_img = input_img.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_img = np.expand_dims(input_img, axis=0)
        
        return input_img

    def _process_output(self, outputs, image_shape):
        """
        Processes YOLO output to extract bounding boxes.

        Args:
            outputs: The output from YOLO model.
            image_shape: Original image shape.

        Returns:
            list: A list of [box, score, class_id] for each detection.
        """
        # This is a simplified implementation
        # For a real implementation, you would need to parse the outputs
        # according to your specific YOLO model's output format
        
        results = []
        
        # If we have a proper output
        if len(outputs) > 0 and outputs[0].size > 0:
            # Get the first output (typically detection output)
            detections = outputs[0]
            
            # Process each detection
            # Note: Format may vary based on your specific model
            for detection in detections:
                # Typically, YOLO outputs are in format [x, y, width, height, confidence, class_id]
                if len(detection) >= 6:
                    confidence = detection[4]
                    class_id = int(detection[5])
                    
                    # Filter by confidence
                    if confidence >= self.conf_threshold:
            # Convert to pixel coordinates
                        orig_h, orig_w = image_shape[:2]
                        
                        # Calculate bbox from center, width, height to xmin, ymin, xmax, ymax
                        x_center, y_center, width, height = detection[:4]
                        
                        x1 = int((x_center - width/2) * orig_w)
                        y1 = int((y_center - height/2) * orig_h)
                        x2 = int((x_center + width/2) * orig_w)
                        y2 = int((y_center + height/2) * orig_h)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(orig_w, x2)
                        y2 = min(orig_h, y2)
                        
                        if confidence >= self.conf_threshold:
                            box = [x1, y1, x2, y2]
                            results.append([box, confidence, class_id])
        
        return results

    def draw_detections(self, image):
        """
        Draws bounding boxes on the detected faces in the image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        return draw_detections(image, self.detection_faces.boxes, self.detection_faces.scores, self.detection_faces.class_ids)


class Yolov8Detector(FaceDetector):
    """
    Implementation of YOLOv8 detector using the ultralytics package.
    This is a placeholder class that requires the ultralytics package.
    """
    
    def __init__(self, model_path="models/yolov8n-face.pt", verbose=False, detection_faces=None):
        self.model_path = model_path
        self.verbose = verbose
        self.detection_faces = detection_faces if detection_faces is not None else DetectionFaces()
        
        try:
            # Only import if the class is actually used
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            if self.verbose:
                print(f"Loaded YOLOv8 model from {model_path}")
        except ImportError:
            print("Error: ultralytics package not found. Please install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            self.model = None

    def detect_faces(self, image):
        """
        Detects faces in the input image using YOLOv8.
        
        Args:
            image (np.ndarray): The input image.
            
        Returns:
            DetectionFaces: The detection results.
        """
        self.detection_faces.reset()
        
        if self.model is None:
            return self.detection_faces
            
        try:
            start_time = time.time()
            # Run inference
            results = self.model(image)
            
            # Process results
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for box, score, class_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    box_list = [x1, y1, x2, y2]
                    cropped_face = get_cropped_faces(image, [box_list])
                    self.detection_faces.add(box_list, float(score), int(class_id), cropped_face)
            
            inference_time = time.time() - start_time
            
            if self.verbose:
                print(f"YOLOv8 Inference Time: {inference_time * 1000:.2f} ms")
                print(f"Detected {len(self.detection_faces)} faces")
                
        except Exception as e:
            if self.verbose:
                print(f"Error in YOLOv8 detection: {str(e)}")
                
        return self.detection_faces

    def draw_detections(self, image):
        """
        Draws bounding boxes on the detected faces in the image.
        
        Args:
            image (np.ndarray): The input image.
            
        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        return draw_detections(image, self.detection_faces.boxes, self.detection_faces.scores, self.detection_faces.class_ids)