import time
import cv2
import numpy as np
import onnxruntime

from Model.Detection.YoloV8OnnxRuntime.utils import xywh2xyxy, nms
from Model.Detection.detection_utilis import draw_detections
from Model.DetectionResult import DetectionResult

class YOLOv8:

    """
    YOLOv8 object detector class.
    """

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, verbose=False):
        """
        Initialize object detector.

        Args:
            path (str): Path to the model.
            conf_thres (float): Confidence threshold for object detection.
            iou_thres (float): IoU threshold for non-maxima suppression.
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.verbose = verbose

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        """
        Detect objects in the given image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            List of bounding boxes, scores and class ids.
        """
        return self.detect_objects(image)

    def initialize_model(self, path):
        """
        Initialize the model.

        Args:
            path (str): Path to the model.
        """
        self.session = onnxruntime.InferenceSession(path,)
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        """
        Detect objects in the given image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            List of bounding boxes, scores and class ids.
        """
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        
        self.results = DetectionResult()

        self.results.boxes = self.boxes
        self.results.scores = self.scores
        self.results.class_ids = self.class_ids
        
        return self.results

    def prepare_input(self, image):
        """
        Prepare the input image for inference.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            Input tensor.
        """
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        """
        Perform inference on the given input tensor.

        Args:
            input_tensor (numpy.ndarray): Input tensor.

        Returns:
            Output of the model.
        """
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        if self.verbose:
            print(f"YOLO Model Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        """
        Process the output of the model.

        Args:
            output (list): Output of the model.

        Returns:
            List of bounding boxes, scores and class ids.
        """
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        """
        Extract bounding boxes from predictions.

        Args:
            predictions (numpy.ndarray): Predictions from the model.

        Returns:
            Bounding boxes.
        """
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        """
        Get input details of the model.
        """
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        """
        Get output details of the model.
        """
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
