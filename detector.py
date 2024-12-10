import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
import cv2
import numpy as np
#from mtcnn.mtcnn import MTCNN
import cv2

class FaceDetectorBase:
    def __init__(self, node: Node, topic_name: str):
        self.node = node
        self.publisher = self.node.create_publisher(Int8, 'face_detected', 10)
        self.cv2_window_name = "Camera Feed with Face Detection"

    def detect_faces(self, img: np.ndarray) -> list:
        raise NotImplementedError("Subclasses must implement this method")

    def process_image(self, msg: Image):
        print('processing')
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        faces = (self.detect_faces(img))
        print(len(faces))
        if (type(faces)!= type(None)) and ( len(faces)>0):
            x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(self.cv2_window_name, img)
            cv2.waitKey(1)
            self.publisher.publish(Int8(data=1))
        else:
            self.publisher.publish(Int8(data=0))


class MTCNNDetector(FaceDetectorBase):
    def __init__(self, node: Node, topic_name: str):
        super().__init__(node, topic_name)
        self.mtcnn = MTCNN()

    def detect_faces(self, img: np.ndarray) -> list:
        faces = self.mtcnn.detect_faces(img)
        bounding_boxes = []
        print(faces)
        if type(faces)!= type(None):
            for face in faces:
                x, y, w, h = face['box']
                bounding_boxes.append((x, y, w, h))
        return bounding_boxes

class HaarCascadeDetector(FaceDetectorBase):
    def __init__(self, node: Node, topic_name: str):
        super().__init__(node, topic_name)
        self.cascade = cv2.CascadeClassifier('faces.xml')

    def detect_faces(self, img: np.ndarray) -> list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))
        return faces


class ImageSubscriber(Node):
    def __init__(self, detector: FaceDetectorBase):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10)
        self.detector = detector

    def listener_callback(self, msg):
        self.detector.process_image(msg)

def main(args=None):
    rclpy.init(args=None)
    node = rclpy.create_node('face_detection_node') # Create the node here
    print('Detecting Begains...')	
    # Choose the detector
    detector_type = "haar"  # Change to "haar" to use Haar Cascade
    if detector_type == "mtcnn":
        detector = MTCNNDetector(node=node, topic_name='image_raw')
    elif detector_type == "haar":
        detector = HaarCascadeDetector(node=node, topic_name='image_raw')
        print('Detecting Ends...')
    else:
        print("Invalid detector type. Choose 'mtcnn' or 'haar'.")
        return
    print('Entering Subscriber...')
    image_subscriber = ImageSubscriber(detector)

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cv2.destroyAllWindows()
        image_subscriber.destroy_node()
        node.destroy_node() #destroy the main node as well
        rclpy.shutdown()

if __name__ == '__main__':
    main()
