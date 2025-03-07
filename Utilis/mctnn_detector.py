import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

class MTCNNDetector(Node):
    def __init__(self):
        super().__init__('mtcnn_detector')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Int8, 'face_detected', 10)
        self.mtcnn = MTCNN()
        self.cv2_window_name = "Camera Feed with MTCNN Detection"  #Window Name
        cv2.namedWindow(self.cv2_window_name) # Create the window

    def listener_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        
        # Display the image before MTCNN processing
        cv2.imshow(self.cv2_window_name, img)
        cv2.waitKey(1) # Update the display

        faces = self.mtcnn.detect_faces(img)
        if faces:
            # Draw bounding boxes around detected faces
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #Show Image with Bounding Boxes after MTCNN
            cv2.imshow(self.cv2_window_name, img)
            cv2.waitKey(1)
            msg = Int8()
            msg.data = 1 # Face detected
            self.publisher_.publish(msg)
        else:
            msg = Int8()
            msg.data = 0 # No face detected
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    mtcnn_detector = MTCNNDetector()
    try:
        rclpy.spin(mtcnn_detector)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cv2.destroyAllWindows() #Clean up OpenCV windows when exiting
        mtcnn_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

