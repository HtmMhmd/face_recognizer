import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.timer = self.create_timer(1/30, self.timer_callback) # 30fps
        self.cap = cv2.VideoCapture(0) # Adjust if needed

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            img_msg = Image()
            img_msg.header.frame_id = 'camera_frame'
            img_msg.height = frame.shape[0]
            img_msg.width = frame.shape[1]
            img_msg.encoding = "bgr8"
            img_msg.step = frame.strides[0]
            img_msg.data = frame.tobytes()
            self.publisher_.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
