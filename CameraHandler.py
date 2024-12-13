import cv2
import numpy as np
import time
from threading import Thread

####-----------------------HYPERPARAMETERS----------------------------------------------------------
RASPI_IP = '192.168.4.13'
####------------------------------------------------------------------------------------------------

class VideoSaver:
    """
    A class to handle saving video frames to a file.
    Attributes:
    -----------
    out : cv2.VideoWriter
        The VideoWriter object used to write frames to the video file.
    Methods:
    --------
    __init__(video_name, frames=20, res=(640, 480), fourcc=[*"XVID"]):
        Initializes the VideoSaver with the specified parameters.
    save_frame(frame):
        Writes a single frame to the video file.
    release():
        Releases the VideoWriter object.
    """
    def __init__(self, video_name, frames=20, res=(640, 480), fourcc=[*"XVID"]):
        fcc = cv2.VideoWriter_fourcc(*fourcc)
        self.out = cv2.VideoWriter(video_name, fcc, frames, res)

    def save_frame(self, frame):
        self.out.write(frame)
    
    def release(self):
        try:
            self.out.release()
        except:
            pass


class CameraHandler:
    """
    A class to handle camera streaming and frame capturing.

    Attributes:
        camera_num (int): The camera number to be used.
        resolution (tuple): The resolution of the captured frames (width, height).
        stream (bool): A flag to control the streaming state.
        frame (numpy.ndarray): The latest captured frame.
        timestamp (float): The timestamp of the latest captured frame.
        cap (cv2.VideoCapture): The video capture object for the camera.
        thread (Thread): The thread object for updating frames.

    Methods:
        update_frame():
            Continuously captures frames from the camera and updates the frame and timestamp attributes.
        
        read():
            Returns the latest timestamp and frame.
        
        release():
            Stops the streaming, joins the thread, and releases the camera resource.
    """
    def __init__(self, camera_num, resolution=(640, 480)):
        self.camera_num = camera_num
        self.resolution = resolution
        self.stream = True
        self.frame = None
        self.timestamp = None

        self.cap = cv2.VideoCapture(f"http://{RASPI_IP}:8081/?action=stream_{camera_num}")

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_num}")

        self.thread = Thread(target=self.update_frame)
        self.thread.start()

    def update_frame(self):
        while self.stream:
            success, frame = self.cap.read()
            if success:
                self.frame = cv2.resize(frame, self.resolution)
                self.timestamp = time.time()

    def read(self):
        return self.timestamp, self.frame

    def release(self):
        self.stream = False
        self.thread.join()
        self.cap.release()
        print(f"Camera {self.camera_num} released")


if __name__ == "__main__":
    camera = CameraHandler(0)
    while True:
        timestamp, frame = camera.read()
        if frame is not None:
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()
    