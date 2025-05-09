import cv2
import numpy as np

class VehicleCamera:
    """
    A class representing a camera mounted on a vehicle.

    Attributes:
        camera_id (int): The unique identifier for the currently active camera.
        cap (cv2.VideoCapture): The OpenCV video capture object for the active camera.
        is_active (bool): Whether the camera is currently capturing frames.

    Methods:
        get_frame(self) -> bytes: Returns the most recent captured frame data.
        update_frame(self) -> bytes: Captures a new frame and returns it.
        change_camera(self, new_camera_id: int) -> None: Attempts to switch cameras (implementation specific).
        deactivate(self) -> None: Deactivates the camera, stopping it from capturing frames).
    """

    def __init__(self, camera_id: int):
        """
        Initializes a VehicleCamera object.

        Args:
            camera_id (int): The initial camera ID to capture from.
        """
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)  # Open video capture for initial camera
        self.is_active = self.cap.isOpened()  # Check if camera opened successfully

    def get_id(self):
        return self.camera_id

    def get_frame(self) -> bytes:
        """
        Returns the most recent captured frame data, or None if no frame is available.
        """
        _, frame = self.cap.read()  # Read a frame
        if frame is None:
            return None

        # Optionally convert frame to bytes for compatibility (if needed)
        ret, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
        return buffer.tobytes() if ret else None

    def update_frame(self) -> bytes:
        """
        Captures a new frame from the camera and returns it.

        Returns:
            bytes: The captured frame data (as JPEG by default), or None if an error occurs.
        """
        if not self.is_active:
            return None

        ret, frame = self.cap.read()  # Read a frame
        if frame is None:
            return None

        # Optionally convert frame to bytes for compatibility (if needed)
        ret, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
        return buffer.tobytes() if ret else None

    def change_camera(self, new_camera_id: int) -> None:
        """
        Attempts to switch cameras on the vehicle.

        This method assumes basic camera switching functionality exists for the vehicle.
        It releases the current camera, opens the new camera with the provided ID,
        updates the internal state, and raises an error if the new camera fails to open.
        """

        if self.camera_id == new_camera_id:
            print("Camera is already active on ID:", new_camera_id)
            return

        # Release the current camera
        self.cap.release()

        # Open the new camera
        self.cap = cv2.VideoCapture(new_camera_id)
        self.camera_id = new_camera_id

        if not self.cap.isOpened():
            raise ValueError("Unable to open new camera")

    def deactivate(self) -> None:
        """
        Deactivates the camera, stopping it from capturing frames.
        """
        if self.is_active:
            self.cap.release()  # Release the OpenCV video capture object
            self.is_active = False


def main():
    camera = VehicleCamera(0)
    MAX_CAMERAS = 8

    id = 0

    while True:
        frame = camera.update_frame()
        if frame is not None:
            # Display the frame using OpenCV (replace with your display logic if needed)
            cv2.imshow(f'Vehicle Camera{id}', cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            if(id < MAX_CAMERAS):
                #camera.deactivate()
                id+=1
                print(id)
                #camera.change_camera(id)
        if key == ord('b'):
            if(id > 0 and id <= MAX_CAMERAS):
                #camera.deactivate()q
                id-=1
                print(id)
                #camera.change_camera(id)
        if key == ord('q'):
            break

main()
