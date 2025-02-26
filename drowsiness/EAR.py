import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import os

# from .mainMediapipe_samples import FaceMeshDetector
from Model.MediapipeDetection.MediapipeFaceLandmarker import FaceMeshDetector
class DrowsinessDetector:
    def __init__(self):
        # self.cap = camera if camera is not None else cv2.VideoCapture(0)

        # if not self.cap.isOpened():
        #     raise RuntimeError("Error: Could not open video capture. Ensure a camera is available.")

        # self.face_mesh_detector = FaceMeshDetector()
        # self.face_mesh_detector.cap = self.cap  # Use the camera instance

        self.count = 0
        self.alarm_threshold = 30  # frames
        self.alarm_playing = False  # flag to check if alarm has been played

        os.environ["SDL_AUDIODRIVER"] = "dummy"
        pygame.mixer.init()

        alarm_path = os.path.join(os.path.dirname(__file__), "alarm2.mp3")
        self.alarm_sound = pygame.mixer.Sound(alarm_path)

        self.alarm_thread = threading.Thread(target=self.alarm_loop)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()


    def calc_ear(self, eye):
        A = np.linalg.norm(np.array(eye[2]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[3]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[1]) - np.array(eye[0]))
        ear = (A + B) / (2.0 * C)
        
        return ear

    def alarm_loop(self):
        while True:
            if self.alarm_playing:
                if not pygame.mixer.get_busy():
                    self.alarm_sound.play(loops=-1)
            else:
                self.alarm_sound.stop()

    def process_frame(self, image, eye_keypoints):
        if "left_eye" not in eye_keypoints or "right_eye" not in eye_keypoints:
            print(" Error: Eye keypoints missing!")
            return image  # Return original image

        left_eye = eye_keypoints["left_eye"]
        right_eye = eye_keypoints["right_eye"]
        

        if not left_eye or not right_eye:
            print(" No eye landmarks detected, skipping drowsiness detection.")
            return image

        left_ear = self.calc_ear(left_eye)
        right_ear = self.calc_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        

        print(f"📉 EAR: {avg_ear}")

        for (x, y) in left_eye + right_eye:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        if avg_ear < 0.25:
            self.count += 1
        else:
            self.count = 0
            self.alarm_playing = False

        if self.count >= self.alarm_threshold:
            print("Drowsiness Detected! Playing Alarm!")
            cv2.putText(image, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self.alarm_playing = True

        return image


    def run(self, image):
        # while self.cap.isOpened():
        #     success, image = self.cap.read()
        #     if not success:
        #         break

        # image = cv2.flip(image, 1)
        processed_image = self.process_frame(image)
        #     cv2.imshow("Drowsiness Detection", processed_image)

        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # self.cap.release()
        # cv2.destroyAllWindows()
        # pygame.mixer.quit()

if __name__ == "__main__":
    # Create a single camera instance
    camera = cv2.VideoCapture(0)
    detector = DrowsinessDetector(camera)
    detector.run()