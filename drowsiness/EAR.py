import cv2
import numpy as np
import subprocess
import os
import threading
import time

class DrowsinessDetector:
    def __init__(self):
        self.count = 0
        self.alarm_threshold = 30  # frames before alarm triggers
        self.alarm_path = "/app/alarm2.mp3"

        # Check if alarm file exists
        if os.path.exists(self.alarm_path):
            print("Alarm file loaded successfully!")
        else:
            print(" Error: Alarm file not found!")

    def calc_ear(self, eye):
        A = np.linalg.norm(np.array(eye[2]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[3]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[1]) - np.array(eye[0]))
        ear = (A + B) / (2.0 * C)
        return ear

    def play_alarm(self):
        """Plays alarm sound inside Docker using PulseAudio"""
        try:
            subprocess.Popen(["paplay", self.alarm_path])
            print(" Alarm triggered! Playing sound...")
        except Exception as e:
            print(f" Error playing alarm: {e}")

    def process_frame(self, image, eye_keypoints):
        if "left_eye" not in eye_keypoints or "right_eye" not in eye_keypoints:
            print("Error: Eye keypoints missing!")
            return image  # Return original image

        left_eye = eye_keypoints["left_eye"]
        right_eye = eye_keypoints["right_eye"]

        if not left_eye or not right_eye:
            print("No eye landmarks detected, skipping drowsiness detection.")
            return image

        left_ear = self.calc_ear(left_eye)
        right_ear = self.calc_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        print(f"📉 EAR: {avg_ear}")

        # Draw eye landmarks on the image
        for (x, y) in left_eye + right_eye:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        if avg_ear < 0.25:
            self.count += 1
        else:
            self.count = 0  # Reset counter if eyes are open

        if self.count >= self.alarm_threshold:
            print("Drowsiness Detected! Playing Alarm!")
            cv2.putText(image, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self.play_alarm()  # Trigger alarm sound

        # if self.count >= self.alarm_threshold:
        #     if not self.alarm_playing:
        #         print("Drowsiness Detected! Playing Alarm!")
        #         threading.Thread(target=self.play_alarm, daemon=True).start()
        #         self.alarm_playing = True
        # else:
        #     if self.alarm_playing:  # Stop the alarm when eyes open
        #         self.alarm_playing = False

       

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