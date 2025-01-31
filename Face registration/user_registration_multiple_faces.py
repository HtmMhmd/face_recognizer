import cv2
import os
import time

class FaceRegistration:
    def __init__(self, user_name, save_path='/home/mona/Face recognition/database', num_images=5):
        self.user_name = user_name
        self.save_path = save_path
        self.num_images = num_images
        self.user_folder = os.path.join(self.save_path, str(self.user_name))
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)
        self.instructions = [
            "Look straight at the camera",
            "Turn your head to the left",
            "Turn your head to the right",
            "Look up",
            "Look down"
        ]
        os.makedirs(self.user_folder, exist_ok=True)
        self.count = 0

    def calculate_distance(self, x, y, w, h, frame):
        face_center = (x + w / 2, y + h / 2)
        frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)
        return ((face_center[0] - frame_center[0]) ** 2 + (face_center[1] - frame_center[1]) ** 2) ** 0.5

    def capture_face(self):
        while self.count < self.num_images:
            ret, frame = self.cap.read()
            if not ret:
                print("Unable to capture video")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            closest_face = None
            min_distance = float('inf')  # Initialize to a large number to ensure the first face is considered

            for (x, y, w, h) in faces:
                distance = self.calculate_distance(x, y, w, h, frame)
                if distance < min_distance:
                    min_distance = distance
                    closest_face = (x, y, w, h)

            if closest_face:
                (x, y, w, h) = closest_face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face = frame[y:y + h, x:x + w]

            if self.count < len(self.instructions):
                text = self.instructions[self.count]
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Adjust your position", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Registration", frame)

            self._process_position(face, x, y, w, h, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Face registration complete!")

    def _process_position(self, face, x, y, w, h, frame):
        if self.count == 0:  # straight
            if abs(x + w / 2 - frame.shape[1] / 2) < 40 and abs(y + h / 2 - frame.shape[0] / 2) < 40:
                self._save_image(face)
        elif self.count == 1:  # left
            if x + w / 2 < frame.shape[1] / 2 - 40:
                self._save_image(face)
        elif self.count == 2:  # right
            if x + w / 2 > frame.shape[1] / 2 + 40:
                self._save_image(face)
        elif self.count == 3:  # up
            if y + h / 2 < frame.shape[0] / 2 - 30:
                self._save_image(face)
        elif self.count == 4:  # down
            if y + h / 2 > frame.shape[0] / 2 + 70:
                self._save_image(face)

    def _save_image(self, face):
        image_path = os.path.join(self.user_folder, f"Position_{self.count}.jpg")
        cv2.imwrite(image_path, face)
        print(f"Image {self.count + 1}/{self.num_images} captured")
        self.count += 1
        time.sleep(1)

# Example Usage
if __name__ == "__main__":
    user_name = input("Enter your name: ")
    registration = FaceRegistration(user_name)
    registration.capture_face()
