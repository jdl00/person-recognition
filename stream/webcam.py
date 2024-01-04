from dataclasses import dataclass
from threading import Thread

import cv2


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@dataclass
class Detection:
    x: float
    y: float
    width: float
    height: float


def get_face(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detections = []
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        detections.append(Detection(x, y, (x+w), (y+h)))

    return detections


class Webcam:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.status, self.frame = self.capture.read()
        self.is_running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.is_running:
            self.status, self.frame = self.capture.read()

    def read(self):
        return self.status, self.frame

    def release(self):
        self.is_running = False
        self.thread.join()
        self.capture.release()
