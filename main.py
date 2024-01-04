import cv2

from stream.webcam import Webcam, get_face


def main():
    stream = Webcam()
    success, frame = stream.read()
    while True:

        success, frame = stream.read()

        detections = get_face(frame)

        for detection in detections:
            cv2.rectangle(frame, (detection.x, detection.y),
                          (detection.width, detection.height), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stream.release()
            break


if __name__ == "__main__":
    main()
