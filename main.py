import cv2

from stream.webcam import Webcam, get_face
from inference import infer


def main():
    stream = Webcam()
    success, frame = stream.read()
    while True:
        success, frame = stream.read()

        detections = get_face(frame)

        for detection in detections:
            # Extract the region of interest (ROI) from the frame
            x, y, w, h = detection.x, detection.y, detection.width, detection.height
            roi = frame[y : y + h, x : x + w]

            # Perform inference on the ROI
            results = infer(roi)

            # Draw the rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Initialize text position
            text_x = x + w + 10
            text_y = y + 20
            line_height = 15

            for result in results:
                text = result.get_output()

                # Draw the text on the frame
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Move to the next line for the next text
                text_y += line_height

        # Display the resulting frame
        cv2.imshow("Face Detection", frame)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stream.release()
            break


if __name__ == "__main__":
    main()
