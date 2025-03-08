from vision_library.gesture_recognizer import MediaPipeGestureRecognizer
import mediapipe as mp
import cv2
import time


def main():
    # Instantiating the Gesture Recognizer Model
    gesture_recognizer_construct = MediaPipeGestureRecognizer()

    # Building the Gesture Recognizer from Options
    gesture_recognizer_options = gesture_recognizer_construct.build_from_options()

    # OpenCV Setup
    capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Infinite Loop for Gesture Recognition
    with gesture_recognizer_construct.GestureRecognizer.create_from_options(gesture_recognizer_options) as recognizer:
        while True:
            ret, frame = capture.read()

            if not ret:
                print("Error accessing LiveStream resource")
                break

            # Converting the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Converting to the MediaPipe image format
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Generating realtime timestamps
            timestamp_ms = int(time.time() * 1000)

            # Recognizing the gestures
            recognition_result = recognizer.recognize_for_video(
                mp_frame, timestamp_ms=timestamp_ms
            )

            # Displaying the results
            if recognition_result.gestures:
                for gesture in recognition_result.gestures[0]:
                    text = f"{gesture.category_name}: {gesture.score:.2f}"
                    frame = cv2.putText(frame, text, (10, 50), font, 1, (255, 0, 0), 2)

            cv2.imshow("Gesture Recognizer", frame)

            # Exit Condition
            if cv2.waitKey(1) == ord("q"):
                print("The livestream is ending ....")
                break

    # Releasing the resources
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
