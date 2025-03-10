from vision_library.gesture_recognizer import MediaPipeGestureRecognizer
from vision_library.hand_landmarker import MediaPipeHandLandmarker
from vision_library.camera_utils import CVLiveStream
from vision_library.mediapipe_utils import MediaPipeUtils

import cv2
import time


TITLE = "Automatic Gesture Recognition and Realtime Learning System"


class GestureRecognizerFSM:
    """Class behaves as the controller for the Gesture Recognition system.

    It will automatically switch states between Gesture Recognition and Hand Landmark tracking to
    learn new gestures based on the state variable which is dicatated by the no of frames where 
    no gestures were picked up by the gesture recognizer."""

    def __init__(self) -> None:
        self.gesture_recognizer = MediaPipeGestureRecognizer()
        self.recognizer = self.gesture_recognizer.build_recognizer()
        self.hand_landmarker = MediaPipeHandLandmarker()
        self.detector = self.hand_landmarker.build_detector()
        self.mediapipe_utilities = MediaPipeUtils()
        self.live_stream = CVLiveStream()

        # State Variable to manage the transitions
        self.state = "recognition"

        # Threashold and Counters to manage the State Variable
        self.no_gesture_counter = 0
        self.landmark_track_counter = 0
        self.threashold_recognizer = 5
        self.threashold_detector = 10

    def recognition_state(self) -> None:
        """Describes the control flow and pipeline for the recognition state."""

        ret, frame = self.live_stream.begin_live_stream()

        # Data Pipeline
        mp_frame = self.mediapipe_utilities.process_frame(
            self.live_stream.convert_to_rgb, frame
        )

        # Model inferencing
        recognition_result = self.recognizer.recognize_for_video(
            mp_frame, self.live_stream.timestamp_ms
        )

        # Displaying the mode
        frame = self.live_stream.display_text_on_stream(frame, "Gesture Recognition Mode", (10, 100))

        # Checking for recognized gestures
        if recognition_result.gestures:
            for gesture in recognition_result.gestures[0]:
                if gesture.category_name is not None:
                    text = f"{gesture.category_name}: {gesture.score:.2f}"
                    frame = self.live_stream.display_text_on_stream(frame, text, (10, 50))
                else:
                    break
        else:
            self.no_gesture_counter += 1
            print(f"No gesture detected for {self.no_gesture_counter} frames.")

        # Displaying the Frames
        self.live_stream.display_live_frame(TITLE, frame)

        # Exit Condition
        if cv2.waitKey(1) == ord("q"):
            self.state = "exit"

        # Checking for transition condition to learn landmarks
        if self.no_gesture_counter >= self.threashold_recognizer:
            self.state = "tracking"
            self.landmark_track_counter = 0

    def tracking_state(self) -> None:
        """Describes the control flow and pipeline for the tracking state."""

        ret, frame = self.live_stream.begin_live_stream()

        # Data Pipeline
        mp_frame = self.mediapipe_utilities.process_frame(
            self.live_stream.convert_to_rgb, frame
        )

        # Displaying the mode
        frame = self.live_stream.display_text_on_stream(frame, "Gesture Tracking Mode", (10, 100))

        # Detection result
        detection_result = self.detector.detect_for_video(
            mp_frame, self.live_stream.timestamp_ms
        )

        if detection_result.hand_landmarks:
            for hand in detection_result.hand_landmarks:
                for landmark in hand:
                    h, w, _ = frame.shape

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    frame = self.live_stream.display_landmark_on_stream(frame, (x, y))

            self.landmark_track_counter += 1

        # Displaying the Landmarks
        self.live_stream.display_live_frame(TITLE, frame)

        # Switching back to Recognition after learning the landmarks
        if self.landmark_track_counter >= self.threashold_detector:
            self.state = "recognition"
            self.no_gesture_counter = 0

    def run(self) -> None:
        """Mainloop for the FSM."""

        while self.state != "exit":
            if self.state == "recognition":
                self.recognition_state()
            elif self.state == "tracking":
                self.tracking_state()
            time.sleep(0.01)

        # Exit application condition
        self.live_stream.clear_live_stream()
