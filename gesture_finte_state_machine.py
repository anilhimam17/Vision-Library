from vision_library.gesture_recognizer import MediaPipeGestureRecognizer
from vision_library.hand_landmarker import MediaPipeHandLandmarker
from vision_library.camera_utils import CVLiveStream
from vision_library.landmark_learner import CustomPredictor, LandmarkTrainer
from vision_library.mediapipe_utils import MediaPipeUtils
from vision_library.landmark_recorder import LandmarkRecorder

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
        self.landmark_recoder = LandmarkRecorder()
        self.landmark_trainer = LandmarkTrainer("svm")
        self.custom_predictor = CustomPredictor()

        # State Variable to manage the transitions
        self.state = "recognition"

        # Buffer for landmark detection persistence
        self.previous_landmarks = []

        # Threashold and Counters to manage the State Variable
        self.no_gesture_counter = 1
        self.threashold_recognizer = 50
        self.gesture_sample_counter = 0
        self.gesture_sample_threshold = 10

    def recognition_state(self) -> None:
        """Describes the control flow and pipeline for the recognition state."""
        ret, frame = self.live_stream.begin_live_stream()
        if not ret:
            print("Error retrieving frames in recognition state")
            return

        # Data Pipeline
        mp_frame = self.mediapipe_utilities.process_frame(
            self.live_stream.convert_to_rgb, frame
        )

        # MediaPipe Gesture Recognizer inference
        mp_result = self.recognizer.recognize_for_video(
            mp_frame, self.live_stream.timestamp_ms
        )

        # MediaPipe Handlandmarker inference
        detection_result = self.detector.detect_for_video(
            mp_frame, timestamp_ms=self.live_stream.timestamp_ms
        )
        _, raw_extracted_landmarks = self.landmark_recoder.extract_landmarks(
            detection_result=detection_result, frame=frame, live_stream_handle=self.live_stream, tracking=False
        )
        svm_label, svm_confidence = self.custom_predictor.make_predictions(raw_extracted_landmarks)

        # Displaying the mode
        frame = self.live_stream.display_text_on_stream(frame, "Gesture Recognition Mode", (10, 50))

        valid_gestures_mp = []
        if mp_result.gestures and len(mp_result.gestures[0]) > 0:
            valid_gestures_mp = [
                gesture for gesture in mp_result.gestures[0]
                if (gesture.category_name is not None and gesture.category_name.lower() != "none")
                and gesture.score >= 0.6
            ]

        # Non-Max suppression
        if valid_gestures_mp:
            mp_most_confident = max(valid_gestures_mp, key=lambda gesture: gesture.score)
            mp_label = mp_most_confident.category_name
            mp_confidence = mp_most_confident.score
        else:
            mp_label = None
            mp_confidence = 0.0

        # Score polling
        if svm_confidence >= 0.7:
            text = f"{svm_label}: {svm_confidence:.2f}"
            frame = self.live_stream.display_text_on_stream(frame, text, (10, 100))
            self.no_gesture_counter = 1
        elif mp_confidence >= 0.6:
            text = f"{mp_label}: {mp_confidence:.2f}"
            frame = self.live_stream.display_text_on_stream(frame, text, (10, 100))
            self.no_gesture_counter = 1
        else:
            self.no_gesture_counter += 1
            frame = self.live_stream.display_text_on_stream(
                frame, f"No gesture detected: {self.no_gesture_counter}/{self.threashold_recognizer}", (10, 100)
            )
            print(f"No gesture detected on {self.no_gesture_counter} frames")

        # Displaying the Frames
        self.live_stream.display_live_frame(TITLE, frame)

        # Exit Condition
        if cv2.waitKey(1) == ord("q"):
            self.state = "exit"
            return

        # Checking for transition condition to learn landmarks
        if self.no_gesture_counter >= self.threashold_recognizer:
            print("Transitioning to tracking state")
            self.state = "tracking"

    def tracking_state(self) -> None:
        """Describes the control flow and pipeline for the tracking state."""

        # Tracking the keystroke for cases
        keyStroke = cv2.waitKey(1)

        ret, frame = self.live_stream.begin_live_stream()
        if not ret:
            print("Error retrieving frames in tracking state")
            return

        # Data Pipeline
        mp_frame = self.mediapipe_utilities.process_frame(
            self.live_stream.convert_to_rgb, frame
        )

        # Displaying the mode
        frame = self.live_stream.display_text_on_stream(frame, "Gesture Tracking Mode", (10, 50))

        # Detection result
        detection_result = self.detector.detect_for_video(
            mp_frame, self.live_stream.timestamp_ms
        )

        # Extracting the landmarks from the tracking state
        current_landmarks, _ = self.landmark_recoder.extract_landmarks(
            detection_result=detection_result, frame=frame, live_stream_handle=self.live_stream, tracking=True
        )

        # Combining the previous and current landmarks for persistance
        if self.previous_landmarks is not None and current_landmarks:
            for (x, y) in self.previous_landmarks:
                frame = self.live_stream.display_landmark_on_stream(frame, (x, y))

        # Moving the current landmarks to cache
        if current_landmarks:
            self.previous_landmarks = current_landmarks

        # Displaying the Landmarks
        self.live_stream.display_live_frame(TITLE, frame)

        # Exit Condition
        if keyStroke == ord("q"):
            self.state = "exit"
            return
        elif keyStroke == ord("r"):
            # Transitioning to learning as an intermediate state before returning to recognition
            self.state = "learning"
        elif keyStroke == ord("n"):
            self.state = "capture"
            self.gesture_sample_counter = 0

    def capturing_state(self) -> None:
        """Describes the pipeline and states for capturing new gestures and storing them."""

        # Tracking the keystroke for cases
        keyStroke = cv2.waitKey(1)

        ret, frame = self.live_stream.begin_live_stream()
        if not ret:
            print("Error retrieving frames in capturing state")
            return

        # Data Pipeline
        mp_frame = self.mediapipe_utilities.process_frame(
            self.live_stream.convert_to_rgb, frame
        )

        # Displaying the mode
        frame = self.live_stream.display_text_on_stream(frame, "Gesture Capturing Mode", (10, 50))

        # Detection result
        detection_result = self.detector.detect_for_video(
            mp_frame, self.live_stream.timestamp_ms
        )

        # Extracting the landmarks to be captured
        extracted_landmarks, raw_extracted_landmarks = self.landmark_recoder.extract_landmarks(
            detection_result=detection_result, frame=frame, live_stream_handle=self.live_stream, tracking=True
        )

        if self.gesture_sample_counter < self.gesture_sample_threshold:
            if keyStroke == ord("s") and extracted_landmarks:
                # Storing the landmarks that are captured
                self.landmark_recoder.store_landmarks(raw_extracted_landmarks)

                # Updating the counter
                self.gesture_sample_counter += 1

                # Visual confirmation on capturing samples
                frame = self.live_stream.display_text_on_stream(
                    frame, text=f"Sample number: {self.gesture_sample_counter} Captured", coord=(10, 100)
                )
                self.live_stream.display_live_frame(TITLE, frame)

                # Slowing down the frames
                _ = cv2.waitKey(350)

            elif keyStroke == ord("q"):
                self.state = "exit"
                return
            else:
                print("No landmarks detected in this frame, try again !!!")
        else:
            print("Retrieved required samples, returning to tracking")

            # Delay to show confirmation and slow down frames
            _ = cv2.waitKey(300)

            self.landmark_recoder.gesture_ctr += 1
            self.state = "tracking"
            return

        # Updating the number of samples captured
        frame = self.live_stream.display_text_on_stream(
            frame,
            coord=(10, 150),
            text=f"Collected: {self.gesture_sample_counter}/{self.gesture_sample_threshold} samples"
        )

        # Updating the number of unique gestures tracked and learned
        frame = self.live_stream.display_text_on_stream(
            frame,
            coord=(10, 200),
            text=f"Unique Gestures Collected: {self.landmark_recoder.gesture_ctr - 1}"
        )

        # Displaying the Landmarks
        self.live_stream.display_live_frame(TITLE, frame)

    def learning_state(self) -> None:
        """Describes the pipeline for learning any of the new gestures that were taught in capture."""

        # Loading the dataset
        landmark_df = self.landmark_recoder.landmarks_df

        # Empty dataframe skip learning process
        if landmark_df.empty:
            self.state = "recognition"
            self.no_gesture_counter = 1
            return

        # Preparing the dataframe for training
        train_set, test_set = self.landmark_trainer.prepare_data(landmark_df)

        # Training the model
        report = self.landmark_trainer.train_model(
            *train_set, *test_set
        )
        print(report)

        # Saving the models
        self.landmark_trainer.save_model()

        # Transitioning back to recognition
        self.state = "recognition"
        self.no_gesture_counter = 1

    def run(self) -> None:
        """Mainloop for the FSM."""

        while self.state != "exit":
            if self.state == "recognition":
                self.recognition_state()
            elif self.state == "tracking":
                self.tracking_state()
            elif self.state == "capture":
                self.capturing_state()
            elif self.state == "learning":
                self.learning_state()
            time.sleep(0.01)

        # Exit application condition
        print("The system has terminated!!!")
        self.live_stream.clear_live_stream()
