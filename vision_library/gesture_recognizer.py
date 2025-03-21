from typing import Any
import mediapipe as mp


class MediaPipeGestureRecognizer:
    """Class responsible for setup of all the mediapipe components for the gesture_recognizer model."""

    def __init__(self) -> None:
        self.BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.model_path = "./models/gesture_recognizer.task"

    @property
    def options(self) -> Any:
        """Building an instance of the gesture_recognizer model from options."""

        return self.GestureRecognizerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=1
        )

    def build_recognizer(self) -> Any:
        """Applying the options to build an instance of the gesture_recognizer."""

        return self.GestureRecognizer.create_from_options(self.options)
