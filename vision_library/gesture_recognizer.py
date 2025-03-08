from typing import Any
import mediapipe as mp


class MediaPipeGestureRecognizer:
    def __init__(self) -> None:
        self.BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.model_path = "./models/gesture_recognizer.task"

    def build_from_options(self) -> Any:
        self.options = self.GestureRecognizerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO,
        )

        return self.options