import mediapipe as mp
from typing import Any


class MediaPipeHandLandmarker:
    """Class responsible for setup of all the mediapipe components for hand_landmark tracking."""

    def __init__(self) -> None:
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.model_path = "./models/hand_landmarker.task"

    @property
    def options(self) -> Any:
        """Building an instance of the hand_landmarker model from options."""

        return self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=2
        )

    def build_detector(self) -> None:
        """Applying the options to create the hand_landmarker model for inference."""
        return self.HandLandmarker.create_from_options(self.options)
