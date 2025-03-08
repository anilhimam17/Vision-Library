from cv2.typing import MatLike
from typing import Any
import mediapipe as mp


class MediaPipeUtils:
    """Class for abstracting additional utilities required by the mediapipe."""

    @staticmethod
    def convert_rgb_to_mp(rgb_frame: MatLike) -> Any:
        """Converting the RGB frame to MediaPipe images accepted by the model for inference."""
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)