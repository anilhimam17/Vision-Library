from cv2.typing import MatLike
from typing import Any
import mediapipe as mp


class MediaPipeUtils:
    """Class for abstracting additional utilities required by the mediapipe."""

    @staticmethod
    def convert_rgb_to_mp(rgb_frame: MatLike) -> Any:
        """Converting the RGB frame to MediaPipe images accepted by the model for inference."""
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def process_frame(self, rgb_converter, frame) -> MatLike:
        """Applies the entire frame processing pipeline."""

        rgb_frame = rgb_converter(frame)
        mp_frame = self.convert_rgb_to_mp(rgb_frame)
        return mp_frame
