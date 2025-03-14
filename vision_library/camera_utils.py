from typing import Any
import cv2
import time

from cv2.typing import MatLike


class CVLiveStream:
    """Class to handle all the OpenCV setup and livestream utilies."""

    def __init__(self) -> None:
        self.capture = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (255, 0, 0)
        self.font_thickness = 2
        self.radius = 7
        self.hand_color = (0, 255, 0)

    @property
    def timestamp_ms(self) -> int:
        """Generating a monotonically increasing timestamp for the livestream."""
        return int(time.time() * 1000)

    def convert_to_rgb(self, frame: MatLike) -> MatLike:
        """Converting the BGR frame read from the livestream in RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def display_text_on_stream(self, frame: MatLike, text: str, coord: tuple[int, int]) -> MatLike:
        """Overlaying text onto the frames of the livestream."""
        return cv2.putText(
            frame, text, coord, self.font, self.font_scale, self.font_color, self.font_thickness
        )

    def display_landmark_on_stream(self, frame: MatLike, coord: tuple[int, int]) -> MatLike:
        return cv2.circle(frame, coord, self.radius, self.hand_color)

    def display_live_frame(self, title: str, frame: MatLike) -> None:
        """Displaying the frames of the livestream."""
        cv2.imshow(title, frame)

    def begin_live_stream(self) -> tuple[bool, Any]:
        """Begins a live_stream and reads frames."""

        ret, frame = self.capture.read()
        if not ret:
            print("Error retrieving livestream feed.")
            return (False, None)
        return (True, frame)

    def clear_live_stream(self) -> None:
        """Deallocating the resources on ending the livestream."""
        self.capture.release()
        cv2.destroyAllWindows()
