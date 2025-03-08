import cv2
import time

from cv2.typing import MatLike


class CVLiveStream:
    """Class to handle all the OpenCV setup and livestream utilies."""

    def __init__(self) -> None:
        self.capture = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_coord = (10, 50)
        self.font_color = (255, 0, 0)
        self.font_thickness = 2

    @property
    def timestamp_ms(self) -> int:
        """Generating a monotonically increasing timestamp for the livestream."""
        return int(time.time() * 1000)

    def convert_to_rgb(self, frame: MatLike) -> MatLike:
        """Converting the BGR frame read from the livestream in RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def display_text_on_stream(self, frame: MatLike, text: str) -> MatLike:
        """Overlaying text onto the frames of the livestream."""
        return cv2.putText(
            frame, text, self.font_coord, self.font, self.font_scale, self.font_color, self.font_thickness 
        )

    def display_live_frame(self, title: str, frame: MatLike) -> None:
        """Displaying the frames of the livestream."""
        cv2.imshow(title, frame)

    def clear_live_stream(self) -> None:
        """Deallocating the resources on ending the livestream."""
        self.capture.release()
        cv2.destroyAllWindows()
