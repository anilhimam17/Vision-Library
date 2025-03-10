from vision_library.gesture_recognizer import MediaPipeGestureRecognizer
from vision_library.mediapipe_utils import MediaPipeUtils
from vision_library.camera_utils import CVLiveStream
import cv2


# Title of the cv2 window
TITLE = "Gesture Recognizer"


class GestureRecognizerApp:
    """Class responsible for constructing the Gesture Recognition Application.

    It load all the components of the pipeline and applies them by running the mainloop
    with the OpenCV livestream running model inference on each frame in realtime."""

    def __init__(self) -> None:
        """Constructing a Gesture Recognition Application."""
        self.live_stream = CVLiveStream()
        self.gesture_recognizer_construct = MediaPipeGestureRecognizer()
        self.gesture_recognizer_options = self.gesture_recognizer_construct.build_from_options()
        self.mediapipe_utilities = MediaPipeUtils()

    def run(self) -> None:
        """The mainloop to compile the pipeline of the Gesture Recognition sytem."""

        # Intialising the Gesture Recognizer and Running Inference on the Livestream
        with (
                self.gesture_recognizer_construct
                .GestureRecognizer.create_from_options(self.gesture_recognizer_options)
        ) as recognizer:
            while True:
                ret, frame = self.live_stream.capture.read()

                if not ret:
                    print("Error retrieving frames from the livestream")
                    break

                # Processing the frames to perform model inference
                rgb_frame = self.live_stream.convert_to_rgb(frame)
                mp_frame = self.mediapipe_utilities.convert_rgb_to_mp(rgb_frame)

                # Model inferencing to find gestures
                recognition_result = recognizer.recognize_for_video(
                    mp_frame, self.live_stream.timestamp_ms
                )

                # Processing the result
                if recognition_result.gestures:
                    for gesture in recognition_result.gestures[0]:
                        text = f"{gesture.category_name}: {gesture.score:.2f}"
                        frame = self.live_stream.display_text_on_stream(frame, text)

                # Rendering the result
                self.live_stream.display_live_frame(TITLE, frame)

                # Exit Condition Checks
                if cv2.waitKey(1) == ord("q"):
                    print("Ending the livestream ....")
                    break

        # Releasing all the resources
        self.live_stream.clear_live_stream()
