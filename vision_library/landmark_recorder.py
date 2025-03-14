import csv
from typing import Any


# Filepath to store the landmarks
LAND_CSV_PATH = "./db/hand_landmarks.csv"


class LandmarkRecorder:
    """Class responsible for making a record of all the gestures are requested to be captured."""

    # Counter for total number of gestures
    gesture_ctr: int = 1

    def __init__(self, filepath: str = LAND_CSV_PATH) -> None:
        self.filepath = filepath

        # Creating a new file for each run
        self.landmark_dataset = open(file=self.filepath, mode="w", newline="")

        # Initialising the file handle and the headers
        self.csv_writer = csv.writer(self.landmark_dataset)
        headers = (
            ["label"] +
            [f"p{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")]
        )
        self.csv_writer.writerow(headers)

    def store_landmarks(self, landmarks) -> None:
        """Storing new landmarks captured by the user for a new gesture."""

        hand = landmarks.hand_landmarks[0]

        # Flattened list of landmarks
        flattened_landmarks = [val for landmark in hand for val in (landmark.x, landmark.y, landmark.z)]

        self.csv_writer.writerow(
            [self.gesture_ctr] + flattened_landmarks
        )

    def update_gesture_counter(self) -> None:
        """Updating the gesture counter after collecting all the samples."""
        self.gesture_ctr += 1

    def extract_landmarks(self, detection_result, frame, live_stream_handle) -> Any:
        """Carries out the landmark extraction process and provides the current detections."""

        current_landmarks = []
        if detection_result.hand_landmarks:
            for hand in detection_result.hand_landmarks:
                for landmark in hand:
                    h, w, _ = frame.shape

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    # Keep note of current landmarks to cache for persistence
                    current_landmarks.append((x, y))
                    frame = live_stream_handle.display_landmark_on_stream(frame, (x, y))

        return current_landmarks

    def close_recorder(self) -> None:
        self.landmark_dataset.close()
        del self.csv_writer
