import csv
import pandas as pd
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

    @property
    def landmarks_df(self) -> pd.DataFrame:
        return pd.read_csv(self.filepath)

    def extract_landmarks(self, detection_result, frame, live_stream_handle, tracking=False) -> tuple[Any, Any]:
        """Carries out the landmark extraction process and provides the current detections."""

        current_landmarks = []
        current_raw_detections = []
        if detection_result.hand_landmarks:
            # Tracking only the first hand for extraction incase of recognition state
            if not tracking:
                first_hand = detection_result.hand_landmarks[0]
                for landmark in first_hand:
                    current_raw_detections.extend([landmark.x, landmark.y, landmark.z])
            # Tracking both the hand in other states, need to make this more standard
            else:
                for hand in detection_result.hand_landmarks:
                    for landmark in hand:
                        h, w, _ = frame.shape

                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        # Keep note of current landmarks to cache for persistence
                        current_landmarks.append((x, y))
                        current_raw_detections.extend([landmark.x, landmark.y, landmark.z])

                        # Display the tracked landmarks only in tracking state
                        if tracking:
                            frame = live_stream_handle.display_landmark_on_stream(frame, (x, y))

        return (current_landmarks, current_raw_detections)

    def store_landmarks(self, landmarks) -> None:
        """Storing new landmarks captured by the user for a new gesture."""

        self.csv_writer.writerow(
            [self.gesture_ctr] + landmarks
        )

        # Flushing the buffer if any values are left behind
        self.landmark_dataset.flush()

    def update_gesture_counter(self) -> None:
        """Updating the gesture counter after collecting all the samples."""
        self.gesture_ctr += 1

    def close_recorder(self) -> None:
        """Releases all the resources allocated for storage of landmarks."""
        self.landmark_dataset.close()
        del self.csv_writer
