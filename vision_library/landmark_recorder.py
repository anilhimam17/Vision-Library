import csv
import numpy as np
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
            [f"p{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")] +
            [f"dist_{i}" for i in range(9)] +
            [f"angle_{i}" for i in range(1, 5)]
        )
        self.csv_writer.writerow(headers)

    @property
    def landmarks_df(self) -> pd.DataFrame:
        return pd.read_csv(self.filepath)

    def extract_landmarks(self, detection_result, frame, live_stream_handle, tracking=False) -> tuple[Any, Any, Any]:
        """Carries out the landmark extraction process and provides the current detections."""

        current_raw_landmarks = []
        current_raw_detections = []
        display_corrected_detections = []
        if detection_result.hand_landmarks:
            # Tracking only the first hand for extraction incase of recognition state
            if not tracking:
                current_raw_landmarks = list(detection_result.hand_landmarks[0])
                for landmark in current_raw_landmarks:
                    current_raw_detections.extend([landmark.x, landmark.y, landmark.z])
            # Tracking both the hand in other states, need to make this more standard
            else:
                for hand in detection_result.hand_landmarks:
                    for landmark in hand:
                        h, w, _ = frame.shape

                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        # Keep note of current landmarks to cache for persistence
                        current_raw_landmarks.append(landmark)
                        current_raw_detections.extend([landmark.x, landmark.y, landmark.z])
                        display_corrected_detections.append((x, y))

                        # Display the tracked landmarks only in tracking state
                        if tracking:
                            frame = live_stream_handle.display_landmark_on_stream(frame, (x, y))

        return (current_raw_landmarks, current_raw_detections, display_corrected_detections)

    def extract_features(self, current_raw_landmarks) -> Any:
        """Function to normalize and extract additional features from the raw landmarks."""

        if not current_raw_landmarks:
            print("No raw detections made for feature extraction.")
            return

        # Grouping all the detections
        x_coords = [landmark.x for landmark in current_raw_landmarks]
        y_coords = [landmark.y for landmark in current_raw_landmarks]

        # Min-Max values wrt each axes
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Height and Width based on frames
        height = max(0.001, x_max - x_min)
        width = max(0.001, y_max - y_min)

        # Wrist landmarks as reference
        wrist_x, wrist_y, wrist_z = current_raw_landmarks[0].x, current_raw_landmarks[0].y, current_raw_landmarks[0].z

        # Normalized Landmarks
        normalized_landmarks = []
        for landmark in current_raw_landmarks:
            norm_x = (landmark.x - wrist_x) / width
            norm_y = (landmark.y - wrist_y) / height
            norm_z = landmark.z - wrist_z

            normalized_landmarks.extend([norm_x, norm_y, norm_z])

        # Landmark Keypoints for Feature Engineering
        landmark_keypoints = [
            (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
            (4, 8), (8, 12), (12, 16), (16, 20)
        ]
        distance_features = []
        for i, j in landmark_keypoints:
            dx = current_raw_landmarks[i].x - current_raw_landmarks[j].x
            dy = current_raw_landmarks[i].y - current_raw_landmarks[j].y
            dz = current_raw_landmarks[i].z - current_raw_landmarks[j].z

            distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            distance_features.append(distance)

        # Angle features for feature engineering
        angle_features = []
        for i in range(1, 5):
            base_idx = i * 1 + 4
            tip_idx = i * 4 + 4

            loc_x = current_raw_landmarks[tip_idx].x - current_raw_landmarks[base_idx].x
            loc_y = current_raw_landmarks[tip_idx].y - current_raw_landmarks[base_idx].y

            angle = np.arctan2(loc_x, loc_y)
            angle_features.append(angle)

        return normalized_landmarks + distance_features + angle_features

    def store_landmarks(self, landmarks) -> None:
        """Storing new landmarks captured by the user for a new gesture."""

        if landmarks:
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
