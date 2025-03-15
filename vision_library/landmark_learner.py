from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


# Filepaths
CLASSIFIER_PATH = "./models/trained_classifier.pkl"
SCALER_PATH = "./models/trained_scaler.pkl"


class LandmarkTrainer:
    """Class responsible for implementing the machine learning models and training them."""

    def __init__(self, model: str) -> None:
        if model == "forest":
            self.model = RandomForestClassifier()
        else:
            self.model = SVC()
        self.scaler = StandardScaler()

    def prepare_data(self, df: pd.DataFrame) -> Any:
        """Function to prepare the dataset for training the model."""

        # Splitting the Features and the Labels
        X = df.drop(["label"], axis=1)
        y = df[["label"]]

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

        # Scaling the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return ((X_train_scaled, y_train), (X_test_scaled, y_test))

    def train_model(self, X_train, y_train, X_test, y_test) -> Any:
        """Function to train the model on the dataset and provide the classification report."""

        # Training the model
        self.model.fit(X_train, y_train)

        # Testing
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)

    def save_model(self) -> None:
        """Function to serialize the trained models."""
        _ = joblib.dump(self.model, CLASSIFIER_PATH)
        _ = joblib.dump(self.scaler, SCALER_PATH)

    def load_model(self) -> tuple[Any, Any]:
        """Function to load the trained models."""
        classifier = joblib.load(CLASSIFIER_PATH)
        scaler = joblib.load(SCALER_PATH)

        return (classifier, scaler)
