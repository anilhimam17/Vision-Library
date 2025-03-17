from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os
import numpy as np


# Filepaths
CLASSIFIER_PATH = "./models/trained_classifier.pkl"
SCALER_PATH = "./models/trained_scaler.pkl"


class LandmarkTrainer:
    """Class responsible for implementing the machine learning models and training them."""

    def __init__(self, model: str) -> None:
        if model == "forest":
            self.model = RandomForestClassifier()
        elif model == "svm":
            self.model = SVC(probability=True)
        self.scaler = StandardScaler()

    def prepare_data(self, df: pd.DataFrame) -> Any:
        """Function to prepare the dataset for training the model."""

        # Splitting the Features and the Labels
        X = df.drop(["label"], axis=1)
        y = df[["label"]]

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

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
        return classification_report(y_test, y_pred, output_dict=True)

    def save_model(self) -> None:
        """Function to serialize the trained models."""
        _ = joblib.dump(self.model, CLASSIFIER_PATH)
        _ = joblib.dump(self.scaler, SCALER_PATH)


class CustomPredictor:
    """Class to load the trained models and make predictions."""
    def __init__(self):
        self.custom_classfier = None
        self.custom_scaler = None

    def load_models(self):
        """Function to load the models dynamically on request."""

        if os.path.exists(CLASSIFIER_PATH):
            self.custom_classfier = joblib.load(CLASSIFIER_PATH)
        if os.path.exists(SCALER_PATH):
            self.custom_scaler = joblib.load(SCALER_PATH)

    def make_predictions(self, processed_features) -> tuple[Any, Any]:
        """Applying the custom pipeline to make predictions from the landmarks."""

        # Loading the latest models
        self.load_models()

        if processed_features and self.custom_classfier is not None:
            feature_vector = np.array(processed_features).reshape(1, -1)
            scaled_feature_vector = self.custom_scaler.transform(feature_vector)
            svm_prediction = self.custom_classfier.predict(scaled_feature_vector)
            svm_confidence = max(self.custom_classfier.predict_proba(scaled_feature_vector)[0])
        else:
            svm_prediction = None
            svm_confidence = 0.0

        return (svm_prediction, svm_confidence)

    def close_predictor(self):
        """Release all the resources and delete pretrained models."""

        os.remove(CLASSIFIER_PATH)
        os.remove(SCALER_PATH)