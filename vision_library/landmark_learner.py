import os
import joblib
import numpy as np
import pandas as pd

from typing import Any

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


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

    def train_model(self, X_train, y_train, X_test, y_test) -> tuple[Any, Any]:
        """Function to train the model on the dataset and provide the classification report."""

        # Sorting the param grids for the different models
        param_grid = {}
        if isinstance(self.model, RandomForestClassifier):
            param_grid = {
                "n_estimators": [5, 10, 25, 50],
                "max_depth": [3, 5, 7, 10],
                "max_features": ["sqrt", "log2"],
                "min_samples_leaf": [1, 2],
                "min_samples_split": [2, 3]
            }
        elif isinstance(self.model, SVC):
            param_grid = {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1, 10],
                "gamma": [0.001, 0.01, 0.1]
            }

        # Initialising the Grid Search
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        # Training the model
        grid_search.fit(X_train, y_train)

        # Best Estimators
        self.model = grid_search.best_estimator_

        # Testing
        y_pred = self.model.predict(X_test)

        test_report = classification_report(y_test, y_pred, output_dict=True)
        return (grid_search.best_params_, test_report)

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
            custom_prediction = self.custom_classfier.predict(scaled_feature_vector)
            custom_confidence = max(self.custom_classfier.predict_proba(scaled_feature_vector)[0])
        else:
            custom_prediction = None
            custom_confidence = 0.0

        return (custom_prediction, custom_confidence)

    def close_predictor(self):
        """Release all the resources and delete pretrained models."""

        for path in [CLASSIFIER_PATH, SCALER_PATH]:
            if os.path.exists(path):
                os.remove(path)
