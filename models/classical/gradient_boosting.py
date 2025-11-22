"""
Gradient Boosting classifier for bearing fault diagnosis.

Purpose:
    Wrapper for sklearn Gradient Boosting classifier.

Reference: Table 7 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBM
import joblib
from pathlib import Path


class GradientBoostingClassifier:
    """
    Gradient Boosting classifier for fault classification.

    Hyperparameters:
    - n_estimators: Number of boosting stages
    - learning_rate: Learning rate shrinks contribution of each tree
    - max_depth: Maximum depth of trees
    - subsample: Fraction of samples for fitting trees

    Example:
        >>> gbm = GradientBoostingClassifier()
        >>> gbm.train(X_train, y_train, hyperparams={'n_estimators': 100})
        >>> predictions = gbm.predict(X_test)
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize Gradient Boosting classifier.

        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        self.model = None
        self.hyperparams_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              hyperparams: Optional[Dict] = None):
        """
        Train Gradient Boosting classifier.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            hyperparams: Dictionary with GBM hyperparameters

        Returns:
            self
        """
        # Default hyperparameters
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 0.8
            }

        self.hyperparams_ = hyperparams

        # Create GBM
        self.model = SklearnGBM(
            n_estimators=hyperparams.get('n_estimators', 100),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            max_depth=hyperparams.get('max_depth', 3),
            subsample=hyperparams.get('subsample', 0.8),
            random_state=self.random_state,
            verbose=0
        )

        # Train
        self.model.fit(X_train, y_train)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X_test: Test features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X_test: Test features

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict_proba(X_test)

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances.

        Returns:
            Feature importances (n_features,)
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        return self.model.feature_importances_

    def get_params(self) -> Dict:
        """Get model hyperparameters."""
        return self.hyperparams_

    def save(self, filepath: Path):
        """
        Save trained model to disk.

        Args:
            filepath: Path to save file
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        joblib.dump({
            'model': self.model,
            'hyperparams': self.hyperparams_
        }, filepath)

    def load(self, filepath: Path):
        """
        Load trained model from disk.

        Args:
            filepath: Path to model file

        Returns:
            self
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.hyperparams_ = data['hyperparams']

        return self

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Accuracy score
        """
        if self.model is None:
            raise ValueError("Model must be trained before scoring")

        return self.model.score(X_test, y_test)

    def staged_predict_proba(self, X_test: np.ndarray):
        """
        Get predictions at each boosting stage.

        Useful for monitoring overfitting.

        Args:
            X_test: Test features

        Returns:
            Generator of probability arrays at each stage
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        return self.model.staged_predict_proba(X_test)
