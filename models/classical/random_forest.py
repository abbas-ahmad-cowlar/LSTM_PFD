"""
Random Forest classifier for bearing fault diagnosis.

Purpose:
    Wrapper for sklearn Random Forest with feature importance extraction.
    Random Forest typically achieves best performance (~95% accuracy).

Reference: Section 9.3 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier as SklearnRF
import joblib
from pathlib import Path


class RandomForestClassifier:
    """
    Random Forest classifier for multi-class fault classification.

    Hyperparameters:
    - n_estimators: Number of trees
    - max_depth: Maximum tree depth
    - min_samples_leaf: Minimum samples per leaf
    - max_features: Features to consider for split

    Example:
        >>> rf = RandomForestClassifier()
        >>> rf.train(X_train, y_train, hyperparams={'n_estimators': 200})
        >>> predictions = rf.predict(X_test)
        >>> importances = rf.get_feature_importances()
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize Random Forest classifier.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.hyperparams_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              hyperparams: Optional[Dict] = None):
        """
        Train Random Forest classifier.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            hyperparams: Dictionary with RF hyperparameters

        Returns:
            self
        """
        # Default hyperparameters
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }

        self.hyperparams_ = hyperparams

        # Create Random Forest
        self.model = SklearnRF(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', None),
            min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
            max_features=hyperparams.get('max_features', 'sqrt'),
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
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
        Get Gini-based feature importances.

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

    def get_n_trees(self) -> int:
        """Get number of trees in forest."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.n_estimators

    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available.

        Returns:
            OOB score or None
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        return getattr(self.model, 'oob_score_', None)
