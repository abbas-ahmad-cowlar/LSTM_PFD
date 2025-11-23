"""
Support Vector Machine classifier with ECOC encoding.

Purpose:
    Wrapper for sklearn SVM with Error-Correcting Output Codes (ECOC)
    for multi-class classification.

Reference: Section 9.2 of technical report

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import numpy as np
from typing import Dict, Optional
from sklearn.svm import SVC
from sklearn.multiclass import OutputCodeClassifier
import joblib
from pathlib import Path


class SVMClassifier:
    """
    SVM classifier with ECOC for multi-class bearing fault classification.

    Hyperparameters:
    - C: Box constraint (regularization)
    - gamma: Kernel scale (RBF kernel)
    - kernel: Kernel type (default: 'rbf')

    Example:
        >>> svm = SVMClassifier()
        >>> svm.train(X_train, y_train, hyperparams={'C': 10, 'gamma': 0.1})
        >>> predictions = svm.predict(X_test)
        >>> proba = svm.predict_proba(X_test)
    """

    def __init__(self, use_ecoc: bool = True, random_state: int = 42):
        """
        Initialize SVM classifier.

        Args:
            use_ecoc: Use Error-Correcting Output Codes
            random_state: Random seed
        """
        self.use_ecoc = use_ecoc
        self.random_state = random_state
        self.model = None
        self.hyperparams_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              hyperparams: Optional[Dict] = None):
        """
        Train SVM classifier.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            hyperparams: Dictionary with 'C', 'gamma', 'kernel'

        Returns:
            self
        """
        # Default hyperparameters
        if hyperparams is None:
            hyperparams = {
                'C': 1.0,
                'gamma': 'scale',
                'kernel': 'rbf'
            }

        self.hyperparams_ = hyperparams

        # Create base SVM
        base_svm = SVC(
            C=hyperparams.get('C', 1.0),
            gamma=hyperparams.get('gamma', 'scale'),
            kernel=hyperparams.get('kernel', 'rbf'),
            probability=True,  # Enable probability estimates
            random_state=self.random_state,
            cache_size=1000  # MB
        )

        # Wrap with ECOC if requested
        if self.use_ecoc:
            self.model = OutputCodeClassifier(
                base_svm,
                random_state=self.random_state
            )
        else:
            self.model = base_svm

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

        # ECOC doesn't support predict_proba directly
        if self.use_ecoc:
            # Use decision function as proxy
            decision = self.model.decision_function(X_test)
            # Softmax to get probabilities
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            proba = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
            return proba
        else:
            return self.model.predict_proba(X_test)

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
            'hyperparams': self.hyperparams_,
            'use_ecoc': self.use_ecoc
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
        self.use_ecoc = data['use_ecoc']

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
