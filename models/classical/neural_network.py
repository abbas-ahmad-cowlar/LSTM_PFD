"""
Multi-Layer Perceptron (MLP) classifier.

Purpose:
    3-layer neural network for bearing fault classification.
    Architecture: 36 → 20 → 10 → 11 (from report Section 9.4)

Reference: Section 9.4 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import numpy as np
from typing import Dict, Optional
from sklearn.neural_network import MLPClassifier as SklearnMLP
import joblib
from pathlib import Path


class MLPClassifier:
    """
    Multi-Layer Perceptron for fault classification.

    Architecture: Input (36) → Hidden1 (20) → Hidden2 (10) → Output (11)

    Hyperparameters:
    - learning_rate_init: Initial learning rate
    - alpha: L2 regularization parameter
    - hidden_layer_sizes: Tuple of hidden layer sizes
    - max_iter: Maximum training iterations

    Example:
        >>> mlp = MLPClassifier()
        >>> mlp.train(X_train, y_train, hyperparams={'learning_rate_init': 0.001})
        >>> predictions = mlp.predict(X_test)
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize MLP classifier.

        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        self.model = None
        self.hyperparams_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              hyperparams: Optional[Dict] = None):
        """
        Train MLP classifier.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            hyperparams: Dictionary with MLP hyperparameters

        Returns:
            self
        """
        # Default hyperparameters
        if hyperparams is None:
            hyperparams = {
                'hidden_layer_sizes': (20, 10),
                'learning_rate_init': 0.001,
                'alpha': 0.0001,
                'max_iter': 500,
                'activation': 'relu'
            }

        self.hyperparams_ = hyperparams

        # Create MLP
        self.model = SklearnMLP(
            hidden_layer_sizes=hyperparams.get('hidden_layer_sizes', (20, 10)),
            learning_rate_init=hyperparams.get('learning_rate_init', 0.001),
            alpha=hyperparams.get('alpha', 0.0001),
            max_iter=hyperparams.get('max_iter', 500),
            activation=hyperparams.get('activation', 'relu'),
            solver='adam',
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
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

    def get_params(self) -> Dict:
        """Get model hyperparameters."""
        return self.hyperparams_

    def get_training_loss(self) -> np.ndarray:
        """
        Get training loss curve.

        Returns:
            Loss values per iteration
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        return np.array(self.model.loss_curve_)

    def get_validation_scores(self) -> np.ndarray:
        """
        Get validation scores during training.

        Returns:
            Validation scores per iteration
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        return np.array(getattr(self.model, 'validation_scores_', []))

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

    def get_n_iterations(self) -> int:
        """Get number of training iterations performed."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.n_iter_
