"""
Stacked ensemble combining multiple base models.

Purpose:
    Meta-learner that combines predictions from SVM, RF, NN, and GBM
    using logistic regression.

Reference: Section 9.1 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import joblib
from pathlib import Path


class StackedEnsemble:
    """
    Stacking ensemble with logistic regression meta-learner.

    Combines predictions from multiple base models to improve accuracy.

    Example:
        >>> from models.classical import SVMClassifier, RandomForestClassifier
        >>> base_models = [SVMClassifier(), RandomForestClassifier()]
        >>> ensemble = StackedEnsemble()
        >>> ensemble.train(X_train, y_train, base_models)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize stacked ensemble.

        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        self.base_models = None
        self.meta_learner = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              base_models: List, cv: int = 5):
        """
        Train stacked ensemble.

        Process:
        1. Train base models using cross-validation
        2. Collect out-of-fold predictions
        3. Train meta-learner on base model predictions

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            base_models: List of base model instances (untrained)
            cv: Number of cross-validation folds

        Returns:
            self
        """
        self.base_models = base_models
        n_samples = X_train.shape[0]
        n_models = len(base_models)

        # Step 1: Generate out-of-fold predictions from base models
        meta_features = np.zeros((n_samples, n_models))

        for i, model in enumerate(base_models):
            # Train model and get out-of-fold predictions
            # This prevents overfitting in meta-learner
            print(f"Training base model {i+1}/{n_models}...")

            # Train on full data
            model.train(X_train, y_train)

            # Get cross-validated predictions for meta-features
            # We'll use a simple approach: train on folds and predict
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            fold_predictions = np.zeros(n_samples)
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train = y_train[train_idx]

                # Create a fresh model instance for this fold
                fold_model = type(model)(random_state=self.random_state)
                fold_model.train(X_fold_train, y_fold_train,
                               hyperparams=model.get_params())

                # Predict on validation fold
                fold_predictions[val_idx] = fold_model.predict(X_fold_val)

            meta_features[:, i] = fold_predictions

        # Step 2: Train meta-learner on meta-features
        print("Training meta-learner...")
        self.meta_learner = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            multi_class='multinomial'
        )
        self.meta_learner.fit(meta_features, y_train)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels using ensemble.

        Args:
            X_test: Test features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        if self.base_models is None or self.meta_learner is None:
            raise ValueError("Ensemble must be trained before prediction")

        # Get predictions from base models
        meta_features = self._get_meta_features(X_test)

        # Meta-learner prediction
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble.

        Args:
            X_test: Test features

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if self.base_models is None or self.meta_learner is None:
            raise ValueError("Ensemble must be trained before prediction")

        # Get predictions from base models
        meta_features = self._get_meta_features(X_test)

        # Meta-learner probability prediction
        return self.meta_learner.predict_proba(meta_features)

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Get meta-features from base model predictions.

        Args:
            X: Input features

        Returns:
            Meta-features (n_samples, n_base_models)
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        for i, model in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X)

        return meta_features

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)

    def save(self, filepath: Path):
        """
        Save trained ensemble to disk.

        Args:
            filepath: Path to save file
        """
        if self.base_models is None or self.meta_learner is None:
            raise ValueError("No trained ensemble to save")

        joblib.dump({
            'base_models': self.base_models,
            'meta_learner': self.meta_learner
        }, filepath)

    def load(self, filepath: Path):
        """
        Load trained ensemble from disk.

        Args:
            filepath: Path to ensemble file

        Returns:
            self
        """
        data = joblib.load(filepath)
        self.base_models = data['base_models']
        self.meta_learner = data['meta_learner']

        return self
