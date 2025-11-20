"""
Feature normalization and standardization.

Purpose:
    Normalize features to zero mean and unit variance for classical ML models.
    Prevents features with large magnitudes from dominating the learning.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    Z-score normalization (standardization) for features.

    Transforms features to have mean=0 and std=1 based on training data statistics.

    Example:
        >>> normalizer = FeatureNormalizer()
        >>> normalizer.fit(X_train)
        >>> X_train_norm = normalizer.transform(X_train)
        >>> X_test_norm = normalizer.transform(X_test)
    """

    def __init__(self, method: str = 'standard'):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('standard', 'zscore', or 'minmax')
                   Note: 'zscore' is an alias for 'standard'
        """
        self.method = method
        # Support 'zscore' as alias for 'standard'
        if method == 'standard' or method == 'zscore':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def fit(self, X: np.ndarray, y=None):
        """
        Compute normalization parameters from training data.

        Args:
            X: Training feature matrix (n_samples, n_features)
            y: Ignored (for compatibility)

        Returns:
            self
        """
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply normalization to features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Normalized feature matrix
        """
        return self.scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Revert normalization (denormalize).

        Args:
            X: Normalized feature matrix

        Returns:
            Original scale features
        """
        return self.scaler.inverse_transform(X)

    def get_params(self, deep=True):
        """Get parameters (for sklearn compatibility)."""
        return {'method': self.method}

    def set_params(self, **params):
        """Set parameters (for sklearn compatibility)."""
        if 'method' in params:
            self.method = params['method']
        return self


class RobustNormalizer(BaseEstimator, TransformerMixin):
    """
    Robust normalization using median and IQR.

    More robust to outliers than standard z-score normalization.

    Example:
        >>> normalizer = RobustNormalizer()
        >>> normalizer.fit(X_train)
        >>> X_norm = normalizer.transform(X_train)
    """

    def __init__(self):
        """Initialize robust normalizer."""
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):
        """
        Compute median and IQR from training data.

        Args:
            X: Training feature matrix
            y: Ignored

        Returns:
            self
        """
        self.median_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        # Avoid division by zero
        self.iqr_[self.iqr_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply robust normalization.

        Formula: (X - median) / IQR

        Args:
            X: Feature matrix

        Returns:
            Normalized features
        """
        if self.median_ is None or self.iqr_ is None:
            raise ValueError("Must fit before transform")

        return (X - self.median_) / self.iqr_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Revert normalization.

        Args:
            X: Normalized features

        Returns:
            Original scale features
        """
        if self.median_ is None or self.iqr_ is None:
            raise ValueError("Must fit before inverse_transform")

        return X * self.iqr_ + self.median_
