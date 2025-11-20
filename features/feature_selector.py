"""
Feature selection using Minimum Redundancy Maximum Relevance (MRMR).

Purpose:
    Select the most informative 15 features from the 36 extracted features
    using MRMR criterion. This is done AFTER train/test split to prevent
    data leakage.

Reference: Section 8.4 of technical report (Innovation #5)

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import List, Optional
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    MRMR-based feature selector for post-split feature selection.

    Selects features that maximize relevance to the target while
    minimizing redundancy among selected features.

    Example:
        >>> selector = FeatureSelector(n_features=15)
        >>> selector.fit(X_train, y_train)
        >>> X_train_selected = selector.transform(X_train)
        >>> X_test_selected = selector.transform(X_test)
    """

    def __init__(self, n_features: int = 15, random_state: int = 42):
        """
        Initialize feature selector.

        Args:
            n_features: Number of features to select (default: 15)
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.random_state = random_state
        self.selected_indices_: Optional[List[int]] = None
        self.feature_names_: Optional[List[str]] = None
        self.relevance_scores_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None):
        """
        Fit the feature selector using MRMR algorithm.

        Algorithm:
        1. Compute relevance: I(feature; target) for all features
        2. Select feature with max relevance
        3. Iteratively select features maximizing:
           MRMR = Relevance - (1/|S|) * sum(Redundancy with selected)

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: Optional list of feature names

        Returns:
            self
        """
        n_samples, n_total_features = X.shape

        if self.n_features > n_total_features:
            self.n_features = n_total_features

        # Compute relevance scores: mutual information with target
        self.relevance_scores_ = mutual_info_classif(
            X, y, random_state=self.random_state
        )

        # Initialize selected features list
        selected = []
        remaining = list(range(n_total_features))

        # Select first feature: max relevance
        first_idx = np.argmax(self.relevance_scores_)
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Iteratively select remaining features
        for _ in range(self.n_features - 1):
            if not remaining:
                break

            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                # Compute relevance
                relevance = self.relevance_scores_[idx]

                # Compute redundancy with already selected features
                redundancy = 0.0
                for sel_idx in selected:
                    # Mutual information between features (approximated)
                    redundancy += self._mutual_info_features(X[:, idx], X[:, sel_idx])

                redundancy /= len(selected)

                # MRMR score
                mrmr_score = relevance - redundancy

                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        self.selected_indices_ = selected

        # Store feature names if provided
        if feature_names is not None:
            self.feature_names_ = [feature_names[i] for i in selected]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform feature matrix by selecting features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Selected features (n_samples, n_selected_features)
        """
        if self.selected_indices_ is None:
            raise ValueError("FeatureSelector must be fitted before transform")

        return X[:, self.selected_indices_]

    def get_selected_features(self) -> List[int]:
        """
        Get indices of selected features.

        Returns:
            List of selected feature indices
        """
        if self.selected_indices_ is None:
            raise ValueError("FeatureSelector must be fitted first")

        return self.selected_indices_

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get names of selected features.

        Returns:
            List of selected feature names (if provided during fit)
        """
        return self.feature_names_

    def _mutual_info_features(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Estimate mutual information between two continuous features.

        Uses histogram-based approximation.

        Args:
            x1: First feature vector
            x2: Second feature vector

        Returns:
            Estimated mutual information
        """
        # Discretize features into bins
        n_bins = 10
        x1_binned = np.digitize(x1, np.histogram_bin_edges(x1, bins=n_bins))
        x2_binned = np.digitize(x2, np.histogram_bin_edges(x2, bins=n_bins))

        # Compute 2D histogram
        hist_2d, _, _ = np.histogram2d(x1_binned, x2_binned, bins=n_bins)
        hist_2d = hist_2d / np.sum(hist_2d)  # Normalize to probability

        # Marginal distributions
        p_x1 = np.sum(hist_2d, axis=1)
        p_x2 = np.sum(hist_2d, axis=0)

        # Compute mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if hist_2d[i, j] > 0:
                    mi += hist_2d[i, j] * np.log(
                        hist_2d[i, j] / (p_x1[i] * p_x2[j] + 1e-12) + 1e-12
                    )

        return max(mi, 0.0)


class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """
    Simple variance threshold feature selector.

    Removes features with variance below threshold (constant or near-constant).

    Example:
        >>> selector = VarianceThresholdSelector(threshold=0.01)
        >>> selector.fit(X_train)
        >>> X_train_filtered = selector.transform(X_train)
    """

    def __init__(self, threshold: float = 0.01):
        """
        Initialize variance threshold selector.

        Args:
            threshold: Minimum variance threshold
        """
        self.threshold = threshold
        self.variances_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y=None):
        """
        Fit selector by computing variances.

        Args:
            X: Feature matrix
            y: Ignored (for compatibility)

        Returns:
            self
        """
        self.variances_ = np.var(X, axis=0)
        self.selected_indices_ = [
            i for i, var in enumerate(self.variances_) if var >= self.threshold
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform by removing low-variance features.

        Args:
            X: Feature matrix

        Returns:
            Filtered features
        """
        if self.selected_indices_ is None:
            raise ValueError("Must fit before transform")

        return X[:, self.selected_indices_]

    def get_selected_features(self) -> List[int]:
        """Get selected feature indices."""
        if self.selected_indices_ is None:
            raise ValueError("Must fit first")
        return self.selected_indices_
