"""
Feature validation utilities.

Purpose:
    Validate extracted features for correctness, check for NaN/Inf values,
    and warn about constant or suspicious features.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


def validate_feature_vector(features: np.ndarray, expected_dim: int = 36) -> Tuple[bool, str]:
    """
    Validate a single feature vector.

    Checks:
    - Correct dimensionality
    - No NaN values
    - No Inf values
    - All finite values

    Args:
        features: Feature vector to validate
        expected_dim: Expected number of features

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> features = np.random.randn(36)
        >>> is_valid, msg = validate_feature_vector(features)
        >>> assert is_valid
    """
    # Check shape
    if features.shape[0] != expected_dim:
        return False, f"Expected {expected_dim} features, got {features.shape[0]}"

    # Check for NaN
    if np.any(np.isnan(features)):
        nan_indices = np.where(np.isnan(features))[0]
        return False, f"Found NaN values at indices: {nan_indices.tolist()}"

    # Check for Inf
    if np.any(np.isinf(features)):
        inf_indices = np.where(np.isinf(features))[0]
        return False, f"Found Inf values at indices: {inf_indices.tolist()}"

    # All checks passed
    return True, "Valid"


def check_feature_distribution(features: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               variance_threshold: float = 1e-6) -> List[str]:
    """
    Check feature distributions and warn about issues.

    Checks for:
    - Constant or near-constant features (low variance)
    - Features with extreme values

    Args:
        features: Feature matrix (n_samples, n_features)
        feature_names: Optional list of feature names
        variance_threshold: Minimum variance threshold

    Returns:
        List of warning messages

    Example:
        >>> features = np.random.randn(100, 36)
        >>> warnings = check_feature_distribution(features)
    """
    warnings_list = []
    n_samples, n_features = features.shape

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Check variance
    variances = np.var(features, axis=0)
    low_variance_indices = np.where(variances < variance_threshold)[0]

    if len(low_variance_indices) > 0:
        low_var_features = [feature_names[i] for i in low_variance_indices]
        warnings_list.append(
            f"Low variance features (< {variance_threshold}): {low_var_features}"
        )

    # Check for outliers (values > 5 std from mean)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)

    for i in range(n_features):
        if stds[i] > 0:
            outliers = np.abs(features[:, i] - means[i]) > 5 * stds[i]
            n_outliers = np.sum(outliers)
            if n_outliers > 0:
                outlier_pct = 100 * n_outliers / n_samples
                if outlier_pct > 5:  # More than 5% outliers
                    warnings_list.append(
                        f"{feature_names[i]}: {outlier_pct:.1f}% outliers (> 5 std)"
                    )

    return warnings_list


def check_for_nans(features: np.ndarray,
                   feature_names: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Check for NaN values in feature matrix.

    Args:
        features: Feature matrix (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Dictionary mapping feature names to NaN counts

    Example:
        >>> nan_counts = check_for_nans(features, feature_names)
        >>> if nan_counts:
        >>>     print(f"Found NaNs: {nan_counts}")
    """
    n_features = features.shape[1]

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    nan_dict = {}
    for i, name in enumerate(feature_names):
        nan_count = np.sum(np.isnan(features[:, i]))
        if nan_count > 0:
            nan_dict[name] = nan_count

    return nan_dict


def replace_nans(features: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Replace NaN values in feature matrix.

    Args:
        features: Feature matrix (n_samples, n_features)
        strategy: Replacement strategy ('mean', 'median', 'zero')

    Returns:
        Feature matrix with NaNs replaced

    Example:
        >>> features_clean = replace_nans(features, strategy='median')
    """
    features_clean = features.copy()

    for i in range(features.shape[1]):
        col = features[:, i]
        if np.any(np.isnan(col)):
            if strategy == 'mean':
                replacement = np.nanmean(col)
            elif strategy == 'median':
                replacement = np.nanmedian(col)
            elif strategy == 'zero':
                replacement = 0.0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            features_clean[np.isnan(col), i] = replacement

    return features_clean


def validate_feature_matrix(features: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None,
                           verbose: bool = True) -> bool:
    """
    Comprehensive validation of feature matrix.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Optional label array for checking class balance
        feature_names: Optional feature names
        verbose: Print warnings

    Returns:
        True if validation passes, False otherwise

    Example:
        >>> is_valid = validate_feature_matrix(X_train, y_train, feature_names)
    """
    all_valid = True

    # Check for NaNs
    nan_dict = check_for_nans(features, feature_names)
    if nan_dict:
        all_valid = False
        if verbose:
            print(f"WARNING: Found NaN values in features: {nan_dict}")

    # Check for Infs
    if np.any(np.isinf(features)):
        all_valid = False
        if verbose:
            print("WARNING: Found Inf values in features")

    # Check distributions
    distribution_warnings = check_feature_distribution(features, feature_names)
    if distribution_warnings and verbose:
        for warning in distribution_warnings:
            print(f"WARNING: {warning}")

    # Check labels if provided
    if labels is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        if verbose:
            print(f"\nClass distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"  Class {label}: {count} samples ({100*count/len(labels):.1f}%)")

        # Check for class imbalance
        min_count = np.min(counts)
        max_count = np.max(counts)
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 10:
            if verbose:
                print(f"WARNING: High class imbalance (ratio: {imbalance_ratio:.1f})")

    return all_valid
