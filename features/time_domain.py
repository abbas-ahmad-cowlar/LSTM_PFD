"""
Time-domain statistical features for vibration signal analysis.

Purpose:
    Compute 7 time-domain features including RMS, kurtosis, skewness,
    and shape factors. These capture basic signal statistics and impulsiveness.

Reference: Section 8.2.1 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Dict
from scipy import stats


def compute_rms(signal: np.ndarray) -> float:
    """
    Compute Root Mean Square (RMS) value.

    RMS = sqrt(mean(x^2))

    Args:
        signal: Input signal array

    Returns:
        RMS value
    """
    return np.sqrt(np.mean(signal ** 2))


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    Compute kurtosis (4th statistical moment).

    Kurtosis measures impulsiveness and peaks in the signal.
    High kurtosis indicates presence of impulses (e.g., bearing defects).

    Args:
        signal: Input signal array

    Returns:
        Kurtosis value (Fisher=True, excess kurtosis)
    """
    return stats.kurtosis(signal, fisher=True)


def compute_skewness(signal: np.ndarray) -> float:
    """
    Compute skewness (3rd statistical moment).

    Skewness measures asymmetry of the distribution.

    Args:
        signal: Input signal array

    Returns:
        Skewness value
    """
    return stats.skew(signal)


def compute_crest_factor(signal: np.ndarray) -> float:
    """
    Compute crest factor.

    Crest Factor = peak / RMS
    High crest factor indicates impulsive behavior.

    Args:
        signal: Input signal array

    Returns:
        Crest factor
    """
    peak = np.max(np.abs(signal))
    rms = compute_rms(signal)
    return peak / rms if rms > 0 else 0.0


def compute_shape_factor(signal: np.ndarray) -> float:
    """
    Compute shape factor.

    Shape Factor = RMS / mean(|x|)

    Args:
        signal: Input signal array

    Returns:
        Shape factor
    """
    rms = compute_rms(signal)
    mean_abs = np.mean(np.abs(signal))
    return rms / mean_abs if mean_abs > 0 else 0.0


def compute_impulse_factor(signal: np.ndarray) -> float:
    """
    Compute impulse factor.

    Impulse Factor = peak / mean(|x|)

    Args:
        signal: Input signal array

    Returns:
        Impulse factor
    """
    peak = np.max(np.abs(signal))
    mean_abs = np.mean(np.abs(signal))
    return peak / mean_abs if mean_abs > 0 else 0.0


def compute_clearance_factor(signal: np.ndarray) -> float:
    """
    Compute clearance factor.

    Clearance Factor = peak / (mean(sqrt(|x|)))^2

    Args:
        signal: Input signal array

    Returns:
        Clearance factor
    """
    peak = np.max(np.abs(signal))
    mean_sqrt = np.mean(np.sqrt(np.abs(signal)))
    return peak / (mean_sqrt ** 2) if mean_sqrt > 0 else 0.0


def extract_time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract all 7 time-domain features.

    Args:
        signal: Input vibration signal (1D array)

    Returns:
        Dictionary with 7 time-domain features:
        - RMS: Root mean square
        - Kurtosis: 4th moment (impulsiveness)
        - Skewness: 3rd moment (asymmetry)
        - CrestFactor: peak/RMS ratio
        - ShapeFactor: RMS/mean ratio
        - ImpulseFactor: peak/mean ratio
        - ClearanceFactor: peak/(mean(sqrt))^2

    Example:
        >>> signal = np.random.randn(10000)
        >>> features = extract_time_domain_features(signal)
        >>> print(f"RMS: {features['RMS']:.3f}")
        >>> print(f"Kurtosis: {features['Kurtosis']:.3f}")
    """
    features = {
        'RMS': compute_rms(signal),
        'Kurtosis': compute_kurtosis(signal),
        'Skewness': compute_skewness(signal),
        'CrestFactor': compute_crest_factor(signal),
        'ShapeFactor': compute_shape_factor(signal),
        'ImpulseFactor': compute_impulse_factor(signal),
        'ClearanceFactor': compute_clearance_factor(signal)
    }

    return features
