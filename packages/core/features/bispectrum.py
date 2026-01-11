"""
Higher-order spectral analysis (bispectrum) features.

Purpose:
    Extract bispectrum features that capture phase coupling and non-linear
    interactions in the signal. Useful for detecting quadratic nonlinearities
    caused by bearing faults.

Reference: Section 8.2.6 of technical report

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, Tuple
from scipy.fft import fft


def compute_bispectrum(signal: np.ndarray, nfft: int = 512) -> np.ndarray:
    """
    Compute bispectrum using direct method.

    Bispectrum B(f1, f2) = E[X(f1) * X(f2) * X*(f1+f2)]
    where X(f) is the Fourier transform.

    The bispectrum detects quadratic phase coupling and deviations
    from Gaussianity.

    Args:
        signal: Input signal array
        nfft: FFT length for segmentation

    Returns:
        Bispectrum magnitude array (simplified 1D summary)
    """
    # Segment the signal
    N = len(signal)
    num_segments = N // nfft
    if num_segments < 1:
        num_segments = 1
        nfft = N

    # Compute bispectrum via averaging over segments
    bispectrum_accum = np.zeros(nfft // 2, dtype=complex)

    for i in range(num_segments):
        segment = signal[i * nfft:(i + 1) * nfft]
        if len(segment) < nfft:
            segment = np.pad(segment, (0, nfft - len(segment)), mode='constant')

        # Compute FFT
        X = fft(segment)
        X_half = X[:nfft // 2]

        # Simplified bispectrum: B(f) ≈ X(f) * X(f) * conj(X(2f))
        # This is a diagonal slice of the full bispectrum
        for k in range(nfft // 4):  # Avoid aliasing
            f1_idx = k
            f2_idx = k
            f_sum_idx = min(2 * k, nfft // 2 - 1)
            bispectrum_accum[k] += X_half[f1_idx] * X_half[f2_idx] * np.conj(X_half[f_sum_idx])

    # Average over segments
    bispectrum = bispectrum_accum / num_segments

    return np.abs(bispectrum)


def compute_bispectrum_peak(bispectrum: np.ndarray) -> float:
    """
    Compute peak value in bispectrum.

    High peak indicates strong phase coupling.

    Args:
        bispectrum: Bispectrum magnitude array

    Returns:
        Peak bispectrum value
    """
    return np.max(bispectrum)


def compute_bispectrum_mean(bispectrum: np.ndarray) -> float:
    """
    Compute mean bispectrum value.

    Args:
        bispectrum: Bispectrum magnitude array

    Returns:
        Mean bispectrum value
    """
    return np.mean(bispectrum)


def compute_bispectrum_entropy(bispectrum: np.ndarray) -> float:
    """
    Compute entropy of bispectrum.

    High entropy indicates distributed phase coupling.

    Args:
        bispectrum: Bispectrum magnitude array

    Returns:
        Entropy value
    """
    # Normalize to probability distribution
    bispectrum_norm = bispectrum / (np.sum(bispectrum) + 1e-12)
    bispectrum_norm = bispectrum_norm[bispectrum_norm > 0]

    # Compute entropy
    entropy = -np.sum(bispectrum_norm * np.log(bispectrum_norm + 1e-12))
    return entropy


def compute_phase_coupling(signal: np.ndarray) -> float:
    """
    Compute phase coupling indicator.

    This measures the strength of quadratic phase coupling using
    the bicoherence metric.

    Args:
        signal: Input signal array

    Returns:
        Phase coupling strength (0 to 1)
    """
    # Simplified bicoherence estimate
    nfft = min(512, len(signal) // 4)
    if nfft < 64:
        nfft = 64

    bispectrum = compute_bispectrum(signal, nfft=nfft)

    # Compute power spectrum for normalization
    fft_vals = fft(signal[:nfft])
    power = np.abs(fft_vals[:nfft // 2]) ** 2

    # Bicoherence ≈ |bispectrum|^2 / (power normalization)
    # Simplified version
    coupling = np.sum(bispectrum) / (np.sum(power[:len(bispectrum)]) + 1e-12)

    # Normalize to [0, 1] range approximately
    coupling = np.tanh(coupling)  # Soft clipping

    return coupling


def compute_nonlinearity_index(signal: np.ndarray) -> float:
    """
    Compute nonlinearity index based on bispectrum.

    Measures deviation from Gaussianity (Gaussian signals have zero bispectrum).

    Args:
        signal: Input signal array

    Returns:
        Nonlinearity index
    """
    nfft = min(512, len(signal) // 4)
    if nfft < 64:
        nfft = 64

    bispectrum = compute_bispectrum(signal, nfft=nfft)

    # Compute skewness as additional nonlinearity measure
    from scipy import stats
    skewness = abs(stats.skew(signal))

    # Combine bispectrum and skewness
    bispectrum_norm = np.sum(bispectrum) / len(bispectrum)
    nonlinearity = (bispectrum_norm + skewness) / 2.0

    return nonlinearity


def extract_bispectrum_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract all 6 higher-order spectral features.

    Args:
        signal: Input vibration signal (1D array)

    Returns:
        Dictionary with 6 bispectrum features:
        - BispectrumPeak: Peak value in bispectrum
        - BispectrumMean: Mean bispectrum value
        - BispectrumEntropy: Entropy of bispectrum
        - PhaseCoupling: Quadratic phase coupling strength
        - NonlinearityIndex: Deviation from Gaussianity
        - BispectrumPeakRatio: Peak/mean ratio

    Example:
        >>> signal = np.random.randn(10000)
        >>> features = extract_bispectrum_features(signal)
        >>> print(f"Phase Coupling: {features['PhaseCoupling']:.3f}")
    """
    # Compute bispectrum
    nfft = min(512, len(signal) // 4)
    if nfft < 64:
        nfft = 64
    bispectrum = compute_bispectrum(signal, nfft=nfft)

    # Extract features
    peak = compute_bispectrum_peak(bispectrum)
    mean = compute_bispectrum_mean(bispectrum)
    entropy = compute_bispectrum_entropy(bispectrum)
    coupling = compute_phase_coupling(signal)
    nonlinearity = compute_nonlinearity_index(signal)

    # Peak ratio
    peak_ratio = peak / (mean + 1e-12)

    features = {
        'BispectrumPeak': peak,
        'BispectrumMean': mean,
        'BispectrumEntropy': entropy,
        'PhaseCoupling': coupling,
        'NonlinearityIndex': nonlinearity,
        'BispectrumPeakRatio': peak_ratio
    }

    return features
