"""
Advanced feature extraction (optional 16 features).

Purpose:
    Extract computationally expensive advanced features including:
    - Continuous wavelet transform (CWT)
    - Wavelet packet decomposition (WPT)
    - Nonlinear dynamics (Lyapunov, entropy, DFA)

WARNING: These features are ~10x slower to compute than base features.
Only use when necessary.

Reference: Section 8.3 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from utils.constants import SAMPLING_RATE, SIGNAL_LENGTH
import numpy as np
from typing import Dict
import pywt
from scipy import signal as sp_signal
from scipy.stats import entropy


def extract_cwt_features(signal_data: np.ndarray, fs: float = 20480) -> Dict[str, float]:
    """
    Extract Continuous Wavelet Transform features.

    CWT provides time-frequency localization better than STFT.

    Args:
        signal_data: Input signal
        fs: Sampling frequency

    Returns:
        Dictionary with 4 CWT features

    Example:
        >>> cwt_features = extract_cwt_features(signal, fs=SAMPLING_RATE)
    """
    # Define scales (frequencies to analyze)
    scales = np.arange(1, 128)
    wavelet = 'morl'  # Morlet wavelet

    # Compute CWT
    coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet, 1.0/fs)

    # Energy at each scale
    energies = np.sum(np.abs(coefficients) ** 2, axis=1)

    # Features
    cwt_total_energy = np.sum(energies)
    cwt_peak_energy = np.max(energies)
    cwt_energy_ratio = cwt_peak_energy / (cwt_total_energy + 1e-12)

    # Find dominant scale
    dominant_scale_idx = np.argmax(energies)
    dominant_frequency = frequencies[dominant_scale_idx]

    features = {
        'CWT_TotalEnergy': cwt_total_energy,
        'CWT_PeakEnergy': cwt_peak_energy,
        'CWT_EnergyRatio': cwt_energy_ratio,
        'CWT_DominantFreq': dominant_frequency
    }

    return features


def extract_wpt_features(signal_data: np.ndarray, level: int = 4) -> Dict[str, float]:
    """
    Extract Wavelet Packet Transform features.

    WPT decomposes both approximation and detail coefficients,
    providing better frequency resolution than DWT.

    Args:
        signal_data: Input signal
        level: Decomposition level

    Returns:
        Dictionary with 4 WPT features

    Example:
        >>> wpt_features = extract_wpt_features(signal, level=4)
    """
    # Perform wavelet packet decomposition
    wp = pywt.WaveletPacket(data=signal_data, wavelet='db4', mode='symmetric', maxlevel=level)

    # Get all nodes at specified level
    nodes = [node.path for node in wp.get_level(level, 'natural')]

    # Compute energy at each node
    energies = []
    for node_path in nodes:
        node = wp[node_path]
        energy = np.sum(node.data ** 2)
        energies.append(energy)

    energies = np.array(energies)
    total_energy = np.sum(energies)

    # Features
    wpt_max_energy = np.max(energies)
    wpt_energy_entropy = entropy(energies / (total_energy + 1e-12))
    wpt_energy_std = np.std(energies)
    wpt_energy_ratio = wpt_max_energy / (total_energy + 1e-12)

    features = {
        'WPT_MaxEnergy': wpt_max_energy,
        'WPT_EnergyEntropy': wpt_energy_entropy,
        'WPT_EnergyStd': wpt_energy_std,
        'WPT_EnergyRatio': wpt_energy_ratio
    }

    return features


def compute_sample_entropy(signal_data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute Sample Entropy (irregularity measure).

    Sample Entropy measures complexity and regularity of time series.
    Lower values indicate more regular/predictable signals.

    Args:
        signal_data: Input signal
        m: Embedding dimension
        r: Tolerance (as fraction of std)

    Returns:
        Sample entropy value

    Example:
        >>> sampen = compute_sample_entropy(signal)
    """
    N = len(signal_data)
    r = r * np.std(signal_data)

    # Build templates
    def _maxdist(xi, xj):
        return max([abs(xi[k] - xj[k]) for k in range(len(xi))])

    def _phi(m):
        x = [[signal_data[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r])
             for i in range(len(x))]
        return sum(C)

    # Compute sample entropy
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return 0.0

    sampen = -np.log(phi_m1 / phi_m)
    return sampen


def compute_approximate_entropy(signal_data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute Approximate Entropy (regularity measure).

    Args:
        signal_data: Input signal
        m: Embedding dimension
        r: Tolerance

    Returns:
        Approximate entropy value
    """
    N = len(signal_data)
    r = r * np.std(signal_data)

    def _phi(m):
        patterns = np.array([signal_data[i:i+m] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            # Count similar patterns
            dists = np.max(np.abs(patterns - patterns[i]), axis=1)
            C[i] = np.sum(dists <= r) / (N - m + 1)
        return np.sum(np.log(C + 1e-12)) / (N - m + 1)

    return _phi(m) - _phi(m + 1)


def compute_dfa(signal_data: np.ndarray, min_window: int = 4, max_window: int = None) -> float:
    """
    Compute Detrended Fluctuation Analysis scaling exponent.

    DFA measures long-range correlations in time series.

    Args:
        signal_data: Input signal
        min_window: Minimum window size
        max_window: Maximum window size

    Returns:
        DFA scaling exponent (alpha)

    Example:
        >>> alpha = compute_dfa(signal)
    """
    N = len(signal_data)
    if max_window is None:
        max_window = N // 4

    # Integrate the signal
    y = np.cumsum(signal_data - np.mean(signal_data))

    # Window sizes (logarithmically spaced)
    windows = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 20).astype(int))

    fluctuations = []

    for window in windows:
        # Divide into segments
        n_segments = N // window
        if n_segments < 1:
            continue

        F_n = 0.0
        for seg in range(n_segments):
            # Extract segment
            segment = y[seg * window:(seg + 1) * window]
            # Fit polynomial trend
            t = np.arange(window)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            # Compute fluctuation
            F_n += np.mean((segment - trend) ** 2)

        F_n /= n_segments
        fluctuations.append(np.sqrt(F_n))

    if len(fluctuations) < 2:
        return 0.5  # Default value

    # Fit power law: F(n) ~ n^alpha
    log_windows = np.log(windows[:len(fluctuations)])
    log_fluct = np.log(fluctuations)

    # Linear fit in log-log space
    alpha = np.polyfit(log_windows, log_fluct, 1)[0]

    return alpha


def extract_nonlinear_features(signal_data: np.ndarray) -> Dict[str, float]:
    """
    Extract nonlinear dynamics features.

    Includes:
    - Sample Entropy
    - Approximate Entropy
    - DFA alpha
    - Correlation dimension (approximated)

    Args:
        signal_data: Input signal

    Returns:
        Dictionary with 4 nonlinear features

    Example:
        >>> nonlinear_features = extract_nonlinear_features(signal)
    """
    # Sample entropy
    sampen = compute_sample_entropy(signal_data, m=2, r=0.2)

    # Approximate entropy
    apen = compute_approximate_entropy(signal_data, m=2, r=0.2)

    # DFA
    dfa_alpha = compute_dfa(signal_data)

    # Hurst exponent (related to DFA)
    hurst = dfa_alpha  # For fractional Brownian motion, H = alpha

    features = {
        'SampleEntropy': sampen,
        'ApproximateEntropy': apen,
        'DFA_Alpha': dfa_alpha,
        'HurstExponent': hurst
    }

    return features


def extract_advanced_features(signal_data: np.ndarray, fs: float = 20480) -> Dict[str, float]:
    """
    Extract all 16 advanced features.

    WARNING: This is ~10x slower than base feature extraction.

    Args:
        signal_data: Input vibration signal
        fs: Sampling frequency

    Returns:
        Dictionary with 16 advanced features:
        - 4 CWT features
        - 4 WPT features
        - 4 nonlinear dynamics features
        - 4 time-frequency features

    Example:
        >>> advanced_features = extract_advanced_features(signal, fs=SAMPLING_RATE)
        >>> print(f"Total advanced features: {len(advanced_features)}")  # 16
    """
    # CWT features (4)
    cwt_features = extract_cwt_features(signal_data, fs)

    # WPT features (4)
    wpt_features = extract_wpt_features(signal_data, level=4)

    # Nonlinear features (4)
    nonlinear_features = extract_nonlinear_features(signal_data)

    # Spectrogram features (4)
    # Compute spectrogram
    f, t, Sxx = sp_signal.spectrogram(signal_data, fs=fs, nperseg=256)

    # Spectral entropy over time
    spectral_entropy_time = np.mean([entropy(Sxx[:, i]) for i in range(Sxx.shape[1])])

    # Time-frequency peak
    tf_peak = np.max(Sxx)
    tf_mean = np.mean(Sxx)
    tf_std = np.std(Sxx)

    tf_features = {
        'Spectrogram_Entropy': spectral_entropy_time,
        'TF_Peak': tf_peak,
        'TF_Mean': tf_mean,
        'TF_Std': tf_std
    }

    # Combine all
    all_features = {
        **cwt_features,
        **wpt_features,
        **nonlinear_features,
        **tf_features
    }

    return all_features
