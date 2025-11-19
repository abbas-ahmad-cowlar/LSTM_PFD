"""
Wavelet transform features for multi-scale signal analysis.

Purpose:
    Extract wavelet-based features that capture transient events and
    multi-scale signal properties. Wavelets are effective for non-stationary
    bearing fault signals.

Reference: Section 8.2.5 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, List
import pywt
from scipy.fft import fft


def compute_dwt_energy(signal: np.ndarray, wavelet: str = 'db4',
                       level: int = 5) -> List[float]:
    """
    Compute energy in each wavelet decomposition level.

    Discrete Wavelet Transform decomposes signal into approximation and
    detail coefficients at multiple scales.

    Args:
        signal: Input signal array
        wavelet: Wavelet type (default: Daubechies-4)
        level: Decomposition level (default: 5)

    Returns:
        List of energies at each level [cA5, cD5, cD4, cD3, cD2, cD1]
    """
    # Perform multilevel DWT
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Compute energy at each level: E = sum(c^2)
    energies = [np.sum(c ** 2) for c in coeffs]

    return energies


def compute_wavelet_energy_ratio(energies: List[float]) -> float:
    """
    Compute ratio of high-frequency to total energy.

    High ratio indicates impulsive content (faults).

    Args:
        energies: Energy at each decomposition level

    Returns:
        Ratio of detail energies to total energy
    """
    total_energy = np.sum(energies)
    # Detail coefficients are all except first (approximation)
    detail_energy = np.sum(energies[1:])

    ratio = detail_energy / (total_energy + 1e-12)
    return ratio


def compute_wavelet_kurtosis(signal: np.ndarray, wavelet: str = 'db4',
                             level: int = 5) -> float:
    """
    Compute kurtosis of wavelet detail coefficients.

    High kurtosis in detail coefficients indicates impulsive transients.

    Args:
        signal: Input signal array
        wavelet: Wavelet type (default: Daubechies-4)
        level: Decomposition level

    Returns:
        Mean kurtosis across all detail coefficient levels
    """
    from scipy import stats

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Compute kurtosis for each detail level (skip approximation)
    kurtosis_values = []
    for c in coeffs[1:]:  # Detail coefficients only
        if len(c) > 3:  # Need enough samples for kurtosis
            kurt = stats.kurtosis(c, fisher=True)
            kurtosis_values.append(kurt)

    # Return mean kurtosis
    return np.mean(kurtosis_values) if kurtosis_values else 0.0


def compute_cepstral_peak_ratio(signal: np.ndarray, fs: float) -> float:
    """
    Compute cepstral peak ratio.

    Cepstrum = IFFT(log(|FFT(signal)|))
    Peak in cepstrum indicates periodic components (e.g., gear mesh).

    Args:
        signal: Input signal array
        fs: Sampling frequency (Hz)

    Returns:
        Ratio of peak to mean in cepstrum
    """
    # Compute power spectrum
    fft_vals = fft(signal)
    power_spectrum = np.abs(fft_vals) ** 2

    # Compute cepstrum (real part of IFFT of log spectrum)
    log_spectrum = np.log(power_spectrum + 1e-12)
    cepstrum = np.real(np.fft.ifft(log_spectrum))

    # Take only positive quefrencies
    N = len(cepstrum)
    cepstrum_pos = np.abs(cepstrum[:N // 2])

    # Compute peak-to-mean ratio
    peak = np.max(cepstrum_pos)
    mean = np.mean(cepstrum_pos)
    ratio = peak / (mean + 1e-12)

    return ratio


def compute_quefrency_centroid(signal: np.ndarray, fs: float) -> float:
    """
    Compute centroid of cepstrum (quefrency centroid).

    Quefrency is the independent variable of cepstrum (time-like).

    Args:
        signal: Input signal array
        fs: Sampling frequency (Hz)

    Returns:
        Quefrency centroid (seconds)
    """
    # Compute power spectrum
    fft_vals = fft(signal)
    power_spectrum = np.abs(fft_vals) ** 2

    # Compute cepstrum
    log_spectrum = np.log(power_spectrum + 1e-12)
    cepstrum = np.real(np.fft.ifft(log_spectrum))

    # Take positive quefrencies
    N = len(cepstrum)
    cepstrum_pos = np.abs(cepstrum[:N // 2])

    # Quefrency bins (in seconds)
    quefrencies = np.arange(N // 2) / fs

    # Compute centroid
    total = np.sum(cepstrum_pos)
    if total > 0:
        centroid = np.sum(quefrencies * cepstrum_pos) / total
    else:
        centroid = 0.0

    return centroid


def extract_wavelet_features(signal: np.ndarray, fs: float = 20480) -> Dict[str, float]:
    """
    Extract all 7 wavelet-based features.

    Args:
        signal: Input vibration signal (1D array)
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary with 7 wavelet features:
        - WaveletEnergyRatio: Ratio of detail to total energy
        - WaveletKurtosis: Mean kurtosis of detail coefficients
        - WaveletEnergy_D1: Energy in detail level 1 (highest freq)
        - WaveletEnergy_D3: Energy in detail level 3 (mid freq)
        - WaveletEnergy_D5: Energy in detail level 5 (low freq)
        - CepstralPeakRatio: Peak/mean in cepstrum
        - QuefrencyCentroid: Centroid of cepstrum

    Example:
        >>> signal = np.random.randn(10000)
        >>> features = extract_wavelet_features(signal, fs=20480)
        >>> print(f"Wavelet Energy Ratio: {features['WaveletEnergyRatio']:.3f}")
    """
    # Compute DWT energies
    energies = compute_dwt_energy(signal, wavelet='db4', level=5)

    # Energy ratio
    energy_ratio = compute_wavelet_energy_ratio(energies)

    # Wavelet kurtosis
    wavelet_kurt = compute_wavelet_kurtosis(signal, wavelet='db4', level=5)

    # Cepstral features
    cepstral_peak = compute_cepstral_peak_ratio(signal, fs)
    quefrency_centroid = compute_quefrency_centroid(signal, fs)

    # Extract specific detail level energies
    # energies = [cA5, cD5, cD4, cD3, cD2, cD1]
    energy_d1 = energies[-1] if len(energies) > 0 else 0.0  # Highest freq
    energy_d3 = energies[-3] if len(energies) > 2 else 0.0  # Mid freq
    energy_d5 = energies[1] if len(energies) > 1 else 0.0   # Low freq (cD5)

    features = {
        'WaveletEnergyRatio': energy_ratio,
        'WaveletKurtosis': wavelet_kurt,
        'WaveletEnergy_D1': energy_d1,
        'WaveletEnergy_D3': energy_d3,
        'WaveletEnergy_D5': energy_d5,
        'CepstralPeakRatio': cepstral_peak,
        'QuefrencyCentroid': quefrency_centroid
    }

    return features
