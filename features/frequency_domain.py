"""
Frequency-domain spectral features for vibration signal analysis.

Purpose:
    Compute 12 frequency-domain features including dominant frequency,
    spectral centroid, entropy, band energies, and harmonic ratios.

Reference: Section 8.2.3 of technical report

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

from utils.constants import SAMPLING_RATE, SIGNAL_LENGTH
import numpy as np
from typing import Dict, Tuple
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq


def compute_fft(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using FFT.

    Args:
        signal: Input signal array
        fs: Sampling frequency (Hz)

    Returns:
        Tuple of (frequencies, power_spectral_density)
    """
    N = len(signal)
    # Compute FFT
    fft_vals = fft(signal)
    # Compute power spectral density (one-sided)
    psd = (2.0 / N) * np.abs(fft_vals[:N // 2])
    # Frequency bins
    freqs = fftfreq(N, 1.0 / fs)[:N // 2]

    return freqs, psd


def compute_dominant_frequency(psd: np.ndarray, freqs: np.ndarray) -> float:
    """
    Find the dominant (peak) frequency in the spectrum.

    Args:
        psd: Power spectral density
        freqs: Frequency bins

    Returns:
        Dominant frequency (Hz)
    """
    peak_idx = np.argmax(psd)
    return freqs[peak_idx]


def compute_spectral_centroid(psd: np.ndarray, freqs: np.ndarray) -> float:
    """
    Compute spectral centroid (center of mass of spectrum).

    Centroid = sum(f * P(f)) / sum(P(f))

    Args:
        psd: Power spectral density
        freqs: Frequency bins

    Returns:
        Spectral centroid (Hz)
    """
    total_power = np.sum(psd)
    if total_power > 0:
        centroid = np.sum(freqs * psd) / total_power
        return centroid
    return 0.0


def compute_spectral_entropy(psd: np.ndarray) -> float:
    """
    Compute Shannon entropy of the power spectrum.

    Entropy = -sum(P_norm * log(P_norm))
    High entropy indicates broadband noise, low entropy indicates tonal content.

    Args:
        psd: Power spectral density

    Returns:
        Spectral entropy (nats)
    """
    # Normalize to probability distribution
    psd_norm = psd / (np.sum(psd) + 1e-12)
    # Remove zeros to avoid log(0)
    psd_norm = psd_norm[psd_norm > 0]
    # Compute entropy
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    return entropy


def compute_band_energy(psd: np.ndarray, freqs: np.ndarray,
                        band_range: Tuple[float, float]) -> float:
    """
    Compute energy in a specific frequency band.

    Args:
        psd: Power spectral density
        freqs: Frequency bins
        band_range: Tuple of (f_low, f_high) in Hz

    Returns:
        Band energy (sum of PSD in range)
    """
    f_low, f_high = band_range
    mask = (freqs >= f_low) & (freqs <= f_high)
    band_energy = np.sum(psd[mask])
    return band_energy


def compute_harmonic_ratios(psd: np.ndarray, freqs: np.ndarray, f0: float,
                            tolerance: float = 5.0) -> Tuple[float, float]:
    """
    Compute harmonic ratios (2X/1X and 3X/1X).

    These ratios indicate specific fault types:
    - High 2X/1X: Misalignment
    - High 3X/1X: Looseness

    Args:
        psd: Power spectral density
        freqs: Frequency bins
        f0: Fundamental frequency (Hz)
        tolerance: Frequency search tolerance (Hz)

    Returns:
        Tuple of (ratio_2X_1X, ratio_3X_1X)
    """
    # Find amplitude at fundamental frequency
    idx_1X = np.argmin(np.abs(freqs - f0))
    amp_1X = psd[idx_1X]

    # Find amplitude at 2X harmonic
    idx_2X = np.argmin(np.abs(freqs - 2 * f0))
    amp_2X = psd[idx_2X]

    # Find amplitude at 3X harmonic
    idx_3X = np.argmin(np.abs(freqs - 3 * f0))
    amp_3X = psd[idx_3X]

    # Compute ratios
    ratio_2X_1X = amp_2X / (amp_1X + 1e-12)
    ratio_3X_1X = amp_3X / (amp_1X + 1e-12)

    return ratio_2X_1X, ratio_3X_1X


def extract_frequency_domain_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract all 12 frequency-domain features.

    Args:
        signal: Input vibration signal (1D array)
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary with 12 frequency-domain features:
        - DominantFreq: Peak frequency in spectrum
        - SpectralCentroid: Center of mass of spectrum
        - SpectralEntropy: Shannon entropy of spectrum
        - LowBandEnergy: Energy in 0-500 Hz
        - MidBandEnergy: Energy in 500-2000 Hz
        - HighBandEnergy: Energy in 2000-5000 Hz
        - VeryHighBandEnergy: Energy in 5000-10000 Hz
        - TotalSpectralPower: Sum of PSD
        - SpectralStd: Standard deviation of PSD
        - Harmonic2X1X: 2X/1X harmonic ratio
        - Harmonic3X1X: 3X/1X harmonic ratio
        - SpectralPeakiness: Peak/mean ratio in spectrum

    Example:
        >>> signal = np.random.randn(10000)
        >>> features = extract_frequency_domain_features(signal, fs=SAMPLING_RATE)
        >>> print(f"Dominant Freq: {features['DominantFreq']:.2f} Hz")
    """
    # Compute FFT
    freqs, psd = compute_fft(signal, fs)

    # Basic spectral features
    dominant_freq = compute_dominant_frequency(psd, freqs)
    centroid = compute_spectral_centroid(psd, freqs)
    entropy = compute_spectral_entropy(psd)

    # Band energies (for bearing diagnostics)
    low_band = compute_band_energy(psd, freqs, (0, 500))
    mid_band = compute_band_energy(psd, freqs, (500, 2000))
    high_band = compute_band_energy(psd, freqs, (2000, 5000))
    very_high_band = compute_band_energy(psd, freqs, (5000, 10000))

    # Statistical measures
    total_power = np.sum(psd)
    spectral_std = np.std(psd)
    spectral_peak = np.max(psd)
    spectral_mean = np.mean(psd)
    spectral_peakiness = spectral_peak / (spectral_mean + 1e-12)

    # Harmonic ratios (assuming base rotation frequency ~60 Hz)
    f0 = dominant_freq if dominant_freq > 10 else 60.0
    ratio_2X_1X, ratio_3X_1X = compute_harmonic_ratios(psd, freqs, f0)

    features = {
        'DominantFreq': dominant_freq,
        'SpectralCentroid': centroid,
        'SpectralEntropy': entropy,
        'LowBandEnergy': low_band,
        'MidBandEnergy': mid_band,
        'HighBandEnergy': high_band,
        'VeryHighBandEnergy': very_high_band,
        'TotalSpectralPower': total_power,
        'SpectralStd': spectral_std,
        'Harmonic2X1X': ratio_2X_1X,
        'Harmonic3X1X': ratio_3X_1X,
        'SpectralPeakiness': spectral_peakiness
    }

    return features
