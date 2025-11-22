"""
Envelope analysis features using Hilbert transform.

Purpose:
    Extract envelope features for detecting modulation patterns caused by
    bearing faults. Hilbert envelope reveals impact patterns.

Reference: Section 8.2.2 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from utils.constants import SAMPLING_RATE, SIGNAL_LENGTH
import numpy as np
from typing import Dict
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute Hilbert envelope of the signal.

    Envelope = |analytic_signal| = |signal + j*Hilbert(signal)|

    Args:
        signal: Input signal array

    Returns:
        Envelope signal (amplitude modulation)
    """
    # Compute analytic signal using Hilbert transform
    analytic_signal = hilbert(signal)
    # Envelope is the magnitude
    envelope = np.abs(analytic_signal)
    return envelope


def compute_envelope_rms(envelope: np.ndarray) -> float:
    """
    Compute RMS of the envelope.

    Args:
        envelope: Envelope signal

    Returns:
        RMS value of envelope
    """
    return np.sqrt(np.mean(envelope ** 2))


def compute_envelope_kurtosis(envelope: np.ndarray) -> float:
    """
    Compute kurtosis of the envelope.

    High kurtosis in envelope indicates impulsive modulation.

    Args:
        envelope: Envelope signal

    Returns:
        Kurtosis of envelope
    """
    from scipy import stats
    return stats.kurtosis(envelope, fisher=True)


def compute_envelope_peak(envelope: np.ndarray) -> float:
    """
    Compute peak value of envelope.

    Args:
        envelope: Envelope signal

    Returns:
        Maximum envelope value
    """
    return np.max(envelope)


def compute_modulation_frequency(envelope: np.ndarray, fs: float) -> float:
    """
    Compute dominant modulation frequency from envelope spectrum.

    The modulation frequency often corresponds to fault characteristic
    frequencies (e.g., ball pass frequency).

    Args:
        envelope: Envelope signal
        fs: Sampling frequency (Hz)

    Returns:
        Dominant modulation frequency (Hz)
    """
    # Compute FFT of envelope
    N = len(envelope)
    fft_vals = fft(envelope - np.mean(envelope))  # Remove DC
    psd = (2.0 / N) * np.abs(fft_vals[:N // 2])
    freqs = fftfreq(N, 1.0 / fs)[:N // 2]

    # Find dominant frequency (excluding DC)
    valid_idx = freqs > 1.0  # Ignore very low frequencies
    if np.sum(valid_idx) > 0:
        peak_idx = np.argmax(psd[valid_idx])
        mod_freq = freqs[valid_idx][peak_idx]
    else:
        mod_freq = 0.0

    return mod_freq


def extract_envelope_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract all 4 envelope analysis features.

    Args:
        signal: Input vibration signal (1D array)
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary with 4 envelope features:
        - EnvelopeRMS: RMS of Hilbert envelope
        - EnvelopeKurtosis: Kurtosis of envelope (impulsiveness)
        - EnvelopePeak: Peak envelope value
        - ModulationFreq: Dominant frequency in envelope spectrum

    Example:
        >>> signal = np.random.randn(10000)
        >>> features = extract_envelope_features(signal, fs=SAMPLING_RATE)
        >>> print(f"Envelope RMS: {features['EnvelopeRMS']:.3f}")
        >>> print(f"Modulation Freq: {features['ModulationFreq']:.2f} Hz")
    """
    # Compute envelope
    envelope = compute_envelope(signal)

    # Extract features from envelope
    features = {
        'EnvelopeRMS': compute_envelope_rms(envelope),
        'EnvelopeKurtosis': compute_envelope_kurtosis(envelope),
        'EnvelopePeak': compute_envelope_peak(envelope),
        'ModulationFreq': compute_modulation_frequency(envelope, fs)
    }

    return features
