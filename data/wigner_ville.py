"""
Wigner-Ville Distribution Module for Phase 5

Implements Wigner-Ville Distribution (WVD) and Pseudo-Wigner-Ville Distribution (PWVD)
for high-resolution time-frequency analysis. WVD provides optimal time-frequency resolution
but suffers from cross-term artifacts.

Author: AI Assistant
Date: 2025-11-20
"""

from utils.constants import SAMPLING_RATE
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional
import warnings


class WignerVilleDistribution:
    """
    Wigner-Ville Distribution generator.

    WVD: W(t,f) = ∫ x(t+τ/2) x*(t-τ/2) e^(-j2πfτ) dτ

    Advantages:
    - Highest time-frequency resolution
    - No window function needed

    Disadvantages:
    - Cross-term interference for multi-component signals
    - Can show negative values (not true energy distribution)
    """

    def __init__(self, fs: int = 20480):
        """
        Initialize WVD generator.

        Args:
            fs: Sampling frequency (Hz)
        """
        self.fs = fs

    def generate_wvd(
        self,
        signal_data: np.ndarray,
        nfft: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Wigner-Ville Distribution.

        Args:
            signal_data: Input signal [N_samples]
            nfft: FFT length (default: next power of 2)

        Returns:
            Tuple containing:
                - wvd: WVD matrix [n_freq, n_time]
                - frequencies: Frequency bins [n_freq]
                - times: Time bins [n_time]
        """
        N = len(signal_data)

        if nfft is None:
            nfft = 2 ** int(np.ceil(np.log2(N)))

        # Initialize WVD matrix
        n_freq = nfft // 2 + 1
        n_time = N
        wvd = np.zeros((n_freq, n_time))

        # Compute WVD
        for t_idx in range(N):
            # Maximum lag for this time instant
            max_lag = min(t_idx, N - 1 - t_idx, nfft // 2)

            # Instantaneous autocorrelation function
            autocorr = np.zeros(nfft, dtype=complex)

            for lag in range(-max_lag, max_lag + 1):
                if 0 <= t_idx + lag < N and 0 <= t_idx - lag < N:
                    autocorr[lag + nfft // 2] = (
                        signal_data[t_idx + lag] * np.conj(signal_data[t_idx - lag])
                    )

            # FFT to get frequency distribution at time t
            spectrum = np.fft.fft(autocorr, n=nfft)
            wvd[:, t_idx] = np.abs(spectrum[:n_freq])

        # Frequency and time axes
        frequencies = np.linspace(0, self.fs / 2, n_freq)
        times = np.arange(N) / self.fs

        return wvd, frequencies, times

    def generate_pseudo_wvd(
        self,
        signal_data: np.ndarray,
        window_size: int = 51,
        nfft: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Pseudo-Wigner-Ville Distribution (PWVD).

        PWVD applies smoothing window to reduce cross-terms.

        Args:
            signal_data: Input signal
            window_size: Smoothing window size (odd number)
            nfft: FFT length

        Returns:
            Smoothed WVD, frequencies, times
        """
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1

        # Generate WVD
        wvd, frequencies, times = self.generate_wvd(signal_data, nfft)

        # Smoothing window (Hamming)
        window = scipy_signal.hamming(window_size)
        window = window / window.sum()

        # Apply smoothing along time axis
        pwvd = np.zeros_like(wvd)
        for freq_idx in range(wvd.shape[0]):
            pwvd[freq_idx, :] = np.convolve(
                wvd[freq_idx, :],
                window,
                mode='same'
            )

        return pwvd, frequencies, times

    def generate_smoothed_pwvd(
        self,
        signal_data: np.ndarray,
        time_window: int = 51,
        freq_window: int = 11,
        nfft: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Smoothed Pseudo-Wigner-Ville Distribution.

        Applies 2D smoothing to further reduce cross-terms.

        Args:
            signal_data: Input signal
            time_window: Time smoothing window size
            freq_window: Frequency smoothing window size
            nfft: FFT length

        Returns:
            Smoothed WVD, frequencies, times
        """
        # Get PWVD
        pwvd, frequencies, times = self.generate_pseudo_wvd(
            signal_data,
            window_size=time_window,
            nfft=nfft
        )

        # Additional frequency smoothing
        if freq_window > 1:
            freq_win = scipy_signal.hamming(freq_window)
            freq_win = freq_win / freq_win.sum()

            spwvd = np.zeros_like(pwvd)
            for time_idx in range(pwvd.shape[1]):
                spwvd[:, time_idx] = np.convolve(
                    pwvd[:, time_idx],
                    freq_win,
                    mode='same'
                )
        else:
            spwvd = pwvd

        return spwvd, frequencies, times

    def generate_normalized_wvd(
        self,
        signal_data: np.ndarray,
        normalization: str = 'log_standardize',
        smoothing: str = 'pseudo'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate normalized WVD for neural network input.

        Args:
            signal_data: Input signal
            normalization: Normalization method
            smoothing: 'none', 'pseudo', or 'smoothed'

        Returns:
            Normalized WVD, frequencies, times
        """
        # Generate WVD with optional smoothing
        if smoothing == 'none':
            wvd, freqs, times = self.generate_wvd(signal_data)
        elif smoothing == 'pseudo':
            wvd, freqs, times = self.generate_pseudo_wvd(signal_data)
        elif smoothing == 'smoothed':
            wvd, freqs, times = self.generate_smoothed_pwvd(signal_data)
        else:
            raise ValueError(f"Unknown smoothing: {smoothing}")

        # Apply log transformation if requested
        if 'log' in normalization:
            wvd = np.log10(wvd + 1e-10)

        # Apply normalization
        if 'standardize' in normalization:
            mean = np.mean(wvd)
            std = np.std(wvd)
            if std > 1e-10:
                wvd = (wvd - mean) / std
        elif 'minmax' in normalization:
            min_val = np.min(wvd)
            max_val = np.max(wvd)
            if max_val - min_val > 1e-10:
                wvd = (wvd - min_val) / (max_val - min_val)

        return wvd, freqs, times


class CohenClassDistribution:
    """
    Cohen's Class time-frequency distributions.

    Includes various TFDs that are members of Cohen's class.
    """

    def __init__(self, fs: int = 20480):
        self.fs = fs

    def born_jordan_distribution(
        self,
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Born-Jordan Distribution (reduced interference).

        Args:
            signal_data: Input signal

        Returns:
            BJD, frequencies, times
        """
        # Simplified implementation using WVD with special kernel
        wvd_gen = WignerVilleDistribution(fs=self.fs)
        # Use smoothed PWVD as approximation
        bjd, freqs, times = wvd_gen.generate_smoothed_pwvd(
            signal_data,
            time_window=25,
            freq_window=5
        )
        return bjd, freqs, times

    def choi_williams_distribution(
        self,
        signal_data: np.ndarray,
        sigma: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Choi-Williams Distribution (exponential kernel for cross-term suppression).

        Args:
            signal_data: Input signal
            sigma: Kernel parameter (higher = more suppression)

        Returns:
            CWD, frequencies, times
        """
        # Simplified implementation
        wvd_gen = WignerVilleDistribution(fs=self.fs)
        # Adaptive smoothing based on sigma
        window_size = int(25 * sigma)
        if window_size % 2 == 0:
            window_size += 1
        cwd, freqs, times = wvd_gen.generate_pseudo_wvd(
            signal_data,
            window_size=window_size
        )
        return cwd, freqs, times


# Convenience function
def generate_wvd(
    signal_data: np.ndarray,
    fs: int = 20480,
    smoothing: str = 'pseudo',
    normalization: str = 'log_standardize'
) -> np.ndarray:
    """
    Convenience function to generate WVD.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        smoothing: Smoothing method
        normalization: Normalization method

    Returns:
        Normalized WVD [n_freq, n_time]
    """
    wvd_gen = WignerVilleDistribution(fs=fs)
    wvd, _, _ = wvd_gen.generate_normalized_wvd(
        signal_data,
        normalization=normalization,
        smoothing=smoothing
    )
    return wvd


if __name__ == "__main__":
    print("Wigner-Ville Distribution - Example Usage\n")

    # Generate test signal with two frequency components
    fs = SAMPLING_RATE
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))

    # Chirp signal (frequency changes over time)
    f0, f1 = 100, 1000
    signal_data = scipy_signal.chirp(t, f0, duration, f1, method='linear')

    # Add noise
    signal_data += 0.1 * np.random.randn(len(signal_data))

    # Generate different WVDs
    wvd_gen = WignerVilleDistribution(fs=fs)

    print("1. Standard WVD")
    wvd, freqs, times = wvd_gen.generate_wvd(signal_data)
    print(f"   WVD shape: {wvd.shape}")
    print(f"   Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")

    print("\n2. Pseudo-WVD (reduced cross-terms)")
    pwvd, _, _ = wvd_gen.generate_pseudo_wvd(signal_data, window_size=51)
    print(f"   PWVD shape: {pwvd.shape}")

    print("\n3. Smoothed Pseudo-WVD")
    spwvd, _, _ = wvd_gen.generate_smoothed_pwvd(signal_data)
    print(f"   SPWVD shape: {spwvd.shape}")

    print("\n4. Normalized WVD")
    norm_wvd, _, _ = wvd_gen.generate_normalized_wvd(signal_data)
    print(f"   Normalized range: [{norm_wvd.min():.2f}, {norm_wvd.max():.2f}]")

    print("\n✓ Wigner-Ville distribution generation successful!")
