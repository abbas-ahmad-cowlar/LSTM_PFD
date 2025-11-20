"""
Spectrogram Generation Module for Phase 5: Time-Frequency Analysis

Implements STFT-based spectrogram generation with various normalization strategies.
Optimized for bearing fault diagnosis with 20.48 kHz sampling rate.

Author: AI Assistant
Date: 2025-11-20
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional, Dict
import warnings


class SpectrogramGenerator:
    """
    Generate spectrograms from vibration signals using Short-Time Fourier Transform (STFT).

    Attributes:
        fs (int): Sampling frequency (Hz)
        nperseg (int): Length of each segment for STFT
        noverlap (int): Number of overlapping samples
        window (str): Window function type
        nfft (Optional[int]): FFT length
    """

    def __init__(
        self,
        fs: int = 20480,
        nperseg: int = 256,
        noverlap: int = 128,
        window: str = 'hann',
        nfft: Optional[int] = None
    ):
        """
        Initialize SpectrogramGenerator.

        Args:
            fs: Sampling frequency in Hz (default: 20480 for bearing data)
            nperseg: Segment length for STFT (default: 256 = 12.5ms window)
            noverlap: Overlap between segments (default: 128 = 50% overlap)
            window: Window function ('hann', 'hamming', 'blackman')
            nfft: FFT length (default: same as nperseg)
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
        self.nfft = nfft if nfft is not None else nperseg

        # Validate parameters
        if self.noverlap >= self.nperseg:
            raise ValueError("noverlap must be less than nperseg")
        if self.nfft < self.nperseg:
            raise ValueError("nfft must be >= nperseg")

    def generate_stft_spectrogram(
        self,
        signal_data: np.ndarray,
        return_phase: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate STFT spectrogram from time-domain signal.

        Args:
            signal_data: Input signal array [N_samples]
            return_phase: If True, return phase information

        Returns:
            Tuple containing:
                - spectrogram: Power spectrogram [n_freq, n_time]
                - frequencies: Frequency bins [n_freq]
                - times: Time bins [n_time]
                - (optional) phase: Phase spectrogram [n_freq, n_time]
        """
        # Compute STFT
        frequencies, times, Sxx = signal.stft(
            signal_data,
            fs=self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            return_onesided=True,  # Only positive frequencies
            boundary='zeros',
            padded=True
        )

        # Compute power spectrogram
        power_spectrogram = np.abs(Sxx) ** 2

        if return_phase:
            phase = np.angle(Sxx)
            return power_spectrogram, frequencies, times, phase
        else:
            return power_spectrogram, frequencies, times

    def generate_log_spectrogram(
        self,
        signal_data: np.ndarray,
        epsilon: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate log-scale (dB) spectrogram.

        Args:
            signal_data: Input signal array
            epsilon: Small constant to avoid log(0)

        Returns:
            Log-scaled spectrogram in dB, frequencies, times
        """
        power_spec, freqs, times = self.generate_stft_spectrogram(signal_data)

        # Convert to dB scale
        log_spec = 10 * np.log10(power_spec + epsilon)

        return log_spec, freqs, times

    def generate_normalized_spectrogram(
        self,
        signal_data: np.ndarray,
        normalization: str = 'log_standardize'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate normalized spectrogram ready for neural network input.

        Args:
            signal_data: Input signal array
            normalization: Normalization method
                - 'log_standardize': Log-scale + z-score normalization
                - 'log_minmax': Log-scale + min-max to [0, 1]
                - 'power_standardize': Power spectrum + z-score
                - 'power_minmax': Power spectrum + min-max

        Returns:
            Normalized spectrogram, frequencies, times
        """
        if normalization in ['log_standardize', 'log_minmax']:
            spec, freqs, times = self.generate_log_spectrogram(signal_data)
        else:
            spec, freqs, times = self.generate_stft_spectrogram(signal_data)

        # Apply normalization
        if normalization in ['log_standardize', 'power_standardize']:
            # Z-score normalization (mean=0, std=1)
            mean = np.mean(spec)
            std = np.std(spec)
            if std > 1e-10:  # Avoid division by zero
                spec = (spec - mean) / std
        elif normalization in ['log_minmax', 'power_minmax']:
            # Min-max normalization to [0, 1]
            min_val = np.min(spec)
            max_val = np.max(spec)
            if max_val - min_val > 1e-10:
                spec = (spec - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

        return spec, freqs, times

    def generate_mel_spectrogram(
        self,
        signal_data: np.ndarray,
        n_mels: int = 128,
        fmin: float = 0,
        fmax: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Mel-scaled spectrogram (perceptually-weighted frequency scale).

        Args:
            signal_data: Input signal array
            n_mels: Number of Mel frequency bins
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz), default is fs/2

        Returns:
            Mel spectrogram, Mel frequencies, times
        """
        # Get STFT spectrogram
        power_spec, freqs, times = self.generate_stft_spectrogram(signal_data)

        if fmax is None:
            fmax = self.fs / 2

        # Create Mel filter bank
        mel_filters = self._create_mel_filters(
            n_mels=n_mels,
            n_fft=self.nfft,
            fs=self.fs,
            fmin=fmin,
            fmax=fmax
        )

        # Apply Mel filters
        mel_spec = np.dot(mel_filters, power_spec)

        # Mel frequencies (for visualization)
        mel_freqs = self._mel_frequencies(n_mels, fmin, fmax)

        return mel_spec, mel_freqs, times

    @staticmethod
    def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
        """Convert Hz to Mel scale."""
        return 2595 * np.log10(1 + frequencies / 700)

    @staticmethod
    def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
        """Convert Mel scale to Hz."""
        return 700 * (10 ** (mels / 2595) - 1)

    def _mel_frequencies(self, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
        """Generate Mel-spaced frequencies."""
        min_mel = self._hz_to_mel(np.array([fmin]))[0]
        max_mel = self._hz_to_mel(np.array([fmax]))[0]
        mels = np.linspace(min_mel, max_mel, n_mels)
        return self._mel_to_hz(mels)

    def _create_mel_filters(
        self,
        n_mels: int,
        n_fft: int,
        fs: int,
        fmin: float,
        fmax: float
    ) -> np.ndarray:
        """
        Create triangular Mel filter bank.

        Returns:
            Mel filter matrix [n_mels, n_fft//2 + 1]
        """
        # Frequency bins in STFT
        n_freqs = n_fft // 2 + 1
        fft_freqs = np.linspace(0, fs / 2, n_freqs)

        # Mel-spaced frequencies
        mel_freqs = self._mel_frequencies(n_mels + 2, fmin, fmax)

        # Create filter bank
        filters = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            # Triangular filter
            left = mel_freqs[i]
            center = mel_freqs[i + 1]
            right = mel_freqs[i + 2]

            # Rising slope
            rising_slope = (fft_freqs - left) / (center - left)
            rising_slope = np.clip(rising_slope, 0, 1)

            # Falling slope
            falling_slope = (right - fft_freqs) / (right - center)
            falling_slope = np.clip(falling_slope, 0, 1)

            # Combine slopes
            filters[i] = np.minimum(rising_slope, falling_slope)

        return filters

    def get_output_shape(self, signal_length: int) -> Tuple[int, int]:
        """
        Calculate output spectrogram shape for given signal length.

        Args:
            signal_length: Length of input signal

        Returns:
            Tuple (n_freq, n_time) representing spectrogram dimensions
        """
        # Number of frequency bins
        n_freq = self.nfft // 2 + 1

        # Number of time frames
        step = self.nperseg - self.noverlap
        n_time = (signal_length - self.noverlap) // step

        return n_freq, n_time

    def batch_generate(
        self,
        signals: np.ndarray,
        normalization: str = 'log_standardize',
        verbose: bool = True
    ) -> np.ndarray:
        """
        Generate spectrograms for a batch of signals.

        Args:
            signals: Array of signals [N, signal_length]
            normalization: Normalization method
            verbose: Print progress

        Returns:
            Spectrograms [N, n_freq, n_time]
        """
        n_signals = signals.shape[0]
        n_freq, n_time = self.get_output_shape(signals.shape[1])

        spectrograms = np.zeros((n_signals, n_freq, n_time), dtype=np.float32)

        for i in range(n_signals):
            spec, _, _ = self.generate_normalized_spectrogram(
                signals[i],
                normalization=normalization
            )
            spectrograms[i] = spec.astype(np.float32)

            if verbose and (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_signals} spectrograms")

        return spectrograms


class SpectrogramConfig:
    """Configuration class for spectrogram generation."""

    DEFAULT_BEARING_CONFIG = {
        'fs': 20480,
        'nperseg': 256,
        'noverlap': 128,
        'window': 'hann',
        'normalization': 'log_standardize'
    }

    FAST_CONFIG = {
        'fs': 20480,
        'nperseg': 128,
        'noverlap': 64,
        'window': 'hann',
        'normalization': 'log_standardize'
    }

    HIGH_RESOLUTION_CONFIG = {
        'fs': 20480,
        'nperseg': 512,
        'noverlap': 256,
        'window': 'hann',
        'normalization': 'log_standardize'
    }

    @staticmethod
    def get_config(config_name: str) -> Dict:
        """Get predefined configuration by name."""
        configs = {
            'default': SpectrogramConfig.DEFAULT_BEARING_CONFIG,
            'fast': SpectrogramConfig.FAST_CONFIG,
            'high_res': SpectrogramConfig.HIGH_RESOLUTION_CONFIG
        }

        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. "
                           f"Available: {list(configs.keys())}")

        return configs[config_name]


# Convenience functions
def generate_spectrogram(
    signal_data: np.ndarray,
    fs: int = 20480,
    nperseg: int = 256,
    normalization: str = 'log_standardize'
) -> np.ndarray:
    """
    Convenience function to generate a single spectrogram.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        nperseg: STFT segment length
        normalization: Normalization method

    Returns:
        Normalized spectrogram [n_freq, n_time]
    """
    generator = SpectrogramGenerator(fs=fs, nperseg=nperseg)
    spec, _, _ = generator.generate_normalized_spectrogram(
        signal_data,
        normalization=normalization
    )
    return spec


if __name__ == "__main__":
    # Example usage
    print("Spectrogram Generator - Example Usage\n")

    # Generate synthetic signal (5 seconds, 20.48 kHz)
    fs = 20480
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))

    # Bearing fault simulation: amplitude modulation at 100 Hz (fault frequency)
    carrier = 5000  # Hz (bearing resonance)
    modulation = 100  # Hz (fault frequency)
    signal_data = np.sin(2 * np.pi * carrier * t) * (1 + 0.3 * np.sin(2 * np.pi * modulation * t))

    # Add noise
    signal_data += 0.1 * np.random.randn(len(t))

    # Generate spectrogram
    generator = SpectrogramGenerator()

    print(f"Signal length: {len(signal_data)} samples ({duration} seconds)")
    print(f"Expected output shape: {generator.get_output_shape(len(signal_data))}")

    # Generate different types of spectrograms
    power_spec, freqs, times = generator.generate_stft_spectrogram(signal_data)
    print(f"\nPower spectrogram shape: {power_spec.shape}")

    log_spec, _, _ = generator.generate_log_spectrogram(signal_data)
    print(f"Log spectrogram shape: {log_spec.shape}")

    norm_spec, _, _ = generator.generate_normalized_spectrogram(signal_data)
    print(f"Normalized spectrogram shape: {norm_spec.shape}")
    print(f"Normalized spectrogram range: [{norm_spec.min():.2f}, {norm_spec.max():.2f}]")

    # Batch generation example
    batch_signals = np.random.randn(10, len(signal_data))
    batch_specs = generator.batch_generate(batch_signals, verbose=False)
    print(f"\nBatch spectrograms shape: {batch_specs.shape}")

    print("\n✓ Spectrogram generation successful!")
