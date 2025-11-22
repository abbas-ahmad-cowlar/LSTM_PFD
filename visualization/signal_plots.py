"""
Signal visualization tools.

Purpose:
    Visualize time-domain signals, frequency spectra, spectrograms.
    Reproduces Figures 2, 3 from technical report.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class SignalPlotter:
    """Signal visualization utilities."""

    @staticmethod
    def plot_signal_examples(signals: np.ndarray, labels: np.ndarray,
                            fs: float, n_examples: int = 3,
                            save_path: Optional[Path] = None):
        """
        Plot example signals in time and frequency domain (Figures 2, 3).

        Args:
            signals: Signal array (n_samples, signal_length)
            labels: Label array
            fs: Sampling frequency
            n_examples: Number of examples to plot
            save_path: Optional path to save figure
        """
        unique_labels = np.unique(labels)
        n_classes = min(len(unique_labels), n_examples)

        fig, axes = plt.subplots(n_classes, 3, figsize=(15, 3 * n_classes))
        if n_classes == 1:
            axes = axes[np.newaxis, :]

        for i, label in enumerate(unique_labels[:n_classes]):
            # Get first signal of this class
            idx = np.where(labels == label)[0][0]
            signal = signals[idx]
            t = np.arange(len(signal)) / fs

            # Time domain
            axes[i, 0].plot(t, signal, linewidth=0.5)
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].set_title(f'Class {label} - Time Domain')
            axes[i, 0].grid(alpha=0.3)

            # Frequency domain
            N = len(signal)
            fft_vals = fft(signal)
            freqs = fftfreq(N, 1.0 / fs)[:N // 2]
            magnitude = 2.0 / N * np.abs(fft_vals[:N // 2])

            axes[i, 1].plot(freqs, magnitude, linewidth=0.5)
            axes[i, 1].set_xlabel('Frequency (Hz)')
            axes[i, 1].set_ylabel('Magnitude')
            axes[i, 1].set_title(f'Class {label} - Frequency Domain')
            axes[i, 1].grid(alpha=0.3)
            axes[i, 1].set_xlim([0, fs / 4])  # Show up to Nyquist/2

            # Spectrogram
            f, t_spec, Sxx = spectrogram(signal, fs, nperseg=256)
            im = axes[i, 2].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                                      shading='gouraud', cmap='viridis')
            axes[i, 2].set_xlabel('Time (s)')
            axes[i, 2].set_ylabel('Frequency (Hz)')
            axes[i, 2].set_title(f'Class {label} - Spectrogram')
            plt.colorbar(im, ax=axes[i, 2], label='Power (dB)')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_signal_comparison(signal1: np.ndarray, signal2: np.ndarray,
                              fs: float, labels: List[str],
                              save_path: Optional[Path] = None):
        """
        Compare two signals side-by-side.

        Args:
            signal1: First signal
            signal2: Second signal
            fs: Sampling frequency
            labels: Labels for the two signals
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        signals = [signal1, signal2]
        for i, (signal, label) in enumerate(zip(signals, labels)):
            t = np.arange(len(signal)) / fs

            # Time domain
            axes[i, 0].plot(t, signal, linewidth=0.5)
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].set_title(f'{label} - Time Domain')
            axes[i, 0].grid(alpha=0.3)

            # Frequency domain
            N = len(signal)
            fft_vals = fft(signal)
            freqs = fftfreq(N, 1.0 / fs)[:N // 2]
            magnitude = 2.0 / N * np.abs(fft_vals[:N // 2])

            axes[i, 1].plot(freqs, magnitude, linewidth=0.5)
            axes[i, 1].set_xlabel('Frequency (Hz)')
            axes[i, 1].set_ylabel('Magnitude')
            axes[i, 1].set_title(f'{label} - Frequency Domain')
            axes[i, 1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
