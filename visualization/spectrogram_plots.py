"""
Spectrogram Visualization Utilities

Provides plotting functions for spectrograms, scalograms, and Wigner-Ville distributions.

Features:
- Side-by-side comparison of different TFR types
- Per-fault spectrogram visualization
- Time-frequency evolution plots
- Spectrogram with fault annotations

Usage:
    from visualization.spectrogram_plots import plot_spectrogram_comparison

    # Compare STFT, CWT, WVD for a signal
    plot_spectrogram_comparison(
        signal=signal,
        fs = SAMPLING_RATE,
        save_path='tfr_comparison.png'
    )

    # Visualize all fault types
    plot_fault_spectrograms_grid(
        signals_by_fault=fault_dict,
        fs = SAMPLING_RATE,
        save_path='fault_spectrograms.png'
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.spectrogram_generator import SpectrogramGenerator
from data.wavelet_transform import WaveletTransform
from data.wigner_ville import generate_wvd
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def plot_spectrogram(
    spectrogram: np.ndarray,
    fs: float = 20480,
    title: str = 'Spectrogram',
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    colorbar: bool = True
) -> plt.Axes:
    """
    Plot a single spectrogram.

    Args:
        spectrogram: Spectrogram array [H, W]
        fs: Sampling frequency (default: 20480 Hz)
        title: Plot title
        ax: Optional matplotlib axes
        cmap: Colormap (default: 'viridis')
        colorbar: Whether to add colorbar (default: True)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    H, W = spectrogram.shape

    # Estimate time and frequency axes
    duration = 5.0  # Assuming 5-second signals
    max_freq = fs / 2

    im = ax.imshow(
        spectrogram,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        extent=[0, duration, 0, max_freq],
        interpolation='bilinear'
    )

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Magnitude (dB)', fontsize=10)

    return ax


def plot_spectrogram_comparison(
    signal: np.ndarray,
    fs: float = 20480,
    tfr_types: List[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Compare different time-frequency representations side-by-side.

    Args:
        signal: Time-domain signal [T]
        fs: Sampling frequency
        tfr_types: List of TFR types to compare (default: ['STFT', 'CWT', 'WVD'])
        save_path: Optional path to save figure
        figsize: Figure size
    """
    if tfr_types is None:
        tfr_types = ['STFT', 'CWT', 'WVD']

    num_tfrs = len(tfr_types)
    fig, axes = plt.subplots(num_tfrs, 1, figsize=figsize)

    if num_tfrs == 1:
        axes = [axes]

    for ax, tfr_type in zip(axes, tfr_types):
        if tfr_type == 'STFT':
            gen = SpectrogramGenerator(fs=fs)
            spec, f, t = gen.generate_stft_spectrogram(signal)
            title = 'STFT Spectrogram'

        elif tfr_type == 'Mel':
            gen = SpectrogramGenerator(fs=fs)
            spec, f, t = gen.generate_mel_spectrogram(signal)
            title = 'Mel Spectrogram'

        elif tfr_type == 'CWT':
            cwt = WaveletTransform(wavelet='morl', scales=128, fs=fs)
            spec, freqs = cwt.generate_cwt_scalogram(signal)
            title = 'CWT Scalogram'

        elif tfr_type == 'WVD':
            spec = generate_wvd(signal, fs)
            title = 'Wigner-Ville Distribution'

        else:
            raise ValueError(f"Unknown TFR type: {tfr_type}")

        plot_spectrogram(spec, fs, title, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved TFR comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_fault_spectrograms_grid(
    signals_by_fault: Dict[str, np.ndarray],
    fs: float = 20480,
    tfr_type: str = 'STFT',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 15)
):
    """
    Plot spectrograms for all fault types in a grid.

    Args:
        signals_by_fault: Dictionary of {fault_name: signal}
        fs: Sampling frequency
        tfr_type: Type of TFR to use (default: 'STFT')
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fault_names = list(signals_by_fault.keys())
    num_faults = len(fault_names)

    # Create grid layout
    ncols = 4
    nrows = (num_faults + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, fault_name in enumerate(fault_names):
        signal = signals_by_fault[fault_name]

        # Generate spectrogram
        if tfr_type == 'STFT':
            spec, f, t = generate_stft_spectrogram(signal, fs)
        elif tfr_type == 'CWT':
            spec, freqs = generate_cwt_scalogram(signal, fs)
        elif tfr_type == 'WVD':
            spec = generate_wvd(signal, fs)
        else:
            raise ValueError(f"Unknown TFR type: {tfr_type}")

        # Plot
        plot_spectrogram(
            spec,
            fs,
            title=fault_name,
            ax=axes[i],
            colorbar=False
        )

    # Hide unused subplots
    for i in range(num_faults, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{tfr_type} Spectrograms for All Fault Types', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fault spectrograms grid to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_spectrogram_with_prediction(
    signal: np.ndarray,
    fs: float,
    true_label: str,
    predicted_label: str,
    confidence: float,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot signal and spectrogram with prediction information.

    Args:
        signal: Time-domain signal
        fs: Sampling frequency
        true_label: Ground truth label
        predicted_label: Predicted label
        confidence: Prediction confidence (0-1)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)

    # Plot signal
    ax1 = fig.add_subplot(gs[0])
    time = np.arange(len(signal)) / fs
    ax1.plot(time, signal, linewidth=0.5, color='steelblue')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Time-Domain Signal', fontsize=12, weight='bold')
    ax1.grid(alpha=0.3)

    # Plot spectrogram
    ax2 = fig.add_subplot(gs[1])
    spec, f, t = generate_stft_spectrogram(signal, fs)
    plot_spectrogram(spec, fs, title='STFT Spectrogram', ax=ax2)

    # Add prediction text
    color = 'green' if true_label == predicted_label else 'red'
    pred_text = f"True: {true_label}\nPredicted: {predicted_label}\nConfidence: {confidence*100:.1f}%"

    ax2.text(
        0.02, 0.98, pred_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2)
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_frequency_evolution(
    signal: np.ndarray,
    fs: float,
    freq_bands: List[Tuple[float, float]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot frequency evolution over time for specific frequency bands.

    Args:
        signal: Time-domain signal
        fs: Sampling frequency
        freq_bands: List of (low_freq, high_freq) tuples to track
        save_path: Optional path to save figure
        figsize: Figure size
    """
    if freq_bands is None:
        # Default: bearing fault frequency bands
        freq_bands = [
            (0, 500),       # Low frequency (imbalance, misalignment)
            (500, 2000),    # Medium frequency (bearing faults)
            (2000, 5000),   # High frequency (oil whirl, cavitation)
        ]

    # Generate spectrogram
    spec, f, t = generate_stft_spectrogram(signal, fs)

    fig, axes = plt.subplots(len(freq_bands) + 1, 1, figsize=figsize)

    # Plot full spectrogram
    plot_spectrogram(spec, fs, title='Full Spectrogram', ax=axes[0])

    # Plot frequency band evolution
    for i, (low_freq, high_freq) in enumerate(freq_bands):
        ax = axes[i + 1]

        # Find frequency bin indices
        freq_idx = np.where((f >= low_freq) & (f <= high_freq))[0]

        # Extract and average energy in this band
        band_energy = spec[freq_idx, :].mean(axis=0)

        # Reconstruct time axis
        time = np.linspace(0, len(signal) / fs, len(band_energy))

        ax.plot(time, band_energy, linewidth=2, color='steelblue')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Energy (dB)', fontsize=11)
        ax.set_title(f'Frequency Band: {low_freq}-{high_freq} Hz', fontsize=12, weight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved frequency evolution to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_spectrogram_statistics(
    spectrograms: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Plot statistical properties of spectrograms per class.

    Args:
        spectrograms: Array of spectrograms [N, H, W]
        labels: Class labels [N]
        class_names: List of class names
        save_path: Optional path to save figure
        figsize: Figure size
    """
    num_classes = len(class_names)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Mean spectrogram per class
    ax = axes[0, 0]
    mean_specs = [spectrograms[labels == i].mean(axis=0) for i in range(num_classes)]

    for i, (mean_spec, class_name) in enumerate(zip(mean_specs, class_names)):
        ax.plot(mean_spec.mean(axis=0), label=class_name, alpha=0.7)

    ax.set_xlabel('Time Bin', fontsize=11)
    ax.set_ylabel('Average Magnitude', fontsize=11)
    ax.set_title('Average Temporal Profile per Class', fontsize=12, weight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Frequency distribution
    ax = axes[0, 1]
    for i, (mean_spec, class_name) in enumerate(zip(mean_specs, class_names)):
        ax.plot(mean_spec.mean(axis=1), label=class_name, alpha=0.7)

    ax.set_xlabel('Frequency Bin', fontsize=11)
    ax.set_ylabel('Average Magnitude', fontsize=11)
    ax.set_title('Average Frequency Profile per Class', fontsize=12, weight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Energy distribution
    ax = axes[1, 0]
    energies = [spectrograms[labels == i].mean(axis=(1, 2)) for i in range(num_classes)]

    positions = np.arange(num_classes)
    bp = ax.boxplot(energies, positions=positions, labels=class_names, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel('Fault Type', fontsize=11)
    ax.set_ylabel('Average Energy', fontsize=11)
    ax.set_title('Energy Distribution per Class', fontsize=12, weight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # 4. Variance distribution
    ax = axes[1, 1]
    variances = [spectrograms[labels == i].var(axis=(1, 2)) for i in range(num_classes)]

    bp = ax.boxplot(variances, positions=positions, labels=class_names, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')

    ax.set_xlabel('Fault Type', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Variance Distribution per Class', fontsize=12, weight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectrogram statistics to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Test visualization
    print("Testing spectrogram visualization...")

    # Generate test signal
    fs = SAMPLING_RATE
    duration = 5.0
    t = np.arange(0, duration, 1/fs)

    # Simulated signal with multiple frequencies
    signal = (
        np.sin(2 * np.pi * 60 * t) +  # Low frequency
        0.5 * np.sin(2 * np.pi * 1000 * t) +  # Medium frequency
        0.3 * np.sin(2 * np.pi * 3000 * t) +  # High frequency
        0.1 * np.random.randn(len(t))  # Noise
    )

    # Test comparison plot
    plot_spectrogram_comparison(signal, fs)

    print("Visualization tests complete!")
