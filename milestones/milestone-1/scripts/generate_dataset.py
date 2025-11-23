#!/usr/bin/env python3
"""
Data Generation Script for Milestone 1: CNN-Based Bearing Fault Diagnosis

Generates synthetic bearing vibration signals in .mat format for CNN training.

Features:
- 11 fault classes (Healthy + 10 fault types)
- Configurable number of samples per class
- Physics-based fault models
- 7-layer realistic noise model
- Variable operating conditions
- Data augmentation
- Output in MATLAB .mat format

Usage:
    # Quick generation (10 samples per class, 110 total):
    python scripts/generate_dataset.py --quick

    # Standard generation (130 samples per class, 1430 total):
    python scripts/generate_dataset.py

    # Custom generation:
    python scripts/generate_dataset.py --samples-per-class 200 --output-dir data/custom

    # Minimal generation for testing (5 samples):
    python scripts/generate_dataset.py --minimal

Author: Bearing Fault Diagnosis Team
Milestone: 1 - CNN-Based Approach
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import argparse
from scipy.io import savemat
from scipy import signal as sp_signal
import time
from typing import Tuple, Dict, Any
import random


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLING_RATE = 20480  # Hz (20.48 kHz)
SIGNAL_DURATION = 5.0  # seconds
NUM_SAMPLES = int(SAMPLING_RATE * SIGNAL_DURATION)  # 102,400 samples
BASE_SPEED_RPM = 3600  # RPM
BASE_SPEED_HZ = BASE_SPEED_RPM / 60  # 60 Hz

# Fault type mapping (Python name -> MATLAB-compatible label)
FAULT_CLASSES = {
    'sain': 'Healthy',
    'desalignement': 'Misalignment',
    'desequilibre': 'Imbalance',
    'jeu': 'Bearing_Clearance',
    'lubrification': 'Lubrication_Issue',
    'cavitation': 'Cavitation',
    'usure': 'Wear',
    'oilwhirl': 'Oil_Whirl',
    'mixed_misalign_imbalance': 'Mixed_Fault_1',
    'mixed_wear_lube': 'Mixed_Fault_2',
    'mixed_cavit_jeu': 'Mixed_Fault_3',
}


# =============================================================================
# SIGNAL GENERATION FUNCTIONS
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def generate_fault_signal(fault_type: str, N: int, fs: int, severity: float = 0.7) -> np.ndarray:
    """
    Generate fault-specific vibration signature.

    Args:
        fault_type: Fault class name
        N: Number of samples
        fs: Sampling frequency
        severity: Fault severity factor (0.0-1.0)

    Returns:
        Fault signal array [N,]
    """
    t = np.arange(N) / fs
    Omega = BASE_SPEED_HZ * (1 + (np.random.rand() - 0.5) * 0.1)  # ±5% speed variation
    omega = 2 * np.pi * Omega

    if fault_type == 'sain':
        # Healthy: minimal vibration
        return 0.05 * np.random.randn(N)

    elif fault_type == 'desalignement':
        # Misalignment: 2X and 3X harmonics
        phase_2X = np.random.rand() * 2 * np.pi
        phase_3X = np.random.rand() * 2 * np.pi
        misalign_2X = 0.35 * severity * np.sin(2 * omega * t + phase_2X)
        misalign_3X = 0.20 * severity * np.sin(3 * omega * t + phase_3X)
        return misalign_2X + misalign_3X

    elif fault_type == 'desequilibre':
        # Imbalance: 1X dominant, speed-squared dependence
        phase_1X = np.random.rand() * 2 * np.pi
        imbalance_1X = 0.5 * severity * np.sin(omega * t + phase_1X)
        return imbalance_1X

    elif fault_type == 'jeu':
        # Bearing clearance: sub-synchronous + harmonics
        sub_freq = (0.43 + 0.05 * np.random.rand()) * Omega
        clearance_sub = 0.25 * severity * np.sin(2 * np.pi * sub_freq * t)
        clearance_1X = 0.18 * severity * np.sin(omega * t)
        clearance_2X = 0.10 * severity * np.sin(2 * omega * t)
        return clearance_sub + clearance_1X + clearance_2X

    elif fault_type == 'lubrification':
        # Lubrication: stick-slip + metal contact events
        stick_slip_freq = 2 + 3 * np.random.rand()
        stick_slip = 0.30 * severity * np.sin(2 * np.pi * stick_slip_freq * t)

        x_fault = stick_slip.copy()

        # Add impact events
        impact_rate = int(1 + 3 * severity)
        for _ in range(impact_rate):
            impact_pos = np.random.randint(0, max(1, N - 20))
            impact_amp = 0.5 * severity
            impact_len = min(20, N - impact_pos)
            decay = np.exp(-0.4 * np.arange(impact_len))
            x_fault[impact_pos:impact_pos + impact_len] += impact_amp * decay * np.random.randn(impact_len)

        return x_fault

    elif fault_type == 'cavitation':
        # Cavitation: high-frequency bursts
        x_fault = np.zeros(N)
        burst_rate = int(2 + 5 * severity)
        burst_len = int(0.008 * fs)  # 8ms bursts

        for _ in range(burst_rate):
            if burst_len >= N:
                continue
            pos = np.random.randint(0, N - burst_len)
            burst_freq = 1500 + 1000 * np.random.rand()
            burst_t = np.arange(burst_len) / fs
            hann_window = sp_signal.windows.hann(burst_len)
            burst = (0.6 * severity * np.sin(2 * np.pi * burst_freq * burst_t) *
                    np.exp(-100 * burst_t) * hann_window)
            x_fault[pos:pos + burst_len] += burst

        return x_fault

    elif fault_type == 'usure':
        # Wear: broadband noise + amplitude modulation
        wear_noise = 0.25 * severity * np.random.randn(N)
        asperity_harm = 0.12 * severity * (np.sin(omega * t) + 0.5 * np.sin(2 * omega * t))
        wear_mod_freq = 0.5 + 1.5 * np.random.rand()
        wear_mod = 1 + 0.3 * np.sin(2 * np.pi * wear_mod_freq * t)
        return (wear_noise + asperity_harm) * wear_mod

    elif fault_type == 'oilwhirl':
        # Oil whirl: sub-synchronous oscillation
        whirl_freq_ratio = 0.42 + 0.06 * np.random.rand()
        whirl_freq = whirl_freq_ratio * Omega
        whirl_amp = 0.40 * severity
        whirl_signal = whirl_amp * np.sin(2 * np.pi * whirl_freq * t)
        subsync_mod_freq = whirl_freq * 0.5
        subsync_mod = 1 + 0.2 * np.sin(2 * np.pi * subsync_mod_freq * t)
        return whirl_signal * subsync_mod

    elif fault_type == 'mixed_misalign_imbalance':
        # Mixed: Misalignment + Imbalance
        phase_2X = np.random.rand() * 2 * np.pi
        phase_3X = np.random.rand() * 2 * np.pi
        misalign_2X = 0.25 * severity * np.sin(2 * omega * t + phase_2X)
        misalign_3X = 0.15 * severity * np.sin(3 * omega * t + phase_3X)

        phase_1X = np.random.rand() * 2 * np.pi
        imbalance_1X = 0.35 * severity * np.sin(omega * t + phase_1X)

        return misalign_2X + misalign_3X + imbalance_1X

    elif fault_type == 'mixed_wear_lube':
        # Mixed: Wear + Lubrication
        wear_noise = 0.18 * severity * np.random.randn(N)
        asperity_harm = 0.08 * severity * (np.sin(omega * t) + 0.5 * np.sin(2 * omega * t))

        stick_slip_freq = 2 + 3 * np.random.rand()
        stick_slip = 0.20 * severity * np.sin(2 * np.pi * stick_slip_freq * t)

        x_fault = wear_noise + asperity_harm + stick_slip

        # Add contact events
        contact_rate = int(2 + 3 * severity)
        for _ in range(contact_rate):
            contact_pos = np.random.randint(0, max(1, N - 10))
            contact_amp = 0.4 * severity
            contact_len = min(10, N - contact_pos)
            decay = np.exp(-0.5 * np.arange(contact_len))
            x_fault[contact_pos:contact_pos + contact_len] += contact_amp * decay * np.random.randn(contact_len)

        return x_fault

    elif fault_type == 'mixed_cavit_jeu':
        # Mixed: Cavitation + Clearance
        x_fault = np.zeros(N)

        # Cavitation bursts
        burst_rate = int(3 + 4 * severity)
        burst_len = int(0.008 * fs)
        for _ in range(burst_rate):
            if burst_len >= N:
                continue
            pos = np.random.randint(0, N - burst_len)
            burst_freq = 1500 + 1000 * np.random.rand()
            burst_t = np.arange(burst_len) / fs
            hann_window = sp_signal.windows.hann(burst_len)
            burst = (0.5 * severity * np.sin(2 * np.pi * burst_freq * burst_t) *
                    np.exp(-100 * burst_t) * hann_window)
            x_fault[pos:pos + burst_len] += burst

        # Clearance components
        sub_freq = (0.43 + 0.05 * np.random.rand()) * Omega
        clearance_sub = 0.22 * severity * np.sin(2 * np.pi * sub_freq * t)
        clearance_1X = 0.15 * severity * np.sin(omega * t)

        x_fault += clearance_sub + clearance_1X
        return x_fault

    else:
        raise ValueError(f"Unknown fault type: {fault_type}")


def apply_noise_layers(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Apply 7-layer realistic noise model.

    Args:
        signal: Clean signal [N,]
        fs: Sampling frequency

    Returns:
        Noisy signal [N,]
    """
    N = len(signal)
    t = np.arange(N) / fs
    x_noisy = signal.copy()

    # 1. Measurement noise (sensor electronics)
    x_noisy += 0.03 * np.random.randn(N)

    # 2. EMI (electromagnetic interference - 50/60 Hz)
    emi_freq = 50 + 10 * np.random.rand()
    emi_amp = 0.01 * (1 + 0.5 * np.random.rand())
    emi_signal = emi_amp * np.sin(2 * np.pi * emi_freq * t + np.random.rand() * 2 * np.pi)
    x_noisy += emi_signal

    # 3. Pink noise (1/f)
    pink_noise = np.cumsum(np.random.randn(N))
    pink_noise = 0.02 * (pink_noise / (np.std(pink_noise) + 1e-10))
    x_noisy += pink_noise

    # 4. Environmental drift
    drift = 0.015 * np.sin(2 * np.pi * (1 / 1.5) * t)
    x_noisy += drift

    # 5. Quantization noise (ADC resolution)
    quant_step = 0.001
    x_noisy = np.round(x_noisy / quant_step) * quant_step

    # 6. Sensor drift (cumulative offset)
    sensor_drift_rate = 0.001 / (N / fs)
    sensor_offset = sensor_drift_rate * t
    x_noisy += sensor_offset

    # 7. Aliasing artifacts (10% probability)
    if np.random.rand() < 0.10:
        alias_freq = fs / 2 + 100 + 200 * np.random.rand()
        alias_signal = 0.005 * np.sin(2 * np.pi * alias_freq * t)
        x_noisy += alias_signal

    # 8. Impulse noise (sporadic impacts)
    num_impulses = int(2 * (N / fs))  # 2 impulses per second
    for _ in range(num_impulses):
        imp_pos = np.random.randint(0, max(1, N - 5))
        imp_amp = 0.02 + 0.03 * np.random.rand()
        imp_len = min(5, N - imp_pos)
        decay = np.exp(-0.3 * np.arange(imp_len))
        x_noisy[imp_pos:imp_pos + imp_len] += imp_amp * decay * np.random.randn(imp_len)

    return x_noisy


def generate_single_sample(fault_type: str, severity: float = None) -> Tuple[np.ndarray, Dict]:
    """
    Generate a single vibration signal sample.

    Args:
        fault_type: Fault class name
        severity: Fault severity (0.0-1.0), random if None

    Returns:
        Tuple of (signal, metadata)
    """
    # Random severity if not specified
    if severity is None:
        severity = 0.5 + 0.5 * np.random.rand()  # 0.5 to 1.0

    # Initialize baseline signal
    x = 0.05 * np.random.randn(NUM_SAMPLES)

    # Apply noise layers to baseline
    x = apply_noise_layers(x, SAMPLING_RATE)

    # Generate and add fault-specific signature
    x_fault = generate_fault_signal(fault_type, NUM_SAMPLES, SAMPLING_RATE, severity)
    x += x_fault

    # Metadata
    metadata = {
        'fault_type': fault_type,
        'fault_label': FAULT_CLASSES[fault_type],
        'severity': float(severity),
        'fs': SAMPLING_RATE,
        'duration_s': SIGNAL_DURATION,
        'num_samples': NUM_SAMPLES,
        'speed_rpm': BASE_SPEED_RPM,
        'generation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return x, metadata


def apply_augmentation(signal: np.ndarray, method: str) -> np.ndarray:
    """
    Apply data augmentation to signal.

    Args:
        signal: Input signal
        method: Augmentation method ('time_shift', 'amplitude_scale', 'noise_injection')

    Returns:
        Augmented signal
    """
    signal_aug = signal.copy()

    if method == 'time_shift':
        shift_max = int(0.02 * len(signal))  # 2% shift
        shift_samples = np.random.randint(-shift_max, shift_max)
        signal_aug = np.roll(signal_aug, shift_samples)

    elif method == 'amplitude_scale':
        scale_factor = 0.85 + (1.15 - 0.85) * np.random.rand()
        signal_aug = signal_aug * scale_factor

    elif method == 'noise_injection':
        extra_noise_level = 0.02 + (0.05 - 0.02) * np.random.rand()
        extra_noise = extra_noise_level * np.random.randn(len(signal))
        signal_aug += extra_noise

    return signal_aug


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_dataset(
    samples_per_class: int = 130,
    output_dir: str = 'data/raw/bearing_data',
    augmentation_ratio: float = 0.0,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate complete bearing fault dataset.

    Args:
        samples_per_class: Number of base samples per fault class
        output_dir: Output directory for .mat files
        augmentation_ratio: Ratio of augmented samples (0.0 to 1.0)
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Dictionary with generation statistics
    """
    # Set seed
    set_seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("  BEARING FAULT DATASET GENERATION - MILESTONE 1")
        print("=" * 70)
        print(f"Output directory: {output_path}")
        print(f"Samples per class: {samples_per_class}")
        print(f"Augmentation ratio: {augmentation_ratio:.1%}")
        print(f"Fault classes: {len(FAULT_CLASSES)}")
        print(f"Random seed: {seed}")
        print("=" * 70)

    start_time = time.time()
    total_signals = 0
    fault_counts = {}

    for fault_type, fault_label in FAULT_CLASSES.items():
        if verbose:
            print(f"\nGenerating fault: {fault_label} ({fault_type})")

        # Create subdirectory for this fault type
        fault_dir = output_path / fault_type
        fault_dir.mkdir(parents=True, exist_ok=True)

        # Calculate samples for this fault
        num_base = samples_per_class
        num_augmented = int(num_base * augmentation_ratio) if augmentation_ratio > 0 else 0
        num_total = num_base + num_augmented

        for i in range(num_total):
            # Generate signal
            is_augmented = (i >= num_base)

            if is_augmented:
                # Generate base signal then augment
                signal, metadata = generate_single_sample(fault_type)
                aug_method = np.random.choice(['time_shift', 'amplitude_scale', 'noise_injection'])
                signal = apply_augmentation(signal, aug_method)
                metadata['is_augmented'] = True
                metadata['augmentation_method'] = aug_method
            else:
                signal, metadata = generate_single_sample(fault_type)
                metadata['is_augmented'] = False

            # Determine filename
            sample_idx = i + 1
            if is_augmented:
                filename = f"sample_{sample_idx:03d}_aug.mat"
            else:
                filename = f"sample_{sample_idx:03d}.mat"

            filepath = fault_dir / filename

            # Prepare MATLAB-compatible structure
            mat_data = {
                'x': signal,
                'fs': SAMPLING_RATE,
                'fault': fault_label,
                'metadata': metadata
            }

            # Save .mat file
            savemat(filepath, mat_data, do_compression=True)

            total_signals += 1

        fault_counts[fault_label] = num_total

        if verbose:
            print(f"  ✓ Generated {num_total} signals ({num_base} base + {num_augmented} augmented)")

    generation_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("  GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total signals: {total_signals}")
        print(f"Total faults: {len(FAULT_CLASSES)}")
        print(f"Generation time: {generation_time:.2f} s ({total_signals/generation_time:.2f} signals/s)")
        print(f"Output directory: {output_path}")
        print("=" * 70)
        print("\nDataset structure:")
        print(f"{output_path}/")
        for fault_type in FAULT_CLASSES.keys():
            fault_dir = output_path / fault_type
            if fault_dir.exists():
                num_files = len(list(fault_dir.glob('*.mat')))
                print(f"  ├── {fault_type}/ ({num_files} files)")
        print("\n" + "=" * 70)
        print("Per-class breakdown:")
        for fault_label, count in fault_counts.items():
            print(f"  {fault_label:<25} {count:>5} signals")
        print("=" * 70)

    return {
        'total_signals': total_signals,
        'num_classes': len(FAULT_CLASSES),
        'generation_time_s': generation_time,
        'signals_per_second': total_signals / generation_time,
        'fault_counts': fault_counts,
        'output_dir': str(output_path)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate bearing fault dataset for Milestone 1 (CNN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick generation (10 samples per class):
  python scripts/generate_dataset.py --quick

  # Standard generation (130 samples per class, 1430 total):
  python scripts/generate_dataset.py

  # Custom generation:
  python scripts/generate_dataset.py --samples-per-class 200 --output-dir data/custom

  # With augmentation (30% more samples):
  python scripts/generate_dataset.py --augmentation-ratio 0.3

  # Minimal for testing (5 samples per class):
  python scripts/generate_dataset.py --minimal

Output:
  .mat files organized by fault type in subdirectories:
    data/raw/bearing_data/
    ├── sain/                    # Healthy
    │   ├── sample_001.mat
    │   ├── sample_002.mat
    │   └── ...
    ├── desalignement/           # Misalignment
    ├── desequilibre/            # Imbalance
    └── [other fault types]/

  Each .mat file contains:
    - x: vibration signal (102,400 samples at 20.48 kHz)
    - fs: sampling frequency
    - fault: fault class label
    - metadata: generation parameters
        """
    )

    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=130,
        help='Number of samples per fault class (default: 130 for 1430 total)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/bearing_data',
        help='Output directory for .mat files (default: data/raw/bearing_data)'
    )

    parser.add_argument(
        '--augmentation-ratio',
        type=float,
        default=0.0,
        help='Augmentation ratio (0.0-1.0). 0.3 means 30%% more samples (default: 0.0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick generation: 10 samples per class (110 total)'
    )

    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Minimal generation for testing: 5 samples per class (55 total)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Handle presets
    if args.quick:
        args.samples_per_class = 10
    elif args.minimal:
        args.samples_per_class = 5

    # Generate dataset
    stats = generate_dataset(
        samples_per_class=args.samples_per_class,
        output_dir=args.output_dir,
        augmentation_ratio=args.augmentation_ratio,
        seed=args.seed,
        verbose=not args.quiet
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
