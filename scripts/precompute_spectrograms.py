"""
Precompute Spectrograms Script

Generates and caches STFT spectrograms, CWT scalograms, or WVD from signals cache.
This is a one-time preprocessing step that dramatically speeds up training.

Usage:
    # Precompute STFT spectrograms
    python scripts/precompute_spectrograms.py \
        --signals_cache data/processed/signals_cache.h5 \
        --output_dir data/spectrograms/stft/ \
        --tfr_type stft

    # Precompute CWT scalograms
    python scripts/precompute_spectrograms.py \
        --tfr_type cwt \
        --output_dir data/spectrograms/cwt/

Author: AI Assistant
Date: 2025-11-20
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import time
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.spectrogram_generator import SpectrogramGenerator
from data.wavelet_transform import WaveletTransform
from data.wigner_ville import WignerVilleDistribution


def precompute_spectrograms(
    signals_cache: str,
    output_dir: str,
    tfr_type: str = 'stft',
    tfr_params: Dict = None,
    normalization: str = 'log_standardize',
    batch_size: int = 100
) -> Dict:
    """
    Precompute time-frequency representations for all signals.

    Args:
        signals_cache: Path to HDF5 signals cache
        output_dir: Output directory for spectrograms
        tfr_type: Type of TFR ('stft', 'cwt', 'wvd')
        tfr_params: Parameters for TFR generation
        normalization: Normalization method
        batch_size: Number of signals to process in memory at once

    Returns:
        Dictionary with processing statistics
    """
    print(f"Precomputing {tfr_type.upper()} spectrograms")
    print(f"Input: {signals_cache}")
    print(f"Output: {output_dir}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default parameters
    if tfr_params is None:
        if tfr_type == 'stft':
            tfr_params = {'fs': 20480, 'nperseg': 256, 'noverlap': 128}
        elif tfr_type == 'cwt':
            tfr_params = {'fs': 20480, 'scales': 128}
        elif tfr_type == 'wvd':
            tfr_params = {'fs': 20480}

    # Initialize TFR generator
    if tfr_type == 'stft':
        tfr_gen = SpectrogramGenerator(**tfr_params)
    elif tfr_type == 'cwt':
        tfr_gen = WaveletTransform(**tfr_params)
    elif tfr_type == 'wvd':
        tfr_gen = WignerVilleDistribution(**tfr_params)
    else:
        raise ValueError(f"Unknown TFR type: {tfr_type}")

    # Load signals
    print("Loading signals...")
    with h5py.File(signals_cache, 'r') as hf:
        signals = hf['signals'][:]
        labels = hf['labels'][:]

        # Load splits if available
        if 'train_indices' in hf:
            train_idx = hf['train_indices'][:]
            val_idx = hf['val_indices'][:]
            test_idx = hf['test_indices'][:]
            use_splits = True
        else:
            use_splits = False

    print(f"Loaded {len(signals)} signals")

    # Get output shape
    if tfr_type == 'stft':
        sample_tfr, _, _ = tfr_gen.generate_normalized_spectrogram(
            signals[0],
            normalization=normalization
        )
    elif tfr_type == 'cwt':
        sample_tfr, _ = tfr_gen.generate_normalized_scalogram(
            signals[0],
            normalization=normalization
        )
    elif tfr_type == 'wvd':
        sample_tfr, _, _ = tfr_gen.generate_normalized_wvd(
            signals[0],
            normalization=normalization,
            smoothing='pseudo'
        )

    tfr_shape = sample_tfr.shape
    print(f"TFR shape: {tfr_shape}")

    # Process all signals
    print(f"\nGenerating {tfr_type.upper()}...")
    start_time = time.time()

    all_tfrs = np.zeros((len(signals), *tfr_shape), dtype=np.float32)

    for i in tqdm(range(0, len(signals), batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, len(signals))
        batch_signals = signals[i:batch_end]

        for j, signal in enumerate(batch_signals):
            idx = i + j

            if tfr_type == 'stft':
                tfr, _, _ = tfr_gen.generate_normalized_spectrogram(
                    signal,
                    normalization=normalization
                )
            elif tfr_type == 'cwt':
                tfr, _ = tfr_gen.generate_normalized_scalogram(
                    signal,
                    normalization=normalization
                )
            elif tfr_type == 'wvd':
                tfr, _, _ = tfr_gen.generate_normalized_wvd(
                    signal,
                    normalization=normalization,
                    smoothing='pseudo'
                )

            all_tfrs[idx] = tfr.astype(np.float32)

    elapsed_time = time.time() - start_time
    print(f"\nProcessing time: {elapsed_time:.2f} seconds")
    print(f"Time per signal: {elapsed_time / len(signals) * 1000:.2f} ms")

    # Save spectrograms
    if use_splits:
        print("\nSaving spectrograms (split by train/val/test)...")

        # Train set
        train_file = output_path / 'train_spectrograms.npz'
        np.savez_compressed(
            train_file,
            spectrograms=all_tfrs[train_idx],
            labels=labels[train_idx]
        )
        print(f"  Train: {train_file} ({len(train_idx)} samples)")

        # Val set
        val_file = output_path / 'val_spectrograms.npz'
        np.savez_compressed(
            val_file,
            spectrograms=all_tfrs[val_idx],
            labels=labels[val_idx]
        )
        print(f"  Val: {val_file} ({len(val_idx)} samples)")

        # Test set
        test_file = output_path / 'test_spectrograms.npz'
        np.savez_compressed(
            test_file,
            spectrograms=all_tfrs[test_idx],
            labels=labels[test_idx]
        )
        print(f"  Test: {test_file} ({len(test_idx)} samples)")

    else:
        print("\nSaving all spectrograms...")
        all_file = output_path / 'all_spectrograms.npz'
        np.savez_compressed(
            all_file,
            spectrograms=all_tfrs,
            labels=labels
        )
        print(f"  Saved: {all_file}")

    # Save metadata
    import json
    metadata = {
        'tfr_type': tfr_type,
        'tfr_params': tfr_params,
        'normalization': normalization,
        'tfr_shape': list(tfr_shape),
        'num_samples': len(signals),
        'processing_time_seconds': elapsed_time,
        'splits': {
            'train': len(train_idx) if use_splits else None,
            'val': len(val_idx) if use_splits else None,
            'test': len(test_idx) if use_splits else None
        }
    }

    metadata_file = output_path / 'tfr_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved: {metadata_file}")
    print("\n✓ Precomputation complete!")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Precompute spectrograms/scalograms")
    parser.add_argument('--signals_cache', type=str,
                      default='data/processed/signals_cache.h5',
                      help='Path to signals HDF5 cache')
    parser.add_argument('--output_dir', type=str,
                      default='data/spectrograms/stft/',
                      help='Output directory for spectrograms')
    parser.add_argument('--tfr_type', type=str, default='stft',
                      choices=['stft', 'cwt', 'wvd'],
                      help='Type of time-frequency representation')
    parser.add_argument('--normalization', type=str, default='log_standardize',
                      help='Normalization method')
    parser.add_argument('--batch_size', type=int, default=100,
                      help='Batch size for processing')

    # STFT-specific parameters
    parser.add_argument('--nperseg', type=int, default=256,
                      help='STFT segment length')
    parser.add_argument('--noverlap', type=int, default=128,
                      help='STFT overlap')

    # CWT-specific parameters
    parser.add_argument('--scales', type=int, default=128,
                      help='Number of CWT scales')
    parser.add_argument('--wavelet', type=str, default='morl',
                      help='Wavelet type for CWT')

    args = parser.parse_args()

    # Build TFR parameters
    if args.tfr_type == 'stft':
        tfr_params = {
            'fs': 20480,
            'nperseg': args.nperseg,
            'noverlap': args.noverlap
        }
    elif args.tfr_type == 'cwt':
        tfr_params = {
            'fs': 20480,
            'scales': args.scales,
            'wavelet': args.wavelet
        }
    elif args.tfr_type == 'wvd':
        tfr_params = {
            'fs': 20480
        }

    # Run precomputation
    metadata = precompute_spectrograms(
        signals_cache=args.signals_cache,
        output_dir=args.output_dir,
        tfr_type=args.tfr_type,
        tfr_params=tfr_params,
        normalization=args.normalization,
        batch_size=args.batch_size
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PRECOMPUTATION SUMMARY")
    print("=" * 60)
    print(f"TFR type: {metadata['tfr_type'].upper()}")
    print(f"TFR shape: {metadata['tfr_shape']}")
    print(f"Number of samples: {metadata['num_samples']}")
    print(f"Processing time: {metadata['processing_time_seconds']:.2f} seconds")

    if metadata['splits']['train'] is not None:
        print("\nSplits:")
        print(f"  Train: {metadata['splits']['train']}")
        print(f"  Val: {metadata['splits']['val']}")
        print(f"  Test: {metadata['splits']['test']}")

    print("\nNext steps:")
    print(f"  1. Train 2D CNN: python scripts/train_spectrogram_cnn.py \\")
    print(f"       --spectrogram_dir {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
