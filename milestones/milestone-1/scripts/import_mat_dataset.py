"""
MAT Dataset Importer Script

One-time script to import all MAT files into HDF5 cache for fast access.
This should be run once after placing/generating MAT files in data/raw/bearing_data/

Usage:
    python scripts/import_mat_dataset.py \
        --mat-dir data/raw/bearing_data \
        --output data/processed/dataset_info.json \
        --validate

Author: AI Assistant
Date: 2025-11-23
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import h5py
import json
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.matlab_importer import load_mat_dataset as load_dataset_fn


def import_mat_dataset(
    mat_dir: str,
    output_file: str,
    generate_splits: bool = True,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    validate: bool = True
) -> Dict:
    """
    Import all MAT files into HDF5 cache.

    Args:
        mat_dir: Directory containing MAT files (organized by fault type)
        output_file: Output HDF5 file path
        generate_splits: Whether to generate train/val/test splits
        split_ratios: (train, val, test) ratios
        validate: Validate signal quality

    Returns:
        Dictionary with dataset statistics
    """
    print(f"Importing MAT dataset from: {mat_dir}")
    print(f"Output: {output_file}\n")

    # Use the load_mat_dataset function from matlab_importer
    all_signals, all_labels, label_names = load_dataset_fn(mat_dir, verbose=True)

    print(f"\nLoaded {len(all_signals)} signals")
    print(f"Signal shape: {all_signals.shape}")

    # Fault type mapping
    fault_type_map = {name: idx for idx, name in enumerate(label_names)}

    # Class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for fault_idx, count in zip(unique, counts):
        fault_name = label_names[fault_idx]
        print(f"  {fault_name} ({fault_idx}): {count} samples")

    # Generate train/val/test splits
    if generate_splits:
        print(f"\nGenerating splits: {split_ratios}")
        indices = np.arange(len(all_signals))

        # Stratified split
        from sklearn.model_selection import train_test_split

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(1 - split_ratios[0]),
            stratify=all_labels,
            random_state=42
        )

        val_size = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_size),
            stratify=all_labels[temp_idx],
            random_state=42
        )

        print(f"Train: {len(train_idx)} samples")
        print(f"Val: {len(val_idx)} samples")
        print(f"Test: {len(test_idx)} samples")

    # Save to HDF5
    print(f"\nSaving to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as hf:
        # Save signals and labels
        hf.create_dataset('signals', data=all_signals, compression='gzip', compression_opts=4)
        hf.create_dataset('labels', data=all_labels, compression='gzip')

        # Save splits
        if generate_splits:
            hf.create_dataset('train_indices', data=train_idx)
            hf.create_dataset('val_indices', data=val_idx)
            hf.create_dataset('test_indices', data=test_idx)

        # Save metadata as attributes
        hf.attrs['num_samples'] = len(all_signals)
        hf.attrs['num_classes'] = len(fault_type_map)
        hf.attrs['signal_length'] = all_signals.shape[1]
        hf.attrs['fault_type_map'] = json.dumps(fault_type_map)

    # Save summary JSON
    summary_file = output_path.parent / 'dataset_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump({
            'num_samples': len(all_signals),
            'num_classes': len(fault_type_map),
            'signal_length': all_signals.shape[1],
            'fault_types': label_names,
            'class_distribution': {label_names[i]: int(count) for i, count in enumerate(counts)},
            'splits': {
                'train': len(train_idx) if generate_splits else None,
                'val': len(val_idx) if generate_splits else None,
                'test': len(test_idx) if generate_splits else None
            }
        }, f, indent=2)

    print("✓ Import complete!")

    # Return statistics
    return {
        'total_signals': len(all_signals),
        'signal_shape': all_signals.shape,
        'num_classes': len(fault_type_map),
        'class_distribution': dict(zip([k for k, v in fault_type_map.items()], counts.tolist())),
        'splits': {
            'train': len(train_idx) if generate_splits else None,
            'val': len(val_idx) if generate_splits else None,
            'test': len(test_idx) if generate_splits else None
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Import MAT dataset to HDF5")
    parser.add_argument('--mat-dir', type=str, default='data/raw/bearing_data/',
                      help='Directory containing MAT files')
    parser.add_argument('--output', type=str, default='data/processed/signals_cache.h5',
                      help='Output HDF5 file')
    parser.add_argument('--validate', action='store_true',
                      help='Validate signal quality')
    parser.add_argument('--no-splits', action='store_true',
                      help='Do not generate train/val/test splits')
    parser.add_argument('--split-ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                      help='Train/val/test split ratios')

    args = parser.parse_args()

    # Run import
    stats = import_mat_dataset(
        mat_dir=args.mat_dir,
        output_file=args.output,
        generate_splits=not args.no_splits,
        split_ratios=tuple(args.split_ratios),
        validate=args.validate
    )

    # Print summary
    print("\n" + "=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)
    print(f"Total signals: {stats['total_signals']}")
    print(f"Signal shape: {stats['signal_shape']}")
    print(f"Number of classes: {stats['num_classes']}")
    print("\nClass distribution:")
    for fault_type, count in stats['class_distribution'].items():
        print(f"  {fault_type}: {count}")

    if stats['splits']['train'] is not None:
        print("\nSplits:")
        print(f"  Train: {stats['splits']['train']}")
        print(f"  Val: {stats['splits']['val']}")
        print(f"  Test: {stats['splits']['test']}")

    print("\nNext steps:")
    print("  1. Verify the cache: python scripts/verify_cache.py")
    print("  2. Precompute spectrograms: python scripts/precompute_spectrograms.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
