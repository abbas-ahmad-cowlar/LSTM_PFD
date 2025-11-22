"""
MAT Dataset Importer Script

One-time script to import all 1430 MAT files into HDF5 cache for fast access.
This should be run once after placing MAT files in data/raw/bearing_data/

Usage:
    python scripts/import_mat_dataset.py \
        --mat_dir data/raw/bearing_data/ \
        --output data/processed/signals_cache.h5

Author: AI Assistant
Date: 2025-11-20
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

from data.matlab_importer import load_mat_signals, convert_matlab_to_pytorch, extract_metadata


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

    mat_dir_path = Path(mat_dir)

    # Find all MAT files
    mat_files = list(mat_dir_path.rglob("*.mat"))
    print(f"Found {len(mat_files)} MAT files")

    if len(mat_files) == 0:
        raise ValueError(f"No MAT files found in {mat_dir}")

    # Load all MAT files
    all_signals = []
    all_labels = []
    all_metadata = []

    # Fault type mapping (modify based on your directory structure)
    fault_type_map = {
        'normal': 0,
        'ball_fault': 1,
        'inner_race': 2,
        'outer_race': 3,
        'combined': 4,
        'imbalance': 5,
        'misalignment': 6,
        'oil_whirl': 7,
        'cavitation': 8,
        'looseness': 9,
        'oil_deficiency': 10
    }

    print("\nLoading MAT files...")
    for mat_file in tqdm(mat_files, desc="Importing"):
        try:
            # Determine fault type from directory structure
            # Assumes structure: mat_dir/fault_type/file.mat
            fault_type_name = mat_file.parent.name

            if fault_type_name not in fault_type_map:
                print(f"Warning: Unknown fault type '{fault_type_name}', skipping {mat_file}")
                continue

            label = fault_type_map[fault_type_name]

            # Load MAT file
            mat_data = load_mat_signals(str(mat_file))

            # Extract signal (assumes MAT contains 'signal' or 'data' field)
            if isinstance(mat_data, dict):
                if 'signal' in mat_data:
                    signal = mat_data['signal'].flatten()
                elif 'data' in mat_data:
                    signal = mat_data['data'].flatten()
                elif 'x' in mat_data:
                    signal = mat_data['x'].flatten()
                else:
                    # Take first array-like field
                    for key, value in mat_data.items():
                        if isinstance(value, np.ndarray) and value.size > 1000:
                            signal = value.flatten()
                            break
            else:
                signal = mat_data.flatten()

            # Validate signal
            if validate:
                if len(signal) < 10000:
                    print(f"Warning: Signal too short ({len(signal)} samples), skipping {mat_file}")
                    continue

                if np.all(signal == 0):
                    print(f"Warning: Signal is all zeros, skipping {mat_file}")
                    continue

            # Truncate or pad to standard length (102400 samples)
            target_length = 102400
            if len(signal) > target_length:
                signal = signal[:target_length]
            elif len(signal) < target_length:
                signal = np.pad(signal, (0, target_length - len(signal)))

            # Store
            all_signals.append(signal.astype(np.float32))
            all_labels.append(label)

            # Metadata
            metadata = {
                'file': mat_file.name,
                'fault_type': fault_type_name,
                'label': label
            }
            all_metadata.append(metadata)

        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            continue

    # Convert to arrays
    all_signals = np.array(all_signals, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    print(f"\nLoaded {len(all_signals)} signals")
    print(f"Signal shape: {all_signals.shape}")

    # Class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for fault_idx, count in zip(unique, counts):
        fault_name = [k for k, v in fault_type_map.items() if v == fault_idx][0]
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

    # Save metadata JSON
    metadata_file = output_path.parent / 'metadata' / 'file_index.json'
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, 'w') as f:
        json.dump({
            'files': [str(meta['file']) for meta in all_metadata],
            'fault_types': [meta['fault_type'] for meta in all_metadata],
            'labels': all_labels.tolist()
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
    parser.add_argument('--mat_dir', type=str, default='data/raw/bearing_data/',
                      help='Directory containing MAT files')
    parser.add_argument('--output', type=str, default='data/processed/signals_cache.h5',
                      help='Output HDF5 file')
    parser.add_argument('--no-splits', action='store_true',
                      help='Do not generate train/val/test splits')
    parser.add_argument('--no-validate', action='store_true',
                      help='Skip signal validation')
    parser.add_argument('--split-ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                      help='Train/val/test split ratios')

    args = parser.parse_args()

    # Run import
    stats = import_mat_dataset(
        mat_dir=args.mat_dir,
        output_file=args.output,
        generate_splits=not args.no_splits,
        split_ratios=tuple(args.split_ratios),
        validate=not args.no_validate
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
