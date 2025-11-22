"""
Standalone feature extraction pipeline with caching support.

Purpose:
    Extract and cache features separately from model training.
    Useful for avoiding recomputation during hyperparameter tuning.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import h5py
import time

from features.feature_extractor import FeatureExtractor
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class FeaturePipeline:
    """
    Feature extraction pipeline with caching.

    Example:
        >>> pipeline = FeaturePipeline()
        >>> features = pipeline.extract_and_save(signals, labels, 'features.h5')
        >>> # Later...
        >>> features, labels = pipeline.load('features.h5')
    """

    def __init__(self, fs: float = 20480):
        """
        Initialize feature pipeline.

        Args:
            fs: Sampling frequency
        """
        self.fs = fs
        self.extractor = FeatureExtractor(fs=fs)

    def extract_and_save(self, signals: np.ndarray, labels: np.ndarray,
                        save_path: Path) -> np.ndarray:
        """
        Extract features and save to HDF5 file.

        Args:
            signals: Signal array (n_samples, signal_length)
            labels: Label array (n_samples,)
            save_path: Path to save HDF5 file

        Returns:
            Extracted features (n_samples, n_features)
        """
        print(f"Extracting features from {len(signals)} signals...")
        start_time = time.time()

        # Extract features
        features = self.extractor.extract_batch(signals)

        elapsed = time.time() - start_time
        print(f"  Feature extraction complete ({elapsed:.1f}s)")
        print(f"  Features shape: {features.shape}")

        # Save to HDF5
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(save_path, 'w') as f:
            f.create_dataset('features', data=features, compression='gzip')
            f.create_dataset('labels', data=labels, compression='gzip')
            f.attrs['feature_names'] = self.extractor.get_feature_names()
            f.attrs['fs'] = self.fs
            f.attrs['n_samples'] = len(signals)
            f.attrs['n_features'] = features.shape[1]

        print(f"  Saved to: {save_path}")

        return features

    def load(self, load_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load cached features from HDF5 file.

        Args:
            load_path: Path to HDF5 file

        Returns:
            Tuple of (features, labels)
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Feature file not found: {load_path}")

        print(f"Loading features from {load_path}...")

        with h5py.File(load_path, 'r') as f:
            features = f['features'][:]
            labels = f['labels'][:]
            print(f"  Loaded {len(features)} samples with {features.shape[1]} features")

        return features, labels

    def extract_dataset_features(self, dataset_dict: Dict,
                                save_dir: Optional[Path] = None) -> Dict:
        """
        Extract features for train/val/test datasets.

        Args:
            dataset_dict: Dictionary with 'train', 'val', 'test' splits
            save_dir: Optional directory to save features

        Returns:
            Dictionary with extracted features
        """
        results = {}

        for split in ['train', 'val', 'test']:
            if split in dataset_dict:
                signals = dataset_dict[split]['signals']
                labels = dataset_dict[split]['labels']

                print(f"\nExtracting {split} features...")
                features = self.extractor.extract_batch(signals)
                results[split] = {
                    'features': features,
                    'labels': labels
                }

                # Save if requested
                if save_dir:
                    save_path = Path(save_dir) / f'{split}_features.h5'
                    self.extract_and_save(signals, labels, save_path)

        return results
