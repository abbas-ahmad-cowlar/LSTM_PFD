"""
PyTorch Dataset for CNN training with raw vibration signals.

Purpose:
    Load raw signals directly (no feature extraction) for end-to-end CNN learning:
    - RawSignalDataset: Loads signals from cache (Phase 0 HDF5)
    - Applies transforms (normalize, augment) on-the-fly
    - Returns signals in CNN-ready format [1, T]

    Difference from Phase 1: No feature extraction, raw waveforms only

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
import h5py

from utils.logging import get_logger
from data.cnn_transforms import get_train_transforms, get_test_transforms

logger = get_logger(__name__)


class RawSignalDataset(Dataset):
    """
    PyTorch Dataset that loads raw vibration signals for CNN training.

    Loads signals from HDF5 cache (created by Phase 0 cache_manager.py).
    No feature extraction - returns raw waveforms [1, T].

    Args:
        signals: NumPy array of signals [N, T] or [N, 1, T]
        labels: NumPy array of fault labels [N]
        metadata: Optional list of metadata dictionaries
        transform: Optional transform pipeline (default: normalize + to_tensor)
        label_to_idx: Optional mapping from label strings to indices

    Example:
        >>> from data.cnn_transforms import get_train_transforms
        >>> transform = get_train_transforms(augment=True)
        >>> dataset = RawSignalDataset(signals, labels, transform=transform)
        >>> signal, label = dataset[0]
        >>> print(f"Signal: {signal.shape}, Label: {label}")  # [1, 102400], 0
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        transform: Optional[Callable] = None,
        label_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize dataset from NumPy arrays.

        Args:
            signals: Signal array [N, T] or [N, 1, T]
            labels: Label array [N] (string or int)
            metadata: Optional metadata list
            transform: Transform pipeline (if None, uses default normalization)
            label_to_idx: Label-to-index mapping (auto-created if None)
        """
        # Store signals as NumPy (transforms expect NumPy input)
        if signals.ndim == 3 and signals.shape[1] == 1:
            # [N, 1, T] → [N, T]
            signals = signals.squeeze(1)
        self.signals = signals.astype(np.float32)

        self.raw_labels = labels
        self.metadata = metadata if metadata is not None else [{}] * len(signals)

        # Set transform (default to test transforms if None)
        if transform is None:
            logger.warning("No transform specified, using default (normalize + to_tensor)")
            transform = get_test_transforms()
        self.transform = transform

        # Create label mapping
        if label_to_idx is None:
            unique_labels = sorted(set(labels))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Convert labels to indices
        self.labels = np.array([self.label_to_idx[label] for label in labels], dtype=np.int64)

        logger.info(f"Created RawSignalDataset: {len(self)} samples, "
                   f"{len(self.label_to_idx)} classes, "
                   f"signal shape: {self.signals.shape}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get signal and label at index.

        Args:
            idx: Sample index

        Returns:
            (signal, label) tuple:
                - signal: Tensor [1, T] (transformed)
                - label: Integer class index
        """
        # Get raw signal (NumPy)
        signal = self.signals[idx]
        label = self.labels[idx]

        # Apply transforms (returns tensor [1, T])
        if self.transform:
            signal = self.transform(signal)

        return signal, label

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for sample.

        Args:
            idx: Sample index

        Returns:
            Metadata dictionary
        """
        return self.metadata[idx]

    def get_label_name(self, idx: int) -> str:
        """
        Get string label for sample.

        Args:
            idx: Sample index

        Returns:
            Label string
        """
        label_idx = self.labels[idx]
        return self.idx_to_label[label_idx]

    def get_class_counts(self) -> Dict[str, int]:
        """
        Get sample counts per class.

        Returns:
            Dictionary mapping class names to counts
        """
        from collections import Counter
        label_names = [self.idx_to_label[label] for label in self.labels]
        return dict(Counter(label_names))


class CachedRawSignalDataset(Dataset):
    """
    Memory-efficient dataset that loads signals from HDF5 on-the-fly.

    Useful for very large datasets that don't fit in RAM.
    Reads directly from HDF5 cache created by Phase 0.

    Args:
        cache_path: Path to HDF5 cache file
        transform: Optional transform pipeline
        split: Dataset split to load ('train', 'val', 'test')

    Example:
        >>> dataset = CachedRawSignalDataset(
        ...     cache_path='/path/to/cache.h5',
        ...     transform=get_train_transforms(augment=True),
        ...     split='train'
        ... )
    """

    def __init__(
        self,
        cache_path: Path,
        transform: Optional[Callable] = None,
        split: str = 'train'
    ):
        """
        Initialize dataset from HDF5 cache.

        Args:
            cache_path: Path to cache file
            transform: Transform pipeline
            split: Data split ('train', 'val', 'test')
        """
        self.cache_path = Path(cache_path)
        self.split = split

        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        # Set transform
        if transform is None:
            transform = get_test_transforms()
        self.transform = transform

        # Load metadata (labels, mapping) - keep in memory
        with h5py.File(self.cache_path, 'r') as f:
            if split not in f:
                raise ValueError(f"Split '{split}' not found in cache. Available: {list(f.keys())}")

            group = f[split]
            self.labels = group['labels'][:]

            # Load label mapping
            if 'label_to_idx' in group.attrs:
                import json
                self.label_to_idx = json.loads(group.attrs['label_to_idx'])
            else:
                # Create mapping from unique labels
                unique_labels = sorted(set(self.labels))
                self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

            # Get dataset size
            self.num_samples = len(self.labels)

        logger.info(f"Created CachedRawSignalDataset: {self.num_samples} samples from {cache_path}")

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get signal and label at index (loads from HDF5).

        Args:
            idx: Sample index

        Returns:
            (signal, label) tuple
        """
        # Load signal from HDF5
        with h5py.File(self.cache_path, 'r') as f:
            signal = f[self.split]['signals'][idx]

        # Convert label to index
        label = self.labels[idx]
        label_idx = self.label_to_idx[label] if isinstance(label, str) else label

        # Apply transforms
        if self.transform:
            signal = self.transform(signal)

        return signal, label_idx


def create_cnn_datasets_from_arrays(
    signals: np.ndarray,
    labels: np.ndarray,
    metadata: Optional[List[Dict]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    augment_train: bool = True,
    random_seed: int = 42
) -> Tuple[RawSignalDataset, RawSignalDataset, RawSignalDataset]:
    """
    Create train/val/test datasets from NumPy arrays.

    Args:
        signals: Signal array [N, T]
        labels: Label array [N]
        metadata: Optional metadata list
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        augment_train: Whether to augment training data
        random_seed: Random seed for reproducibility

    Returns:
        (train_dataset, val_dataset, test_dataset)

    Example:
        >>> signals = np.random.randn(1000, 102400)
        >>> labels = np.random.randint(0, 11, 1000)
        >>> train_ds, val_ds, test_ds = create_cnn_datasets_from_arrays(signals, labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"

    # Set random seed
    np.random.seed(random_seed)

    # Shuffle indices
    num_samples = len(signals)
    indices = np.random.permutation(num_samples)

    # Split indices
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create label mapping from all data
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Create datasets
    train_dataset = RawSignalDataset(
        signals=signals[train_indices],
        labels=labels[train_indices],
        metadata=[metadata[i] for i in train_indices] if metadata else None,
        transform=get_train_transforms(augment=augment_train),
        label_to_idx=label_to_idx
    )

    val_dataset = RawSignalDataset(
        signals=signals[val_indices],
        labels=labels[val_indices],
        metadata=[metadata[i] for i in val_indices] if metadata else None,
        transform=get_test_transforms(),  # No augmentation for val
        label_to_idx=label_to_idx
    )

    test_dataset = RawSignalDataset(
        signals=signals[test_indices],
        labels=labels[test_indices],
        metadata=[metadata[i] for i in test_indices] if metadata else None,
        transform=get_test_transforms(),  # No augmentation for test
        label_to_idx=label_to_idx
    )

    logger.info(f"Created CNN datasets: train={len(train_dataset)}, "
               f"val={len(val_dataset)}, test={len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def test_cnn_dataset():
    """Test CNN dataset classes."""
    print("=" * 60)
    print("Testing CNN Dataset Classes")
    print("=" * 60)

    # Create dummy data
    num_samples = 100
    signal_length = 102400
    num_classes = 11

    signals = np.random.randn(num_samples, signal_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)

    # Test RawSignalDataset
    print("\n1. Testing RawSignalDataset...")
    from data.cnn_transforms import get_train_transforms
    dataset = RawSignalDataset(
        signals=signals,
        labels=labels,
        transform=get_train_transforms(augment=True)
    )
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Number of classes: {len(dataset.label_to_idx)}")

    # Test __getitem__
    signal, label = dataset[0]
    print(f"   Sample 0: signal shape={signal.shape}, label={label}")
    assert signal.shape == (1, signal_length), f"Expected [1, {signal_length}], got {signal.shape}"
    assert isinstance(label, (int, np.integer))

    # Test batch loading
    print("\n2. Testing batch loading...")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch_signals, batch_labels = next(iter(loader))
    print(f"   Batch signals: {batch_signals.shape}")
    print(f"   Batch labels: {batch_labels.shape}")
    assert batch_signals.shape == (8, 1, signal_length)
    assert batch_labels.shape == (8,)

    # Test create_cnn_datasets_from_arrays
    print("\n3. Testing dataset splitting...")
    train_ds, val_ds, test_ds = create_cnn_datasets_from_arrays(
        signals=signals,
        labels=labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print(f"   Train: {len(train_ds)} samples")
    print(f"   Val: {len(val_ds)} samples")
    print(f"   Test: {len(test_ds)} samples")
    assert len(train_ds) + len(val_ds) + len(test_ds) == num_samples

    # Test class counts
    print("\n4. Testing class distribution...")
    class_counts = train_ds.get_class_counts()
    print(f"   Class counts: {class_counts}")

    print("\n" + "=" * 60)
    print("✅ All CNN dataset tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cnn_dataset()
