"""
Optimized DataLoaders for CNN training.

Purpose:
    Create PyTorch DataLoaders with optimizations for CNN training:
    - Pin memory for faster GPU transfer
    - Parallel data loading (num_workers)
    - Persistent workers to reduce overhead
    - Custom collate functions for batching

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Optional, List
import numpy as np

from utils.logging import get_logger
from data.cnn_dataset import RawSignalDataset

logger = get_logger(__name__)


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching signals.

    Stacks signals into batch tensor [B, 1, T] and labels into [B].

    Args:
        batch: List of (signal, label) tuples from Dataset

    Returns:
        (batch_signals, batch_labels) tuple:
            - batch_signals: Tensor [B, 1, T]
            - batch_labels: Tensor [B]
    """
    signals, labels = zip(*batch)

    # Stack signals: List of [1, T] → [B, 1, T]
    batch_signals = torch.stack(signals, dim=0)

    # Stack labels: List of int → [B]
    batch_labels = torch.tensor(labels, dtype=torch.long)

    return batch_signals, batch_labels


def create_cnn_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True
) -> DataLoader:
    """
    Create optimized DataLoader for CNN training.

    Optimizations:
    - pin_memory=True: Faster CPU→GPU transfer (requires CUDA)
    - num_workers=4: Parallel data loading (adjust based on CPU cores)
    - persistent_workers=True: Reduce worker initialization overhead
    - prefetch_factor=2: Prefetch batches for next iteration

    Args:
        dataset: PyTorch Dataset
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of parallel workers (0 = main thread only)
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop last incomplete batch
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        DataLoader instance

    Example:
        >>> from data.cnn_dataset import RawSignalDataset
        >>> dataset = RawSignalDataset(signals, labels)
        >>> loader = create_cnn_dataloader(dataset, batch_size=32, shuffle=True)
        >>> for signals, labels in loader:
        ...     print(signals.shape)  # [32, 1, 102400]
    """
    # Check if CUDA available for pin_memory
    if pin_memory and not torch.cuda.is_available():
        logger.warning("pin_memory=True but CUDA not available. Setting to False.")
        pin_memory = False

    # Persistent workers only make sense with num_workers > 0
    if num_workers == 0:
        persistent_workers = False

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )

    logger.info(f"Created DataLoader: batch_size={batch_size}, shuffle={shuffle}, "
               f"num_workers={num_workers}, pin_memory={pin_memory}")

    return loader


def create_cnn_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders with appropriate settings.

    Training loader: shuffle=True, persistent_workers=True
    Val/test loaders: shuffle=False, drop_last=False

    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of parallel workers
        pin_memory: Whether to pin memory

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders (if provided)

    Example:
        >>> loaders = create_cnn_dataloaders(
        ...     train_dataset=train_ds,
        ...     val_dataset=val_ds,
        ...     test_dataset=test_ds,
        ...     batch_size=32
        ... )
        >>> train_loader = loaders['train']
        >>> val_loader = loaders['val']
    """
    loaders = {}

    # Training loader (shuffle, drop last incomplete batch)
    loaders['train'] = create_cnn_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last for consistent batch size
        persistent_workers=True if num_workers > 0 else False
    )

    # Validation loader (no shuffle, keep all samples)
    if val_dataset is not None:
        loaders['val'] = create_cnn_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,  # Keep all validation samples
            persistent_workers=True if num_workers > 0 else False
        )

    # Test loader (no shuffle, keep all samples)
    if test_dataset is not None:
        loaders['test'] = create_cnn_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,  # Keep all test samples
            persistent_workers=True if num_workers > 0 else False
        )

    logger.info(f"Created {len(loaders)} DataLoaders: {list(loaders.keys())}")

    return loaders


class DataLoaderConfig:
    """
    Configuration class for DataLoader settings.

    Provides presets for different scenarios (fast training, debugging, etc.)
    """

    @staticmethod
    def fast_training(batch_size: int = 64) -> dict:
        """
        Configuration for fast training (maximize throughput).

        Returns:
            Config dict for create_cnn_dataloader()
        """
        return {
            'batch_size': batch_size,
            'num_workers': 8,  # More workers for parallelism
            'pin_memory': True,
            'persistent_workers': True
        }

    @staticmethod
    def debugging(batch_size: int = 8) -> dict:
        """
        Configuration for debugging (single-threaded, no prefetch).

        Returns:
            Config dict for create_cnn_dataloader()
        """
        return {
            'batch_size': batch_size,
            'num_workers': 0,  # Single-threaded for debugging
            'pin_memory': False,
            'persistent_workers': False
        }

    @staticmethod
    def memory_efficient(batch_size: int = 16) -> dict:
        """
        Configuration for limited memory (smaller batch, fewer workers).

        Returns:
            Config dict for create_cnn_dataloader()
        """
        return {
            'batch_size': batch_size,
            'num_workers': 2,  # Fewer workers to save memory
            'pin_memory': False,
            'persistent_workers': False
        }

    @staticmethod
    def default(batch_size: int = 32) -> dict:
        """
        Default balanced configuration.

        Returns:
            Config dict for create_cnn_dataloader()
        """
        return {
            'batch_size': batch_size,
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True
        }


def test_cnn_dataloader():
    """Test DataLoader creation and batching."""
    print("=" * 60)
    print("Testing CNN DataLoader")
    print("=" * 60)

    # Create dummy dataset
    from data.cnn_dataset import RawSignalDataset
    from data.cnn_transforms import get_train_transforms

    num_samples = 100
    signal_length = 102400
    signals = np.random.randn(num_samples, signal_length).astype(np.float32)
    labels = np.random.randint(0, 11, num_samples)

    dataset = RawSignalDataset(
        signals=signals,
        labels=labels,
        transform=get_train_transforms(augment=False)
    )

    # Test single DataLoader
    print("\n1. Testing single DataLoader...")
    loader = create_cnn_dataloader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0  # Use 0 for testing (no multiprocessing issues)
    )
    print(f"   DataLoader created with {len(loader)} batches")

    # Test batch loading
    batch_signals, batch_labels = next(iter(loader))
    print(f"   Batch signals: {batch_signals.shape}")
    print(f"   Batch labels: {batch_labels.shape}")
    assert batch_signals.shape == (16, 1, signal_length)
    assert batch_labels.shape == (16,)

    # Test full iteration
    print("\n2. Testing full iteration...")
    total_samples = 0
    for signals_batch, labels_batch in loader:
        total_samples += len(signals_batch)
    print(f"   Total samples iterated: {total_samples}")
    # Note: drop_last=True in training, so total may be < num_samples

    # Test create_cnn_dataloaders
    print("\n3. Testing multiple DataLoaders...")
    from data.cnn_dataset import create_cnn_datasets_from_arrays
    train_ds, val_ds, test_ds = create_cnn_datasets_from_arrays(
        signals=signals,
        labels=labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    loaders = create_cnn_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=8,
        num_workers=0
    )
    print(f"   Created loaders: {list(loaders.keys())}")
    assert 'train' in loaders
    assert 'val' in loaders
    assert 'test' in loaders

    # Test DataLoaderConfig presets
    print("\n4. Testing DataLoaderConfig presets...")
    fast_config = DataLoaderConfig.fast_training(batch_size=32)
    print(f"   Fast training config: {fast_config}")
    debug_config = DataLoaderConfig.debugging(batch_size=8)
    print(f"   Debugging config: {debug_config}")

    print("\n" + "=" * 60)
    print("✅ All DataLoader tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cnn_dataloader()
