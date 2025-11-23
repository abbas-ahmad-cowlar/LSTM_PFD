"""
DataLoader factory and utilities for efficient data loading.

Purpose:
    Simplified DataLoader creation with best practices:
    - Multi-worker data loading
    - Pin memory for GPU transfer
    - Reproducible shuffling
    - Batch size auto-tuning

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable, Dict, Any
import multiprocessing as mp

from utils.logging import get_logger
from data.dataset import BearingFaultDataset, collate_fn_with_metadata

logger = get_logger(__name__)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    worker_init_fn: Optional[Callable] = None,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Create DataLoader with sensible defaults for bearing fault diagnosis.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size (default: 32)
        shuffle: Whether to shuffle data (default: True for training)
        num_workers: Number of worker processes (default: auto-detect)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        drop_last: Drop incomplete last batch (default: False)
        collate_fn: Custom collate function (default: None)
        worker_init_fn: Worker initialization function (default: None)
        persistent_workers: Keep workers alive between epochs (default: False)

    Returns:
        Configured DataLoader

    Example:
        >>> train_loader = create_dataloader(
        ...     train_dataset,
        ...     batch_size=64,
        ...     shuffle=True,
        ...     num_workers=4
        ... )
        >>> for batch in train_loader:
        ...     signals, labels = batch
    """
    # Auto-detect number of workers
    if num_workers is None:
        num_workers = min(4, max(1, mp.cpu_count() // 2))
        logger.debug(f"Auto-detected num_workers={num_workers}")

    # Check if CUDA available for pin_memory
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False
        logger.debug("CUDA not available, disabling pin_memory")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers and num_workers > 0
    )

    logger.info(
        f"Created DataLoader: {len(dataset)} samples, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )

    return dataloader


def create_train_val_test_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders with appropriate settings.

    Train loader: shuffle=True, drop_last=True (for batch norm stability)
    Val/Test loaders: shuffle=False, drop_last=False (evaluate all samples)

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of workers (auto-detect if None)
        pin_memory: Pin memory for GPU
        collate_fn: Custom collate function

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders

    Example:
        >>> loaders = create_train_val_test_loaders(
        ...     train_ds, val_ds, test_ds,
        ...     batch_size=64
        ... )
        >>> train_loader = loaders['train']
        >>> val_loader = loaders['val']
    """
    # Auto-detect workers
    if num_workers is None:
        num_workers = min(4, max(1, mp.cpu_count() // 2))

    # Train loader (shuffle, drop_last)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    # Val loader (no shuffle, keep all)
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    # Test loader (no shuffle, keep all)
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    logger.info(
        f"Created train/val/test loaders: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def worker_init_fn_seed(worker_id: int):
    """
    Worker initialization function for reproducible data loading.

    Sets unique random seed for each worker based on worker_id.

    Args:
        worker_id: Worker process ID

    Example:
        >>> loader = create_dataloader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=worker_init_fn_seed
        ... )
    """
    import numpy as np
    import random

    # Set unique seed per worker
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def estimate_optimal_batch_size(
    dataset: Dataset,
    model: torch.nn.Module,
    device: torch.device,
    max_batch_size: int = 512,
    initial_batch_size: int = 32
) -> int:
    """
    Estimate optimal batch size via binary search on GPU memory.

    Finds largest batch size that fits in GPU memory without OOM.

    Args:
        dataset: Dataset to test with
        model: Model to test
        device: Device to run on
        max_batch_size: Maximum batch size to try
        initial_batch_size: Starting batch size

    Returns:
        Optimal batch size

    Example:
        >>> optimal_bs = estimate_optimal_batch_size(
        ...     dataset, model, device=torch.device('cuda')
        ... )
        >>> print(f"Optimal batch size: {optimal_bs}")
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, returning initial batch size")
        return initial_batch_size

    logger.info("Estimating optimal batch size...")

    model = model.to(device)
    model.eval()

    # Get sample input shape
    sample_signal, _ = dataset[0]
    input_shape = sample_signal.shape

    # Binary search for max batch size
    low, high = 1, max_batch_size
    optimal_batch_size = initial_batch_size

    while low <= high:
        mid = (low + high) // 2

        try:
            # Create dummy batch
            dummy_batch = torch.randn(mid, *input_shape, device=device)

            # Forward pass
            with torch.no_grad():
                _ = model(dummy_batch)

            # Success - try larger
            optimal_batch_size = mid
            low = mid + 1

            # Clear cache
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                # OOM - try smaller
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise e

    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def compute_class_weights(dataset: BearingFaultDataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Weights are inversely proportional to class frequencies.
    Useful for loss weighting with imbalanced classes.

    Args:
        dataset: BearingFaultDataset

    Returns:
        Tensor of class weights (num_classes,)

    Example:
        >>> weights = compute_class_weights(train_dataset)
        >>> criterion = nn.CrossEntropyLoss(weight=weights)
    """
    # Get class distribution
    class_counts = []
    num_classes = len(dataset.label_to_idx)

    for class_idx in range(num_classes):
        count = (dataset.labels == class_idx).sum().item()
        class_counts.append(count)

    class_counts = torch.tensor(class_counts, dtype=torch.float32)

    # Compute weights (inverse frequency)
    total_samples = len(dataset)
    weights = total_samples / (num_classes * class_counts)

    # Normalize to sum to num_classes
    weights = weights / weights.sum() * num_classes

    logger.info(f"Computed class weights: {weights.tolist()}")

    return weights


def get_dataloader_stats(dataloader: DataLoader) -> Dict[str, Any]:
    """
    Compute statistics about DataLoader configuration.

    Args:
        dataloader: DataLoader to analyze

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = get_dataloader_stats(train_loader)
        >>> print(f"Batches per epoch: {stats['num_batches']}")
    """
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size

    stats = {
        'num_samples': len(dataset),
        'batch_size': batch_size,
        'num_batches': len(dataloader),
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'drop_last': dataloader.drop_last,
        'shuffle': dataloader.sampler is not None,
    }

    # Compute iteration time estimate (rough)
    # Assume ~10ms per batch for data loading
    estimated_time_per_epoch_sec = stats['num_batches'] * 0.01

    stats['estimated_time_per_epoch_sec'] = estimated_time_per_epoch_sec

    return stats


class InfiniteDataLoader:
    """
    Wrapper for DataLoader that loops infinitely.

    Useful for training loops that use iteration count instead of epochs.

    Example:
        >>> infinite_loader = InfiniteDataLoader(train_loader)
        >>> for i, batch in enumerate(infinite_loader):
        ...     if i >= 1000:
        ...         break
        ...     # Train on batch
    """

    def __init__(self, dataloader: DataLoader):
        """
        Initialize infinite loader.

        Args:
            dataloader: DataLoader to wrap
        """
        self.dataloader = dataloader
        self.iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch, restart if exhausted."""
        if self.iterator is None:
            self.iterator = iter(self.dataloader)

        try:
            batch = next(self.iterator)
        except StopIteration:
            # Restart iterator
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        return batch


def prefetch_to_device(
    dataloader: DataLoader,
    device: torch.device,
    non_blocking: bool = True
):
    """
    Generator that prefetches batches to device.

    Overlaps data transfer with computation for faster training.

    Args:
        dataloader: DataLoader to wrap
        device: Device to transfer to
        non_blocking: Use async transfer (requires pin_memory=True)

    Yields:
        Batches on target device

    Example:
        >>> for batch in prefetch_to_device(train_loader, device):
        ...     outputs = model(batch['signals'])
    """
    for batch in dataloader:
        # Move to device
        if isinstance(batch, (tuple, list)):
            batch = tuple(b.to(device, non_blocking=non_blocking) for b in batch)
        elif isinstance(batch, dict):
            batch = {k: v.to(device, non_blocking=non_blocking)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        else:
            batch = batch.to(device, non_blocking=non_blocking)

        yield batch
