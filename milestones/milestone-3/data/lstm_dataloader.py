"""
LSTM DataLoader utilities for train/val/test split creation.

Author: Bearing Fault Diagnosis Team
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional, List
import numpy as np
from sklearn.model_selection import train_test_split

from .lstm_dataset import LSTMDataset, create_dataset_from_directory
from utils.constants import SIGNAL_LENGTH


def create_lstm_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    signal_length: int = SIGNAL_LENGTH,
    normalize: bool = True,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test DataLoaders for LSTM training.

    Args:
        data_dir: Directory containing .MAT files organized by fault type
        batch_size: Batch size for training
        val_batch_size: Batch size for validation/test (defaults to batch_size * 2)
        num_workers: Number of data loading workers
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
        test_ratio: Ratio of data for testing (default: 0.15)
        signal_length: Fixed signal length
        normalize: Whether to normalize signals
        shuffle: Whether to shuffle training data
        random_seed: Random seed for reproducibility

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader

    Example:
        >>> train_loader, val_loader, test_loader = create_lstm_dataloaders(
        ...     data_dir='data/raw/bearing_data',
        ...     batch_size=32,
        ...     num_workers=4
        ... )
    """
    # Validate split ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    # Default val_batch_size
    if val_batch_size is None:
        val_batch_size = batch_size * 2

    # Load all .MAT files and labels
    print(f"Loading dataset from: {data_dir}")
    mat_files, labels = create_dataset_from_directory(
        data_dir=data_dir,
        signal_length=signal_length,
        normalize=normalize
    )

    total_samples = len(mat_files)
    print(f"Total samples: {total_samples}")

    # Split into train/val/test (stratified by label)
    # First split: train vs (val + test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        mat_files,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_seed
    )

    # Second split: val vs test
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_seed
    )

    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")

    # Create datasets
    train_dataset = LSTMDataset(
        mat_files=train_files,
        labels=train_labels,
        signal_length=signal_length,
        normalize=normalize,
        transform=None  # Add augmentation transforms here if needed
    )

    val_dataset = LSTMDataset(
        mat_files=val_files,
        labels=val_labels,
        signal_length=signal_length,
        normalize=normalize,
        transform=None
    )

    test_dataset = LSTMDataset(
        mat_files=test_files,
        labels=test_labels,
        signal_length=signal_length,
        normalize=normalize,
        transform=None
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def get_dataloader_info(dataloader: DataLoader) -> dict:
    """
    Get information about a DataLoader.

    Args:
        dataloader: PyTorch DataLoader

    Returns:
        info: Dictionary with dataloader information
    """
    dataset = dataloader.dataset

    info = {
        'num_samples': len(dataset),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
    }

    # Add class distribution if available
    if hasattr(dataset, 'get_class_distribution'):
        info['class_distribution'] = dataset.get_class_distribution()

    return info
