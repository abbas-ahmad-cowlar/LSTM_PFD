"""
Time-Frequency Representation Dataset for Phase 5

PyTorch Dataset classes for loading precomputed spectrograms, scalograms, and WVDs.
Supports both precomputed (fast) and on-the-fly (flexible) generation modes.

Author: AI Assistant
Date: 2025-11-20
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import h5py

from data.spectrogram_generator import SpectrogramGenerator
from data.wavelet_transform import WaveletTransform
from data.wigner_ville import WignerVilleDistribution


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for precomputed spectrograms.

    Fast loading from cached NPZ files. Recommended for training.
    """

    def __init__(
        self,
        spectrogram_file: str,
        labels_file: Optional[str] = None,
        transform: Optional[callable] = None,
        return_metadata: bool = False
    ):
        """
        Initialize dataset from precomputed spectrograms.

        Args:
            spectrogram_file: Path to .npz file with spectrograms
            labels_file: Path to .npy file with labels (or included in spectrogram_file)
            transform: Optional augmentation transform
            return_metadata: If True, return (spectrogram, label, metadata)
        """
        super().__init__()

        self.transform = transform
        self.return_metadata = return_metadata

        # Load spectrograms
        data = np.load(spectrogram_file, mmap_mode='r')

        if 'spectrograms' in data:
            self.spectrograms = data['spectrograms']
        else:
            # Assume first array is spectrograms
            self.spectrograms = data[list(data.keys())[0]]

        # Load labels
        if labels_file is not None:
            self.labels = np.load(labels_file)
        elif 'labels' in data:
            self.labels = data['labels']
        else:
            raise ValueError("Labels not found in spectrogram file and labels_file not provided")

        # Optional metadata
        if 'metadata' in data:
            self.metadata = data['metadata']
        else:
            self.metadata = None

        # Validate shapes
        assert len(self.spectrograms) == len(self.labels), \
            f"Mismatch: {len(self.spectrograms)} spectrograms, {len(self.labels)} labels"

        print(f"Loaded {len(self)} spectrograms with shape {self.spectrograms.shape[1:]}")

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get spectrogram
        spectrogram = self.spectrograms[idx].astype(np.float32)

        # Add channel dimension if needed [H, W] -> [1, H, W]
        if spectrogram.ndim == 2:
            spectrogram = spectrogram[np.newaxis, :]

        # Apply augmentation
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        # Convert to tensor
        spectrogram_tensor = torch.from_numpy(spectrogram)

        # Get label
        label = int(self.labels[idx])
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_metadata and self.metadata is not None:
            return spectrogram_tensor, label_tensor, self.metadata[idx]
        else:
            return spectrogram_tensor, label_tensor


class OnTheFlyTFRDataset(Dataset):
    """
    PyTorch Dataset that computes TFRs on-the-fly.

    Slower but more flexible. Useful for:
    - Experimenting with different TFR parameters
    - Limited disk space
    - Augmentation requiring original signals
    """

    def __init__(
        self,
        signals_cache: str,
        tfr_type: str = 'stft',
        tfr_params: Optional[Dict] = None,
        transform: Optional[callable] = None,
        cache_in_memory: bool = False
    ):
        """
        Initialize dataset with on-the-fly TFR computation.

        Args:
            signals_cache: Path to HDF5 file with raw signals
            tfr_type: Type of TFR ('stft', 'cwt', 'wvd')
            tfr_params: Parameters for TFR generation
            transform: Optional augmentation
            cache_in_memory: Load all signals to RAM (faster but memory-intensive)
        """
        super().__init__()

        self.tfr_type = tfr_type
        self.transform = transform
        self.cache_in_memory = cache_in_memory

        # Default TFR parameters
        if tfr_params is None:
            tfr_params = {}

        # Load signals
        self.h5_file_path = signals_cache
        self.h5_file = h5py.File(signals_cache, 'r')

        # Assuming HDF5 structure: /signals -> [N, signal_length], /labels -> [N]
        self.signals = self.h5_file['signals']
        self.labels = self.h5_file['labels'][:]  # Load labels to memory

        # Cache signals in memory if requested
        if cache_in_memory:
            print("Loading all signals to memory...")
            self.signals = self.signals[:]
            self.h5_file.close()

        # Initialize TFR generator
        if tfr_type == 'stft':
            self.tfr_generator = SpectrogramGenerator(**tfr_params)
        elif tfr_type == 'cwt':
            self.tfr_generator = WaveletTransform(**tfr_params)
        elif tfr_type == 'wvd':
            self.tfr_generator = WignerVilleDistribution(**tfr_params)
        else:
            raise ValueError(f"Unknown TFR type: {tfr_type}")

        print(f"Initialized OnTheFlyTFRDataset with {len(self)} signals ({tfr_type})")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get signal
        signal = self.signals[idx].astype(np.float32)

        # Compute TFR
        if self.tfr_type == 'stft':
            tfr, _, _ = self.tfr_generator.generate_normalized_spectrogram(signal)
        elif self.tfr_type == 'cwt':
            tfr, _ = self.tfr_generator.generate_normalized_scalogram(signal)
        elif self.tfr_type == 'wvd':
            tfr, _, _ = self.tfr_generator.generate_normalized_wvd(signal)

        # Add channel dimension [H, W] -> [1, H, W]
        if tfr.ndim == 2:
            tfr = tfr[np.newaxis, :]

        # Apply augmentation
        if self.transform is not None:
            tfr = self.transform(tfr)

        # Convert to tensor
        tfr_tensor = torch.from_numpy(tfr.astype(np.float32))
        label_tensor = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return tfr_tensor, label_tensor

    def __del__(self):
        # Close HDF5 file
        if hasattr(self, 'h5_file') and not self.cache_in_memory:
            try:
                self.h5_file.close()
            except:
                pass


class MultiTFRDataset(Dataset):
    """
    Dataset that returns multiple TFR types for the same signal.

    Useful for multi-stream models that process different TFRs in parallel.
    """

    def __init__(
        self,
        signals_cache: str,
        tfr_types: List[str] = ['stft', 'cwt'],
        tfr_params: Optional[Dict[str, Dict]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize multi-TFR dataset.

        Args:
            signals_cache: Path to signals HDF5
            tfr_types: List of TFR types to compute
            tfr_params: Dict mapping tfr_type -> params
            transform: Optional augmentation
        """
        super().__init__()

        self.tfr_types = tfr_types
        self.transform = transform

        # Load signals
        self.h5_file = h5py.File(signals_cache, 'r')
        self.signals = self.h5_file['signals']
        self.labels = self.h5_file['labels'][:]

        # Initialize generators for each TFR type
        self.tfr_generators = {}
        for tfr_type in tfr_types:
            params = tfr_params.get(tfr_type, {}) if tfr_params else {}

            if tfr_type == 'stft':
                self.tfr_generators[tfr_type] = SpectrogramGenerator(**params)
            elif tfr_type == 'cwt':
                self.tfr_generators[tfr_type] = WaveletTransform(**params)
            elif tfr_type == 'wvd':
                self.tfr_generators[tfr_type] = WignerVilleDistribution(**params)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        signal = self.signals[idx].astype(np.float32)

        tfrs = {}
        for tfr_type in self.tfr_types:
            # Generate TFR
            if tfr_type == 'stft':
                tfr, _, _ = self.tfr_generators[tfr_type].generate_normalized_spectrogram(signal)
            elif tfr_type == 'cwt':
                tfr, _ = self.tfr_generators[tfr_type].generate_normalized_scalogram(signal)
            elif tfr_type == 'wvd':
                tfr, _, _ = self.tfr_generators[tfr_type].generate_normalized_wvd(signal)

            # Add channel dimension
            if tfr.ndim == 2:
                tfr = tfr[np.newaxis, :]

            tfrs[tfr_type] = torch.from_numpy(tfr.astype(np.float32))

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return tfrs, label

    def __del__(self):
        if hasattr(self, 'h5_file'):
            try:
                self.h5_file.close()
            except:
                pass


def create_tfr_dataloaders(
    train_spectrogram_file: str,
    val_spectrogram_file: str,
    test_spectrogram_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[callable] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test sets.

    Args:
        train_spectrogram_file: Path to training spectrograms
        val_spectrogram_file: Path to validation spectrograms
        test_spectrogram_file: Path to test spectrograms
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_transform: Augmentation for training set
        **dataset_kwargs: Additional arguments for SpectrogramDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SpectrogramDataset(
        train_spectrogram_file,
        transform=train_transform,
        **dataset_kwargs
    )

    val_dataset = SpectrogramDataset(
        val_spectrogram_file,
        transform=None,  # No augmentation for validation
        **dataset_kwargs
    )

    test_dataset = SpectrogramDataset(
        test_spectrogram_file,
        transform=None,
        **dataset_kwargs
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("TFR Dataset - Example Usage\n")

    # This is a demonstration - actual usage requires precomputed spectrograms
    # or signals cache file

    print("1. Precomputed Spectrogram Dataset (Recommended)")
    print("   Usage:")
    print("   dataset = SpectrogramDataset('data/spectrograms/stft/train.npz')")
    print("   spectrogram, label = dataset[0]")
    print("   print(f'Shape: {spectrogram.shape}, Label: {label}')")

    print("\n2. On-the-Fly TFR Dataset (Flexible)")
    print("   Usage:")
    print("   dataset = OnTheFlyTFRDataset(")
    print("       signals_cache='data/processed/signals_cache.h5',")
    print("       tfr_type='stft'")
    print("   )")

    print("\n3. Multi-TFR Dataset (For fusion models)")
    print("   Usage:")
    print("   dataset = MultiTFRDataset(")
    print("       signals_cache='data/processed/signals_cache.h5',")
    print("       tfr_types=['stft', 'cwt']")
    print("   )")
    print("   tfrs, label = dataset[0]")
    print("   print(f'STFT shape: {tfrs['stft'].shape}')")
    print("   print(f'CWT shape: {tfrs['cwt'].shape}')")

    print("\n✓ TFR Dataset classes ready for use!")
