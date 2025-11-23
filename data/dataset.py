"""
PyTorch Dataset classes for bearing fault diagnosis signals.

Purpose:
    Dataset wrappers for generated signals with PyTorch integration:
    - In-memory dataset for preprocessed signals
    - On-the-fly generation dataset
    - Cached dataset with disk caching
    - Train/val/test splitting utilities

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
import pickle
from collections import Counter

from utils.logging import get_logger
from data.signal_generator import SignalGenerator
from data.augmentation import SignalAugmenter, random_augment
from config.data_config import DataConfig

logger = get_logger(__name__)


class BearingFaultDataset(Dataset):
    """
    In-memory PyTorch Dataset for bearing fault signals.

    Loads all signals into memory for fast training. Best for datasets
    that fit in RAM (< 10GB).

    Example:
        >>> dataset = BearingFaultDataset(signals, labels, metadata)
        >>> signal, label = dataset[0]
        >>> print(f"Signal shape: {signal.shape}, Label: {label}")
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
        Initialize dataset from arrays.

        Args:
            signals: Array of signals (num_signals, signal_length)
            labels: Array of fault labels (num_signals,)
            metadata: Optional list of metadata dictionaries
            transform: Optional transform to apply to signals
            label_to_idx: Optional mapping from label strings to indices
        """
        self.signals = torch.FloatTensor(signals)
        self.raw_labels = labels
        self.metadata = metadata
        self.transform = transform

        # Create label mapping
        if label_to_idx is None:
            unique_labels = sorted(set(labels))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Convert labels to indices
        self.labels = torch.LongTensor([self.label_to_idx[label] for label in labels])

        logger.info(f"Created BearingFaultDataset: {len(self)} samples, "
                   f"{len(self.label_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get signal and label at index.

        Args:
            idx: Sample index

        Returns:
            (signal, label) tuple
            - signal: (signal_length,) float tensor
            - label: integer class index
        """
        signal = self.signals[idx]
        label = self.labels[idx]

        # Apply transform if specified
        if self.transform:
            signal = self.transform(signal)

        return signal, label

    def get_metadata(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get metadata for sample at index."""
        if self.metadata is not None:
            return self.metadata[idx]
        return None

    def get_label_name(self, idx: int) -> str:
        """Get string label for sample at index."""
        label_idx = self.labels[idx].item()
        return self.idx_to_label[label_idx]

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution.

        Returns:
            Dictionary mapping label names to counts
        """
        label_indices = self.labels.numpy()
        counter = Counter(label_indices)
        return {self.idx_to_label[idx]: count for idx, count in counter.items()}

    def get_signals_by_class(self, class_name: str) -> torch.Tensor:
        """
        Get all signals for a specific class.

        Args:
            class_name: Fault class name

        Returns:
            Tensor of signals (num_samples, signal_length)
        """
        class_idx = self.label_to_idx[class_name]
        mask = self.labels == class_idx
        return self.signals[mask]

    @classmethod
    def from_generator_output(
        cls,
        generator_output: Dict[str, Any],
        transform: Optional[Callable] = None
    ) -> 'BearingFaultDataset':
        """
        Create dataset from SignalGenerator output.

        Args:
            generator_output: Output from SignalGenerator.generate_dataset()
            transform: Optional transform

        Returns:
            BearingFaultDataset instance

        Example:
            >>> generator = SignalGenerator(config)
            >>> output = generator.generate_dataset()
            >>> dataset = BearingFaultDataset.from_generator_output(output)
        """
        signals = generator_output['signals']
        labels = generator_output['labels']
        metadata = generator_output['metadata']

        return cls(signals, labels, metadata, transform)

    @classmethod
    def from_mat_file(
        cls,
        mat_path: Path,
        transform: Optional[Callable] = None
    ) -> 'BearingFaultDataset':
        """
        Load dataset from .mat file.

        Args:
            mat_path: Path to .mat file with 'signals' and 'labels' fields
            transform: Optional transform

        Returns:
            BearingFaultDataset instance
        """
        import scipy.io as sio

        mat_data = sio.loadmat(mat_path)
        signals = mat_data['signals']
        labels = mat_data['labels'].flatten()

        # Convert bytes to strings if needed
        if labels.dtype.kind == 'O':
            labels = np.array([str(label) for label in labels])

        return cls(signals, labels, None, transform)

    @classmethod
    def from_hdf5(
        cls,
        hdf5_path: Path,
        split: str = 'train',
        transform: Optional[Callable] = None
    ) -> 'BearingFaultDataset':
        """
        Load dataset from HDF5 file.

        HDF5 File Structure (created by signal_generator._save_as_hdf5()):
        ----------------------------------------------------------------
        Dataset Groups:
            f['train']/          - Training split
                'signals'        - (N_train, signal_length) float32 array
                'labels'         - (N_train,) int64 array of fault type indices
            f['val']/            - Validation split
                'signals'        - (N_val, signal_length) float32 array
                'labels'         - (N_val,) int64 array
            f['test']/           - Test split
                'signals'        - (N_test, signal_length) float32 array
                'labels'         - (N_test,) int64 array

        File Attributes:
            f.attrs['num_classes']      - Total number of fault classes (int)
            f.attrs['fault_types']      - List of fault type names (strings)
            f.attrs['signal_length']    - Length of each signal (int, e.g., 102400)
            f.attrs['sampling_rate']    - Sampling rate in Hz (int, e.g., 20480)
            f.attrs['train_samples']    - Number of training samples (int)
            f.attrs['val_samples']      - Number of validation samples (int)
            f.attrs['test_samples']     - Number of test samples (int)

        Label Encoding:
            Labels are stored as integers (0 to num_classes-1) corresponding to
            indices in utils.constants.FAULT_TYPES. Use label_to_idx mapping to
            convert between string names and integer indices.

        Args:
            hdf5_path: Path to HDF5 file
            split: Which split to load ('train', 'val', or 'test')
            transform: Optional transform to apply to signals

        Returns:
            BearingFaultDataset instance with loaded signals and labels

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            ValueError: If specified split doesn't exist in file

        Example:
            >>> # Load training set
            >>> train_dataset = BearingFaultDataset.from_hdf5(
            ...     Path('data/processed/dataset.h5'),
            ...     split='train'
            ... )
            >>> print(f"Loaded {len(train_dataset)} training samples")

            >>> # Load validation set with transforms
            >>> val_dataset = BearingFaultDataset.from_hdf5(
            ...     Path('data/processed/dataset.h5'),
            ...     split='val',
            ...     transform=my_transform
            ... )

            >>> # Check HDF5 file attributes
            >>> import h5py
            >>> with h5py.File('data/processed/dataset.h5', 'r') as f:
            ...     print(f"Classes: {f.attrs['num_classes']}")
            ...     print(f"Signal length: {f.attrs['signal_length']}")
            ...     print(f"Fault types: {list(f.attrs['fault_types'])}")
        """
        import h5py

        hdf5_path = Path(hdf5_path)

        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        logger.info(f"Loading {split} set from HDF5: {hdf5_path}")

        with h5py.File(hdf5_path, 'r') as f:
            if split not in f:
                available_splits = list(f.keys())
                raise ValueError(
                    f"Split '{split}' not found in HDF5. "
                    f"Available splits: {available_splits}"
                )

            signals = f[split]['signals'][:]
            labels = f[split]['labels'][:]

        logger.info(f"Loaded {len(signals)} samples from {split} split")

        return cls(signals, labels, None, transform)


class AugmentedBearingDataset(BearingFaultDataset):
    """
    Dataset with on-the-fly augmentation.

    Applies random augmentations during training for better generalization.

    Example:
        >>> dataset = AugmentedBearingDataset(
        ...     signals, labels,
        ...     augmentation_prob=0.5,
        ...     augmentation_methods=['time_warp', 'jittering']
        ... )
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        transform: Optional[Callable] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        augmentation_prob: float = 0.5,
        augmentation_methods: List[str] = ['time_warp', 'jittering'],
        rng_seed: Optional[int] = None
    ):
        """
        Initialize augmented dataset.

        Args:
            signals: Array of signals
            labels: Array of labels
            metadata: Optional metadata
            transform: Optional transform
            label_to_idx: Optional label mapping
            augmentation_prob: Probability of applying augmentation
            augmentation_methods: List of augmentation method names
            rng_seed: Random seed for augmentation
        """
        super().__init__(signals, labels, metadata, transform, label_to_idx)

        self.augmentation_prob = augmentation_prob
        self.augmentation_methods = augmentation_methods

        # Create augmenter with fixed seed for reproducibility
        rng = np.random.default_rng(rng_seed)
        self.augmenter = SignalAugmenter(rng=rng)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get signal with optional augmentation."""
        signal = self.signals[idx]
        label = self.labels[idx]

        # Apply augmentation with probability
        if np.random.rand() < self.augmentation_prob:
            # Convert to numpy for augmentation
            signal_np = signal.numpy()
            signal_np = random_augment(
                signal_np,
                methods=self.augmentation_methods,
                rng=self.augmenter.rng
            )
            signal = torch.FloatTensor(signal_np)

        # Apply transform if specified
        if self.transform:
            signal = self.transform(signal)

        return signal, label


class CachedBearingDataset(Dataset):
    """
    Dataset with disk caching for large-scale generation.

    Generates signals on-the-fly and caches to disk for reuse.
    Useful when dataset doesn't fit in memory.

    Example:
        >>> dataset = CachedBearingDataset(
        ...     config=config,
        ...     cache_dir=Path('./cache'),
        ...     num_samples=1000
        ... )
    """

    def __init__(
        self,
        config: DataConfig,
        cache_dir: Path,
        num_samples: int,
        transform: Optional[Callable] = None,
        regenerate: bool = False
    ):
        """
        Initialize cached dataset.

        Args:
            config: Data generation configuration
            cache_dir: Directory for cached signals
            num_samples: Total number of samples
            transform: Optional transform
            regenerate: Force regeneration of cache
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.num_samples = num_samples
        self.transform = transform

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize label mapping
        from utils.constants import FAULT_TYPES
        self.label_to_idx = {label: idx for idx, label in enumerate(FAULT_TYPES)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Check if cache exists
        self.index_path = self.cache_dir / 'index.pkl'
        if self.index_path.exists() and not regenerate:
            self._load_index()
            logger.info(f"Loaded cached dataset index: {len(self)} samples")
        else:
            self._generate_and_cache()

    def _generate_and_cache(self):
        """Generate dataset and cache to disk."""
        logger.info(f"Generating and caching {self.num_samples} samples...")

        generator = SignalGenerator(self.config)
        dataset = generator.generate_dataset()

        signals = dataset['signals']
        labels = dataset['labels']
        metadata = dataset['metadata']

        # Truncate or extend to desired number of samples
        if len(signals) > self.num_samples:
            signals = signals[:self.num_samples]
            labels = labels[:self.num_samples]
            metadata = metadata[:self.num_samples]

        # Save individual samples
        self.index = []
        for i, (signal, label, meta) in enumerate(zip(signals, labels, metadata)):
            sample_path = self.cache_dir / f'sample_{i:06d}.pkl'
            with open(sample_path, 'wb') as f:
                pickle.dump({'signal': signal, 'label': label, 'metadata': meta}, f)

            self.index.append({
                'path': sample_path,
                'label': label,
                'metadata': meta
            })

        # Save index
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)

        logger.info(f"Cached {len(self.index)} samples to {self.cache_dir}")

    def _load_index(self):
        """Load index from disk."""
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load signal from cache."""
        sample_info = self.index[idx]

        # Load from disk
        with open(sample_info['path'], 'rb') as f:
            sample = pickle.load(f)

        signal = torch.FloatTensor(sample['signal'])

        # Convert label string to index using label_to_idx mapping
        label_str = sample['label']
        if label_str not in self.label_to_idx:
            logger.warning(f"Unknown label '{label_str}', defaulting to 0")
            label = 0
        else:
            label = self.label_to_idx[label_str]

        if self.transform:
            signal = self.transform(signal)

        return signal, label


def train_val_test_split(
    dataset: BearingFaultDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    rng_seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: BearingFaultDataset to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        stratify: Whether to maintain class balance
        rng_seed: Random seed

    Returns:
        (train_dataset, val_dataset, test_dataset) tuple of Subsets

    Example:
        >>> train_ds, val_ds, test_ds = train_val_test_split(
        ...     dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        ... )
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")

    rng = np.random.default_rng(rng_seed)
    n_samples = len(dataset)

    if stratify:
        # Stratified split (maintain class balance)
        train_indices, val_indices, test_indices = [], [], []

        for class_name in dataset.label_to_idx.keys():
            # Get indices for this class
            class_idx = dataset.label_to_idx[class_name]
            class_mask = dataset.labels == class_idx
            class_indices = np.where(class_mask)[0]

            # Shuffle
            rng.shuffle(class_indices)

            # Split
            n_class = len(class_indices)
            n_train = int(n_class * train_ratio)
            n_val = int(n_class * val_ratio)

            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train:n_train + n_val])
            test_indices.extend(class_indices[n_train + n_val:])

        # Shuffle combined indices
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

    else:
        # Random split
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

    # Create Subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    logger.info(f"Dataset split: Train={len(train_indices)}, "
               f"Val={len(val_indices)}, Test={len(test_indices)}")

    return train_dataset, val_dataset, test_dataset


def collate_fn_with_metadata(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that preserves metadata.

    Args:
        batch: List of (signal, label) tuples

    Returns:
        Dictionary with 'signals' and 'labels' tensors

    Example:
        >>> loader = DataLoader(dataset, collate_fn=collate_fn_with_metadata)
    """
    signals = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])

    return {
        'signals': signals,
        'labels': labels
    }
