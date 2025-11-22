"""
LSTM Dataset for Bearing Fault Diagnosis

PyTorch Dataset class for loading vibration signals for LSTM training.

Author: Bearing Fault Diagnosis Team
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import scipy.io as sio

from utils.constants import FAULT_TYPES, NUM_CLASSES, SIGNAL_LENGTH


class LSTMDataset(Dataset):
    """
    PyTorch Dataset for LSTM-based bearing fault diagnosis.

    Loads raw vibration signals from .MAT files for LSTM processing.
    Unlike CNNs that can handle varying lengths through adaptive pooling,
    LSTMs require fixed-length sequences.

    Args:
        mat_files: List of paths to .MAT files
        labels: List of fault type labels (integers 0-10)
        signal_length: Fixed signal length (default: 102400)
        normalize: Whether to normalize signals (default: True)
        transform: Optional transform to apply to signals

    Example:
        >>> dataset = LSTMDataset(mat_files, labels)
        >>> signal, label = dataset[0]
        >>> print(signal.shape)  # torch.Size([1, 102400])
    """

    def __init__(
        self,
        mat_files: List[str],
        labels: List[int],
        signal_length: int = SIGNAL_LENGTH,
        normalize: bool = True,
        transform: Optional[callable] = None
    ):
        self.mat_files = mat_files
        self.labels = labels
        self.signal_length = signal_length
        self.normalize = normalize
        self.transform = transform

        assert len(mat_files) == len(labels), \
            f"Number of files ({len(mat_files)}) must match number of labels ({len(labels)})"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.mat_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            signal: Vibration signal tensor [1, signal_length]
            label: Fault type label (int)
        """
        # Load .MAT file
        mat_path = self.mat_files[idx]
        signal = self._load_signal(mat_path)

        # Ensure correct length
        signal = self._process_signal_length(signal)

        # Normalize
        if self.normalize:
            signal = self._normalize_signal(signal)

        # Convert to tensor [1, L]
        signal = torch.FloatTensor(signal).unsqueeze(0)

        # Apply transform if specified
        if self.transform is not None:
            signal = self.transform(signal)

        # Get label
        label = self.labels[idx]

        return signal, label

    def _load_signal(self, mat_path: str) -> np.ndarray:
        """
        Load signal from .MAT file.

        Args:
            mat_path: Path to .MAT file

        Returns:
            signal: 1D numpy array
        """
        try:
            mat_data = sio.loadmat(mat_path)

            # Common variable names in bearing datasets
            possible_keys = ['data', 'signal', 'vibration', 'accel',
                           'DE_time', 'FE_time', 'BA_time', 'X']

            signal = None
            for key in possible_keys:
                if key in mat_data:
                    signal = mat_data[key]
                    break

            if signal is None:
                # Try to find any numeric array
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and value.dtype in [np.float32, np.float64]:
                        signal = value
                        break

            if signal is None:
                raise ValueError(f"No signal data found in {mat_path}")

            # Flatten to 1D
            signal = signal.flatten()

            return signal

        except Exception as e:
            raise RuntimeError(f"Error loading {mat_path}: {e}")

    def _process_signal_length(self, signal: np.ndarray) -> np.ndarray:
        """
        Ensure signal has correct length.

        If signal is longer, truncate it.
        If signal is shorter, pad with zeros.

        Args:
            signal: Input signal

        Returns:
            signal: Fixed-length signal
        """
        current_length = len(signal)

        if current_length > self.signal_length:
            # Truncate (take first signal_length samples)
            signal = signal[:self.signal_length]
        elif current_length < self.signal_length:
            # Pad with zeros
            padding = self.signal_length - current_length
            signal = np.pad(signal, (0, padding), mode='constant', constant_values=0)

        return signal

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal to zero mean and unit variance.

        Args:
            signal: Input signal

        Returns:
            normalized_signal: Normalized signal
        """
        mean = np.mean(signal)
        std = np.std(signal)

        if std > 0:
            signal = (signal - mean) / std
        else:
            signal = signal - mean

        return signal

    def get_class_distribution(self) -> dict:
        """Get distribution of classes in the dataset."""
        class_counts = {}
        for label in self.labels:
            fault_name = FAULT_TYPES[label]
            class_counts[fault_name] = class_counts.get(fault_name, 0) + 1

        return class_counts


def create_dataset_from_directory(
    data_dir: str,
    signal_length: int = SIGNAL_LENGTH,
    normalize: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Create dataset from directory structure.

    Expected structure:
        data_dir/
        ├── sain/ (label 0)
        │   ├── sample_001.mat
        │   └── ...
        ├── desalignement/ (label 1)
        └── ...

    Args:
        data_dir: Root directory containing fault type subdirectories
        signal_length: Fixed signal length
        normalize: Whether to normalize signals

    Returns:
        mat_files: List of .MAT file paths
        labels: List of corresponding labels
    """
    data_path = Path(data_dir)

    mat_files = []
    labels = []

    # Map fault type names to labels
    fault_type_to_label = {fault_type: idx for idx, fault_type in enumerate(FAULT_TYPES)}

    # Iterate through fault type directories
    for fault_dir in sorted(data_path.iterdir()):
        if not fault_dir.is_dir():
            continue

        fault_name = fault_dir.name

        if fault_name not in fault_type_to_label:
            print(f"Warning: Unknown fault type '{fault_name}', skipping")
            continue

        label = fault_type_to_label[fault_name]

        # Find all .MAT files in this directory
        for mat_file in sorted(fault_dir.glob('*.mat')):
            mat_files.append(str(mat_file))
            labels.append(label)

    print(f"Found {len(mat_files)} samples across {len(set(labels))} fault types")

    return mat_files, labels
