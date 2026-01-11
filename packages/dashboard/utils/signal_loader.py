"""
Signal Loading Utilities for XAI Dashboard.
Handles loading signals from datasets (HDF5 files) and experiments.
"""
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple, List
from database.connection import get_db_session
from models.signal import Signal
from models.dataset import Dataset
from models.experiment import Experiment
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SignalLoader:
    """Utility class for loading signals from various sources."""

    @staticmethod
    def load_signal_by_id(signal_id: int) -> Optional[torch.Tensor]:
        """
        Load signal tensor from database by signal ID.

        Args:
            signal_id: Signal database ID

        Returns:
            Signal tensor [1, 1, T] or None if failed
        """
        try:
            with get_db_session() as session:
                signal_record = session.query(Signal).filter_by(id=signal_id).first()

                if not signal_record:
                    logger.error(f"Signal {signal_id} not found in database")
                    return None

                dataset = signal_record.dataset

                if not dataset or not dataset.file_path:
                    logger.error(f"Dataset for signal {signal_id} has no file path")
                    return None

                # Load from HDF5
                dataset_path = Path(dataset.file_path)
                if not dataset_path.exists():
                    logger.error(f"Dataset file not found: {dataset_path}")
                    return None

                with h5py.File(dataset_path, 'r') as f:
                    # Try to find signal by ID
                    signal_key = signal_record.signal_id

                    if signal_key in f:
                        # Signal is a group with 'data' dataset
                        if isinstance(f[signal_key], h5py.Group) and 'data' in f[signal_key]:
                            signal_data = f[signal_key]['data'][:]
                        else:
                            signal_data = f[signal_key][:]
                    else:
                        logger.error(f"Signal {signal_key} not found in HDF5 file")
                        return None

                # Convert to tensor
                signal_tensor = torch.tensor(signal_data, dtype=torch.float32)

                # Ensure proper dimensions [1, 1, T]
                if signal_tensor.dim() == 1:
                    signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
                elif signal_tensor.dim() == 2:
                    signal_tensor = signal_tensor.unsqueeze(0)

                logger.info(f"Loaded signal {signal_id} with shape {signal_tensor.shape}")
                return signal_tensor

        except Exception as e:
            logger.error(f"Failed to load signal {signal_id}: {e}", exc_info=True)
            return None

    @staticmethod
    def load_background_data(
        experiment_id: int,
        num_samples: int = 100,
        balance_classes: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Load background dataset for SHAP from experiment's training data.

        Args:
            experiment_id: Experiment ID
            num_samples: Number of background samples
            balance_classes: Whether to balance samples across classes

        Returns:
            Background data tensor [N, 1, T] or None if failed
        """
        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()

                if not experiment or not experiment.dataset_id:
                    logger.error(f"Experiment {experiment_id} has no dataset")
                    return None

                dataset = session.query(Dataset).filter_by(id=experiment.dataset_id).first()

                if not dataset or not dataset.file_path:
                    logger.error(f"Dataset {dataset.id} has no file path")
                    return None

                dataset_path = Path(dataset.file_path)
                if not dataset_path.exists():
                    logger.error(f"Dataset file not found: {dataset_path}")
                    return None

                # Load samples from HDF5
                with h5py.File(dataset_path, 'r') as f:
                    # Get all signal keys
                    signal_keys = [k for k in f.keys() if k.startswith('signal_')]

                    if not signal_keys:
                        # Try alternative structure (signals dataset)
                        if 'signals' in f:
                            signals_data = f['signals'][:]
                            # Random sample
                            if len(signals_data) > num_samples:
                                indices = np.random.choice(len(signals_data), num_samples, replace=False)
                                signals_data = signals_data[indices]
                            else:
                                signals_data = signals_data[:num_samples]

                            # Convert to tensor
                            background = torch.tensor(signals_data, dtype=torch.float32)

                            # Ensure [N, 1, T] shape
                            if background.dim() == 2:
                                background = background.unsqueeze(1)

                            logger.info(f"Loaded {len(background)} background samples from signals dataset")
                            return background

                    # Load from individual signal groups
                    if balance_classes:
                        # Group by fault class
                        class_signals = {}
                        for key in signal_keys:
                            if 'fault_type' in f[key].attrs:
                                fault_type = f[key].attrs['fault_type']
                                if fault_type not in class_signals:
                                    class_signals[fault_type] = []
                                class_signals[fault_type].append(key)

                        # Sample evenly from each class
                        samples_per_class = num_samples // len(class_signals) if class_signals else num_samples
                        selected_keys = []

                        for fault_type, keys in class_signals.items():
                            n_to_sample = min(samples_per_class, len(keys))
                            sampled = np.random.choice(keys, n_to_sample, replace=False).tolist()
                            selected_keys.extend(sampled)

                        # Add random samples if needed
                        if len(selected_keys) < num_samples:
                            remaining = num_samples - len(selected_keys)
                            available = [k for k in signal_keys if k not in selected_keys]
                            if available:
                                additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                                selected_keys.extend(additional.tolist())

                    else:
                        # Random sampling
                        n_to_sample = min(num_samples, len(signal_keys))
                        selected_keys = np.random.choice(signal_keys, n_to_sample, replace=False).tolist()

                    # Load selected signals
                    background_signals = []
                    for key in selected_keys:
                        if 'data' in f[key]:
                            signal_data = f[key]['data'][:]
                        else:
                            signal_data = f[key][:]
                        background_signals.append(signal_data)

                    # Convert to tensor
                    background = torch.tensor(np.array(background_signals), dtype=torch.float32)

                    # Ensure [N, 1, T] shape
                    if background.dim() == 2:
                        background = background.unsqueeze(1)

                    logger.info(f"Loaded {len(background)} background samples for experiment {experiment_id}")
                    return background

        except Exception as e:
            logger.error(f"Failed to load background data for experiment {experiment_id}: {e}", exc_info=True)
            return None

    @staticmethod
    def load_signals_from_dataset(
        dataset_id: int,
        num_signals: Optional[int] = None,
        fault_class: Optional[str] = None
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Load multiple signals from a dataset.

        Args:
            dataset_id: Dataset ID
            num_signals: Maximum number of signals to load (None = all)
            fault_class: Filter by specific fault class (None = all)

        Returns:
            Tuple of (list of signal tensors, list of signal IDs)
        """
        try:
            with get_db_session() as session:
                query = session.query(Signal).filter_by(dataset_id=dataset_id)

                if fault_class:
                    query = query.filter_by(fault_class=fault_class)

                if num_signals:
                    query = query.limit(num_signals)

                signal_records = query.all()

                if not signal_records:
                    logger.warning(f"No signals found for dataset {dataset_id}")
                    return [], []

                # Load each signal
                signals = []
                signal_ids = []

                for record in signal_records:
                    signal_tensor = SignalLoader.load_signal_by_id(record.id)
                    if signal_tensor is not None:
                        signals.append(signal_tensor)
                        signal_ids.append(record.id)

                logger.info(f"Loaded {len(signals)} signals from dataset {dataset_id}")
                return signals, signal_ids

        except Exception as e:
            logger.error(f"Failed to load signals from dataset {dataset_id}: {e}", exc_info=True)
            return [], []

    @staticmethod
    def get_signal_metadata(signal_id: int) -> Optional[dict]:
        """
        Get metadata for a signal.

        Args:
            signal_id: Signal database ID

        Returns:
            Dictionary with signal metadata or None
        """
        try:
            with get_db_session() as session:
                signal = session.query(Signal).filter_by(id=signal_id).first()

                if not signal:
                    return None

                return {
                    'signal_id': signal.signal_id,
                    'fault_class': signal.fault_class,
                    'severity': signal.severity,
                    'rms': signal.rms,
                    'kurtosis': signal.kurtosis,
                    'dominant_frequency': signal.dominant_frequency,
                    'dataset_id': signal.dataset_id,
                    'file_path': signal.file_path,
                }

        except Exception as e:
            logger.error(f"Failed to get metadata for signal {signal_id}: {e}")
            return None

    @staticmethod
    def normalize_signal(signal: torch.Tensor, method: str = 'standard') -> torch.Tensor:
        """
        Normalize signal.

        Args:
            signal: Input signal tensor
            method: Normalization method ('standard', 'minmax', 'rms')

        Returns:
            Normalized signal
        """
        if method == 'standard':
            # Zero mean, unit variance
            mean = signal.mean()
            std = signal.std()
            if std > 0:
                return (signal - mean) / std
            return signal - mean

        elif method == 'minmax':
            # Scale to [0, 1]
            min_val = signal.min()
            max_val = signal.max()
            if max_val > min_val:
                return (signal - min_val) / (max_val - min_val)
            return signal - min_val

        elif method == 'rms':
            # Normalize by RMS
            rms = torch.sqrt(torch.mean(signal ** 2))
            if rms > 0:
                return signal / rms
            return signal

        else:
            logger.warning(f"Unknown normalization method: {method}, returning original signal")
            return signal
