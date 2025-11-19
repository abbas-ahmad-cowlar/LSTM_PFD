"""Data generation and management module for LSTM_PFD pipeline."""

from .signal_generator import SignalGenerator, FaultModeler, NoiseGenerator
from .matlab_importer import MatlabImporter, MatlabSignalData, load_matlab_reference
from .data_validator import SignalValidator, ValidationResult, validate_against_matlab
from .augmentation import SignalAugmenter, random_augment
from .dataset import (
    BearingFaultDataset,
    AugmentedBearingDataset,
    CachedBearingDataset,
    train_val_test_split,
    collate_fn_with_metadata
)
from .dataloader import (
    create_dataloader,
    create_train_val_test_loaders,
    worker_init_fn_seed,
    compute_class_weights,
    InfiniteDataLoader,
    prefetch_to_device
)

__all__ = [
    # Generation
    'SignalGenerator',
    'FaultModeler',
    'NoiseGenerator',
    # MATLAB import/validation
    'MatlabImporter',
    'MatlabSignalData',
    'load_matlab_reference',
    'SignalValidator',
    'ValidationResult',
    'validate_against_matlab',
    # Augmentation
    'SignalAugmenter',
    'random_augment',
    # Datasets
    'BearingFaultDataset',
    'AugmentedBearingDataset',
    'CachedBearingDataset',
    'train_val_test_split',
    'collate_fn_with_metadata',
    # DataLoaders
    'create_dataloader',
    'create_train_val_test_loaders',
    'worker_init_fn_seed',
    'compute_class_weights',
    'InfiniteDataLoader',
    'prefetch_to_device'
]
