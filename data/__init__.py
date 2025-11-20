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
from .transforms import (
    Compose,
    Normalize,
    Resample,
    BandpassFilter,
    LowpassFilter,
    HighpassFilter,
    ToTensor,
    Unsqueeze,
    Detrend,
    get_default_transform
)
from .cache_manager import (
    CacheManager,
    cache_dataset_simple,
    load_cached_dataset_simple
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
    'prefetch_to_device',
    # Transforms
    'Compose',
    'Normalize',
    'Resample',
    'BandpassFilter',
    'LowpassFilter',
    'HighpassFilter',
    'ToTensor',
    'Unsqueeze',
    'Detrend',
    'get_default_transform',
    # Caching
    'CacheManager',
    'cache_dataset_simple',
    'load_cached_dataset_simple'
]
