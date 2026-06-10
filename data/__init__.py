"""Data generation and management module for the LSTM_PFD pipeline.

Kept surface (Convergence Plan Part I §5): the physics-based signal
generator, BearingFaultDataset (+HDF5), dataloaders, transforms,
signal validation, augmentation, and caching. CNN/TFR/streaming dataset
variants were pruned 2026-06 (tag `pre-convergence-2026-06`).
"""

from .signal_generation import (
    SignalGenerator,
    FaultModeler,
    NoiseGenerator,
    SignalMetadata,
)
from .matlab_importer import MatlabImporter, MatlabSignalData, load_matlab_reference
from .dataset import (
    BearingFaultDataset,
    AugmentedBearingDataset,
    CachedBearingDataset,
    WindowedView,
    train_val_test_split,
    collate_fn_with_metadata
)
from .dataloader import (
    create_dataloader,
    create_cnn_dataloader,
    create_cnn_dataloaders,
    create_train_val_test_loaders,
    worker_init_fn_seed,
    compute_class_weights,
    InfiniteDataLoader,
    prefetch_to_device,
    collate_signals,
    DataLoaderConfig,
)
from .signal_validation import (
    validate_signal,
    validate_batch,
    ValidationReport,
    SignalValidationError,
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
    'SignalMetadata',
    # MATLAB import
    'MatlabImporter',
    'MatlabSignalData',
    'load_matlab_reference',
    # Datasets
    'BearingFaultDataset',
    'AugmentedBearingDataset',
    'CachedBearingDataset',
    'WindowedView',
    'train_val_test_split',
    'collate_fn_with_metadata',
    # DataLoaders
    'create_dataloader',
    'create_cnn_dataloader',
    'create_cnn_dataloaders',
    'create_train_val_test_loaders',
    'worker_init_fn_seed',
    'compute_class_weights',
    'InfiniteDataLoader',
    'prefetch_to_device',
    'collate_signals',
    'DataLoaderConfig',
    # Signal validation
    'validate_signal',
    'validate_batch',
    'ValidationReport',
    'SignalValidationError',
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
