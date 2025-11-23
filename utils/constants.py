"""
Project-wide constants for LSTM_PFD bearing fault diagnosis.

This module centralizes all magic numbers and constant values used throughout
the project to ensure consistency and ease of maintenance.

Purpose:
    - Signal processing parameters (sampling rate, length, duration)
    - Model architecture constants (number of classes, default dimensions)
    - Fault type definitions and mappings
    - Default hyperparameters
    - File paths and directory names

Author: Syed Abbas Ahmad
Date: 2025-11-21
Created as part of critical refactoring to eliminate magic numbers
"""

from typing import List, Dict, Tuple
from pathlib import Path

# ==============================================================================
# SIGNAL PARAMETERS
# ==============================================================================

# Sampling and signal characteristics
SAMPLING_RATE: int = 20480  # Hz - Sampling frequency
SIGNAL_DURATION: float = 5.0  # seconds - Duration of each signal
SIGNAL_LENGTH: int = 102400  # samples - Total samples per signal (fs * T = 20480 * 5)

# Frequency analysis
NYQUIST_FREQUENCY: float = SAMPLING_RATE / 2.0  # Hz - Maximum frequency without aliasing
TIME_STEP: float = 1.0 / SAMPLING_RATE  # seconds - Time between samples

# Operating conditions
NOMINAL_ROTATION_SPEED: float = 60.0  # Hz - Base rotation speed (3600 RPM)
NOMINAL_ROTATION_SPEED_RPM: int = 3600  # RPM - Alternative unit


# ==============================================================================
# FAULT CLASSIFICATION
# ==============================================================================

# Number of fault classes
NUM_CLASSES: int = 11  # Total number of bearing fault types

# Fault type names (matching data generation configuration)
FAULT_TYPES: List[str] = [
    'sain',                      # 0: Healthy/Normal operation
    'desalignement',             # 1: Misalignment
    'desequilibre',              # 2: Imbalance
    'jeu',                       # 3: Bearing clearance/looseness
    'lubrification',             # 4: Lubrication issues
    'cavitation',                # 5: Cavitation
    'usure',                     # 6: Wear
    'oilwhirl',                  # 7: Oil whirl
    'mixed_misalign_imbalance',  # 8: Mixed: Misalignment + Imbalance
    'mixed_wear_lube',           # 9: Mixed: Wear + Lubrication
    'mixed_cavit_jeu',           # 10: Mixed: Cavitation + Clearance
]

# Fault type ID mapping (for compatibility with different naming conventions)
FAULT_TYPE_TO_ID: Dict[str, int] = {fault: idx for idx, fault in enumerate(FAULT_TYPES)}

# English translations for display
FAULT_TYPE_DISPLAY_NAMES: Dict[str, str] = {
    'sain': 'Healthy',
    'desalignement': 'Misalignment',
    'desequilibre': 'Imbalance',
    'jeu': 'Bearing Clearance',
    'lubrification': 'Lubrication Issue',
    'cavitation': 'Cavitation',
    'usure': 'Wear',
    'oilwhirl': 'Oil Whirl',
    'mixed_misalign_imbalance': 'Misalignment + Imbalance',
    'mixed_wear_lube': 'Wear + Lubrication',
    'mixed_cavit_jeu': 'Cavitation + Clearance',
}

# Single fault types (excluding healthy and mixed faults)
SINGLE_FAULT_TYPES: List[str] = [
    'desalignement', 'desequilibre', 'jeu', 'lubrification',
    'cavitation', 'usure', 'oilwhirl'
]

# Mixed fault types
MIXED_FAULT_TYPES: List[str] = [
    'mixed_misalign_imbalance', 'mixed_wear_lube', 'mixed_cavit_jeu'
]

# Phase 6 PINN-compatible fault labels (English naming for physics models)
# These map to the same indices as FAULT_TYPES but use English/physics terminology
FAULT_LABELS_PINN = {
    0: 'healthy',
    1: 'misalignment',
    2: 'imbalance',
    3: 'outer_race',
    4: 'inner_race',
    5: 'ball',
    6: 'looseness',
    7: 'oil_whirl',
    8: 'cavitation',
    9: 'wear',
    10: 'lubrication'
}

# Use PINN labels as default alias for backward compatibility with Phase 6
FAULT_LABELS = FAULT_LABELS_PINN


# ==============================================================================
# MODEL ARCHITECTURE DEFAULTS
# ==============================================================================

# Default CNN architecture parameters
DEFAULT_CNN_CHANNELS: List[int] = [32, 64, 128, 256]
DEFAULT_CNN_KERNEL_SIZES: List[int] = [15, 11, 7, 5]
DEFAULT_CNN_STRIDES: List[int] = [2, 2, 2, 2]
DEFAULT_CNN_POOL_SIZES: List[int] = [4, 4, 4, 4]

# Default ResNet architecture parameters
DEFAULT_RESNET_BLOCKS: List[int] = [2, 2, 2, 2]  # ResNet-18 style
DEFAULT_RESNET_CHANNELS: List[int] = [64, 128, 256, 512]
DEFAULT_RESNET_KERNEL_SIZE: int = 7

# Default fully connected layer dimensions
DEFAULT_FC_HIDDEN_DIMS: List[int] = [512, 256]

# Regularization defaults
DEFAULT_DROPOUT: float = 0.3
DEFAULT_BATCH_NORM: bool = True


# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

# Optimization defaults
DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_NUM_EPOCHS: int = 100
DEFAULT_WEIGHT_DECAY: float = 1e-4

# Learning rate scheduler
DEFAULT_LR_PATIENCE: int = 10
DEFAULT_LR_FACTOR: float = 0.5

# Early stopping
DEFAULT_EARLY_STOPPING_PATIENCE: int = 20
DEFAULT_MIN_DELTA: float = 1e-4


# ==============================================================================
# DATA GENERATION
# ==============================================================================

# Dataset size
DEFAULT_NUM_SIGNALS_PER_FAULT: int = 100

# Noise levels
DEFAULT_NOISE_LEVEL: float = 0.03
DEFAULT_EMI_LEVEL: float = 0.01
DEFAULT_PINK_NOISE_LEVEL: float = 0.02

# Severity levels
SEVERITY_LEVELS: List[str] = ['incipient', 'mild', 'moderate', 'severe']

# Severity factor ranges (non-overlapping)
SEVERITY_RANGES: Dict[str, Tuple[float, float]] = {
    'incipient': (0.20, 0.45),
    'mild': (0.45, 0.70),
    'moderate': (0.70, 0.90),
    'severe': (0.90, 1.00),
}


# ==============================================================================
# FILE PATHS AND DIRECTORIES
# ==============================================================================

# Default directory names (relative to project root)
DEFAULT_DATA_DIR: str = 'data_signaux_sep_production'
DEFAULT_CHECKPOINT_DIR: str = 'checkpoints'
DEFAULT_LOG_DIR: str = 'logs'
DEFAULT_RESULTS_DIR: str = 'results'
DEFAULT_PLOTS_DIR: str = 'plots'

# File extensions
CHECKPOINT_EXTENSION: str = '.pt'
CONFIG_EXTENSION: str = '.yaml'
LOG_EXTENSION: str = '.log'


# ==============================================================================
# PHYSICS-INFORMED PARAMETERS
# ==============================================================================

# Sommerfeld number
DEFAULT_SOMMERFELD_BASE: float = 0.15

# Reynolds number range
DEFAULT_REYNOLDS_RANGE: Tuple[float, float] = (500.0, 5000.0)

# Clearance ratio
DEFAULT_CLEARANCE_RATIO_RANGE: Tuple[float, float] = (0.001, 0.003)

# Physics loss weight
DEFAULT_PHYSICS_LOSS_WEIGHT: float = 0.1


# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================

# Computation device preferences
DEVICE_PREFERENCE: List[str] = ['cuda', 'mps', 'cpu']  # Order of preference

# Mixed precision training
DEFAULT_USE_AMP: bool = True  # Automatic Mixed Precision


# ==============================================================================
# RANDOM SEEDS (for reproducibility)
# ==============================================================================

DEFAULT_RANDOM_SEED: int = 42
NUMPY_RANDOM_SEED: int = 42
TORCH_RANDOM_SEED: int = 42


# ==============================================================================
# VISUALIZATION
# ==============================================================================

# Plot parameters
DEFAULT_FIGSIZE: Tuple[int, int] = (12, 8)
DEFAULT_DPI: int = 100

# Color maps
DEFAULT_COLORMAP: str = 'viridis'
CONFUSION_MATRIX_COLORMAP: str = 'Blues'


# ==============================================================================
# SPECTROGRAM PARAMETERS
# ==============================================================================

# STFT parameters for spectrogram generation
DEFAULT_NPERSEG: int = 256  # Number of points per segment
DEFAULT_NOVERLAP: int = 128  # Overlap between segments
DEFAULT_NFFT: int = 256  # FFT size

# Mel-spectrogram parameters
DEFAULT_N_MELS: int = 128  # Number of mel bands
DEFAULT_FMIN: float = 0.0  # Minimum frequency
DEFAULT_FMAX: float = NYQUIST_FREQUENCY  # Maximum frequency


# ==============================================================================
# VALIDATION AND TESTING
# ==============================================================================

# Data split ratios
DEFAULT_TRAIN_RATIO: float = 0.7
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15

# Cross-validation
DEFAULT_NUM_FOLDS: int = 5


# ==============================================================================
# API AND DEPLOYMENT
# ==============================================================================

# API limits
MAX_SIGNAL_LENGTH: int = SIGNAL_LENGTH * 10  # Maximum accepted signal length
MIN_SIGNAL_LENGTH: int = 1024  # Minimum signal length for processing

# Batch inference
MAX_BATCH_SIZE: int = 128


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_fault_id(fault_name: str) -> int:
    """
    Get numeric ID for a fault type name.

    Args:
        fault_name: Name of the fault type

    Returns:
        Numeric ID (0-10)

    Raises:
        ValueError: If fault name is not recognized
    """
    if fault_name not in FAULT_TYPE_TO_ID:
        raise ValueError(
            f"Unknown fault type: '{fault_name}'. "
            f"Valid types: {list(FAULT_TYPE_TO_ID.keys())}"
        )
    return FAULT_TYPE_TO_ID[fault_name]


def get_fault_name(fault_id: int) -> str:
    """
    Get fault type name from numeric ID.

    Args:
        fault_id: Numeric ID (0-10)

    Returns:
        Fault type name

    Raises:
        ValueError: If fault ID is out of range
    """
    if not 0 <= fault_id < NUM_CLASSES:
        raise ValueError(
            f"Fault ID {fault_id} out of range. "
            f"Valid range: 0-{NUM_CLASSES-1}"
        )
    return FAULT_TYPES[fault_id]


def get_fault_display_name(fault_name: str) -> str:
    """
    Get English display name for a fault type.

    Args:
        fault_name: Internal fault type name

    Returns:
        Human-readable display name
    """
    return FAULT_TYPE_DISPLAY_NAMES.get(fault_name, fault_name)


def validate_signal_length(length: int) -> bool:
    """
    Validate that a signal length is within acceptable bounds.

    Args:
        length: Signal length to validate

    Returns:
        True if valid, False otherwise
    """
    return MIN_SIGNAL_LENGTH <= length <= MAX_SIGNAL_LENGTH


# ==============================================================================
# WEB APPLICATION - FILE SIZE LIMITS
# ==============================================================================

# File upload limits
MAX_UPLOAD_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB - Maximum file upload size
MAX_MODEL_FILE_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB - Maximum model file size
MAX_DATASET_FILE_SIZE_BYTES: int = 1024 * 1024 * 1024  # 1 GB - Maximum dataset file size

# File size conversion constants
BYTES_PER_KB: int = 1024  # Bytes in 1 kilobyte
BYTES_PER_MB: int = 1024 * 1024  # Bytes in 1 megabyte
BYTES_PER_GB: int = 1024 * 1024 * 1024  # Bytes in 1 gigabyte


# ==============================================================================
# WEB APPLICATION - SAMPLE AND DATA LIMITS
# ==============================================================================

# Default sample counts for various operations
DEFAULT_NUM_SIGNALS_PER_FAULT: int = 100  # Default signals per fault type in dataset generation
DEFAULT_BACKGROUND_SAMPLES: int = 100  # Default number of background samples for XAI methods
DEFAULT_PERTURBATIONS: int = 1000  # Default number of perturbations for LIME

# Maximum sample limits to prevent memory issues
MAX_SAMPLES_PER_DATASET: int = 20480  # Maximum samples per dataset to prevent memory overflow
MAX_VISUALIZATION_SAMPLES: int = 1000  # Maximum samples to visualize at once
MAX_PREVIEW_SAMPLES: int = 100  # Maximum samples for preview displays
MAX_DATASET_LIST_LIMIT: int = 100  # Maximum datasets to list at once


# ==============================================================================
# WEB APPLICATION - PAGINATION
# ==============================================================================

DEFAULT_PAGE_SIZE: int = 50  # Default number of items per page
MAX_PAGE_SIZE: int = 500  # Maximum page size to prevent performance issues
DEFAULT_RECENT_ITEMS_LIMIT: int = 5  # Number of recent items to show in lists


# ==============================================================================
# WEB APPLICATION - CACHE TTL (Time To Live)
# ==============================================================================

CACHE_TTL_SHORT: int = 300  # 5 minutes - For frequently changing data
CACHE_TTL_MEDIUM: int = 600  # 10 minutes - For moderately stable data
CACHE_TTL_LONG: int = 3600  # 1 hour - For stable data


# ==============================================================================
# WEB APPLICATION - RATE LIMITS
# ==============================================================================

API_RATE_LIMIT_PER_HOUR: int = 1000  # Maximum API requests per hour
EMAIL_RATE_LIMIT_PER_MINUTE: int = 100  # Maximum emails to send per minute
WEBHOOK_RATE_LIMIT_PER_SECOND: int = 1  # Maximum webhook calls per second


# ==============================================================================
# WEB APPLICATION - TRAINING LIMITS
# ==============================================================================

# Epoch limits
MIN_EPOCHS: int = 1  # Minimum training epochs
MAX_EPOCHS: int = 1000  # Maximum training epochs
DEFAULT_EPOCHS_FALLBACK: int = 100  # Fallback value when epochs not specified

# Learning rate defaults
DEFAULT_LEARNING_RATE_FALLBACK: float = 0.001  # Fallback learning rate

# Training timeouts
MAX_TRAINING_DURATION_SECONDS: int = 7200  # 2 hours - Maximum training time
MAX_TESTING_DURATION_SECONDS: int = 1800  # 30 minutes - Maximum testing time
MAX_TASK_DURATION_SECONDS: int = 600  # 10 minutes - Maximum general task time
MAX_OPERATION_TIMEOUT_SECONDS: int = 300  # 5 minutes - Maximum operation timeout


# ==============================================================================
# WEB APPLICATION - STRING LENGTHS
# ==============================================================================

MAX_TAG_LENGTH: int = 50  # Maximum length of tags
MAX_NAME_LENGTH: int = 255  # Maximum length of names/titles
MAX_DESCRIPTION_LENGTH: int = 2000  # Maximum length of descriptions


# ==============================================================================
# WEB APPLICATION - DATA GENERATION DEFAULTS
# ==============================================================================

# Default percentage values (stored as integers, converted to decimals in code)
DEFAULT_SPEED_VARIATION_PERCENT: int = 10  # Default speed variation percentage
DEFAULT_AUGMENTATION_RATIO_PERCENT: int = 30  # Default augmentation ratio percentage
DEFAULT_LOAD_RANGE_MIN_PERCENT: int = 30  # Default minimum load percentage
DEFAULT_LOAD_RANGE_MAX_PERCENT: int = 100  # Default maximum load percentage

# Temperature range defaults
DEFAULT_TEMP_RANGE_MIN: int = 40  # Default minimum temperature (°C)
DEFAULT_TEMP_RANGE_MAX: int = 80  # Default maximum temperature (°C)

# Estimation constants
SIGNALS_PER_MINUTE_GENERATION: int = 50  # Estimated signals generated per minute
FILES_PER_MINUTE_IMPORT: int = 10  # Estimated files imported per minute


# ==============================================================================
# WEB APPLICATION - HYPERPARAMETER OPTIMIZATION RANGES
# ==============================================================================

# Random Forest defaults
RF_N_ESTIMATORS_MIN: int = 100  # Minimum number of trees
RF_N_ESTIMATORS_MAX: int = 1000  # Maximum number of trees
RF_N_ESTIMATORS_STEP: int = 100  # Step size for n_estimators
RF_MAX_DEPTH_MIN: int = 10  # Minimum tree depth
RF_MAX_DEPTH_MAX: int = 100  # Maximum tree depth
RF_MAX_DEPTH_STEP: int = 10  # Step size for max_depth
RF_N_ESTIMATORS_DEFAULT: int = 100  # Default number of estimators for feature importance

# SVM defaults
SVM_GAMMA_MIN: float = 0.001  # Minimum gamma value
SVM_GAMMA_MAX: float = 1.0  # Maximum gamma value
SVM_GAMMA_STEP: float = 0.001  # Step size for gamma

# Neural network architecture ranges
NN_FILTERS_MIN: int = 32  # Minimum number of filters/channels
NN_FILTERS_MAX: int = 256  # Maximum number of filters/channels
NN_FILTERS_STEP: int = 32  # Step size for filters
NN_CHANNEL_SIZES_OPTIONS: list = [16, 32, 64, 128]  # Channel size options for NAS
NN_D_MODEL_OPTIONS: list = [128, 256, 512]  # Transformer d_model options

# Transformer defaults
TRANSFORMER_D_MODEL_MIN: int = 128  # Minimum d_model dimension
TRANSFORMER_D_MODEL_MAX: int = 512  # Maximum d_model dimension
TRANSFORMER_D_MODEL_STEP: int = 64  # Step size for d_model

# Progressive training defaults
PROGRESSIVE_START_SIZE_DEFAULT: int = 51200  # Default starting signal size for progressive training
PROGRESSIVE_END_SIZE_DEFAULT: int = 102400  # Default ending signal size for progressive training


# ==============================================================================
# WEB APPLICATION - DEPLOYMENT CONSTANTS
# ==============================================================================

# ONNX export defaults
DEFAULT_ONNX_OPSET_VERSION: int = 14  # Default ONNX opset version
DEFAULT_ONNX_INPUT_SHAPE: tuple = (1, 1, 102400)  # Default input shape for ONNX export

# Model benchmarking
DEFAULT_BENCHMARK_RUNS: int = 100  # Number of runs for benchmarking
BENCHMARK_WARMUP_RUNS: int = 10  # Number of warmup runs before benchmarking
MILLISECONDS_PER_SECOND: int = 1000  # Conversion factor for ms to seconds

# Model pruning defaults
DEFAULT_PRUNING_AMOUNT: float = 0.3  # Default pruning amount (30% of parameters)


# ==============================================================================
# WEB APPLICATION - VISUALIZATION DEFAULTS
# ==============================================================================

# Plot dimensions
DEFAULT_PLOT_HEIGHT: int = 600  # Default plot height in pixels
SMALL_PLOT_HEIGHT: int = 400  # Smaller plot height
DYNAMIC_PLOT_HEIGHT_PER_ITEM: int = 20  # Height per item for dynamic plots
MIN_PLOT_HEIGHT: int = 400  # Minimum plot height

# Spectrogram scales
WAVELET_SCALES_MIN: int = 1  # Minimum wavelet scale
WAVELET_SCALES_MAX: int = 128  # Maximum wavelet scale

# Sample data sizes for visualizations
POWER_SPECTRUM_DISPLAY_SAMPLES: int = 500  # Number of power spectrum samples to display
RANDOM_SAMPLE_SIZE: int = 100  # Default random sample size for visualizations
SALIENCY_MAP_SIZE: int = 1000  # Default saliency map size
ACTIVATION_MAP_FILTERS: int = 32  # Number of activation map filters
ACTIVATION_MAP_LENGTH: int = 100  # Length of activation maps

# Counterfactual visualization
COUNTERFACTUAL_SIGNAL_LENGTH: int = 1000  # Length of counterfactual signals
COUNTERFACTUAL_NOISE_FACTOR: float = 0.3  # Noise factor for counterfactual generation


# ==============================================================================
# WEB APPLICATION - HTTP STATUS CODES
# ==============================================================================

# Success status codes
HTTP_STATUS_OK: int = 200  # Standard success response
HTTP_STATUS_CREATED: int = 201  # Resource created successfully
HTTP_STATUS_ACCEPTED: int = 202  # Request accepted for processing

# Client error status codes
HTTP_STATUS_BAD_REQUEST: int = 400  # Bad request
HTTP_STATUS_UNAUTHORIZED: int = 401  # Unauthorized
HTTP_STATUS_FORBIDDEN: int = 403  # Forbidden
HTTP_STATUS_NOT_FOUND: int = 404  # Not found

# Success range
HTTP_SUCCESS_MIN: int = 200  # Minimum success status code
HTTP_SUCCESS_MAX: int = 300  # Maximum success status code (exclusive)
HTTP_ERROR_MIN: int = 400  # Minimum error status code


# ==============================================================================
# WEB APPLICATION - PERCENTAGE CONVERSION
# ==============================================================================

PERCENT_MULTIPLIER: int = 100  # Multiplier to convert decimal to percentage
PERCENT_DIVISOR: int = 100  # Divisor to convert percentage to decimal


# ==============================================================================
# WEB APPLICATION - MONITORING AND METRICS
# ==============================================================================

# API monitoring defaults
DEFAULT_MONITORING_HOURS: int = 24  # Default hours to look back for monitoring data
DEFAULT_RECENT_REQUESTS_LIMIT: int = 100  # Default limit for recent requests

# System monitoring
BYTES_TO_GB_DIVISOR: int = 1024 ** 3  # Divisor to convert bytes to gigabytes


# ==============================================================================
# WEB APPLICATION - TESTING DEFAULTS
# ==============================================================================

DEFAULT_TEST_SAMPLES: int = 100  # Default number of test samples
MAX_PASS_RATE_PERCENT: int = 100  # Maximum pass rate percentage


# ==============================================================================
# WEB APPLICATION - NOISE LAYERS
# ==============================================================================

# Number of available noise layers in data generation
TOTAL_NOISE_LAYERS: int = 7  # Total number of noise layer options


# ==============================================================================
# WEB APPLICATION - API KEY DEFAULTS
# ==============================================================================

DEFAULT_API_KEY_EXPIRY_DAYS: int = 365  # Default API key expiration (1 year)


# ==============================================================================
# WEB APPLICATION - LOCAL DEVELOPMENT
# ==============================================================================

# Local server defaults
DEFAULT_LOCAL_PORT: int = 8050  # Default port for local Dash server
DEFAULT_LOCAL_HOST: str = "localhost"  # Default hostname


# ==============================================================================
# VERSION INFO
# ==============================================================================

__version__ = '1.0.0'
__author__ = 'Syed Abbas Ahmad'
__date__ = '2025-11-21'
