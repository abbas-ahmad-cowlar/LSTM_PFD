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

Author: LSTM_PFD Team
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
# VERSION INFO
# ==============================================================================

__version__ = '1.0.0'
__author__ = 'LSTM_PFD Team'
__date__ = '2025-11-21'
