"""
Global constants for the Dash application.
"""
import sys
from pathlib import Path

from dashboard_config import FAULT_CLASSES, COLOR_PALETTE

# Decoupled from parent project to avoid circular imports
NUM_CLASSES = 11
SIGNAL_LENGTH = 102400
SAMPLING_RATE = 20480

# Fault class to index mapping
FAULT_CLASS_TO_IDX = {fault: idx for idx, fault in enumerate(FAULT_CLASSES)}
IDX_TO_FAULT_CLASS = {idx: fault for idx, fault in enumerate(FAULT_CLASSES)}

# Fault class display names (aligned with FAULT_CLASSES order)
FAULT_CLASS_NAMES = {
    "normal": "Normal",
    "misalignment": "Misalignment",
    "imbalance": "Imbalance",
    "looseness": "Looseness",
    "lubrication": "Lubrication Issue",
    "cavitation": "Cavitation",
    "wear": "Wear",
    "oil_whirl": "Oil Whirl",
    "combined_misalign_imbalance": "Misalignment + Imbalance",
    "combined_wear_lube": "Wear + Lubrication",
    "combined_cavit_jeu": "Cavitation + Looseness"
}

# Color mapping for fault classes
FAULT_CLASS_COLORS = {fault: color for fault, color in zip(FAULT_CLASSES, COLOR_PALETTE)}

# Severity levels
SEVERITY_LEVELS = ["incipient", "mild", "moderate", "severe"]

# Model types
MODEL_TYPES = [
    "random_forest", "svm", "cnn1d", "resnet18", "resnet34", "resnet50",
    "efficientnet", "transformer", "spectrogram_cnn", "pinn", "ensemble"
]

MODEL_TYPE_NAMES = {
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "cnn1d": "1D CNN",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "efficientnet": "EfficientNet",
    "transformer": "Transformer",
    "spectrogram_cnn": "Spectrogram CNN",
    "pinn": "Physics-Informed NN",
    "ensemble": "Ensemble"
}

# Experiment status
EXPERIMENT_STATUS = ["pending", "running", "paused", "completed", "failed", "cancelled"]

# HPO methods
HPO_METHODS = ["grid_search", "random_search", "bayesian", "hyperband"]
HPO_METHOD_NAMES = {
    "grid_search": "Grid Search",
    "random_search": "Random Search",
    "bayesian": "Bayesian Optimization",
    "hyperband": "Hyperband"
}

# XAI methods
XAI_METHODS = ["shap", "lime", "integrated_gradients", "grad_cam", "attention"]
XAI_METHOD_NAMES = {
    "shap": "SHAP",
    "lime": "LIME",
    "integrated_gradients": "Integrated Gradients",
    "grad_cam": "Grad-CAM",
    "attention": "Attention Weights"
}

# Feature names (36 features from Phase 1)
FEATURE_NAMES = [
    # Time domain (7 features)
    "rms", "kurtosis", "skewness", "peak_value", "crest_factor", "impulse_factor", "clearance_factor",
    # Frequency domain (12 features)
    "spectral_centroid", "spectral_spread", "spectral_entropy", "spectral_flatness",
    "dominant_frequency", "peak_frequency_amplitude", "frequency_std",
    "spectral_kurtosis", "spectral_skewness", "total_power", "band_power_low", "band_power_high",
    # Envelope features (4 features)
    "envelope_rms", "envelope_kurtosis", "envelope_peak", "envelope_mean",
    # Additional features (13 features)
    "zero_crossing_rate", "mean_abs_deviation", "variance", "std_dev",
    "median_abs_deviation", "interquartile_range", "energy",
    "shape_factor", "margin_factor", "sample_entropy", "approximate_entropy",
    "permutation_entropy", "correlation_dimension"
]

# ============================================================================
# Data Generation Constants
# ============================================================================
DEFAULT_NUM_SIGNALS_PER_FAULT = 100
DEFAULT_SPEED_VARIATION_PERCENT = 10
DEFAULT_LOAD_RANGE_MIN_PERCENT = 30
DEFAULT_LOAD_RANGE_MAX_PERCENT = 100
DEFAULT_TEMP_RANGE_MIN = 40.0
DEFAULT_TEMP_RANGE_MAX = 80.0
DEFAULT_AUGMENTATION_RATIO_PERCENT = 30
DEFAULT_RANDOM_SEED = 42
PERCENT_DIVISOR = 100
PERCENT_MULTIPLIER = 100
SIGNALS_PER_MINUTE_GENERATION = 100  # Estimated signals generated per minute
DEFAULT_RECENT_ITEMS_LIMIT = 10
TOTAL_NOISE_LAYERS = 7

# ============================================================================
# Dataset and Experiment Constants
# ============================================================================
MAX_DATASET_LIST_LIMIT = 100
MAX_EXPERIMENT_LIST_LIMIT = 500

# ============================================================================
# Hyperparameter Ranges - Random Forest
# ============================================================================
RF_N_ESTIMATORS_MIN = 10
RF_N_ESTIMATORS_MAX = 500
RF_N_ESTIMATORS_STEP = 10
RF_MAX_DEPTH_MIN = 2
RF_MAX_DEPTH_MAX = 50
RF_MAX_DEPTH_STEP = 2

# ============================================================================
# Hyperparameter Ranges - SVM
# ============================================================================
SVM_GAMMA_MIN = 0.001
SVM_GAMMA_MAX = 1.0
SVM_GAMMA_STEP = 0.001

# ============================================================================
# Hyperparameter Ranges - Neural Networks
# ============================================================================
NN_FILTERS_MIN = 16
NN_FILTERS_MAX = 256
NN_FILTERS_STEP = 16

# ============================================================================
# Hyperparameter Ranges - Transformer
# ============================================================================
TRANSFORMER_D_MODEL_MIN = 64
TRANSFORMER_D_MODEL_MAX = 512
TRANSFORMER_D_MODEL_STEP = 64

# ============================================================================
# Training Defaults
# ============================================================================
DEFAULT_EPOCHS_FALLBACK = 100
DEFAULT_LEARNING_RATE_FALLBACK = 0.001
PROGRESSIVE_START_SIZE_DEFAULT = 50
PROGRESSIVE_END_SIZE_DEFAULT = 100

# ============================================================================
# File and Data Size Constants
# ============================================================================
BYTES_PER_MB = 1024 * 1024
FILES_PER_MINUTE_IMPORT = 10  # Estimated files imported per minute

# ============================================================================
# Deployment Constants
# ============================================================================
DEFAULT_ONNX_INPUT_SHAPE = (1, 1, SIGNAL_LENGTH)  # Batch, Channels, Signal Length
DEFAULT_ONNX_OPSET_VERSION = 11
DEFAULT_BENCHMARK_RUNS = 100
BENCHMARK_WARMUP_RUNS = 10
MILLISECONDS_PER_SECOND = 1000
DEFAULT_PRUNING_AMOUNT = 0.3  # 30% pruning by default

# ============================================================================
# Pagination Constants
# ============================================================================
DEFAULT_PAGE_SIZE = 20
