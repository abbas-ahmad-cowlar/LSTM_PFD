"""
Global constants for the Dash application.
"""
import sys
from pathlib import Path

# Add parent directory to path to import from main project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FAULT_CLASSES, COLOR_PALETTE
import utils.constants as parent_constants

# Import core constants from parent project
NUM_CLASSES = parent_constants.NUM_CLASSES
SIGNAL_LENGTH = parent_constants.SIGNAL_LENGTH
SAMPLING_RATE = parent_constants.SAMPLING_RATE

# Fault class to index mapping
FAULT_CLASS_TO_IDX = {fault: idx for idx, fault in enumerate(FAULT_CLASSES)}
IDX_TO_FAULT_CLASS = {idx: fault for idx, fault in enumerate(FAULT_CLASSES)}

# Fault class display names
FAULT_CLASS_NAMES = {
    "normal": "Normal",
    "ball_fault": "Ball Fault",
    "inner_race": "Inner Race Fault",
    "outer_race": "Outer Race Fault",
    "combined": "Combined Fault",
    "imbalance": "Imbalance",
    "misalignment": "Misalignment",
    "oil_whirl": "Oil Whirl",
    "cavitation": "Cavitation",
    "looseness": "Looseness",
    "oil_deficiency": "Oil Deficiency"
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
