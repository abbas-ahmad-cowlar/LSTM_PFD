"""
Configuration Validator

Validates master configuration file for unified pipeline.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

import yaml
import json
from typing import Dict, List, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate master configuration file.

    Checks:
    - All required sections present
    - Value ranges are valid
    - File paths exist
    - Hyperparameter consistency

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> config = load_config('configs/default_config.yaml')
        >>> is_valid = validate_config(config)
        >>> assert is_valid, "Invalid configuration"
    """
    logger.info("Validating configuration...")

    # Check required sections
    _validate_required_sections(config)

    # Check value ranges
    _validate_value_ranges(config)

    # Check file paths
    _validate_file_paths(config)

    # Check hyperparameter consistency
    _validate_hyperparameters(config)

    logger.info("✓ Configuration validated successfully")
    return True


def _validate_required_sections(config: Dict[str, Any]):
    """Validate all required sections are present."""
    required_sections = [
        'data',
        'classical',
        'deep_learning',
        'deployment'
    ]

    missing = []
    for section in required_sections:
        if section not in config:
            missing.append(section)

    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    logger.info(f"✓ All required sections present: {required_sections}")


def _validate_value_ranges(config: Dict[str, Any]):
    """Validate configuration values are in valid ranges."""
    # Data config
    if 'data' in config:
        data_config = config['data']

        # Train ratio
        if 'train_ratio' in data_config:
            train_ratio = data_config['train_ratio']
            if not (0.5 <= train_ratio <= 0.9):
                raise ValueError(
                    f"Invalid train_ratio: {train_ratio} (must be 0.5-0.9)"
                )

        # Val ratio
        if 'val_ratio' in data_config:
            val_ratio = data_config['val_ratio']
            if not (0.05 <= val_ratio <= 0.3):
                raise ValueError(
                    f"Invalid val_ratio: {val_ratio} (must be 0.05-0.3)"
                )

    # Deep learning config
    if 'deep_learning' in config:
        dl_config = config['deep_learning']

        # Batch size
        if 'batch_size' in dl_config:
            batch_size = dl_config['batch_size']
            if batch_size <= 0:
                raise ValueError(
                    f"Invalid batch_size: {batch_size} (must be > 0)"
                )

        # Learning rate
        if 'learning_rate' in dl_config:
            lr = dl_config['learning_rate']
            if not (1e-6 <= lr <= 1.0):
                raise ValueError(
                    f"Invalid learning_rate: {lr} (must be 1e-6 to 1.0)"
                )

        # Epochs
        if 'epochs' in dl_config:
            epochs = dl_config['epochs']
            if epochs <= 0 or epochs > 10000:
                raise ValueError(
                    f"Invalid epochs: {epochs} (must be 1-10000)"
                )

    logger.info("✓ All value ranges validated")


def _validate_file_paths(config: Dict[str, Any]):
    """Validate file paths exist."""
    # Check data directories
    if 'data' in config:
        data_config = config['data']

        if 'signal_dirs' in data_config:
            for signal_dir in data_config['signal_dirs']:
                path = Path(signal_dir)
                if not path.exists():
                    logger.warning(
                        f"Data directory not found: {signal_dir} "
                        "(will be created if needed)"
                    )

        if 'cache_path' in data_config:
            cache_path = Path(data_config['cache_path'])
            if not cache_path.exists():
                logger.warning(
                    f"Cache file not found: {cache_path} "
                    "(will be created if needed)"
                )

    logger.info("✓ File paths validated")


def _validate_hyperparameters(config: Dict[str, Any]):
    """Validate hyperparameter consistency across phases."""
    # Check if batch sizes are consistent
    batch_sizes = []

    if 'deep_learning' in config and 'batch_size' in config['deep_learning']:
        batch_sizes.append(('deep_learning', config['deep_learning']['batch_size']))

    if 'deployment' in config and 'batch_size' in config['deployment']:
        batch_sizes.append(('deployment', config['deployment']['batch_size']))

    # Warn if batch sizes differ significantly
    if len(batch_sizes) > 1:
        values = [bs for _, bs in batch_sizes]
        if max(values) / min(values) > 2:
            logger.warning(
                f"Batch sizes vary significantly across phases: {batch_sizes}"
            )

    logger.info("✓ Hyperparameters validated")


def suggest_config_optimizations(config: Dict[str, Any]) -> List[str]:
    """
    Suggest configuration optimizations based on best practices.

    Args:
        config: Configuration dictionary

    Returns:
        List of optimization suggestions

    Example:
        >>> suggestions = suggest_config_optimizations(config)
        >>> for suggestion in suggestions:
        ...     print(f"- {suggestion}")
    """
    suggestions = []

    # Check batch size for GPU memory efficiency
    if 'deep_learning' in config:
        dl_config = config['deep_learning']
        batch_size = dl_config.get('batch_size', 32)

        if batch_size < 32:
            suggestions.append(
                f"Consider increasing batch_size from {batch_size} to 32-64 "
                "for better GPU utilization"
            )
        elif batch_size > 128:
            suggestions.append(
                f"Batch size {batch_size} may be too large and cause "
                "OOM errors on smaller GPUs"
            )

    # Check learning rate scheduler
    if 'deep_learning' in config:
        dl_config = config['deep_learning']
        if 'scheduler' not in dl_config:
            suggestions.append(
                "Consider adding a learning rate scheduler "
                "(e.g., cosine annealing) for better convergence"
            )

    # Check data augmentation
    if 'data' in config:
        data_config = config['data']
        if not data_config.get('augmentation', False):
            suggestions.append(
                "Consider enabling data augmentation to improve "
                "model generalization"
            )

    return suggestions


def generate_config_template(output_path: str = 'configs/template_config.yaml'):
    """
    Generate configuration template for new users.

    Args:
        output_path: Path to save template

    Example:
        >>> generate_config_template('configs/my_config.yaml')
    """
    template = {
        'data': {
            'signal_dirs': ['data/raw/bearing_data'],
            'cache_path': 'data/processed/signals_cache.h5',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'augmentation': True,
            'num_signals_per_fault': 130
        },
        'classical': {
            'models': ['random_forest', 'svm', 'xgboost'],
            'feature_selection': 'mrmr',
            'n_features': 15,
            'optimize_hyperparams': True
        },
        'deep_learning': {
            'model': 'resnet34',
            'batch_size': 64,
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'mixed_precision': True
        },
        'ensemble': {
            'voting': 'soft',
            'weights': None  # Auto-optimize
        },
        'deployment': {
            'quantization': 'dynamic',
            'export_onnx': True,
            'batch_size': 32
        },
        'run_phase_0': True,
        'run_phase_1': True,
        'run_phase_2_4': True,
        'run_phase_5': False,
        'run_phase_6': False,
        'run_phase_7': False,
        'run_phase_8': True,
        'run_phase_9': True
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✓ Configuration template saved: {output_path}")


# ------------------------------------------------------------------ #
#  Model-specific config validation  (used by adapters)
# ------------------------------------------------------------------ #

# Classical ML model types (not in MODEL_REGISTRY — they use sklearn)
CLASSICAL_MODEL_TYPES = {'rf', 'svm', 'gbm', 'random_forest', 'gradient_boosting'}


def validate_model_config(
    model_type: str,
    hyperparameters: Dict[str, Any] | None = None,
) -> bool:
    """
    Validate a model training config before dispatching to an adapter.

    Checks:
      - model_type is known (in MODEL_REGISTRY or classical list)
      - learning_rate is in (0, 10)
      - batch_size is in [1, 4096]
      - num_epochs / n_trials is positive
      - weight_decay is non-negative

    Args:
        model_type:       e.g. 'resnet34_1d', 'rf', 'cnn_transformer'
        hyperparameters:  optional dict of training hyperparameters

    Returns:
        True if valid

    Raises:
        ValueError: with detail message if invalid

    Example:
        >>> validate_model_config('resnet34_1d', {'lr': 0.001})
        True
        >>> validate_model_config('nonexistent_model')
        ValueError: Unknown model_type 'nonexistent_model'. ...
    """
    errors: List[str] = []
    hp = hyperparameters or {}

    # ---- model_type ---- #
    mt_lower = model_type.lower().strip()
    if mt_lower not in CLASSICAL_MODEL_TYPES:
        try:
            import sys
            from pathlib import Path as _Path
            project_root = str(_Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from packages.core.models.model_factory import list_available_models
            available = list_available_models()
            if mt_lower not in available:
                errors.append(
                    f"Unknown model_type '{model_type}'. "
                    f"Available DL models: {available[:10]}... "
                    f"Classical models: {sorted(CLASSICAL_MODEL_TYPES)}"
                )
        except ImportError:
            # model_factory not importable — skip check
            pass

    # ---- learning rate ---- #
    lr = hp.get('lr', hp.get('learning_rate'))
    if lr is not None:
        lr = float(lr)
        if not (0 < lr < 10):
            errors.append(f"learning_rate={lr} out of range (0, 10)")

    # ---- batch size ---- #
    bs = hp.get('batch_size')
    if bs is not None:
        bs = int(bs)
        if not (1 <= bs <= 4096):
            errors.append(f"batch_size={bs} out of range [1, 4096]")

    # ---- epochs / trials ---- #
    for key in ('num_epochs', 'epochs', 'n_trials'):
        val = hp.get(key)
        if val is not None and int(val) < 1:
            errors.append(f"{key}={val} must be ≥ 1")

    # ---- weight decay ---- #
    wd = hp.get('weight_decay')
    if wd is not None and float(wd) < 0:
        errors.append(f"weight_decay={wd} must be ≥ 0")

    if errors:
        raise ValueError(
            f"Config validation failed for model_type='{model_type}': "
            + "; ".join(errors)
        )

    return True
