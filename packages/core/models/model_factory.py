"""
Model Factory for Creating and Managing Models

Provides a centralized interface for:
- Model instantiation
- Loading pretrained weights
- Model registration
- Configuration-based creation

The registry holds exactly the curated zoo (Convergence Plan Part I §4):
Tier 1 benchmark models + Tier 2 extension models. One honest key per
architecture — no aliases. Pruned architectures are recoverable from tag
`pre-convergence-2026-06`.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from pathlib import Path

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .base_model import BaseModel

# Tier 1
from .cnn.cnn_1d import CNN1D, create_cnn1d
from .cnn.attention_cnn import AttentionCNN1D
from .hybrid.cnn_lstm import CNNLSTM, create_cnn_lstm
from .resnet.resnet_1d import ResNet1D, create_resnet18_1d
from .transformer.patchtst import PatchTST
from .pinn.hybrid_pinn import HybridPINN, create_hybrid_pinn
from .pinn.physics_constrained_cnn import (
    PhysicsConstrainedCNN,
    create_physics_constrained_cnn,
)
from .pinn.multitask_pinn import MultitaskPINN, create_multitask_pinn
from .ensemble.voting_ensemble import VotingEnsemble, create_voting_ensemble

# Tier 2 (extension — smoke-tested, benchmark-optional)
from .cnn.multi_scale_cnn import MultiScaleCNN1D
from .resnet.se_resnet import SEResNet1D, create_se_resnet18_1d
from .transformer.signal_transformer import SignalTransformer, create_transformer


# ---------------------------------------------------------------------------
# Factory wrapper functions for models that lack them in their modules
# ---------------------------------------------------------------------------

def create_patchtst(num_classes: int = NUM_CLASSES, **kwargs) -> PatchTST:
    """Create a PatchTST model (Nie et al., 2023)."""
    return PatchTST(num_classes=num_classes, **kwargs)


def create_attention_cnn(num_classes: int = NUM_CLASSES, **kwargs) -> AttentionCNN1D:
    """Create an AttentionCNN1D model."""
    return AttentionCNN1D(num_classes=num_classes, **kwargs)


def create_multi_scale_cnn(num_classes: int = NUM_CLASSES, **kwargs) -> MultiScaleCNN1D:
    """Create a MultiScaleCNN1D model."""
    return MultiScaleCNN1D(num_classes=num_classes, **kwargs)


def _create_multitask_pinn(num_classes: int = NUM_CLASSES, **kwargs) -> MultitaskPINN:
    """Adapter for create_multitask_pinn (uses num_fault_classes internally)."""
    return create_multitask_pinn(num_fault_classes=num_classes, **kwargs)


# Model registry: one honest key per curated architecture.
MODEL_REGISTRY = {
    # Tier 1 — core benchmark zoo
    'cnn1d': create_cnn1d,
    'attention_cnn': create_attention_cnn,
    'cnn_lstm': create_cnn_lstm,
    'resnet18': create_resnet18_1d,
    'patchtst': create_patchtst,
    'hybrid_pinn': create_hybrid_pinn,
    'physics_constrained_cnn': create_physics_constrained_cnn,
    'multitask_pinn': _create_multitask_pinn,

    # Tier 2 — extension zoo
    'multi_scale_cnn': create_multi_scale_cnn,
    'se_resnet18': create_se_resnet18_1d,
    'signal_transformer': create_transformer,
}


def register_model(name_or_fn=None, name: str = None):
    """
    Register a new model type.

    Can be used as a function or as a decorator:

        # As a function:
        register_model('my_model', my_creation_fn)

        # As a decorator:
        @register_model('my_model')
        def create_my_model(num_classes=11, **kwargs):
            ...

    Args:
        name_or_fn: Model name (str) when used as decorator, or
                     model name (str) when used as function with 2 args.
        name: Model name when used as function with keyword args.
    """
    # Usage: register_model('name', fn)
    if isinstance(name_or_fn, str) and name is None:
        model_name = name_or_fn
        def decorator(fn):
            MODEL_REGISTRY[model_name.lower()] = fn
            return fn
        return decorator
    # Usage: register_model(name='name', creation_fn=fn)  (legacy)
    elif name_or_fn is not None and callable(name_or_fn) and name is not None:
        MODEL_REGISTRY[name.lower()] = name_or_fn
    # Usage: register_model('name', fn)  via positional args
    elif isinstance(name_or_fn, str):
        def _register(creation_fn):
            MODEL_REGISTRY[name_or_fn.lower()] = creation_fn
        return _register
    else:
        raise ValueError("register_model requires a name string")


def list_available_models() -> List[str]:
    """
    List all registered model names.

    Returns:
        List of model names
    """
    return sorted(MODEL_REGISTRY.keys())


def create_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    **kwargs
) -> BaseModel:
    """
    Create a model by name.

    Args:
        model_name: Name of the model ('cnn1d', 'resnet18', etc.)
        num_classes: Number of output classes (default: NUM_CLASSES from constants)
        **kwargs: Additional model-specific arguments

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not registered

    Example:
        >>> model = create_model('cnn1d', num_classes=NUM_CLASSES, dropout=0.3)
        >>> model = create_model('resnet18', num_classes=NUM_CLASSES)
    """
    model_name_lower = model_name.lower()

    if model_name_lower not in MODEL_REGISTRY:
        available = ', '.join(list_available_models())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )

    creation_fn = MODEL_REGISTRY[model_name_lower]
    model = creation_fn(num_classes=num_classes, **kwargs)

    return model


def create_model_from_config(config: Dict[str, Any]) -> BaseModel:
    """
    Create a model from a configuration dictionary.

    Args:
        config: Configuration dictionary with keys:
            - model_name: Name of the model
            - num_classes: Number of classes (optional, default: NUM_CLASSES)
            - Additional model-specific parameters

    Returns:
        Instantiated model

    Example:
        >>> config = {
        ...     'model_name': 'cnn1d',
        ...     'num_classes': NUM_CLASSES,
        ...     'dropout': 0.3,
        ...     'input_channels': 1
        ... }
        >>> model = create_model_from_config(config)
    """
    config = dict(config)  # Make a copy
    model_name = config.pop('model_name')
    return create_model(model_name, **config)


def load_pretrained(
    model_name: str,
    checkpoint_path: str,
    num_classes: int = NUM_CLASSES,
    device: str = 'cpu',
    strict: bool = True,
    **kwargs
) -> BaseModel:
    """
    Load a pretrained model from checkpoint.

    Args:
        model_name: Name of the model architecture
        checkpoint_path: Path to checkpoint file (.pt or .pth)
        num_classes: Number of output classes
        device: Device to load model to
        strict: Whether to strictly enforce that keys match
        **kwargs: Additional model creation arguments

    Returns:
        Model with loaded weights

    Example:
        >>> model = load_pretrained(
        ...     'cnn1d',
        ...     'checkpoints/best_model.pt',
        ...     num_classes=NUM_CLASSES
        ... )
    """
    # Create model
    model = create_model(model_name, num_classes=num_classes, **kwargs)

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return model


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        optimizer: Optional optimizer state
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        **kwargs: Additional items to save in checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if metrics is not None:
        checkpoint['metrics'] = metrics

    # Add any additional items
    checkpoint.update(kwargs)

    # Create directory if needed
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    torch.save(checkpoint, checkpoint_path)


def create_ensemble(
    model_names: List[str],
    checkpoint_paths: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    num_classes: int = NUM_CLASSES,
    device: str = 'cpu',
    **kwargs
) -> VotingEnsemble:
    """
    Create a soft-voting ensemble of models.

    (Stacking/boosting/MoE ensembles were pruned in the 2026-06 convergence;
    soft voting is the single kept ensemble strategy.)

    Args:
        model_names: List of model names to ensemble
        checkpoint_paths: Optional list of checkpoint paths (must match model_names)
        weights: Optional weights for voting
        num_classes: Number of output classes
        device: Device to load models to
        **kwargs: Additional arguments for model creation

    Returns:
        VotingEnsemble instance

    Example:
        >>> ensemble = create_ensemble(
        ...     model_names=['cnn1d', 'resnet18', 'patchtst'],
        ...     checkpoint_paths=['cnn.pt', 'resnet.pt', 'patchtst.pt'],
        ...     weights=[0.3, 0.4, 0.3],
        ...     num_classes=NUM_CLASSES
        ... )
    """
    models = []

    for i, model_name in enumerate(model_names):
        if checkpoint_paths and i < len(checkpoint_paths):
            # Load pretrained model
            model = load_pretrained(
                model_name,
                checkpoint_paths[i],
                num_classes=num_classes,
                device=device,
                **kwargs
            )
        else:
            # Create untrained model
            model = create_model(model_name, num_classes=num_classes, **kwargs)
            model.to(device)

        models.append(model)

    ensemble = create_voting_ensemble(
        models=models,
        weights=weights,
        voting_type='soft',
        num_classes=num_classes
    )

    ensemble.to(device)
    return ensemble


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model type.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model information
    """
    model_name_lower = model_name.lower()

    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")

    # Create a temporary model to get info
    model = create_model(model_name, num_classes=NUM_CLASSES)

    info = {
        'model_name': model_name,
        'config': model.get_config(),
        'num_parameters': model.get_num_params(),
    }

    return info


def print_model_summary(model: nn.Module, input_shape: tuple = (1, 1, 5000)):
    """
    Print a summary of the model architecture.

    Args:
        model: Model to summarize
        input_shape: Shape of input tensor (B, C, T)
    """
    print(f"\nModel: {model.__class__.__name__}")
    print(f"{'='*60}")

    # Print configuration if available
    if hasattr(model, 'get_config'):
        config = model.get_config()
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Print number of parameters
    if hasattr(model, 'get_num_params'):
        total_params = model.get_num_params()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-trainable: {total_params - trainable_params:,}")

    # Test forward pass
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\nInput shape: {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"\nCould not test forward pass: {e}")

    print(f"{'='*60}\n")
