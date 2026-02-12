"""
Model Factory for Creating and Managing Models

Provides a centralized interface for:
- Model instantiation
- Loading pretrained weights
- Model registration
- Configuration-based creation

Supports all model architectures:
- CNN1D
- ResNet1D
- Transformer
- HybridPINN
- Ensemble models
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from pathlib import Path

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .base_model import BaseModel
from .cnn_1d import CNN1D, create_cnn1d
from .resnet_1d import ResNet1D, create_resnet18_1d, create_resnet34_1d
from .transformer import SignalTransformer, create_transformer
from .hybrid_pinn import HybridPINN, create_hybrid_pinn
from .legacy_ensemble import (
    EnsembleModel,
    VotingEnsemble,
    StackedEnsemble,
    create_voting_ensemble,
    create_stacked_ensemble
)

# Phase 4: Advanced Transformer Variants
from .transformer.vision_transformer_1d import (
    VisionTransformer1D,
    create_vit_1d,
    vit_tiny_1d,
    vit_small_1d,
    vit_base_1d
)
from .hybrid.cnn_transformer import (
    CNNTransformerHybrid,
    create_cnn_transformer_hybrid,
    cnn_transformer_small,
    cnn_transformer_base,
    cnn_transformer_large
)

# PatchTST and TSMixer
from .transformer.patchtst import PatchTST
from .transformer.tsmixer import TSMixer

# EfficientNet 1D
from .efficientnet import (
    EfficientNet1D,
    create_efficientnet_b0,
    create_efficientnet_b1,
    create_efficientnet_b2,
    create_efficientnet_b3,
    create_efficientnet_b4,
    create_efficientnet_b5,
    create_efficientnet_b6,
    create_efficientnet_b7,
)

# Spectrogram CNN (2D models)
from .spectrogram_cnn import (
    ResNet2DSpectrogram,
    resnet18_2d,
    resnet34_2d,
    resnet50_2d,
    EfficientNet2DSpectrogram,
    efficientnet_b0 as efficientnet_2d_b0,
    efficientnet_b1 as efficientnet_2d_b1,
    efficientnet_b3 as efficientnet_2d_b3,
)

# Dual-Stream CNN
from .spectrogram_cnn.dual_stream_cnn import DualStreamCNN

# Fusion models
from .fusion import (
    EarlyFusion,
    create_early_fusion,
    LateFusion,
    create_late_fusion,
)

# PINN variants
from .pinn.physics_constrained_cnn import (
    PhysicsConstrainedCNN,
    AdaptivePhysicsConstrainedCNN,
    create_physics_constrained_cnn,
)
from .pinn.multitask_pinn import (
    MultitaskPINN,
    AdaptiveMultitaskPINN,
    create_multitask_pinn,
)
from .pinn.knowledge_graph_pinn import KnowledgeGraphPINN

# Contrastive models
from .contrastive.signal_encoder import SignalEncoder
from .contrastive.classifier import ContrastiveClassifier


# ---------------------------------------------------------------------------
# Factory wrapper functions for models that lack them
# ---------------------------------------------------------------------------

def create_patchtst(num_classes: int = NUM_CLASSES, **kwargs) -> PatchTST:
    """Create a PatchTST model (Nie et al., 2023)."""
    return PatchTST(num_classes=num_classes, **kwargs)


def create_tsmixer(num_classes: int = NUM_CLASSES, **kwargs) -> TSMixer:
    """Create a TSMixer model (Chen et al., 2023)."""
    return TSMixer(num_classes=num_classes, **kwargs)


def create_dual_stream_cnn(num_classes: int = NUM_CLASSES, **kwargs) -> DualStreamCNN:
    """Create a DualStreamCNN model (time + frequency branches)."""
    return DualStreamCNN(num_classes=num_classes, **kwargs)


def _create_multitask_pinn(num_classes: int = NUM_CLASSES, **kwargs) -> MultitaskPINN:
    """Adapter for create_multitask_pinn (uses num_fault_classes internally)."""
    return create_multitask_pinn(num_fault_classes=num_classes, **kwargs)


def _create_adaptive_multitask_pinn(num_classes: int = NUM_CLASSES, **kwargs):
    """Create AdaptiveMultitaskPINN via adapter."""
    return create_multitask_pinn(num_fault_classes=num_classes, adaptive=True, **kwargs)


def create_knowledge_graph_pinn(num_classes: int = NUM_CLASSES, **kwargs) -> KnowledgeGraphPINN:
    """Create a KnowledgeGraphPINN model."""
    return KnowledgeGraphPINN(num_classes=num_classes, **kwargs)


def create_signal_encoder(num_classes: int = NUM_CLASSES, **kwargs) -> SignalEncoder:
    """Create a SignalEncoder for contrastive learning."""
    return SignalEncoder(num_classes=num_classes, **kwargs)


def create_contrastive_classifier(num_classes: int = NUM_CLASSES, **kwargs) -> ContrastiveClassifier:
    """Create a ContrastiveClassifier with default encoder."""
    return ContrastiveClassifier(num_classes=num_classes, **kwargs)


# Model registry: Maps model names to creation functions
MODEL_REGISTRY = {
    # CNN models
    'cnn1d': create_cnn1d,
    'cnn_1d': create_cnn1d,

    # ResNet models
    'resnet18': create_resnet18_1d,
    'resnet18_1d': create_resnet18_1d,
    'resnet34': create_resnet34_1d,
    'resnet34_1d': create_resnet34_1d,

    # Transformer
    'transformer': create_transformer,
    'signal_transformer': create_transformer,

    # Vision Transformer 1D
    'vit_1d': create_vit_1d,
    'vision_transformer_1d': create_vit_1d,
    'vit_tiny_1d': vit_tiny_1d,
    'vit_small_1d': vit_small_1d,
    'vit_base_1d': vit_base_1d,

    # PatchTST (Nie et al., 2023)
    'patchtst': create_patchtst,
    'patch_tst': create_patchtst,

    # TSMixer (Chen et al., 2023)
    'tsmixer': create_tsmixer,
    'ts_mixer': create_tsmixer,

    # CNN-Transformer Hybrid
    'cnn_transformer': create_cnn_transformer_hybrid,
    'cnn_transformer_hybrid': create_cnn_transformer_hybrid,
    'cnn_transformer_small': cnn_transformer_small,
    'cnn_transformer_base': cnn_transformer_base,
    'cnn_transformer_large': cnn_transformer_large,

    # EfficientNet 1D
    'efficientnet_b0': create_efficientnet_b0,
    'efficientnet_b1': create_efficientnet_b1,
    'efficientnet_b2': create_efficientnet_b2,
    'efficientnet_b3': create_efficientnet_b3,
    'efficientnet_b4': create_efficientnet_b4,
    'efficientnet_b5': create_efficientnet_b5,
    'efficientnet_b6': create_efficientnet_b6,
    'efficientnet_b7': create_efficientnet_b7,

    # Spectrogram CNN (2D models — input shape: [B, C, H, W])
    'resnet18_2d': resnet18_2d,
    'resnet34_2d': resnet34_2d,
    'resnet50_2d': resnet50_2d,
    'efficientnet_2d_b0': efficientnet_2d_b0,
    'efficientnet_2d_b1': efficientnet_2d_b1,
    'efficientnet_2d_b3': efficientnet_2d_b3,

    # Dual-Stream CNN (time + frequency branches)
    'dual_stream': create_dual_stream_cnn,
    'dual_stream_cnn': create_dual_stream_cnn,

    # Fusion models (require multi-input pipelines)
    'early_fusion': create_early_fusion,
    'late_fusion': create_late_fusion,

    # Physics-informed
    'pinn': create_hybrid_pinn,
    'hybrid_pinn': create_hybrid_pinn,
    'physics_informed': create_hybrid_pinn,

    # Physics-Constrained CNN
    'physics_cnn': create_physics_constrained_cnn,
    'physics_constrained_cnn': create_physics_constrained_cnn,
    'adaptive_physics_cnn': lambda **kw: create_physics_constrained_cnn(adaptive=True, **kw),

    # Multi-task PINN
    'multitask_pinn': _create_multitask_pinn,
    'adaptive_multitask_pinn': _create_adaptive_multitask_pinn,

    # Knowledge Graph PINN
    'knowledge_graph_pinn': create_knowledge_graph_pinn,
    'kg_pinn': create_knowledge_graph_pinn,

    # Contrastive learning models
    'signal_encoder': create_signal_encoder,
    'contrastive_classifier': create_contrastive_classifier,
}


def register_model(name: str, creation_fn: callable):
    """
    Register a new model type.

    Args:
        name: Model name (used in create_model)
        creation_fn: Function that creates the model
    """
    MODEL_REGISTRY[name.lower()] = creation_fn


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

    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    ensemble_type: str = 'voting',
    weights: Optional[List[float]] = None,
    num_classes: int = NUM_CLASSES,
    device: str = 'cpu',
    **kwargs
) -> EnsembleModel:
    """
    Create an ensemble of models.

    Args:
        model_names: List of model names to ensemble
        checkpoint_paths: Optional list of checkpoint paths (must match model_names)
        ensemble_type: Type of ensemble ('voting' or 'stacking')
        weights: Optional weights for voting ensemble
        num_classes: Number of output classes
        device: Device to load models to
        **kwargs: Additional arguments for model creation

    Returns:
        EnsembleModel instance

    Example:
        >>> ensemble = create_ensemble(
        ...     model_names=['cnn1d', 'resnet18', 'transformer'],
        ...     checkpoint_paths=['cnn.pt', 'resnet.pt', 'transformer.pt'],
        ...     ensemble_type='voting',
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

    # Create ensemble
    if ensemble_type == 'voting':
        ensemble = create_voting_ensemble(
            models=models,
            weights=weights,
            voting_type='soft',
            num_classes=num_classes
        )
    elif ensemble_type == 'stacking':
        ensemble = create_stacked_ensemble(
            base_models=models,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

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
