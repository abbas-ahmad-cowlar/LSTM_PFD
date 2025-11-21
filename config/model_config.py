"""
Model architecture configurations for deep learning models.

Purpose:
    Configuration classes for all model architectures:
    - CNN-1D models
    - ResNet-1D models
    - Transformer models
    - Hybrid LSTM models
    - Ensemble models

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from config.base_config import BaseConfig
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


@dataclass
class CNN1DConfig(BaseConfig):
    """
    Configuration for 1D CNN model.

    Architecture:
        Conv1D -> BatchNorm -> ReLU -> MaxPool -> ... -> FC -> Softmax

    Example:
        >>> config = CNN1DConfig(
        ...     input_length=SIGNAL_LENGTH,
        ...     num_classes=NUM_CLASSES,
        ...     conv_channels=[32, 64, 128],
        ...     kernel_sizes=[15, 11, 7]
        ... )
    """
    # Input/Output
    input_length: int = SIGNAL_LENGTH
    num_classes: int = NUM_CLASSES

    # Convolutional layers
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [15, 11, 7, 5])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    pool_sizes: List[int] = field(default_factory=lambda: [4, 4, 4, 4])

    # Fully connected layers
    fc_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])

    # Regularization
    dropout_prob: float = 0.5
    batch_norm: bool = True

    # Activation
    activation: str = 'relu'  # 'relu', 'leaky_relu', 'elu', 'gelu'

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_length": {"type": "integer", "minimum": 1},
                "num_classes": {"type": "integer", "minimum": 2},
                "conv_channels": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "dropout_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }


@dataclass
class ResNet1DConfig(BaseConfig):
    """
    Configuration for 1D ResNet model.

    Architecture:
        Input -> Conv -> ResBlock -> ResBlock -> ... -> GAP -> FC

    ResBlock:
        Conv1D -> BN -> ReLU -> Conv1D -> BN -> (+residual) -> ReLU

    Example:
        >>> config = ResNet1DConfig(
        ...     input_length=SIGNAL_LENGTH,
        ...     num_classes=NUM_CLASSES,
        ...     blocks=[2, 2, 2, 2],  # ResNet-18 style
        ...     channels=[64, 128, 256, 512]
        ... )
    """
    # Input/Output
    input_length: int = SIGNAL_LENGTH
    num_classes: int = NUM_CLASSES

    # ResNet structure
    blocks: List[int] = field(default_factory=lambda: [2, 2, 2, 2])  # Blocks per stage
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])  # Channels per stage
    initial_kernel_size: int = 15
    initial_stride: int = 2

    # ResBlock configuration
    residual_kernel_size: int = 7
    use_bottleneck: bool = False  # Use 1x1 convs for dimensionality reduction

    # Regularization
    dropout_prob: float = 0.3
    batch_norm: bool = True

    # Pooling
    use_global_avg_pool: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_length": {"type": "integer", "minimum": 1},
                "num_classes": {"type": "integer", "minimum": 2},
                "blocks": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "dropout_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }


@dataclass
class TransformerConfig(BaseConfig):
    """
    Configuration for Transformer model for time series.

    Architecture:
        Embedding -> Positional Encoding -> Transformer Encoder -> FC

    Example:
        >>> config = TransformerConfig(
        ...     input_length=SIGNAL_LENGTH,
        ...     num_classes=NUM_CLASSES,
        ...     d_model=256,
        ...     nhead=8,
        ...     num_layers=6
        ... )
    """
    # Input/Output
    input_length: int = SIGNAL_LENGTH
    num_classes: int = NUM_CLASSES

    # Transformer architecture
    d_model: int = 256  # Model dimension
    nhead: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of encoder layers
    dim_feedforward: int = 1024  # FFN hidden dimension

    # Positional encoding
    use_positional_encoding: bool = True
    max_seq_length: int = 10000

    # Regularization
    dropout_prob: float = 0.1
    attention_dropout: float = 0.1

    # Patch/Segment embedding (optional)
    use_patch_embedding: bool = True
    patch_size: int = 256  # Segment length for embedding
    patch_stride: int = 128  # Stride for patching

    # Activation
    activation: str = 'gelu'

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_length": {"type": "integer", "minimum": 1},
                "num_classes": {"type": "integer", "minimum": 2},
                "d_model": {"type": "integer", "minimum": 1},
                "nhead": {"type": "integer", "minimum": 1},
                "num_layers": {"type": "integer", "minimum": 1}
            }
        }


@dataclass
class LSTMConfig(BaseConfig):
    """
    Configuration for LSTM model.

    Architecture:
        LSTM -> LSTM -> ... -> FC -> Softmax

    Example:
        >>> config = LSTMConfig(
        ...     input_length=SIGNAL_LENGTH,
        ...     num_classes=NUM_CLASSES,
        ...     hidden_size=256,
        ...     num_layers=3
        ... )
    """
    # Input/Output
    input_length: int = SIGNAL_LENGTH
    num_classes: int = NUM_CLASSES
    input_size: int = 1  # Number of features per timestep

    # LSTM architecture
    hidden_size: int = 256
    num_layers: int = 3
    bidirectional: bool = True

    # Regularization
    dropout_prob: float = 0.3

    # FC layers after LSTM
    fc_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_length": {"type": "integer", "minimum": 1},
                "num_classes": {"type": "integer", "minimum": 2},
                "hidden_size": {"type": "integer", "minimum": 1},
                "num_layers": {"type": "integer", "minimum": 1}
            }
        }


@dataclass
class HybridPINNConfig(BaseConfig):
    """
    Configuration for Hybrid Physics-Informed Neural Network.

    Combines:
    - Data-driven feature extraction (CNN/Transformer)
    - Physics-informed constraints (Sommerfeld, bearing equations)
    - Domain knowledge integration

    Example:
        >>> config = HybridPINNConfig(
        ...     input_length=SIGNAL_LENGTH,
        ...     num_classes=NUM_CLASSES,
        ...     backbone='resnet1d',
        ...     physics_loss_weight=0.1
        ... )
    """
    # Input/Output
    input_length: int = SIGNAL_LENGTH
    num_classes: int = NUM_CLASSES

    # Backbone architecture
    backbone: str = 'resnet1d'  # 'cnn1d', 'resnet1d', 'transformer'
    backbone_config: Optional[Dict[str, Any]] = None

    # Physics-informed components
    use_physics_branch: bool = True
    physics_features: List[str] = field(default_factory=lambda: [
        'sommerfeld', 'reynolds', 'speed', 'load', 'temperature'
    ])
    physics_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    # Fusion strategy
    fusion_method: str = 'concat'  # 'concat', 'attention', 'gating'

    # Physics loss
    use_physics_loss: bool = True
    physics_loss_weight: float = 0.1

    # Regularization
    dropout_prob: float = 0.3

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_length": {"type": "integer", "minimum": 1},
                "num_classes": {"type": "integer", "minimum": 2},
                "backbone": {"type": "string", "enum": ["cnn1d", "resnet1d", "transformer"]},
                "physics_loss_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }


@dataclass
class EnsembleConfig(BaseConfig):
    """
    Configuration for ensemble of models.

    Combines multiple models for improved robustness:
    - Voting (hard/soft)
    - Stacking
    - Boosting

    Example:
        >>> config = EnsembleConfig(
        ...     num_classes=NUM_CLASSES,
        ...     model_configs=[cnn_cfg, resnet_cfg, transformer_cfg],
        ...     ensemble_method='soft_voting'
        ... )
    """
    # Input/Output
    num_classes: int = NUM_CLASSES

    # Ensemble configuration
    model_configs: List[Dict[str, Any]] = field(default_factory=list)
    model_types: List[str] = field(default_factory=lambda: ['cnn1d', 'resnet1d', 'transformer'])

    # Ensemble method
    ensemble_method: str = 'soft_voting'  # 'hard_voting', 'soft_voting', 'stacking', 'weighted'

    # Weights for weighted voting
    model_weights: Optional[List[float]] = None

    # Stacking meta-learner
    meta_learner: str = 'logistic'  # 'logistic', 'random_forest', 'mlp'

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "num_classes": {"type": "integer", "minimum": 2},
                "ensemble_method": {"type": "string",
                                   "enum": ["hard_voting", "soft_voting", "stacking", "weighted"]},
                "model_types": {"type": "array", "items": {"type": "string"}}
            }
        }


@dataclass
class ModelConfig(BaseConfig):
    """
    Master model configuration.

    Aggregates all model-specific configs and provides factory pattern.

    Example:
        >>> config = ModelConfig(
        ...     model_type='resnet1d',
        ...     resnet1d=ResNet1DConfig(blocks=[3, 4, 6, 3])
        ... )
        >>> config.to_yaml('config/model_config.yaml')
    """
    # Model type selection
    model_type: str = 'cnn1d'  # 'cnn1d', 'resnet1d', 'transformer', 'lstm', 'hybrid_pinn', 'ensemble'

    # Model-specific configs
    cnn1d: CNN1DConfig = field(default_factory=CNN1DConfig)
    resnet1d: ResNet1DConfig = field(default_factory=ResNet1DConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    hybrid_pinn: HybridPINNConfig = field(default_factory=HybridPINNConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    # Training behavior
    pretrained: bool = False
    pretrained_path: Optional[str] = None

    def get_active_config(self) -> BaseConfig:
        """
        Get configuration for currently selected model type.

        Returns:
            Active model configuration

        Example:
            >>> model_cfg = config.get_active_config()
            >>> if config.model_type == 'resnet1d':
            ...     assert isinstance(model_cfg, ResNet1DConfig)
        """
        config_map = {
            'cnn1d': self.cnn1d,
            'resnet1d': self.resnet1d,
            'transformer': self.transformer,
            'lstm': self.lstm,
            'hybrid_pinn': self.hybrid_pinn,
            'ensemble': self.ensemble
        }

        if self.model_type not in config_map:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return config_map[self.model_type]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_type": {"type": "string",
                              "enum": ["cnn1d", "resnet1d", "transformer", "lstm",
                                      "hybrid_pinn", "ensemble"]},
                "pretrained": {"type": "boolean"}
            }
        }
