"""
CNN Models for Bearing Fault Diagnosis

This package provides state-of-the-art CNN architectures for bearing fault diagnosis:
- Basic 1D CNNs: Multi-scale convolutions for raw signal processing
- ResNet variants: Deep residual networks (ResNet-18/34/50, SE-ResNet, Wide-ResNet)
- EfficientNet: Compound-scaled efficient architectures

All models are designed for 1D vibration signal classification.

Author: Bearing Fault Diagnosis Team
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

# Import CNN models
from models.cnn.cnn_1d import CNN1D
from models.cnn.attention_cnn import AttentionCNN1D, LightweightAttentionCNN
from models.cnn.multi_scale_cnn import MultiScaleCNN1D, DilatedMultiScaleCNN
from models.cnn.conv_blocks import ConvBlock1D, ResidualConvBlock1D, SeparableConv1D

# Import ResNet models
from models.resnet.resnet_1d import ResNet1D, create_resnet18_1d, create_resnet34_1d, create_resnet50_1d
from models.resnet.se_resnet import SEResNet1D, create_se_resnet18_1d, create_se_resnet34_1d
from models.resnet.wide_resnet import WideResNet1D, create_wide_resnet16_8, create_wide_resnet28_10
from models.resnet.residual_blocks import BasicBlock1D, Bottleneck1D

# Import EfficientNet models
from models.efficientnet.efficientnet_1d import (
    EfficientNet1D,
    create_efficientnet_b0,
    create_efficientnet_b1,
    create_efficientnet_b2,
    create_efficientnet_b3,
    create_efficientnet_b4
)
from models.efficientnet.mbconv_block import MBConvBlock, FusedMBConvBlock

__all__ = [
    # Basic CNN Models
    'CNN1D',
    'AttentionCNN1D',
    'LightweightAttentionCNN',
    'MultiScaleCNN1D',
    'DilatedMultiScaleCNN',
    'ConvBlock1D',
    'ResidualConvBlock1D',
    'SeparableConv1D',

    # ResNet Models
    'ResNet1D',
    'create_resnet18_1d',
    'create_resnet34_1d',
    'create_resnet50_1d',
    'SEResNet1D',
    'create_se_resnet18_1d',
    'create_se_resnet34_1d',
    'WideResNet1D',
    'create_wide_resnet16_8',
    'create_wide_resnet28_10',
    'BasicBlock1D',
    'Bottleneck1D',

    # EfficientNet Models
    'EfficientNet1D',
    'create_efficientnet_b0',
    'create_efficientnet_b1',
    'create_efficientnet_b2',
    'create_efficientnet_b3',
    'create_efficientnet_b4',
    'MBConvBlock',
    'FusedMBConvBlock',
]


def create_model(model_name: str, num_classes: int = NUM_CLASSES, **kwargs):
    """
    Factory function to create CNN models by name.

    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes (default: 11)
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model

    Available models:
        - 'cnn1d': Basic 1D CNN
        - 'attention_cnn': CNN with attention mechanisms
        - 'multiscale_cnn': Multi-scale CNN
        - 'resnet18', 'resnet34', 'resnet50': ResNet variants
        - 'se_resnet18', 'se_resnet34': SE-ResNet variants
        - 'wide_resnet16', 'wide_resnet28': Wide ResNet variants
        - 'efficientnet_b0' to 'efficientnet_b4': EfficientNet variants
    """
    model_map = {
        'cnn1d': CNN1D,
        'attention_cnn': AttentionCNN1D,
        'attention_cnn_lite': LightweightAttentionCNN,
        'multiscale_cnn': MultiScaleCNN1D,
        'dilated_cnn': DilatedMultiScaleCNN,
        'resnet18': lambda num_classes, **kw: create_resnet18_1d(num_classes=num_classes, **kw),
        'resnet34': lambda num_classes, **kw: create_resnet34_1d(num_classes=num_classes, **kw),
        'resnet50': lambda num_classes, **kw: create_resnet50_1d(num_classes=num_classes, **kw),
        'se_resnet18': lambda num_classes, **kw: create_se_resnet18_1d(num_classes=num_classes, **kw),
        'se_resnet34': lambda num_classes, **kw: create_se_resnet34_1d(num_classes=num_classes, **kw),
        'wide_resnet16': lambda num_classes, **kw: create_wide_resnet16_8(num_classes=num_classes, **kw),
        'wide_resnet28': lambda num_classes, **kw: create_wide_resnet28_10(num_classes=num_classes, **kw),
        'efficientnet_b0': lambda num_classes, **kw: create_efficientnet_b0(num_classes=num_classes, **kw),
        'efficientnet_b1': lambda num_classes, **kw: create_efficientnet_b1(num_classes=num_classes, **kw),
        'efficientnet_b2': lambda num_classes, **kw: create_efficientnet_b2(num_classes=num_classes, **kw),
        'efficientnet_b3': lambda num_classes, **kw: create_efficientnet_b3(num_classes=num_classes, **kw),
        'efficientnet_b4': lambda num_classes, **kw: create_efficientnet_b4(num_classes=num_classes, **kw),
    }

    if model_name not in model_map:
        available = ', '.join(model_map.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_fn = model_map[model_name]
    return model_fn(num_classes=num_classes, **kwargs)


def list_available_models():
    """List all available CNN model architectures."""
    return [
        'cnn1d', 'attention_cnn', 'attention_cnn_lite', 'multiscale_cnn', 'dilated_cnn',
        'resnet18', 'resnet34', 'resnet50',
        'se_resnet18', 'se_resnet34',
        'wide_resnet16', 'wide_resnet28',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4'
    ]
