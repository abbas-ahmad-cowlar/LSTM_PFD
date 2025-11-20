"""
ResNet architectures for 1D signal processing.

This package provides various ResNet implementations adapted for time-series
bearing fault diagnosis:
- ResNet-18/34/50: Standard ResNet architectures
- SE-ResNet: ResNet with Squeeze-and-Excitation blocks
- Wide-ResNet: Wider but shallower networks
"""

from .residual_blocks import (
    BasicBlock1D,
    Bottleneck1D,
    PreActBlock1D,
    make_downsample_layer
)

from .resnet_1d import (
    ResNet1D,
    create_resnet18_1d,
    create_resnet34_1d,
    create_resnet50_1d
)

from .se_resnet import (
    SEResNet1D,
    SEBlock,
    create_se_resnet18_1d,
    create_se_resnet34_1d,
    create_se_resnet50_1d
)

from .wide_resnet import (
    WideResNet1D,
    create_wide_resnet16_8,
    create_wide_resnet16_10,
    create_wide_resnet22_8,
    create_wide_resnet28_10
)

__all__ = [
    # Basic blocks
    'BasicBlock1D',
    'Bottleneck1D',
    'PreActBlock1D',
    'make_downsample_layer',
    # Standard ResNet
    'ResNet1D',
    'create_resnet18_1d',
    'create_resnet34_1d',
    'create_resnet50_1d',
    # SE-ResNet
    'SEResNet1D',
    'SEBlock',
    'create_se_resnet18_1d',
    'create_se_resnet34_1d',
    'create_se_resnet50_1d',
    # Wide ResNet
    'WideResNet1D',
    'create_wide_resnet16_8',
    'create_wide_resnet16_10',
    'create_wide_resnet22_8',
    'create_wide_resnet28_10',
]
