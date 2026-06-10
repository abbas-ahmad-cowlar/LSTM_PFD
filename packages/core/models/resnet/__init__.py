"""
ResNet architectures for 1D signal processing.

- ResNet-18/34/50 (1D): standard residual networks (T1: resnet18)
- SE-ResNet: ResNet with Squeeze-and-Excitation blocks (T2: se_resnet18)
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
]
