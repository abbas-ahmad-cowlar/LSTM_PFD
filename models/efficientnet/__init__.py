"""
EfficientNet architectures for 1D signal processing.

This package provides EfficientNet implementations with compound scaling:
- EfficientNet-B0 to B7: Progressively scaled models
- MBConv blocks: Mobile inverted bottleneck convolutions
- Efficient attention mechanisms
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .mbconv_block import (
    MBConvBlock,
    FusedMBConvBlock,
    DepthwiseSeparableConv1D,
    SEBlock1D
)

from .efficientnet_1d import (
    EfficientNet1D,
    create_efficientnet_b0,
    create_efficientnet_b1,
    create_efficientnet_b2,
    create_efficientnet_b3,
    create_efficientnet_b4,
    create_efficientnet_b5,
    create_efficientnet_b6,
    create_efficientnet_b7,
    calculate_scaling_params
)

__all__ = [
    # MBConv blocks
    'MBConvBlock',
    'FusedMBConvBlock',
    'DepthwiseSeparableConv1D',
    'SEBlock1D',
    # EfficientNet models
    'EfficientNet1D',
    'create_efficientnet_b0',
    'create_efficientnet_b1',
    'create_efficientnet_b2',
    'create_efficientnet_b3',
    'create_efficientnet_b4',
    'create_efficientnet_b5',
    'create_efficientnet_b6',
    'create_efficientnet_b7',
    'calculate_scaling_params',
]
