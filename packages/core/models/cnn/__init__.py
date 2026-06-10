"""
1D CNN models for end-to-end learning from raw vibration signals.

- CNN1D: 5-block baseline CNN (Tier 1; the proven reference model)
- AttentionCNN1D: CNN with attention for XAI-friendly saliency (Tier 1)
- MultiScaleCNN1D: parallel multi-band kernels (Tier 2)
- Conv blocks: reusable Conv1D building blocks
"""

from .conv_blocks import (
    ConvBlock1D,
    ResidualConvBlock1D,
    SeparableConv1D
)
from .cnn_1d import CNN1D, create_cnn1d
from .attention_cnn import AttentionCNN1D
from .multi_scale_cnn import MultiScaleCNN1D

__all__ = [
    'ConvBlock1D',
    'ResidualConvBlock1D',
    'SeparableConv1D',
    'CNN1D',
    'create_cnn1d',
    'AttentionCNN1D',
    'MultiScaleCNN1D',
]
