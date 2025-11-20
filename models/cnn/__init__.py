"""
CNN models for end-to-end learning from raw vibration signals.

Purpose:
    1D Convolutional Neural Networks for bearing fault diagnosis:
    - Conv blocks: Reusable Conv1D-BN-ReLU-Dropout-Pool modules
    - CNN architectures: Configurable depth and width
    - Attention mechanisms: SE blocks, CBAM
    - Model variants: Shallow, deep, residual architectures

Author: LSTM_PFD Team
Date: 2025-11-20
"""

from models.cnn.conv_blocks import (
    ConvBlock1D,
    ResidualConvBlock1D,
    SeparableConv1D
)

__all__ = [
    'ConvBlock1D',
    'ResidualConvBlock1D',
    'SeparableConv1D'
]
