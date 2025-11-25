"""
Mobile Inverted Bottleneck Convolution (MBConv) Block

Core building block for EfficientNet architectures.
Combines:
- Inverted bottleneck (expand → depthwise → project)
- Depthwise separable convolutions (parameter efficient)
- Squeeze-and-Excitation attention
- Skip connections

Reference:
- Sandler et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for CNNs"
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise Separable Convolution for 1D signals.

    Splits standard convolution into:
    1. Depthwise: Each channel processed independently
    2. Pointwise: 1x1 conv to mix channels

    Parameter reduction: ~8-9x compared to standard conv

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise conv
        stride: Stride
        padding: Padding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        # Depthwise: Each channel processed separately
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: groups=in_channels
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_channels)

        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU6(inplace=True)  # ReLU6 for mobile networks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Depthwise
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Pointwise
        x = self.pointwise(x)
        x = self.bn2(x)

        return x


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention (1D version).

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        reduced_channels = max(channels // reduction, 1)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention."""
        attention = self.squeeze(x)
        attention = self.excitation(attention)
        return x * attention


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block.

    Structure:
        Input [B, C_in, T]
          ↓ Expansion (1x1 conv if expand_ratio > 1)
        [B, C_in * expand_ratio, T]
          ↓ Depthwise conv (with stride)
        [B, C_in * expand_ratio, T']
          ↓ Squeeze-Excitation (optional)
        [B, C_in * expand_ratio, T']
          ↓ Projection (1x1 conv)
        [B, C_out, T']
          ↓ Skip connection (if stride=1 and C_in=C_out)
        Output [B, C_out, T']

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size for depthwise conv
        stride: Stride for depthwise conv
        expand_ratio: Expansion ratio for inverted bottleneck
        se_ratio: SE reduction ratio (0 to disable SE)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        dropout: float = 0.2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio

        # Determine if skip connection should be used
        self.use_skip_connection = (stride == 1) and (in_channels == out_channels)

        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.has_expansion = expand_ratio != 1

        if self.has_expansion:
            self.expand_conv = nn.Conv1d(
                in_channels,
                expanded_channels,
                kernel_size=1,
                bias=False
            )
            self.expand_bn = nn.BatchNorm1d(expanded_channels)
            self.expand_relu = nn.ReLU6(inplace=True)

        # Depthwise convolution
        padding = kernel_size // 2
        self.depthwise_conv = nn.Conv1d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=expanded_channels,
            bias=False
        )
        self.depthwise_bn = nn.BatchNorm1d(expanded_channels)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        # Squeeze-and-Excitation
        self.has_se = se_ratio > 0
        if self.has_se:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SEBlock1D(expanded_channels, reduction=expanded_channels // se_channels)

        # Projection phase
        self.project_conv = nn.Conv1d(
            expanded_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.project_bn = nn.BatchNorm1d(out_channels)

        # Dropout for regularization (only applied to skip connection path)
        self.dropout = nn.Dropout(dropout) if dropout > 0 and self.use_skip_connection else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with inverted bottleneck."""
        identity = x

        # Expansion
        if self.has_expansion:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_relu(x)

        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)

        # Squeeze-and-Excitation
        if self.has_se:
            x = self.se(x)

        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)

        # Skip connection
        if self.use_skip_connection:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity

        return x


class FusedMBConvBlock(nn.Module):
    """
    Fused Mobile Inverted Bottleneck Convolution Block.

    Replaces separate expansion + depthwise with a single regular conv.
    Faster on some hardware but uses more parameters.

    Used in EfficientNetV2.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        stride: Stride
        expand_ratio: Expansion ratio
        se_ratio: SE reduction ratio
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        dropout: float = 0.2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.use_skip_connection = (stride == 1) and (in_channels == out_channels)

        expanded_channels = in_channels * expand_ratio

        # Fused expansion + depthwise
        padding = kernel_size // 2
        self.fused_conv = nn.Conv1d(
            in_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.fused_bn = nn.BatchNorm1d(expanded_channels)
        self.fused_relu = nn.ReLU6(inplace=True)

        # Squeeze-and-Excitation
        self.has_se = se_ratio > 0
        if self.has_se:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SEBlock1D(expanded_channels, reduction=expanded_channels // se_channels)

        # Projection
        self.project_conv = nn.Conv1d(
            expanded_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.project_bn = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout) if dropout > 0 and self.use_skip_connection else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x

        # Fused expansion + depthwise
        x = self.fused_conv(x)
        x = self.fused_bn(x)
        x = self.fused_relu(x)

        # Squeeze-and-Excitation
        if self.has_se:
            x = self.se(x)

        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)

        # Skip connection
        if self.use_skip_connection:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity

        return x


# Test the blocks
if __name__ == "__main__":
    print("Testing MBConv blocks...")

    # Test standard MBConv
    print("\nTesting MBConvBlock...")
    block = MBConvBlock(in_channels=64, out_channels=128, stride=2, expand_ratio=6)
    x = torch.randn(2, 64, 1000)
    y = block(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 128, 500), f"Expected (2, 128, 500), got {y.shape}"

    # Test with skip connection
    print("\nTesting MBConvBlock with skip connection...")
    block = MBConvBlock(in_channels=128, out_channels=128, stride=1, expand_ratio=6)
    x = torch.randn(2, 128, 1000)
    y = block(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 128, 1000)

    # Test Fused MBConv
    print("\nTesting FusedMBConvBlock...")
    block = FusedMBConvBlock(in_channels=64, out_channels=128, stride=2)
    x = torch.randn(2, 64, 1000)
    y = block(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 128, 500)

    # Test Depthwise Separable Conv
    print("\nTesting DepthwiseSeparableConv1D...")
    conv = DepthwiseSeparableConv1D(64, 128, kernel_size=3)
    x = torch.randn(2, 64, 1000)
    y = conv(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 128, 1000)

    print("\n✓ All MBConv tests passed!")
