"""
Modular convolutional blocks for CNN architectures.

Purpose:
    Reusable building blocks for 1D CNN construction:
    - ConvBlock1D: Standard Conv-BN-ReLU-Dropout-Pool block
    - ResidualConvBlock1D: Conv block with skip connection
    - SeparableConv1D: Depthwise separable convolution (efficient)

    These blocks enable modular architecture design and facilitate
    architecture search in Phase 4.

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConvBlock1D(nn.Module):
    """
    Standard 1D convolutional block: Conv → BN → ReLU → Dropout → Pool.

    Architecture:
        Input [B, in_channels, T]
        ├─ Conv1D(in_channels → out_channels, kernel_size, stride)
        ├─ BatchNorm1D(out_channels)
        ├─ ReLU
        ├─ Dropout(p=dropout)
        └─ MaxPool1D(pool_size) or AdaptiveAvgPool1D
        Output [B, out_channels, T']

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Stride for convolution
        padding: Padding mode ('same' or int)
        dropout: Dropout probability (0.0 = no dropout)
        pool_size: Max pooling kernel size (None = no pooling)
        use_batch_norm: Whether to apply batch normalization
        activation: Activation function ('relu', 'leaky_relu', 'elu')

    Example:
        >>> block = ConvBlock1D(1, 32, kernel_size=64, stride=4)
        >>> x = torch.randn(8, 1, SIGNAL_LENGTH)  # Batch of 8 signals
        >>> out = block(x)  # Shape: [8, 32, 25600]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = 'same',
        dropout: float = 0.0,
        pool_size: Optional[int] = None,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super(ConvBlock1D, self).__init__()

        # Compute padding for 'same' mode
        if padding == 'same':
            # For stride > 1, adjust padding to maintain size before pooling
            if stride == 1:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0  # Let stride reduce dimension
        else:
            pad = padding

        # Convolutional layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=not use_batch_norm  # No bias if using BN
        )

        # Batch normalization (optional)
        self.batch_norm = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout (optional)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Pooling (optional)
        self.pool = nn.MaxPool1d(kernel_size=pool_size) if pool_size else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, in_channels, T]

        Returns:
            Output tensor [B, out_channels, T']
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class ResidualConvBlock1D(nn.Module):
    """
    Residual convolutional block with skip connection.

    Architecture:
        Input [B, in_channels, T]
        ├─ ConvBlock1D (main path)
        ├─ Conv1D 1x1 (skip connection, if channels differ)
        └─ Add(main + skip) → ReLU
        Output [B, out_channels, T']

    Prevents gradient vanishing in deep networks by providing
    direct gradient flow through skip connections.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Stride for convolution
        dropout: Dropout probability

    Example:
        >>> block = ResidualConvBlock1D(64, 128, kernel_size=16, stride=2)
        >>> x = torch.randn(8, 64, 12800)
        >>> out = block(x)  # Shape: [8, 128, 6400]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.0
    ):
        super(ResidualConvBlock1D, self).__init__()

        # Main convolutional path
        self.conv_block = ConvBlock1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            pool_size=None  # No pooling in residual blocks
        )

        # Skip connection: 1x1 conv if channels/dimensions change
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip_connection = nn.Identity()

        # Final activation after addition
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor [B, in_channels, T]

        Returns:
            Output tensor [B, out_channels, T']
        """
        # Main path
        out = self.conv_block(x)

        # Skip connection
        skip = self.skip_connection(x)

        # Add and activate
        out = out + skip
        out = self.final_activation(out)

        return out


class SeparableConv1D(nn.Module):
    """
    Depthwise separable 1D convolution for parameter efficiency.

    Architecture:
        Input [B, in_channels, T]
        ├─ Depthwise Conv1D (each channel separately)
        ├─ Pointwise Conv1D (1x1 conv to mix channels)
        ├─ BatchNorm → ReLU → Dropout
        Output [B, out_channels, T']

    Reduces parameters by ~9× compared to standard convolution:
        Standard: in_channels × out_channels × kernel_size
        Separable: in_channels × kernel_size + in_channels × out_channels

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of depthwise kernel
        stride: Stride for depthwise convolution
        dropout: Dropout probability

    Example:
        >>> block = SeparableConv1D(128, 256, kernel_size=8, stride=2)
        >>> x = torch.randn(8, 128, 6400)
        >>> out = block(x)  # Shape: [8, 256, 3200]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.0
    ):
        super(SeparableConv1D, self).__init__()

        # Depthwise convolution (each channel separately)
        padding = (kernel_size - 1) // 2 if stride == 1 else 0
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: groups = in_channels for depthwise
            bias=False
        )

        # Pointwise convolution (1x1 to mix channels)
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(out_channels)

        # Activation and dropout
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, in_channels, T]

        Returns:
            Output tensor [B, out_channels, T']
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


def test_conv_blocks():
    """Test convolutional blocks with sample inputs."""
    print("Testing ConvBlock1D...")
    block1 = ConvBlock1D(1, 32, kernel_size=64, stride=4)
    x1 = torch.randn(8, 1, SIGNAL_LENGTH)
    out1 = block1(x1)
    print(f"  Input: {x1.shape} → Output: {out1.shape}")
    assert out1.shape == (8, 32, 25600), f"Expected [8, 32, 25600], got {out1.shape}"

    print("\nTesting ResidualConvBlock1D...")
    block2 = ResidualConvBlock1D(64, 128, kernel_size=16, stride=2)
    x2 = torch.randn(8, 64, 12800)
    out2 = block2(x2)
    print(f"  Input: {x2.shape} → Output: {out2.shape}")
    assert out2.shape == (8, 128, 6400), f"Expected [8, 128, 6400], got {out2.shape}"

    print("\nTesting SeparableConv1D...")
    block3 = SeparableConv1D(128, 256, kernel_size=8, stride=2)
    x3 = torch.randn(8, 128, 6400)
    out3 = block3(x3)
    print(f"  Input: {x3.shape} → Output: {out3.shape}")
    assert out3.shape == (8, 256, 3200), f"Expected [8, 256, 3200], got {out3.shape}"

    print("\n✅ All conv block tests passed!")


if __name__ == "__main__":
    test_conv_blocks()
