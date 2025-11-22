"""
Residual Blocks for 1D ResNet Architectures

Implements various residual block variants for time-series signal processing:
- BasicBlock1D: Standard residual block (2 conv layers)
- Bottleneck1D: Efficient bottleneck design (1x1 -> 3x3 -> 1x1)
- PreActBlock1D: Pre-activation variant (BN-ReLU-Conv)

Reference:
- He et al. (2016). "Deep Residual Learning for Image Recognition"
- He et al. (2016). "Identity Mappings in Deep Residual Networks" (Pre-activation)
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicBlock1D(nn.Module):
    """
    Basic residual block for ResNet-18/34.

    Structure:
        x -> Conv1 -> BN -> ReLU -> Conv2 -> BN -> (+) -> ReLU
        |___________________________________________|
                    (skip connection)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolutions
        stride: Stride for first convolution (for downsampling)
        downsample: Downsample layer for skip connection (if needed)
        dropout: Dropout probability
    """
    expansion = 1  # Output channels multiplier

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        # First conv layer (may downsample)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second conv layer
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Downsample layer for skip connection (if channels change)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape [B, C_in, T]

        Returns:
            Output tensor of shape [B, C_out, T']
        """
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply dropout
        if self.dropout is not None:
            out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out += identity
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    """
    Bottleneck residual block for ResNet-50/101/152.

    Structure:
        x -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> (+) -> ReLU
        |__________________________________________________________________|
                                (skip connection)

    More efficient than BasicBlock for deep networks:
    - Reduces parameters: 256->64->64->256 vs. 256->256->256
    - Same receptive field with fewer FLOPs

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (before expansion)
        kernel_size: Kernel size for middle convolution
        stride: Stride for middle convolution (for downsampling)
        downsample: Downsample layer for skip connection (if needed)
        dropout: Dropout probability
    """
    expansion = 4  # Output channels = out_channels * 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        # 1x1 conv for dimensionality reduction
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 3x3 conv for feature extraction (may downsample)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1x1 conv for dimensionality expansion
        self.conv3 = nn.Conv1d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Downsample layer for skip connection
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape [B, C_in, T]

        Returns:
            Output tensor of shape [B, C_out * expansion, T']
        """
        identity = x

        # Bottleneck path
        # 1x1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Apply dropout
        if self.dropout is not None:
            out = self.dropout(out)

        # 1x1 expand
        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out += identity
        out = self.relu(out)

        return out


class PreActBlock1D(nn.Module):
    """
    Pre-activation residual block.

    Structure:
        x -> BN -> ReLU -> Conv1 -> BN -> ReLU -> Conv2 -> (+)
        |______________________________________________|
                    (skip connection)

    Advantages over post-activation (BasicBlock1D):
    - Better gradient flow (identity path is completely clean)
    - Easier optimization for very deep networks (100+ layers)
    - Slightly better accuracy in practice

    Reference:
    - He et al. (2016). "Identity Mappings in Deep Residual Networks"

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolutions
        stride: Stride for first convolution (for downsampling)
        downsample: Downsample layer for skip connection (if needed)
        dropout: Dropout probability
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        # Pre-activation BN and ReLU
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # First conv layer (may downsample)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )

        # Second pre-activation and conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Downsample layer for skip connection
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-activation and residual connection.

        Args:
            x: Input tensor of shape [B, C_in, T]

        Returns:
            Output tensor of shape [B, C_out, T']
        """
        identity = x

        # Pre-activation
        out = self.bn1(x)
        out = self.relu(out)

        # First conv
        out = self.conv1(out)

        # Pre-activation
        out = self.bn2(out)
        out = self.relu(out)

        # Apply dropout
        if self.dropout is not None:
            out = self.dropout(out)

        # Second conv
        out = self.conv2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection (no ReLU after addition)
        out += identity

        return out


def make_downsample_layer(
    in_channels: int,
    out_channels: int,
    stride: int,
    expansion: int = 1
) -> nn.Sequential:
    """
    Create downsample layer for skip connection when channels or spatial size change.

    Args:
        in_channels: Input channels
        out_channels: Output channels (before expansion)
        stride: Stride for downsampling
        expansion: Channel expansion factor

    Returns:
        Sequential module with conv and batch norm
    """
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels * expansion,
            kernel_size=1,
            stride=stride,
            bias=False
        ),
        nn.BatchNorm1d(out_channels * expansion)
    )


# Test the blocks
if __name__ == "__main__":
    # Test BasicBlock1D
    print("Testing BasicBlock1D...")
    block = BasicBlock1D(64, 128, stride=2, downsample=make_downsample_layer(64, 128, 2))
    x = torch.randn(2, 64, 1000)
    y = block(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 128, 500), f"Expected (2, 128, 500), got {y.shape}"

    # Test Bottleneck1D
    print("\nTesting Bottleneck1D...")
    block = Bottleneck1D(256, 64, stride=2, downsample=make_downsample_layer(256, 64, 2, expansion=4))
    x = torch.randn(2, 256, 1000)
    y = block(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 256, 500), f"Expected (2, 256, 500), got {y.shape}"

    # Test PreActBlock1D
    print("\nTesting PreActBlock1D...")
    block = PreActBlock1D(64, 128, stride=2, downsample=make_downsample_layer(64, 128, 2))
    x = torch.randn(2, 64, 1000)
    y = block(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 128, 500), f"Expected (2, 128, 500), got {y.shape}"

    print("\n✓ All tests passed!")
