"""
Multi-Scale 1D CNN for Bearing Fault Diagnosis

Implements multi-scale feature extraction to capture fault signatures at
different temporal resolutions simultaneously.

Key Concepts:
- Parallel conv branches with different kernel sizes (multi-scale)
- Inception-style modules adapted for 1D signals
- Multi-resolution feature fusion
- Dilated convolutions for expanded receptive fields

Multi-scale processing is beneficial for bearing fault diagnosis because:
- Different faults have characteristic frequencies at different scales
- Early faults may have subtle high-frequency components
- Severe faults manifest in low-frequency envelope modulations

Expected Performance: 94-96% test accuracy

Author: Phase 2 - CNN Implementation
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class InceptionModule1D(nn.Module):
    """
    Inception module adapted for 1D signals

    Processes input with multiple parallel convolutional paths:
    - 1×1 conv (dimensionality reduction)
    - 3×1 conv (local patterns)
    - 5×1 conv (medium-scale patterns)
    - 7×1 conv (large-scale patterns)
    - Max pooling path

    Outputs are concatenated channel-wise.

    Args:
        in_channels: Number of input channels
        out_channels_1x1: Output channels for 1x1 conv
        out_channels_3x1: Output channels for 3x1 conv
        out_channels_5x1: Output channels for 5x1 conv
        out_channels_7x1: Output channels for 7x1 conv
        out_channels_pool: Output channels for pooling path
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels_1x1: int,
        out_channels_3x1: int,
        out_channels_5x1: int,
        out_channels_7x1: int,
        out_channels_pool: int,
        dropout: float = 0.2
    ):
        super().__init__()

        # 1x1 conv branch (point-wise)
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_1x1, kernel_size=1),
            nn.BatchNorm1d(out_channels_1x1),
            nn.ReLU(inplace=True)
        )

        # 3x1 conv branch
        self.branch3x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_3x1, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels_3x1),
            nn.ReLU(inplace=True)
        )

        # 5x1 conv branch
        self.branch5x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_5x1, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels_5x1),
            nn.ReLU(inplace=True)
        )

        # 7x1 conv branch
        self.branch7x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_7x1, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels_7x1),
            nn.ReLU(inplace=True)
        )

        # Pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels_pool, kernel_size=1),
            nn.BatchNorm1d(out_channels_pool),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch, in_channels, length]

        Returns:
            Concatenated multi-scale features [batch, total_out_channels, length]
        """
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x1(x)
        branch3 = self.branch5x1(x)
        branch4 = self.branch7x1(x)
        branch5 = self.branch_pool(x)

        # Concatenate along channel dimension
        out = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
        out = self.dropout(out)

        return out


class DilatedConvBlock(nn.Module):
    """
    Dilated convolutional block for expanded receptive field

    Dilated convolutions allow the network to have a large receptive field
    without increasing the number of parameters or reducing resolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size
        dilation: Dilation rate
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MultiScaleCNN1D(nn.Module):
    """
    Multi-Scale 1D CNN with Inception modules

    Architecture:
        - Initial conv for feature extraction
        - 3 Inception modules (multi-scale parallel processing)
        - Max pooling between modules for downsampling
        - Global average pooling
        - Fully connected classifier

    Args:
        num_classes: Number of fault classes (default: 11)
        input_length: Input signal length (default: 102400)
        in_channels: Number of input channels (default: 1)
        dropout: Dropout probability (default: 0.3)

    Examples:
        >>> model = MultiScaleCNN1D(num_classes=11, input_length=102400)
        >>> signal = torch.randn(16, 1, 102400)
        >>> output = model(signal)
        >>> print(output.shape)  # [16, 11]
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_length: int = 102400,
        in_channels: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_length = input_length
        self.in_channels = in_channels

        # Initial convolution
        self.conv_initial = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Inception module 1
        self.inception1 = InceptionModule1D(
            in_channels=32,
            out_channels_1x1=16,
            out_channels_3x1=32,
            out_channels_5x1=16,
            out_channels_7x1=8,
            out_channels_pool=8,
            dropout=dropout
        )
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Inception module 2
        self.inception2 = InceptionModule1D(
            in_channels=80,  # 16+32+16+8+8
            out_channels_1x1=32,
            out_channels_3x1=64,
            out_channels_5x1=32,
            out_channels_7x1=16,
            out_channels_pool=16,
            dropout=dropout
        )
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Inception module 3
        self.inception3 = InceptionModule1D(
            in_channels=160,  # 32+64+32+16+16
            out_channels_1x1=64,
            out_channels_3x1=128,
            out_channels_5x1=64,
            out_channels_7x1=32,
            out_channels_pool=32,
            dropout=dropout
        )
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, 256),  # 64+128+64+32+32
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input signal [batch, 1, length]

        Returns:
            Class logits [batch, num_classes]
        """
        # Initial conv
        x = self.conv_initial(x)  # [B, 32, L/4]

        # Inception modules with pooling
        x = self.inception1(x)    # [B, 80, L/4]
        x = self.pool1(x)          # [B, 80, L/8]

        x = self.inception2(x)    # [B, 160, L/8]
        x = self.pool2(x)          # [B, 160, L/16]

        x = self.inception3(x)    # [B, 320, L/16]
        x = self.pool3(x)          # [B, 320, L/32]

        # Global pooling
        x = self.global_pool(x)   # [B, 320, 1]

        # Classification
        x = self.classifier(x)    # [B, num_classes]

        return x

    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DilatedMultiScaleCNN(nn.Module):
    """
    Multi-scale CNN using dilated convolutions

    Uses dilated convolutions with increasing dilation rates to capture
    features at multiple temporal scales efficiently.

    Dilation rates: 1, 2, 4, 8, 16 (exponentially increasing)

    Args:
        num_classes: Number of fault classes
        input_length: Input signal length
        in_channels: Number of input channels
        base_channels: Base number of channels
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_length: int = 102400,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_length = input_length

        # Initial conv
        self.conv_initial = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Dilated conv blocks with increasing dilation
        self.dilated1 = DilatedConvBlock(base_channels, base_channels*2,
                                        kernel_size=3, dilation=1, dropout=dropout)
        self.dilated2 = DilatedConvBlock(base_channels*2, base_channels*4,
                                        kernel_size=3, dilation=2, dropout=dropout)
        self.dilated3 = DilatedConvBlock(base_channels*4, base_channels*8,
                                        kernel_size=3, dilation=4, dropout=dropout)
        self.dilated4 = DilatedConvBlock(base_channels*8, base_channels*16,
                                        kernel_size=3, dilation=8, dropout=dropout)

        # Downsampling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels*16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input signal [batch, 1, length]

        Returns:
            Class logits [batch, num_classes]
        """
        x = self.conv_initial(x)  # [B, 32, L/4]

        x = self.dilated1(x)      # [B, 64, L/4], dilation=1
        x = self.pool(x)           # [B, 64, L/8]

        x = self.dilated2(x)      # [B, 128, L/8], dilation=2
        x = self.pool(x)           # [B, 128, L/16]

        x = self.dilated3(x)      # [B, 256, L/16], dilation=4
        x = self.pool(x)           # [B, 256, L/32]

        x = self.dilated4(x)      # [B, 512, L/32], dilation=8

        x = self.global_pool(x)   # [B, 512, 1]
        x = self.classifier(x)    # [B, num_classes]

        return x

    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_multi_scale_cnn():
    """Test multi-scale CNN models"""
    print("Testing Multi-Scale CNN Models...\n")

    # Test InceptionModule1D
    print("=" * 60)
    print("Testing InceptionModule1D")
    print("=" * 60)

    inception = InceptionModule1D(
        in_channels=32,
        out_channels_1x1=16,
        out_channels_3x1=32,
        out_channels_5x1=16,
        out_channels_7x1=8,
        out_channels_pool=8
    )

    x = torch.randn(4, 32, 1000)
    out = inception(x)
    print(f"✓ Inception module:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape[1] == 16 + 32 + 16 + 8 + 8, "Channel mismatch"

    # Test MultiScaleCNN1D
    print("\n" + "=" * 60)
    print("Testing MultiScaleCNN1D")
    print("=" * 60)

    model = MultiScaleCNN1D(num_classes=11, input_length=102400)
    print(f"✓ Model created")
    print(f"  Parameters: {model.get_num_params():,}")

    # Forward pass
    signal = torch.randn(4, 1, 102400)
    output = model(signal)

    print(f"\n✓ Forward pass:")
    print(f"  Input:  {signal.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == (4, 11), "Output shape mismatch"

    # Backward pass
    loss = output.sum()
    loss.backward()
    print(f"\n✓ Backward pass successful")

    # Test DilatedMultiScaleCNN
    print("\n" + "=" * 60)
    print("Testing DilatedMultiScaleCNN")
    print("=" * 60)

    dilated_model = DilatedMultiScaleCNN(num_classes=11, input_length=102400)
    print(f"✓ Dilated model created")
    print(f"  Parameters: {dilated_model.get_num_params():,}")

    output = dilated_model(signal)
    print(f"\n✓ Forward pass:")
    print(f"  Input:  {signal.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == (4, 11)

    # Compare models
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"MultiScaleCNN1D:        {model.get_num_params():>10,} parameters")
    print(f"DilatedMultiScaleCNN:   {dilated_model.get_num_params():>10,} parameters")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_multi_scale_cnn()
