"""
SE-ResNet: ResNet with Squeeze-and-Excitation Blocks

Implements ResNet enhanced with Squeeze-and-Excitation (SE) channel attention.
SE blocks adaptively recalibrate channel-wise feature responses.

Expected benefit: +1-2% accuracy improvement over standard ResNet.

Reference:
- Hu et al. (2018). "Squeeze-and-Excitation Networks" (CVPR 2018)
- Won ImageNet 2017 classification challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from models.base_model import BaseModel
from models.resnet.residual_blocks import BasicBlock1D, Bottleneck1D, make_downsample_layer


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Architecture:
        Input [B, C, T]
          ↓ Squeeze (Global pooling)
        [B, C, 1]
          ↓ Excitation (FC → ReLU → FC → Sigmoid)
        [B, C, 1] (attention weights)
          ↓ Recalibration (element-wise multiplication)
        Output [B, C, T]

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        reduced_channels = max(channels // reduction, 1)

        # Squeeze: Global average pooling
        self.squeeze = nn.AdaptiveAvgPool1d(1)

        # Excitation: 2-layer FC with bottleneck
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SE channel attention.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Recalibrated tensor [B, C, T]
        """
        batch_size, channels, _ = x.shape

        # Squeeze: Global pooling
        squeeze = self.squeeze(x)  # [B, C, 1]
        squeeze = squeeze.view(batch_size, channels)  # [B, C]

        # Excitation: Channel attention weights
        attention = self.excitation(squeeze)  # [B, C]
        attention = attention.view(batch_size, channels, 1)  # [B, C, 1]

        # Recalibration: Scale channels by attention
        return x * attention


class SEBasicBlock1D(nn.Module):
    """
    Basic residual block with SE attention.

    Structure:
        x -> Conv1 -> BN -> ReLU -> Conv2 -> BN -> SE -> (+) -> ReLU
        |___________________________________________________|

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size
        stride: Stride for first convolution
        downsample: Downsample layer for skip connection
        dropout: Dropout probability
        reduction: SE reduction ratio
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1,
        reduction: int = 16
    ):
        super().__init__()

        # Two conv layers (same as BasicBlock1D)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE block
        self.se = SEBlock(out_channels, reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SE attention."""
        identity = x

        # Conv path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE attention
        out = self.se(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBottleneck1D(nn.Module):
    """
    Bottleneck residual block with SE attention.

    Structure:
        x -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> SE -> (+) -> ReLU
        |__________________________________________________________________________|

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (before expansion)
        kernel_size: Kernel size for middle conv
        stride: Stride for middle convolution
        downsample: Downsample layer for skip connection
        dropout: Dropout probability
        reduction: SE reduction ratio
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1,
        reduction: int = 16
    ):
        super().__init__()

        # Bottleneck: 1x1 → 3x3 → 1x1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        # SE block (applied to expanded channels)
        self.se = SEBlock(out_channels * self.expansion, reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SE attention."""
        identity = x

        # Bottleneck path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply SE attention
        out = self.se(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNet1D(BaseModel):
    """
    SE-ResNet architecture for 1D signals.

    Same as ResNet but with SE blocks for channel attention.

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        block: Block type (SEBasicBlock1D or SEBottleneck1D)
        layers: Number of blocks in each layer
        dropout: Dropout probability
        reduction: SE reduction ratio
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 1,
        block: type = SEBasicBlock1D,
        layers: List[int] = None,
        dropout: float = 0.1,
        reduction: int = 16
    ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]  # SE-ResNet-18

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.in_channels = 64
        self.dropout = dropout
        self.reduction = reduction

        # Initial convolution
        self.conv1 = nn.Conv1d(
            input_channels, 64,
            kernel_size=64, stride=4, padding=32, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

        # Residual layers with SE blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        block: type,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create residual layer with SE blocks."""
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = make_downsample_layer(
                self.in_channels,
                out_channels,
                stride,
                block.expansion
            )

        layers = []

        # First block
        layers.append(
            block(
                self.in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                downsample=downsample,
                dropout=self.dropout,
                reduction=self.reduction
            )
        )

        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    downsample=None,
                    dropout=self.dropout,
                    reduction=self.reduction
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        logits = self.fc(x)

        return logits

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'SEResNet1D',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'num_parameters': self.get_num_params(),
            'dropout': self.dropout,
            'se_reduction': self.reduction
        }


def create_se_resnet18_1d(num_classes: int = 11, **kwargs) -> SEResNet1D:
    """Create SE-ResNet-18."""
    return SEResNet1D(
        num_classes=num_classes,
        block=SEBasicBlock1D,
        layers=[2, 2, 2, 2],
        **kwargs
    )


def create_se_resnet34_1d(num_classes: int = 11, **kwargs) -> SEResNet1D:
    """Create SE-ResNet-34."""
    return SEResNet1D(
        num_classes=num_classes,
        block=SEBasicBlock1D,
        layers=[3, 4, 6, 3],
        **kwargs
    )


def create_se_resnet50_1d(num_classes: int = 11, **kwargs) -> SEResNet1D:
    """Create SE-ResNet-50."""
    return SEResNet1D(
        num_classes=num_classes,
        block=SEBottleneck1D,
        layers=[3, 4, 6, 3],
        **kwargs
    )


# Test
if __name__ == "__main__":
    print("Testing SE-ResNet...")

    model = create_se_resnet18_1d(num_classes=11)
    x = torch.randn(2, 1, 102400)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11)

    print("\n✓ SE-ResNet tests passed!")
