"""
Wide ResNet: Wider but Shallower Networks

Instead of going deeper, Wide ResNet increases channel width.
Trade-off: More parameters but shallower network (faster training, parallelizable).

Key idea:
- Standard ResNet-18: [64, 128, 256, 512] channels
- Wide ResNet-16-8: [64*8, 128*8, 256*8, 512*8] = [512, 1024, 2048, 4096] channels

Reference:
- Zagoruyko & Komodakis (2016). "Wide Residual Networks" (BMVC)
- Showed wider networks can match or beat deeper networks
"""

import torch
import torch.nn as nn
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_model import BaseModel
from resnet.residual_blocks import BasicBlock1D, make_downsample_layer


class WideResNet1D(BaseModel):
    """
    Wide ResNet for 1D signals.

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        depth: Total network depth (e.g., 16, 22, 28, 40)
        widen_factor: Channel width multiplier (e.g., 2, 4, 8, 10)
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 1,
        depth: int = 16,
        widen_factor: int = 8,
        dropout: float = 0.3  # Wide networks benefit from higher dropout
    ):
        super().__init__()

        # Calculate number of blocks per layer
        # depth = 4 + 6*n (for 3 layers)
        # Example: depth=16 → n=2, depth=22 → n=3, depth=28 → n=4
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n_blocks = (depth - 4) // 6

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.depth = depth
        self.widen_factor = widen_factor
        self.dropout = dropout

        # Base channels widened by widen_factor
        base_channels = [16, 16, 32, 64]
        channels = [c * widen_factor for c in base_channels]

        self.in_channels = channels[0]

        # Initial convolution
        self.conv1 = nn.Conv1d(
            input_channels,
            channels[0],
            kernel_size=64,
            stride=4,
            padding=32,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

        # Wide residual layers
        self.layer1 = self._make_layer(BasicBlock1D, channels[1], n_blocks, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, channels[2], n_blocks, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, channels[3], n_blocks, stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[3] * BasicBlock1D.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        block: type,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create wide residual layer."""
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
                dropout=self.dropout
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
                    dropout=self.dropout
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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        logits = self.fc(x)

        return logits

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'WideResNet1D',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'depth': self.depth,
            'widen_factor': self.widen_factor,
            'num_parameters': self.get_num_params(),
            'dropout': self.dropout
        }


def create_wide_resnet16_8(num_classes: int = 11, **kwargs) -> WideResNet1D:
    """
    Create Wide ResNet-16-8 (16 layers, 8× width).

    Approximately 5-10M parameters.
    Good balance of width and depth.
    """
    return WideResNet1D(
        num_classes=num_classes,
        depth=16,
        widen_factor=8,
        **kwargs
    )


def create_wide_resnet16_10(num_classes: int = 11, **kwargs) -> WideResNet1D:
    """
    Create Wide ResNet-16-10 (16 layers, 10× width).

    Approximately 10-15M parameters.
    Very wide network.
    """
    return WideResNet1D(
        num_classes=num_classes,
        depth=16,
        widen_factor=10,
        **kwargs
    )


def create_wide_resnet22_8(num_classes: int = 11, **kwargs) -> WideResNet1D:
    """
    Create Wide ResNet-22-8 (22 layers, 8× width).

    Approximately 8-12M parameters.
    Deeper and wider.
    """
    return WideResNet1D(
        num_classes=num_classes,
        depth=22,
        widen_factor=8,
        **kwargs
    )


def create_wide_resnet28_10(num_classes: int = 11, **kwargs) -> WideResNet1D:
    """
    Create Wide ResNet-28-10 (28 layers, 10× width).

    Approximately 15-20M parameters.
    Very large network, highest capacity.
    """
    return WideResNet1D(
        num_classes=num_classes,
        depth=28,
        widen_factor=10,
        **kwargs
    )


# Test
if __name__ == "__main__":
    print("Testing Wide ResNet...")

    model = create_wide_resnet16_8(num_classes=11)
    x = torch.randn(2, 1, 102400)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11)

    print(f"\nWide ResNet-16-8 config: {model.get_config()}")

    print("\n✓ Wide ResNet tests passed!")
