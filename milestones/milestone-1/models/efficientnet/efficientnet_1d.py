"""
EfficientNet for 1D Signals

Implements EfficientNet with compound scaling for time-series classification.

Key innovations:
- Compound scaling: Scale depth, width, and resolution together
- MBConv blocks: Efficient mobile inverted bottleneck convolutions
- Progressive scaling: B0 (baseline) → B7 (largest)

Scaling rule: α × β² × γ² ≈ 2
- α (depth): Number of layers
- β (width): Number of channels
- γ (resolution): Input signal length

Reference:
- Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for CNNs"
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

from models.base_model import BaseModel
from models.efficientnet.mbconv_block import MBConvBlock, FusedMBConvBlock


def round_channels(channels: int, width_multiplier: float, divisor: int = 8) -> int:
    """
    Round number of channels to nearest divisor.

    Args:
        channels: Base number of channels
        width_multiplier: Width scaling factor
        divisor: Ensure channels are divisible by this

    Returns:
        Rounded channels
    """
    channels *= width_multiplier
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)

    # Make sure rounding doesn't decrease by more than 10%
    if new_channels < 0.9 * channels:
        new_channels += divisor

    return int(new_channels)


def round_repeats(repeats: int, depth_multiplier: float) -> int:
    """
    Round number of block repeats.

    Args:
        repeats: Base number of repeats
        depth_multiplier: Depth scaling factor

    Returns:
        Rounded repeats
    """
    return int(math.ceil(depth_multiplier * repeats))


class EfficientNet1D(BaseModel):
    """
    EfficientNet architecture for 1D signals.

    Architecture:
        Input [B, 1, 102400]
        ├─ Stem: Conv 1→32
        ├─ Stage 1: MBConv1 (k=3, expand=1)
        ├─ Stage 2: MBConv6 (k=3, expand=6)
        ├─ Stage 3: MBConv6 (k=5, expand=6)
        ├─ Stage 4: MBConv6 (k=3, expand=6)
        ├─ Stage 5: MBConv6 (k=5, expand=6)
        ├─ Stage 6: MBConv6 (k=5, expand=6)
        ├─ Stage 7: MBConv6 (k=3, expand=6)
        ├─ Head: Conv 1280
        ├─ GlobalAvgPool
        └─ FC → num_classes

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        width_multiplier: Width scaling factor (β)
        depth_multiplier: Depth scaling factor (α)
        dropout_rate: Dropout rate
        use_se: Use Squeeze-and-Excitation blocks
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        dropout_rate: float = 0.2,
        use_se: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.dropout_rate = dropout_rate

        # Define base architecture (EfficientNet-B0)
        # [expand_ratio, channels, num_blocks, stride, kernel_size]
        base_config = [
            [1,  16,  1, 1, 3],   # Stage 1
            [6,  24,  2, 2, 3],   # Stage 2
            [6,  40,  2, 2, 5],   # Stage 3
            [6,  80,  3, 2, 3],   # Stage 4
            [6,  112, 3, 1, 5],   # Stage 5
            [6,  192, 4, 2, 5],   # Stage 6
            [6,  320, 1, 1, 3],   # Stage 7
        ]

        # Stem
        stem_channels = round_channels(32, width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU6(inplace=True)
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        in_channels = stem_channels

        for expand_ratio, channels, num_blocks, stride, kernel_size in base_config:
            out_channels = round_channels(channels, width_multiplier)
            num_blocks = round_repeats(num_blocks, depth_multiplier)

            for i in range(num_blocks):
                # Only first block in each stage has stride > 1
                block_stride = stride if i == 0 else 1

                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25 if use_se else 0,
                        dropout=dropout_rate
                    )
                )

                in_channels = out_channels

        # Head
        head_channels = round_channels(1280, width_multiplier)
        self.head = nn.Sequential(
            nn.Conv1d(
                in_channels,
                head_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm1d(head_channels),
            nn.ReLU6(inplace=True)
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channels, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Stem
        x = self.stem(x)

        # Blocks
        for block in self.blocks:
            x = block(x)

        # Head
        x = self.head(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'EfficientNet1D',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'width_multiplier': self.width_multiplier,
            'depth_multiplier': self.depth_multiplier,
            'num_parameters': self.get_num_params(),
            'dropout_rate': self.dropout_rate
        }


def calculate_scaling_params(phi: float) -> Tuple[float, float, float]:
    """
    Calculate compound scaling parameters for given phi.

    Constraint: α × β² × γ² ≈ 2^phi

    Args:
        phi: Compound coefficient

    Returns:
        (depth_multiplier, width_multiplier, resolution_multiplier)
    """
    # From EfficientNet paper
    alpha = 1.2  # Depth
    beta = 1.1   # Width
    gamma = 1.15 # Resolution

    # Scale by phi
    depth_multiplier = alpha ** phi
    width_multiplier = beta ** phi
    resolution_multiplier = gamma ** phi

    return depth_multiplier, width_multiplier, resolution_multiplier


def create_efficientnet_b0(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet1D:
    """
    EfficientNet-B0 (baseline).

    Parameters: ~1M
    Target accuracy: 94-95%
    """
    return EfficientNet1D(
        num_classes=num_classes,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        dropout_rate=0.2,
        **kwargs
    )


def create_efficientnet_b1(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet1D:
    """
    EfficientNet-B1.

    Parameters: ~1.5M
    """
    depth, width, _ = calculate_scaling_params(0.5)
    return EfficientNet1D(
        num_classes=num_classes,
        width_multiplier=width,
        depth_multiplier=depth,
        dropout_rate=0.2,
        **kwargs
    )


def create_efficientnet_b2(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet1D:
    """
    EfficientNet-B2.

    Parameters: ~2M
    """
    depth, width, _ = calculate_scaling_params(1.0)
    return EfficientNet1D(
        num_classes=num_classes,
        width_multiplier=width,
        depth_multiplier=depth,
        dropout_rate=0.3,
        **kwargs
    )


def create_efficientnet_b3(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet1D:
    """
    EfficientNet-B3 (recommended for accuracy-efficiency balance).

    Parameters: ~5M
    Target accuracy: 96-97%
    """
    depth, width, _ = calculate_scaling_params(1.8)
    return EfficientNet1D(
        num_classes=num_classes,
        width_multiplier=width,
        depth_multiplier=depth,
        dropout_rate=0.3,
        **kwargs
    )


def create_efficientnet_b4(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet1D:
    """
    EfficientNet-B4.

    Parameters: ~8M
    """
    depth, width, _ = calculate_scaling_params(2.2)
    return EfficientNet1D(
        num_classes=num_classes,
        width_multiplier=width,
        depth_multiplier=depth,
        dropout_rate=0.4,
        **kwargs
    )


def create_efficientnet_b5(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet1D:
    """
    EfficientNet-B5.

    Parameters: ~12M
    """
    depth, width, _ = calculate_scaling_params(2.6)
    return EfficientNet1D(
        num_classes=num_classes,
        width_multiplier=width,
        depth_multiplier=depth,
        dropout_rate=0.4,
        **kwargs
    )


def create_efficientnet_b6(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet1D:
    """
    EfficientNet-B6.

    Parameters: ~15M
    """
    depth, width, _ = calculate_scaling_params(2.8)
    return EfficientNet1D(
        num_classes=num_classes,
        width_multiplier=width,
        depth_multiplier=depth,
        dropout_rate=0.5,
        **kwargs
    )


def create_efficientnet_b7(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet1D:
    """
    EfficientNet-B7 (largest).

    Parameters: ~20M
    Target accuracy: 97-98%
    """
    depth, width, _ = calculate_scaling_params(3.1)
    return EfficientNet1D(
        num_classes=num_classes,
        width_multiplier=width,
        depth_multiplier=depth,
        dropout_rate=0.5,
        **kwargs
    )


# Test
if __name__ == "__main__":
    print("Testing EfficientNet-1D...")

    # Test B0
    print("\nEfficientNet-B0:")
    model = create_efficientnet_b0(num_classes=NUM_CLASSES)
    x = torch.randn(2, 1, SIGNAL_LENGTH)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11)

    # Test B3
    print("\nEfficientNet-B3:")
    model = create_efficientnet_b3(num_classes=NUM_CLASSES)
    y = model(x)
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11)

    print("\n✓ EfficientNet tests passed!")
