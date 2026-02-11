"""
EfficientNet-2D for Spectrogram Classification

Implements parameter-efficient EfficientNet architecture for spectrograms.
Uses compound scaling (width, depth, resolution) for optimal efficiency.

Reference:
- Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for CNNs"
- Adapted for single-channel spectrograms

Input: [B, 1, H, W] where H=n_freq (129), W=n_time (400)
Output: [B, 11] for 11 fault classes
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
import math


from packages.core.models.base_model import BaseModel


class SwishActivation(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    Also known as SiLU (Sigmoid Linear Unit)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Recalibrates channel-wise feature responses by modeling
    interdependencies between channels.

    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for bottleneck (default: 4)
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()

        reduced_channels = max(1, in_channels // reduction_ratio)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            SwishActivation(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        scale = self.squeeze(x)       # [B, C, 1, 1]
        scale = self.excitation(scale)  # [B, C, 1, 1]
        return x * scale               # Broadcast multiplication


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block.

    Structure:
        Input → [Expand → DWConv → SE → Project] → Output
                └────────────────────────────────┘
                         (Skip connection if stride=1)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise conv (3 or 5)
        stride: Stride for depthwise conv (1 or 2)
        expand_ratio: Expansion ratio for inverted bottleneck (1, 4, or 6)
        se_ratio: Squeeze-excitation reduction ratio (default: 4)
        dropout_rate: Dropout rate for stochastic depth (default: 0.0)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: int = 4,
        dropout_rate: float = 0.0
    ):
        super().__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.dropout_rate = dropout_rate

        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                SwishActivation()
            )
        else:
            self.expand = nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels, expanded_channels,
                kernel_size=kernel_size, stride=stride,
                padding=kernel_size // 2, groups=expanded_channels,
                bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            SwishActivation()
        )

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(expanded_channels, se_ratio)

        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Stochastic depth (dropout) for regularization
        if dropout_rate > 0 and self.use_residual:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Expansion
        x = self.expand(x)

        # Depthwise convolution
        x = self.depthwise(x)

        # Squeeze-and-Excitation
        x = self.se(x)

        # Projection
        x = self.project(x)

        # Skip connection with stochastic depth
        if self.use_residual:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity

        return x


class EfficientNet2DSpectrogram(BaseModel):
    """
    EfficientNet architecture for spectrogram classification.

    Configuration (EfficientNet-B0):
        Input [B, 1, 129, 400]
        ├─ Stem: Conv3x3, 1→32 → [B, 32, 65, 200]
        ├─ MBConv1 (k3, e1): 32→16, 1 block
        ├─ MBConv6 (k3, e6): 16→24, 2 blocks, s=2
        ├─ MBConv6 (k5, e6): 24→40, 2 blocks, s=2
        ├─ MBConv6 (k3, e6): 40→80, 3 blocks, s=2
        ├─ MBConv6 (k5, e6): 80→112, 3 blocks
        ├─ MBConv6 (k5, e6): 112→192, 4 blocks, s=2
        ├─ MBConv6 (k3, e6): 192→320, 1 block
        ├─ Head: Conv1x1, 320→1280 → [B, 1280, *, *]
        ├─ GlobalAvgPool → [B, 1280]
        └─ FC: 1280 → 11

    Args:
        num_classes: Number of output classes (default: 11)
        input_channels: Number of input channels (default: 1)
        width_mult: Width multiplier for scaling (default: 1.0 for B0)
        depth_mult: Depth multiplier for scaling (default: 1.0 for B0)
        dropout_rate: Dropout rate before final FC (default: 0.2)
        drop_connect_rate: Drop connect rate for stochastic depth (default: 0.2)
    """

    # EfficientNet-B0 configuration
    # (expand_ratio, out_channels, num_blocks, kernel_size, stride)
    BLOCK_CONFIGS = [
        (1, 16, 1, 3, 1),    # Stage 1
        (6, 24, 2, 3, 2),    # Stage 2
        (6, 40, 2, 5, 2),    # Stage 3
        (6, 80, 3, 3, 2),    # Stage 4
        (6, 112, 3, 5, 1),   # Stage 5
        (6, 192, 4, 5, 2),   # Stage 6
        (6, 320, 1, 3, 1),   # Stage 7
    ]

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate

        # Stem: Initial convolution
        stem_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, stem_channels,
                kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(stem_channels),
            SwishActivation()
        )

        # Build MBConv blocks
        self.blocks = nn.ModuleList([])
        in_channels = stem_channels
        total_blocks = sum(self._round_repeats(cfg[2], depth_mult) for cfg in self.BLOCK_CONFIGS)
        block_idx = 0

        for expand_ratio, out_channels, num_blocks, kernel_size, stride in self.BLOCK_CONFIGS:
            out_channels = self._round_filters(out_channels, width_mult)
            num_blocks = self._round_repeats(num_blocks, depth_mult)

            for i in range(num_blocks):
                # Stochastic depth: linearly increase dropout rate
                drop_rate = drop_connect_rate * block_idx / total_blocks

                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,  # Only first block uses stride
                        expand_ratio=expand_ratio,
                        dropout_rate=drop_rate
                    )
                )

                in_channels = out_channels
                block_idx += 1

        # Head
        head_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            SwishActivation()
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channels, num_classes)

        self._initialize_weights()

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            'model': 'EfficientNet2DSpectrogram',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'dropout_rate': self.dropout_rate,
        }

    def _round_filters(self, filters: int, width_mult: float) -> int:
        """Round number of filters based on width multiplier."""
        if width_mult == 1.0:
            return filters

        filters = int(filters * width_mult)
        # Make divisible by 8 for efficient computation
        new_filters = max(8, int(filters + 8 / 2) // 8 * 8)

        # Ensure we don't reduce by more than 10%
        if new_filters < 0.9 * filters:
            new_filters += 8

        return new_filters

    def _round_repeats(self, repeats: int, depth_mult: float) -> int:
        """Round number of block repeats based on depth multiplier."""
        if depth_mult == 1.0:
            return repeats
        return int(math.ceil(depth_mult * repeats))

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrogram [B, 1, H, W]

        Returns:
            Logits [B, num_classes]
        """
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Extract intermediate feature maps."""
        features = {}

        x = self.stem(x)
        features['stem'] = x

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [0, 2, 4, 6, 9, 14]:  # Sample some layers
                features[f'block_{i}'] = x

        x = self.head(x)
        features['head'] = x

        return features


def efficientnet_b0(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet2DSpectrogram:
    """EfficientNet-B0 (baseline, 5.3M params)."""
    return EfficientNet2DSpectrogram(
        num_classes=num_classes,
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.2,
        **kwargs
    )


def efficientnet_b1(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet2DSpectrogram:
    """EfficientNet-B1 (7.8M params)."""
    return EfficientNet2DSpectrogram(
        num_classes=num_classes,
        width_mult=1.0,
        depth_mult=1.1,
        dropout_rate=0.2,
        **kwargs
    )


def efficientnet_b3(num_classes: int = NUM_CLASSES, **kwargs) -> EfficientNet2DSpectrogram:
    """EfficientNet-B3 (12M params)."""
    return EfficientNet2DSpectrogram(
        num_classes=num_classes,
        width_mult=1.2,
        depth_mult=1.4,
        dropout_rate=0.3,
        **kwargs
    )


if __name__ == '__main__':
    # Test the model
    model = efficientnet_b0(num_classes=NUM_CLASSES)

    # Test forward pass
    batch_size = 4
    spectrogram = torch.randn(batch_size, 1, 129, 400)
    output = model(spectrogram)

    print(f"Input shape: {spectrogram.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test feature extraction
    features = model.get_feature_maps(spectrogram)
    print("\nFeature map shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
