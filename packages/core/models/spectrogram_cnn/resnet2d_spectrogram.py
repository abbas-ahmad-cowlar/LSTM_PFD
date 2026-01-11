"""
ResNet-2D for Spectrogram Classification

Adapts ResNet architecture for 2D spectrograms with transfer learning from ImageNet.
Processes time-frequency representations for bearing fault diagnosis.

Reference:
- He et al. (2016). "Deep Residual Learning for Image Recognition"
- Adapted for single-channel spectrograms with optional transfer learning

Input: [B, 1, H, W] where H=n_freq (129), W=n_time (400)
Output: [B, 11] for 11 fault classes
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type, Union


from packages.core.models.base_model import BaseModel


class BasicBlock2D(nn.Module):
    """
    Basic residual block for 2D convolutions.

    Structure:
        x → [Conv2D → BN → ReLU → Conv2D → BN] → (+) → ReLU
        └────────────────────────────────────────┘
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck2D(nn.Module):
    """
    Bottleneck residual block for 2D convolutions (used in ResNet-50+).

    Structure:
        x → [Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN] → (+) → ReLU
        └──────────────────────────────────────────────────────────────┘
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2DSpectrogram(BaseModel):
    """
    ResNet architecture for spectrogram classification.

    Architecture (ResNet-18):
        Input [B, 1, 129, 400]
        ├─ Conv1: 1→64, k=7, s=2 → [B, 64, 65, 200]
        ├─ MaxPool: k=3, s=2 → [B, 64, 33, 100]
        ├─ Layer1: 2× BasicBlock, 64 channels → [B, 64, 33, 100]
        ├─ Layer2: 2× BasicBlock, 128 channels, s=2 → [B, 128, 17, 50]
        ├─ Layer3: 2× BasicBlock, 256 channels, s=2 → [B, 256, 9, 25]
        ├─ Layer4: 2× BasicBlock, 512 channels, s=2 → [B, 512, 5, 13]
        ├─ AdaptiveAvgPool → [B, 512]
        └─ FC: 512 → 11

    Args:
        num_classes: Number of output classes (default: 11)
        input_channels: Number of input channels (default: 1 for grayscale spectrograms)
        block: Block type (BasicBlock2D or Bottleneck2D)
        layers: Number of blocks in each layer (default: [2,2,2,2] for ResNet-18)
        pretrained: Whether to use ImageNet pretrained weights (default: False)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        block: Type[Union[BasicBlock2D, Bottleneck2D]] = BasicBlock2D,
        layers: List[int] = None,
        pretrained: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]  # ResNet-18 configuration

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.in_channels = 64
        self.dropout = dropout
        self.block = block

        # Initial convolution layer
        # Modified for single-channel spectrograms (vs. 3-channel RGB in ImageNet)
        self.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Transfer learning from ImageNet
        if pretrained:
            self._load_pretrained_weights()
        else:
            self._initialize_weights()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock2D, Bottleneck2D]],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        downsample = None

        # Downsampling needed if stride != 1 or channel count changes
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self):
        """
        Load ImageNet pretrained weights from torchvision.

        Note: Conv1 layer is reinitialized since spectrograms have 1 channel
        vs. 3 channels in ImageNet. Rest of the network uses pretrained weights.
        """
        try:
            from torchvision.models import resnet18
            pretrained_model = resnet18(pretrained=True)

            # Transfer weights except conv1 (different input channels)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()

            # Filter out conv1 weights and FC layer
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and 'conv1' not in k and 'fc' not in k
                and model_dict[k].shape == v.shape
            }

            # Update model dictionary
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

            # Initialize conv1 and fc separately
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.normal_(self.fc.weight, 0, 0.01)
            nn.init.constant_(self.fc.bias, 0)

            print(f"✓ Loaded ImageNet pretrained weights (except conv1 and fc)")

        except Exception as e:
            print(f"⚠ Could not load pretrained weights: {e}")
            print("  Proceeding with random initialization")
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrogram [B, 1, H, W]

        Returns:
            Logits [B, num_classes]
        """
        # Input: [B, 1, 129, 400]
        x = self.conv1(x)        # [B, 64, 65, 200]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)      # [B, 64, 33, 100]

        x = self.layer1(x)       # [B, 64, 33, 100]
        x = self.layer2(x)       # [B, 128, 17, 50]
        x = self.layer3(x)       # [B, 256, 9, 25]
        x = self.layer4(x)       # [B, 512, 5, 13]

        x = self.avgpool(x)      # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.dropout_layer(x)
        x = self.fc(x)           # [B, 11]

        return x

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for visualization.

        Args:
            x: Input spectrogram [B, 1, H, W]

        Returns:
            Dictionary of feature maps at different layers
        """
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features['conv1'] = x

        x = self.layer1(x)
        features['layer1'] = x

        x = self.layer2(x)
        features['layer2'] = x

        x = self.layer3(x)
        features['layer3'] = x

        x = self.layer4(x)
        features['layer4'] = x

        return features


def resnet18_2d(num_classes: int = NUM_CLASSES, pretrained: bool = False, **kwargs) -> ResNet2DSpectrogram:
    """ResNet-18 for spectrograms."""
    return ResNet2DSpectrogram(
        num_classes=num_classes,
        block=BasicBlock2D,
        layers=[2, 2, 2, 2],
        pretrained=pretrained,
        **kwargs
    )


def resnet34_2d(num_classes: int = NUM_CLASSES, pretrained: bool = False, **kwargs) -> ResNet2DSpectrogram:
    """ResNet-34 for spectrograms."""
    return ResNet2DSpectrogram(
        num_classes=num_classes,
        block=BasicBlock2D,
        layers=[3, 4, 6, 3],
        pretrained=pretrained,
        **kwargs
    )


def resnet50_2d(num_classes: int = NUM_CLASSES, pretrained: bool = False, **kwargs) -> ResNet2DSpectrogram:
    """ResNet-50 for spectrograms (uses Bottleneck blocks)."""
    return ResNet2DSpectrogram(
        num_classes=num_classes,
        block=Bottleneck2D,
        layers=[3, 4, 6, 3],
        pretrained=pretrained,
        **kwargs
    )


if __name__ == '__main__':
    # Test the model
    model = resnet18_2d(num_classes=NUM_CLASSES, pretrained=False)

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
