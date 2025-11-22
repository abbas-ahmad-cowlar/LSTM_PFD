"""
ResNet-18 Adapted for 1D Signals

Implements ResNet architecture for time-series bearing fault diagnosis.
Uses residual connections to enable deeper networks and better gradient flow.

Reference:
- He et al. (2016). "Deep Residual Learning for Image Recognition"
- Adapted for 1D signals with appropriate kernel sizes

Input: [B, 1, T] where T is signal length (102400)
Output: [B, 11] for 11 fault classes
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type, Union

from models.base_model import BaseModel
from models.resnet.residual_blocks import BasicBlock1D, Bottleneck1D, make_downsample_layer


class ResNet1D(BaseModel):
    """
    ResNet architecture adapted for 1D signal classification.

    Architecture (ResNet-18):
        Input [B, 1, 102400]
        ├─ Conv1: 1→64, k=64, s=4 → [B, 64, 25600]
        ├─ MaxPool: k=4, s=4 → [B, 64, 6400]
        ├─ Layer1: 2× BasicBlock, 64 channels → [B, 64, 6400]
        ├─ Layer2: 2× BasicBlock, 128 channels, s=2 → [B, 128, 3200]
        ├─ Layer3: 2× BasicBlock, 256 channels, s=2 → [B, 256, 1600]
        ├─ Layer4: 2× BasicBlock, 512 channels, s=2 → [B, 512, 800]
        ├─ AdaptiveAvgPool → [B, 512]
        └─ FC: 512 → 11

    Args:
        num_classes: Number of output classes (default: 11)
        input_channels: Number of input channels (default: 1)
        block: Block type to use (BasicBlock1D or Bottleneck1D)
        layers: Number of blocks in each layer (default: [2, 2, 2, 2] for ResNet-18)
        dropout: Dropout probability (default: 0.1)
        input_length: Expected input signal length (default: 102400)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        block: Type[Union[BasicBlock1D, Bottleneck1D]] = BasicBlock1D,
        layers: List[int] = None,
        dropout: float = 0.1,
        input_length: int = SIGNAL_LENGTH
    ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]  # ResNet-18 configuration

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.in_channels = 64
        self.dropout = dropout
        self.input_length = input_length
        self.block = block

        # Initial convolution - adapted for long signals
        # Kernel size 64 to capture low-frequency bearing signatures
        self.conv1 = nn.Conv1d(
            input_channels,
            64,
            kernel_size=64,
            stride=4,
            padding=32,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling for further downsampling
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock1D, Bottleneck1D]],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a residual layer with multiple blocks.

        Args:
            block: Block class to use (BasicBlock1D or Bottleneck1D)
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for first block (for downsampling)

        Returns:
            Sequential layer containing blocks
        """
        downsample = None

        # If stride != 1 or channels change, need downsample layer for skip connection
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = make_downsample_layer(
                self.in_channels,
                out_channels,
                stride,
                block.expansion
            )

        layers = []

        # First block (may downsample)
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

        # Update in_channels for subsequent blocks
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
        """Initialize weights using He initialization for ReLU networks."""
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
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, T] or [B, T]

        Returns:
            logits: Output tensor of shape [B, num_classes]
        """
        # Ensure input is 3D [B, C, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)  # [B, 512 * expansion, 1]
        x = torch.flatten(x, 1)  # [B, 512 * expansion]

        # Classification
        logits = self.fc(x)

        return logits

    def get_feature_extractor(self) -> nn.Module:
        """
        Return the feature extraction backbone (all layers except fc).
        Useful for transfer learning and feature extraction.

        Returns:
            Feature extractor module
        """
        return nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool
        )

    def freeze_backbone(self):
        """Freeze feature extraction layers for transfer learning."""
        for param in self.get_feature_extractor().parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze feature extraction layers."""
        for param in self.get_feature_extractor().parameters():
            param.requires_grad = True

    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """
        Get intermediate feature maps from each layer.
        Useful for visualization and analysis.

        Args:
            x: Input tensor of shape [B, C, T]

        Returns:
            Dictionary of feature maps at each layer
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        features = {}

        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['stem'] = x.clone()
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        features['layer1'] = x.clone()

        x = self.layer2(x)
        features['layer2'] = x.clone()

        x = self.layer3(x)
        features['layer3'] = x.clone()

        x = self.layer4(x)
        features['layer4'] = x.clone()

        x = self.avgpool(x)
        features['pooled'] = x.clone()

        return features

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'ResNet1D',
            'block_type': self.block.__name__,
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'input_length': self.input_length,
            'num_parameters': self.get_num_params(),
            'dropout': self.dropout
        }


def create_resnet18_1d(num_classes: int = NUM_CLASSES, **kwargs) -> ResNet1D:
    """
    Factory function to create ResNet-18 for 1D signals.

    ResNet-18: [2, 2, 2, 2] blocks, ~2.5M parameters

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to ResNet1D

    Returns:
        ResNet1D model instance
    """
    return ResNet1D(
        num_classes=num_classes,
        block=BasicBlock1D,
        layers=[2, 2, 2, 2],
        **kwargs
    )


def create_resnet34_1d(num_classes: int = NUM_CLASSES, **kwargs) -> ResNet1D:
    """
    Factory function to create ResNet-34 for 1D signals.

    ResNet-34: [3, 4, 6, 3] blocks, ~5M parameters

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to ResNet1D

    Returns:
        ResNet1D model instance
    """
    return ResNet1D(
        num_classes=num_classes,
        block=BasicBlock1D,
        layers=[3, 4, 6, 3],
        **kwargs
    )


def create_resnet50_1d(num_classes: int = NUM_CLASSES, **kwargs) -> ResNet1D:
    """
    Factory function to create ResNet-50 for 1D signals.

    ResNet-50: [3, 4, 6, 3] bottleneck blocks, ~10M parameters

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to ResNet1D

    Returns:
        ResNet1D model instance
    """
    return ResNet1D(
        num_classes=num_classes,
        block=Bottleneck1D,
        layers=[3, 4, 6, 3],
        **kwargs
    )


# Test the model
if __name__ == "__main__":
    print("Testing ResNet-18...")
    model = create_resnet18_1d(num_classes=NUM_CLASSES)
    x = torch.randn(2, 1, SIGNAL_LENGTH)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11), f"Expected (2, 11), got {y.shape}"

    print("\nTesting ResNet-34...")
    model = create_resnet34_1d(num_classes=NUM_CLASSES)
    y = model(x)
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11)

    print("\nTesting ResNet-50...")
    model = create_resnet50_1d(num_classes=NUM_CLASSES)
    y = model(x)
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11)

    print("\n✓ All tests passed!")
