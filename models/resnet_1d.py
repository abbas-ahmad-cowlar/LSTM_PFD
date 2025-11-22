"""
ResNet-18 Adapted for 1D Signals

Implements ResNet architecture for time-series bearing fault diagnosis.
Uses residual connections to enable deeper networks and better gradient flow.

Reference:
- He et al. (2016). "Deep Residual Learning for Image Recognition"
- Adapted for 1D signals with appropriate kernel sizes

Input: [B, 1, T] where T is signal length
Output: [B, 11] for 11 fault classes
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type
from .base_model import BaseModel


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
        stride: Stride for first convolution (for downsampling)
        downsample: Downsample layer for skip connection (if needed)
        dropout: Dropout probability
    """
    expansion = 1  # Output channels multiplier

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        # First conv layer (may downsample)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second conv layer
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply dropout before adding skip connection
        if self.dropout is not None:
            out = self.dropout(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(BaseModel):
    """
    ResNet-18 architecture adapted for 1D signal classification.

    Architecture:
        - Initial conv: 1 -> 64 channels
        - Layer1: 2 blocks, 64 channels
        - Layer2: 2 blocks, 128 channels (stride 2)
        - Layer3: 2 blocks, 256 channels (stride 2)
        - Layer4: 2 blocks, 512 channels (stride 2)
        - Global Average Pooling
        - Fully Connected: 512 -> num_classes

    Args:
        num_classes: Number of output classes (default: 11)
        input_channels: Number of input channels (default: 1)
        dropout: Dropout probability (default: 0.2)
        layers: Number of blocks in each layer (default: [2, 2, 2, 2] for ResNet-18)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        dropout: float = 0.2,
        layers: List[int] = None
    ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]  # ResNet-18 configuration

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.in_channels = 64
        self.dropout = dropout

        # Initial convolution
        self.conv1 = nn.Conv1d(
            input_channels,
            64,
            kernel_size=15,
            stride=2,
            padding=7,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(BasicBlock1D, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, layers[3], stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock1D],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a residual layer with multiple blocks.

        Args:
            block: Block class to use
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for first block (for downsampling)

        Returns:
            Sequential layer containing blocks
        """
        downsample = None

        # If stride != 1 or channels change, need downsample layer
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels * block.expansion)
            )

        layers = []

        # First block (may downsample)
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
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
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, T]

        Returns:
            logits: Output tensor of shape [B, num_classes]
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        # Initial convolution
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
        x = self.avgpool(x)  # [B, 512, 1]
        x = torch.flatten(x, 1)  # [B, 512]

        # Classification
        logits = self.fc(x)

        return logits

    def get_feature_extractor(self) -> nn.Module:
        """
        Return the feature extraction backbone (all layers except fc).

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

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'ResNet1D',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'num_parameters': self.get_num_params()
        }


def create_resnet18_1d(num_classes: int = NUM_CLASSES, **kwargs) -> ResNet1D:
    """
    Factory function to create ResNet-18 for 1D signals.

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to ResNet1D

    Returns:
        ResNet1D model instance
    """
    return ResNet1D(num_classes=num_classes, layers=[2, 2, 2, 2], **kwargs)


def create_resnet34_1d(num_classes: int = NUM_CLASSES, **kwargs) -> ResNet1D:
    """
    Factory function to create ResNet-34 for 1D signals.

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to ResNet1D

    Returns:
        ResNet1D model instance
    """
    return ResNet1D(num_classes=num_classes, layers=[3, 4, 6, 3], **kwargs)
