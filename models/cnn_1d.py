"""
1D CNN for Raw Signal Classification

Architecture:
- 6 convolutional layers with increasing channels
- Batch normalization and dropout for regularization
- Adaptive pooling for variable input lengths
- 2 fully connected layers for classification

Input: [B, 1, T] where T is signal length
Output: [B, 11] for 11 fault classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .base_model import BaseModel


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv-BN-ReLU-Dropout pattern.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Stride for convolution
        padding: Padding for convolution
        dropout: Dropout probability
        use_bn: Whether to use batch normalization
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.2,
        use_bn: bool = True
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn
        )
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CNN1D(BaseModel):
    """
    1D CNN for bearing fault classification.

    Architecture:
        - Conv1: 1 -> 32 channels
        - Conv2: 32 -> 64 channels
        - MaxPool
        - Conv3: 64 -> 128 channels
        - Conv4: 128 -> 128 channels
        - MaxPool
        - Conv5: 128 -> 256 channels
        - Conv6: 256 -> 256 channels
        - Adaptive Average Pooling
        - FC1: 256 -> 128
        - FC2: 128 -> num_classes

    Args:
        num_classes: Number of output classes (default: 11)
        input_channels: Number of input channels (default: 1)
        dropout: Dropout probability (default: 0.3)
        use_bn: Whether to use batch normalization (default: True)
    """
    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 1,
        dropout: float = 0.3,
        use_bn: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels

        # Feature extraction layers
        self.conv1 = ConvBlock(input_channels, 32, kernel_size=7, padding=3, dropout=dropout, use_bn=use_bn)
        self.conv2 = ConvBlock(32, 64, kernel_size=5, padding=2, dropout=dropout, use_bn=use_bn)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1, dropout=dropout, use_bn=use_bn)
        self.conv4 = ConvBlock(128, 128, kernel_size=3, padding=1, dropout=dropout, use_bn=use_bn)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(128, 256, kernel_size=3, padding=1, dropout=dropout, use_bn=use_bn)
        self.conv6 = ConvBlock(256, 256, kernel_size=3, padding=1, dropout=dropout, use_bn=use_bn)

        # Adaptive pooling for variable input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.fc1 = nn.Linear(256, 128)
        self.fc1_bn = nn.BatchNorm1d(128) if use_bn else nn.Identity()
        self.fc1_dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
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

        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)

        # Global pooling
        x = self.adaptive_pool(x)  # [B, 256, 1]
        x = x.squeeze(-1)  # [B, 256]

        # Classification
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc1_dropout(x)

        logits = self.fc2(x)

        return logits

    def get_feature_extractor(self) -> nn.Module:
        """
        Return the feature extraction backbone (all conv layers).

        Returns:
            Feature extractor module
        """
        return nn.Sequential(
            self.conv1,
            self.conv2,
            self.pool1,
            self.conv3,
            self.conv4,
            self.pool2,
            self.conv5,
            self.conv6,
            self.adaptive_pool
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
            'model_type': 'CNN1D',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'num_parameters': self.get_num_params()
        }


def create_cnn1d(num_classes: int = 11, **kwargs) -> CNN1D:
    """
    Factory function to create CNN1D model.

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to CNN1D

    Returns:
        CNN1D model instance
    """
    return CNN1D(num_classes=num_classes, **kwargs)
