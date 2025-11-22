"""
1D Convolutional Neural Network for bearing fault diagnosis.

Purpose:
    End-to-end learning from raw vibration signals (102,400 samples).
    Bypasses manual feature engineering through hierarchical feature learning.

    Architecture: 5 conv blocks → Global pooling → 2 FC layers
    Target: 93-96% test accuracy (match classical ML baseline)

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from models.base_model import BaseModel
from models.cnn.conv_blocks import ConvBlock1D
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class CNN1D(BaseModel):
    """
    1D Convolutional Neural Network for raw signal classification.

    Architecture:
        Input: [B, 1, SIGNAL_LENGTH]  # 5 sec @ SAMPLING_RATE Hz
        ├─ ConvBlock1: Conv1D(1→32, k=64, s=4) + Pool → [B, 32, 25600]
        ├─ ConvBlock2: Conv1D(32→64, k=32, s=2) + Pool → [B, 64, 12800]
        ├─ ConvBlock3: Conv1D(64→128, k=16, s=2) + Pool → [B, 128, 6400]
        ├─ ConvBlock4: Conv1D(128→256, k=8, s=2) + Pool → [B, 256, 3200]
        ├─ ConvBlock5: Conv1D(256→512, k=4, s=2) + Pool → [B, 512, 1600]
        ├─ GlobalAvgPool → [B, 512]
        ├─ FC1: 512 → 256, ReLU, Dropout(0.5)
        └─ FC2: 256 → NUM_CLASSES (fault types)

    Parameters: ~1.2M (lightweight for real-time deployment)

    Args:
        num_classes: Number of fault classes (default: NUM_CLASSES from constants)
        input_channels: Number of input channels (default: 1 for mono signal)
        dropout: Dropout probability in FC layers (default: 0.5)
        use_batch_norm: Apply batch normalization (default: True)

    Example:
        >>> model = CNN1D(num_classes=NUM_CLASSES)
        >>> x = torch.randn(8, 1, SIGNAL_LENGTH)  # Batch of 8 signals
        >>> logits = model(x)  # Shape: [8, NUM_CLASSES]
        >>> print(model.count_parameters())
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        super(CNN1D, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Convolutional feature extractor
        # Progressive downsampling: SIGNAL_LENGTH → 25600 → 12800 → 6400 → 3200 → 1600
        self.conv_block1 = ConvBlock1D(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=64,
            stride=4,  # Aggressive downsampling in first layer
            dropout=0.2,
            use_batch_norm=use_batch_norm
        )

        self.conv_block2 = ConvBlock1D(
            in_channels=32,
            out_channels=64,
            kernel_size=32,
            stride=2,
            dropout=0.2,
            use_batch_norm=use_batch_norm
        )

        self.conv_block3 = ConvBlock1D(
            in_channels=64,
            out_channels=128,
            kernel_size=16,
            stride=2,
            dropout=0.3,
            use_batch_norm=use_batch_norm
        )

        self.conv_block4 = ConvBlock1D(
            in_channels=128,
            out_channels=256,
            kernel_size=8,
            stride=2,
            dropout=0.3,
            use_batch_norm=use_batch_norm
        )

        self.conv_block5 = ConvBlock1D(
            in_channels=256,
            out_channels=512,
            kernel_size=4,
            stride=2,
            dropout=0.4,
            use_batch_norm=use_batch_norm
        )

        # Global average pooling (replaces flatten, reduces overfitting)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected classifier
        self.fc1 = nn.Linear(512, 256)
        self.fc1_activation = nn.ReLU(inplace=True)
        self.fc1_dropout = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(256, num_classes)

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
            x: Input tensor [B, 1, SIGNAL_LENGTH] or [B, SIGNAL_LENGTH]

        Returns:
            Logits [B, num_classes]
        """
        # Handle 2D input (B, T) by adding channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] → [B, 1, T]

        # Convolutional feature extraction
        x = self.conv_block1(x)  # [B, 1, SIGNAL_LENGTH] → [B, 32, 25600]
        x = self.conv_block2(x)  # [B, 32, 25600] → [B, 64, 12800]
        x = self.conv_block3(x)  # [B, 64, 12800] → [B, 128, 6400]
        x = self.conv_block4(x)  # [B, 128, 6400] → [B, 256, 3200]
        x = self.conv_block5(x)  # [B, 256, 3200] → [B, 512, 1600]

        # Global pooling
        x = self.global_pool(x)  # [B, 512, 1600] → [B, 512, 1]
        x = x.squeeze(-1)  # [B, 512, 1] → [B, 512]

        # Fully connected classifier
        x = self.fc1(x)  # [B, 512] → [B, 256]
        x = self.fc1_activation(x)
        x = self.fc1_dropout(x)

        x = self.fc2(x)  # [B, 256] → [B, num_classes]

        return x

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """
        Extract features at a specific intermediate layer.

        Useful for visualization, transfer learning, and interpretability.

        Args:
            x: Input tensor [B, 1, SIGNAL_LENGTH]
            layer_name: Name of layer ('conv1', 'conv2', ..., 'conv5', 'fc1')

        Returns:
            Features at specified layer

        Example:
            >>> model = CNN1D(num_classes=NUM_CLASSES)
            >>> x = torch.randn(8, 1, SIGNAL_LENGTH)
            >>> conv3_features = model.get_intermediate_features(x, 'conv3')
            >>> print(conv3_features.shape)  # [8, 128, 6400]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Forward through layers until reaching target
        x = self.conv_block1(x)
        if layer_name == 'conv1':
            return x

        x = self.conv_block2(x)
        if layer_name == 'conv2':
            return x

        x = self.conv_block3(x)
        if layer_name == 'conv3':
            return x

        x = self.conv_block4(x)
        if layer_name == 'conv4':
            return x

        x = self.conv_block5(x)
        if layer_name == 'conv5':
            return x

        x = self.global_pool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.fc1_activation(x)
        if layer_name == 'fc1':
            return x

        raise ValueError(f"Unknown layer name: {layer_name}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'model_type': 'CNN1D',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm,
            'architecture': '5-layer CNN with global pooling'
        }

    def get_layer_names(self) -> list:
        """
        Get list of named layers for feature extraction.

        Returns:
            List of layer names
        """
        return ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1']


def test_cnn_1d():
    """Test CNN1D architecture with sample inputs."""
    print("=" * 60)
    print("Testing CNN1D Architecture")
    print("=" * 60)

    # Create model
    model = CNN1D(num_classes=NUM_CLASSES)
    print(f"\nModel created: {model.get_config()}")

    # Test forward pass
    print("\n1. Testing forward pass...")
    batch_size = 8
    x = torch.randn(batch_size, 1, SIGNAL_LENGTH)
    logits = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    assert logits.shape == (batch_size, NUM_CLASSES), f"Expected [8, {NUM_CLASSES}], got {logits.shape}"

    # Test with 2D input
    print("\n2. Testing with 2D input (auto-unsqueeze)...")
    x_2d = torch.randn(batch_size, SIGNAL_LENGTH)
    logits_2d = model(x_2d)
    print(f"   Input shape: {x_2d.shape}")
    print(f"   Output shape: {logits_2d.shape}")
    assert logits_2d.shape == (batch_size, NUM_CLASSES)

    # Test intermediate features
    print("\n3. Testing intermediate feature extraction...")
    for layer_name in model.get_layer_names():
        features = model.get_intermediate_features(x, layer_name)
        print(f"   {layer_name}: {features.shape}")

    # Count parameters
    print("\n4. Model statistics...")
    param_counts = model.count_parameters()
    print(f"   Total parameters: {param_counts['total']:,}")
    print(f"   Trainable parameters: {param_counts['trainable']:,}")
    print(f"   Non-trainable parameters: {param_counts['non_trainable']:,}")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    has_gradients = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"   All parameters have gradients: {has_gradients}")
    assert has_gradients, "Gradient flow broken!"

    print("\n" + "=" * 60)
    print("✅ All CNN1D tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cnn_1d()
