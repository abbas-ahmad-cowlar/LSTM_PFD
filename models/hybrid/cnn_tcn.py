"""
CNN-TCN Hybrid Architecture

Combines CNN feature extraction with Temporal Convolutional Network (TCN).

TCN advantages over LSTM:
- Parallelizable (no sequential dependency)
- Larger receptive field with dilated convolutions
- Faster training and inference
- More stable gradients

Architecture:
    Input [B, 1, 102400]
      ↓
    CNN Backbone → [B, 512, 800]
      ↓
    TCN (dilated causal convolutions) → [B, 512, 800]
      ↓
    Global Average Pooling → [B, 512]
      ↓
    FC → [B, 11]

Reference:
- Bai et al. (2018). "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_model import BaseModel


class CausalConv1d(nn.Module):
    """
    Causal convolution: Output at time t depends only on inputs up to time t.

    Ensures no information leakage from future to past.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        dilation: Dilation factor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()

        # Padding to ensure causality
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal padding."""
        x = self.conv(x)

        # Remove right padding to maintain causality
        if self.padding > 0:
            x = x[:, :, :-self.padding]

        return x


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block.

    Structure:
        x → CausalConv → ReLU → Dropout → CausalConv → ReLU → Dropout → (+) → ReLU
        |__________________________________________________________________|
                                (residual connection)

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        dilation: Dilation factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        # Two causal convolutions with dilations
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        # First causal conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second causal conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network.

    Stack of TCN blocks with exponentially increasing dilation.

    Args:
        num_channels: List of channels for each TCN layer
        kernel_size: Kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_channels = num_channels[i - 1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]

            layers.append(
                TCNBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN."""
        return self.network(x)


class CNNTCN(BaseModel):
    """
    CNN-TCN hybrid architecture.

    Combines:
    - CNN backbone for feature extraction
    - TCN for temporal modeling with large receptive field

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        cnn_backbone: CNN backbone type
        tcn_channels: List of TCN layer channels
        tcn_kernel_size: TCN kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 1,
        cnn_backbone: str = 'simple',
        tcn_channels: Optional[List[int]] = None,
        tcn_kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_p = dropout

        # CNN backbone
        self.cnn_backbone = self._build_cnn_backbone(cnn_backbone, input_channels)
        self.cnn_output_channels = 512

        # TCN
        if tcn_channels is None:
            # Default: 4 layers with constant channels
            tcn_channels = [512, 512, 512, 512]

        # Ensure first TCN channel matches CNN output
        tcn_channels[0] = self.cnn_output_channels

        self.tcn = TemporalConvNet(
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(tcn_channels[-1], num_classes)

        self._initialize_weights()

    def _build_cnn_backbone(self, backbone: str, input_channels: int) -> nn.Module:
        """Build CNN backbone."""
        if backbone == 'simple':
            return nn.Sequential(
                # Block 1
                nn.Conv1d(input_channels, 64, kernel_size=64, stride=4, padding=32),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=4, stride=4),

                # Block 2
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                # Block 3
                nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),

                # Block 4
                nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # CNN feature extraction
        x = self.cnn_backbone(x)

        # TCN temporal modeling
        x = self.tcn(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'CNNTCN',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'num_parameters': self.get_num_params(),
            'dropout': self.dropout_p
        }


def create_cnn_tcn(num_classes: int = 11, **kwargs) -> CNNTCN:
    """
    Factory function to create CNN-TCN model.

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments

    Returns:
        CNNTCN model instance
    """
    return CNNTCN(num_classes=num_classes, **kwargs)


# Test
if __name__ == "__main__":
    print("Testing CNN-TCN...")

    model = create_cnn_tcn(num_classes=11)
    x = torch.randn(2, 1, 102400)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11)

    print("\n✓ CNN-TCN tests passed!")
