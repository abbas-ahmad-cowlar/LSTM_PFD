"""
CNN-LSTM Hybrid Architecture

Combines CNN feature extraction with LSTM temporal modeling for bearing fault diagnosis.

Architecture:
    Input [B, 1, 102400]
      ↓
    CNN Backbone (ResNet-18) → [B, 512, 800]  # Feature maps
      ↓
    Permute: [B, 512, 800] → [B, 800, 512]  # Time steps, features
      ↓
    BiLSTM: 512 → 256 hidden → [B, 800, 512]  # 256×2 = 512 (bidirectional)
      ↓
    Attention Pooling: [B, 800, 512] → [B, 512]  # Weighted average over time
      ↓
    FC: 512 → 11

Rationale:
- CNN extracts local patterns and hierarchical features
- LSTM captures long-term temporal dependencies CNN might miss
- Attention focuses on most important time steps

Reference:
- Zhao et al. (2017). "Deep Learning and Its Applications to Machine Health Monitoring"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.base_model import BaseModel


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over time dimension.

    Learns which time steps are most important for classification.

    Args:
        input_dim: Dimension of input features
    """

    def __init__(self, input_dim: int):
        super().__init__()

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention pooling.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            pooled: Pooled features [B, D]
            attention_weights: Attention weights [B, T]
        """
        # Compute attention scores
        attention_scores = self.attention(x)  # [B, T, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, T, 1]

        # Weighted sum
        pooled = torch.sum(x * attention_weights, dim=1)  # [B, D]

        return pooled, attention_weights.squeeze(-1)


class CNNLSTM(BaseModel):
    """
    CNN-LSTM hybrid architecture for bearing fault diagnosis.

    Combines:
    - CNN backbone for feature extraction
    - Bidirectional LSTM for temporal modeling
    - Attention pooling for classification

    Args:
        num_classes: Number of output classes (default: 11)
        input_channels: Number of input channels (default: 1)
        cnn_backbone: CNN backbone to use ('resnet18', 'resnet34', 'simple')
        lstm_hidden: LSTM hidden size (default: 256)
        lstm_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        bidirectional: Use bidirectional LSTM (default: True)
        use_attention: Use attention pooling (default: True)
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 1,
        cnn_backbone: str = 'resnet18',
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.dropout_p = dropout

        # Build CNN backbone
        self.cnn_backbone = self._build_cnn_backbone(cnn_backbone, input_channels)
        self.cnn_output_channels = self._get_cnn_output_channels(cnn_backbone)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate LSTM output dimension
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        # Attention pooling or simple pooling
        if use_attention:
            self.pooling = AttentionPooling(lstm_output_dim)
        else:
            self.pooling = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.fc = nn.Linear(lstm_output_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _build_cnn_backbone(self, backbone: str, input_channels: int) -> nn.Module:
        """
        Build CNN backbone for feature extraction.

        Args:
            backbone: Backbone architecture name
            input_channels: Number of input channels

        Returns:
            CNN backbone module
        """
        if backbone == 'resnet18':
            # Import ResNet-18 and use as feature extractor
            try:
                from resnet.resnet_1d import create_resnet18_1d
                resnet = create_resnet18_1d(num_classes=11, input_channels=input_channels)
                # Remove final FC layer and pooling - we'll use LSTM instead
                backbone_layers = [
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3,
                    resnet.layer4
                ]
                return nn.Sequential(*backbone_layers)
            except ImportError:
                print("Warning: ResNet not available, using simple CNN backbone")
                return self._build_simple_cnn(input_channels)

        elif backbone == 'resnet34':
            try:
                from resnet.resnet_1d import create_resnet34_1d
                resnet = create_resnet34_1d(num_classes=11, input_channels=input_channels)
                backbone_layers = [
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                    resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
                ]
                return nn.Sequential(*backbone_layers)
            except ImportError:
                return self._build_simple_cnn(input_channels)

        elif backbone == 'simple':
            return self._build_simple_cnn(input_channels)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _build_simple_cnn(self, input_channels: int) -> nn.Module:
        """
        Build simple CNN backbone if ResNet not available.

        Args:
            input_channels: Number of input channels

        Returns:
            Simple CNN module
        """
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

    def _get_cnn_output_channels(self, backbone: str) -> int:
        """Get number of output channels from CNN backbone."""
        if backbone in ['resnet18', 'resnet34', 'simple']:
            return 512
        else:
            return 512

    def _initialize_weights(self):
        """Initialize LSTM and FC weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, T] or [B, T]

        Returns:
            logits: Output tensor [B, num_classes]
        """
        # Ensure input is 3D [B, C, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)  # [B, 512, T']

        # Permute for LSTM: [B, C, T] -> [B, T, C]
        lstm_input = cnn_features.permute(0, 2, 1)  # [B, T', 512]

        # LSTM temporal modeling
        lstm_output, (h_n, c_n) = self.lstm(lstm_input)  # [B, T', lstm_output_dim]

        # Pooling
        if self.use_attention:
            pooled, attention_weights = self.pooling(lstm_output)  # [B, lstm_output_dim]
            self.last_attention_weights = attention_weights  # Store for visualization
        else:
            # Simple average pooling
            pooled = torch.mean(lstm_output, dim=1)  # [B, lstm_output_dim]

        # Dropout
        pooled = self.dropout(pooled)

        # Classification
        logits = self.fc(pooled)  # [B, num_classes]

        return logits

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from last forward pass.

        Returns:
            Attention weights [B, T] or None if attention not used
        """
        if self.use_attention and hasattr(self, 'last_attention_weights'):
            return self.last_attention_weights
        return None

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features without LSTM.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            CNN features [B, C', T']
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.cnn_backbone(x)

    def get_lstm_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN + LSTM features without classification.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            LSTM features [B, T', D]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        cnn_features = self.cnn_backbone(x)
        lstm_input = cnn_features.permute(0, 2, 1)
        lstm_output, _ = self.lstm(lstm_input)

        return lstm_output

    def freeze_cnn(self):
        """Freeze CNN backbone for fine-tuning."""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        """Unfreeze CNN backbone."""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = True

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'CNNLSTM',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'num_parameters': self.get_num_params()
        }


def create_cnn_lstm(
    num_classes: int = 11,
    backbone: str = 'resnet18',
    **kwargs
) -> CNNLSTM:
    """
    Factory function to create CNN-LSTM model.

    Args:
        num_classes: Number of output classes
        backbone: CNN backbone to use
        **kwargs: Additional arguments

    Returns:
        CNNLSTM model instance
    """
    return CNNLSTM(num_classes=num_classes, cnn_backbone=backbone, **kwargs)


# Test the model
if __name__ == "__main__":
    print("Testing CNN-LSTM...")

    # Test with simple backbone
    model = create_cnn_lstm(num_classes=11, backbone='simple')
    x = torch.randn(2, 1, 102400)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11), f"Expected (2, 11), got {y.shape}"

    # Test attention weights
    if model.use_attention:
        attention = model.get_attention_weights()
        print(f"Attention weights shape: {attention.shape}")

    # Test feature extraction
    cnn_features = model.get_cnn_features(x)
    print(f"CNN features: {cnn_features.shape}")

    lstm_features = model.get_lstm_features(x)
    print(f"LSTM features: {lstm_features.shape}")

    print("\n✓ CNN-LSTM tests passed!")
