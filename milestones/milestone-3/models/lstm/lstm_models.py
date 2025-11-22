"""
LSTM Models for Bearing Fault Diagnosis

This module implements LSTM-based architectures for bearing fault diagnosis
using raw vibration signals. LSTMs are effective for capturing temporal
dependencies and long-term patterns in sequential data.

Author: Bearing Fault Diagnosis Team
Milestone: 2 - LSTM Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class BaseLSTMModel(nn.Module):
    """
    Base class for LSTM models with common functionality.
    """

    def __init__(self):
        super().__init__()

    def _initialize_weights(self):
        """Initialize LSTM and linear layer weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases: zeros, except forget gate bias = 1
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
            elif 'weight' in name and 'fc' in name:
                # Fully connected weights: Xavier
                nn.init.xavier_uniform_(param.data)


class VanillaLSTM(BaseLSTMModel):
    """
    Vanilla LSTM for bearing fault diagnosis.

    Architecture:
        Input: Raw vibration signal [B, 1, 102400]
          ↓
        Reshape to sequence: [B, seq_len, feature_dim]
          ↓
        LSTM layers (unidirectional)
          ↓
        Global average pooling over time
          ↓
        Dropout + Fully connected → num_classes

    Args:
        num_classes: Number of output classes (default: 11)
        input_size: Input feature dimension per time step (default: 1)
        hidden_size: LSTM hidden dimension (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.3)
        seq_len: Sequence length (default: 102400)

    Example:
        >>> model = VanillaLSTM(num_classes=11, hidden_size=128, num_layers=2)
        >>> x = torch.randn(32, 1, 102400)  # Batch of 32 signals
        >>> output = model(x)  # [32, 11]
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = SIGNAL_LENGTH
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.seq_len = seq_len

        # LSTM layer (unidirectional)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, L] where:
               - B: batch size
               - C: channels (1 for vibration signal)
               - L: signal length (102400)

        Returns:
            logits: Class logits [B, num_classes]
        """
        batch_size = x.size(0)

        # Reshape: [B, 1, L] → [B, L, 1]
        # This treats each time step as a 1D feature
        x = x.transpose(1, 2)  # [B, L, 1]

        # LSTM forward pass
        # lstm_out: [B, L, hidden_size]
        # h_n: [num_layers, B, hidden_size]
        # c_n: [num_layers, B, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Global average pooling over sequence length
        # Take mean across time dimension
        pooled = torch.mean(lstm_out, dim=1)  # [B, hidden_size]

        # Apply dropout
        pooled = self.dropout(pooled)

        # Classifier
        logits = self.fc(pooled)  # [B, num_classes]

        return logits

    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'VanillaLSTM',
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_p,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)
        }


class BiLSTM(BaseLSTMModel):
    """
    Bidirectional LSTM for bearing fault diagnosis.

    BiLSTM processes the sequence in both forward and backward directions,
    allowing it to capture context from both past and future time steps.
    This is particularly useful for offline fault diagnosis where the entire
    signal is available.

    Architecture:
        Input: Raw vibration signal [B, 1, 102400]
          ↓
        Reshape to sequence: [B, seq_len, feature_dim]
          ↓
        Bidirectional LSTM layers
          ↓
        Global average pooling over time
          ↓
        Dropout + Fully connected → num_classes

    Args:
        num_classes: Number of output classes (default: 11)
        input_size: Input feature dimension per time step (default: 1)
        hidden_size: LSTM hidden dimension per direction (default: 128)
        num_layers: Number of BiLSTM layers (default: 2)
        dropout: Dropout probability (default: 0.3)
        seq_len: Sequence length (default: 102400)
        pooling: Pooling method ('mean', 'max', 'last') (default: 'mean')

    Note:
        - Total output dimension is hidden_size * 2 (forward + backward)
        - More parameters than vanilla LSTM due to bidirectional processing

    Example:
        >>> model = BiLSTM(num_classes=11, hidden_size=128, num_layers=2)
        >>> x = torch.randn(32, 1, 102400)
        >>> output = model(x)  # [32, 11]
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = SIGNAL_LENGTH,
        pooling: str = 'mean'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.seq_len = seq_len
        self.pooling = pooling

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Key difference from Vanilla LSTM
        )

        # Output dimension is hidden_size * 2 (forward + backward)
        lstm_output_dim = hidden_size * 2

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, L] where:
               - B: batch size
               - C: channels (1 for vibration signal)
               - L: signal length (102400)

        Returns:
            logits: Class logits [B, num_classes]
        """
        batch_size = x.size(0)

        # Reshape: [B, 1, L] → [B, L, 1]
        x = x.transpose(1, 2)  # [B, L, 1]

        # BiLSTM forward pass
        # lstm_out: [B, L, hidden_size * 2]
        # h_n: [num_layers * 2, B, hidden_size]  # 2 for bidirectional
        # c_n: [num_layers * 2, B, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Pooling over sequence length
        if self.pooling == 'mean':
            # Global average pooling
            pooled = torch.mean(lstm_out, dim=1)  # [B, hidden_size * 2]
        elif self.pooling == 'max':
            # Global max pooling
            pooled = torch.max(lstm_out, dim=1)[0]  # [B, hidden_size * 2]
        elif self.pooling == 'last':
            # Take last time step output
            pooled = lstm_out[:, -1, :]  # [B, hidden_size * 2]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Apply dropout
        pooled = self.dropout(pooled)

        # Classifier
        logits = self.fc(pooled)  # [B, num_classes]

        return logits

    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'BiLSTM',
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_p,
            'pooling': self.pooling,
            'bidirectional': True,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)
        }


def create_lstm_model(
    model_type: str = 'vanilla',
    num_classes: int = NUM_CLASSES,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create LSTM models.

    Args:
        model_type: Type of LSTM ('vanilla' or 'bilstm')
        num_classes: Number of output classes
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        **kwargs: Additional model-specific arguments

    Returns:
        LSTM model instance

    Example:
        >>> model = create_lstm_model('bilstm', num_classes=11, hidden_size=256)
    """
    model_map = {
        'vanilla': VanillaLSTM,
        'vanilla_lstm': VanillaLSTM,
        'lstm': VanillaLSTM,
        'bilstm': BiLSTM,
        'bi_lstm': BiLSTM,
        'bidirectional_lstm': BiLSTM,
    }

    model_type_lower = model_type.lower()

    if model_type_lower not in model_map:
        available = ', '.join(model_map.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {available}"
        )

    model_class = model_map[model_type_lower]

    return model_class(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs
    )
