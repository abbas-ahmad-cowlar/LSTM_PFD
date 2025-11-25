"""
Configurable CNN-LSTM Hybrid Architecture

Flexible hybrid model that can combine ANY CNN with ANY LSTM for bearing fault diagnosis.

Architecture:
    Raw Signal [B, 1, 102400]
        ↓
    CNN Backbone (configurable: CNN1D, ResNet, EfficientNet, etc.)
        → Extracts spatial/frequency features
        → Output: [B, features, time_steps]
        ↓
    Reshape for LSTM: [B, time_steps, features]
        ↓
    LSTM (configurable: Vanilla LSTM, BiLSTM)
        → Models temporal dependencies
        → Output: [B, time_steps, lstm_hidden * directions]
        ↓
    Temporal Pooling (mean/max/last)
        ↓
    Classifier → [B, num_classes]

Author: Bearing Fault Diagnosis Team
Milestone: 3 - CNN-LSTM Hybrid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class HybridCNNLSTM(nn.Module):
    """
    Configurable CNN-LSTM Hybrid Model.

    This architecture combines the strengths of CNNs and LSTMs:
    - CNN: Extracts spatial/local patterns and hierarchical features
    - LSTM: Captures temporal dependencies and sequential patterns

    The model is fully configurable - you can combine ANY CNN backbone
    with ANY LSTM variant.

    Args:
        cnn_model: Pre-configured CNN model (from milestone-1)
        lstm_hidden_size: Hidden size for LSTM layers
        lstm_num_layers: Number of LSTM layers
        lstm_bidirectional: Whether to use bidirectional LSTM
        lstm_dropout: Dropout for LSTM
        pooling_method: Temporal pooling ('mean', 'max', 'last', 'attention')
        num_classes: Number of output classes
        freeze_cnn: Whether to freeze CNN weights during training

    Example:
        >>> from models.resnet import create_resnet34_1d
        >>> cnn = create_resnet34_1d(num_classes=11)
        >>> hybrid = HybridCNNLSTM(
        ...     cnn_model=cnn,
        ...     lstm_hidden_size=256,
        ...     lstm_bidirectional=True
        ... )
    """

    def __init__(
        self,
        cnn_model: nn.Module,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_dropout: float = 0.3,
        pooling_method: str = 'mean',
        num_classes: int = NUM_CLASSES,
        freeze_cnn: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.pooling_method = pooling_method
        self.freeze_cnn = freeze_cnn

        # CNN Backbone (feature extractor)
        self.cnn_backbone = self._prepare_cnn_backbone(cnn_model)

        # Freeze CNN if requested
        if freeze_cnn:
            for param in self.cnn_backbone.parameters():
                param.requires_grad = False

        # Get CNN output dimension
        self.cnn_output_dim = self._get_cnn_output_dim()

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=lstm_bidirectional
        )

        # Calculate LSTM output dimension
        lstm_output_dim = lstm_hidden_size * (2 if lstm_bidirectional else 1)

        # Attention pooling (if selected)
        if pooling_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 4),
                nn.Tanh(),
                nn.Linear(lstm_output_dim // 4, 1)
            )
        else:
            self.attention = None

        # Dropout
        self.dropout = nn.Dropout(lstm_dropout)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(lstm_dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )

        # Initialize weights
        self._initialize_lstm_weights()

    def _prepare_cnn_backbone(self, cnn_model: nn.Module) -> nn.Module:
        """
        Prepare CNN backbone by removing final classification layer.

        We want CNN to output feature maps, not class predictions.
        """
        # Remove the final FC layer (classifier)
        # Most CNN models have it as 'fc' or 'classifier'
        if hasattr(cnn_model, 'fc'):
            # For ResNet-style models
            cnn_output_dim = cnn_model.fc.in_features
            cnn_model.fc = nn.Identity()
        elif hasattr(cnn_model, 'classifier'):
            # For other models with 'classifier'
            if isinstance(cnn_model.classifier, nn.Sequential):
                cnn_output_dim = cnn_model.classifier[-1].in_features
                cnn_model.classifier = nn.Sequential(*list(cnn_model.classifier.children())[:-1])
            else:
                cnn_output_dim = cnn_model.classifier.in_features
                cnn_model.classifier = nn.Identity()

        return cnn_model

    def _get_cnn_output_dim(self) -> int:
        """
        Determine CNN output dimension by running a dummy forward pass.
        """
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, SIGNAL_LENGTH)
            cnn_output = self.cnn_backbone(dummy_input)

            # CNN output could be:
            # - [B, features] (after global pooling)
            # - [B, features, time_steps] (feature maps)

            if len(cnn_output.shape) == 2:
                # [B, features] - already pooled
                return cnn_output.shape[1]
            elif len(cnn_output.shape) == 3:
                # [B, features, time_steps] - feature maps
                return cnn_output.shape[1]  # Return feature dimension
            else:
                raise ValueError(f"Unexpected CNN output shape: {cnn_output.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 1, L] where L=102400

        Returns:
            logits: Class logits [B, num_classes]
        """
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)  # [B, features] or [B, features, time_steps]

        # Prepare for LSTM
        if len(cnn_features.shape) == 2:
            # [B, features] - expand time dimension
            cnn_features = cnn_features.unsqueeze(1)  # [B, 1, features]
        else:
            # [B, features, time_steps] - transpose for LSTM
            cnn_features = cnn_features.transpose(1, 2)  # [B, time_steps, features]

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        # lstm_out: [B, time_steps, lstm_hidden * directions]

        # Temporal pooling
        if self.pooling_method == 'mean':
            pooled = torch.mean(lstm_out, dim=1)  # [B, lstm_hidden * directions]
        elif self.pooling_method == 'max':
            pooled = torch.max(lstm_out, dim=1)[0]  # [B, lstm_hidden * directions]
        elif self.pooling_method == 'last':
            pooled = lstm_out[:, -1, :]  # [B, lstm_hidden * directions]
        elif self.pooling_method == 'attention':
            # Attention-based pooling
            attention_scores = self.attention(lstm_out)  # [B, time_steps, 1]
            attention_weights = F.softmax(attention_scores, dim=1)  # [B, time_steps, 1]
            pooled = torch.sum(lstm_out * attention_weights, dim=1)  # [B, lstm_hidden * directions]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Dropout
        pooled = self.dropout(pooled)

        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]

        return logits

    def _initialize_lstm_weights(self):
        """Initialize LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)  # Forget gate bias = 1

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in self.cnn_backbone.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())

        return {
            'model_name': 'HybridCNNLSTM',
            'cnn_backbone': self.cnn_backbone.__class__.__name__,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'lstm_bidirectional': self.lstm_bidirectional,
            'pooling_method': self.pooling_method,
            'freeze_cnn': self.freeze_cnn,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'cnn_params': cnn_params,
            'lstm_params': lstm_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)
        }


def create_hybrid_model(
    cnn_type: str = 'resnet34',
    lstm_type: str = 'bilstm',
    lstm_hidden_size: int = 256,
    lstm_num_layers: int = 2,
    pooling_method: str = 'mean',
    num_classes: int = NUM_CLASSES,
    freeze_cnn: bool = False,
    **kwargs
) -> HybridCNNLSTM:
    """
    Factory function to create configurable hybrid models.

    Args:
        cnn_type: Type of CNN backbone
            - 'cnn1d': Basic 1D CNN
            - 'resnet18', 'resnet34', 'resnet50': ResNet variants
            - 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4': EfficientNet
        lstm_type: Type of LSTM ('lstm' or 'bilstm')
        lstm_hidden_size: LSTM hidden dimension
        lstm_num_layers: Number of LSTM layers
        pooling_method: Temporal pooling method
        num_classes: Number of output classes
        freeze_cnn: Whether to freeze CNN weights
        **kwargs: Additional arguments

    Returns:
        Hybrid CNN-LSTM model

    Examples:
        >>> # ResNet34 + BiLSTM (recommended)
        >>> model = create_hybrid_model('resnet34', 'bilstm', lstm_hidden_size=256)

        >>> # EfficientNet + LSTM (efficient)
        >>> model = create_hybrid_model('efficientnet_b2', 'lstm', lstm_hidden_size=128)

        >>> # CNN1D + BiLSTM (lightweight)
        >>> model = create_hybrid_model('cnn1d', 'bilstm', lstm_hidden_size=128)
    """
    # Import CNN models (using relative imports)
    from ..cnn.cnn_1d import CNN1D
    from ..resnet.resnet_1d import create_resnet18_1d, create_resnet34_1d, create_resnet50_1d
    from ..efficientnet.efficientnet_1d import create_efficientnet_b0, create_efficientnet_b2, create_efficientnet_b4

    # Create CNN backbone
    cnn_map = {
        'cnn1d': lambda: CNN1D(num_classes=num_classes),
        'resnet18': lambda: create_resnet18_1d(num_classes=num_classes),
        'resnet34': lambda: create_resnet34_1d(num_classes=num_classes),
        'resnet50': lambda: create_resnet50_1d(num_classes=num_classes),
        'efficientnet_b0': lambda: create_efficientnet_b0(num_classes=num_classes),
        'efficientnet_b2': lambda: create_efficientnet_b2(num_classes=num_classes),
        'efficientnet_b4': lambda: create_efficientnet_b4(num_classes=num_classes),
    }

    if cnn_type.lower() not in cnn_map:
        available = ', '.join(cnn_map.keys())
        raise ValueError(f"Unknown CNN type: {cnn_type}. Available: {available}")

    cnn_model = cnn_map[cnn_type.lower()]()

    # Determine if bidirectional
    lstm_bidirectional = lstm_type.lower() in ['bilstm', 'bi_lstm', 'bidirectional_lstm']

    # Create hybrid model
    hybrid = HybridCNNLSTM(
        cnn_model=cnn_model,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_bidirectional=lstm_bidirectional,
        pooling_method=pooling_method,
        num_classes=num_classes,
        freeze_cnn=freeze_cnn,
        **kwargs
    )

    return hybrid


# Recommended configurations
def create_recommended_hybrid_1(**kwargs):
    """
    Recommended Configuration 1: ResNet34 + BiLSTM

    Best balance of accuracy and efficiency.
    Expected accuracy: 96-98%
    """
    return create_hybrid_model(
        cnn_type='resnet34',
        lstm_type='bilstm',
        lstm_hidden_size=256,
        lstm_num_layers=2,
        pooling_method='mean',
        **kwargs
    )


def create_recommended_hybrid_2(**kwargs):
    """
    Recommended Configuration 2: EfficientNet-B2 + BiLSTM

    Efficient and accurate.
    Expected accuracy: 96-98%
    """
    return create_hybrid_model(
        cnn_type='efficientnet_b2',
        lstm_type='bilstm',
        lstm_hidden_size=256,
        lstm_num_layers=2,
        pooling_method='attention',
        **kwargs
    )


def create_recommended_hybrid_3(**kwargs):
    """
    Recommended Configuration 3: ResNet18 + LSTM

    Faster training, good for quick experiments.
    Expected accuracy: 94-96%
    """
    return create_hybrid_model(
        cnn_type='resnet18',
        lstm_type='lstm',
        lstm_hidden_size=128,
        lstm_num_layers=2,
        pooling_method='mean',
        **kwargs
    )
