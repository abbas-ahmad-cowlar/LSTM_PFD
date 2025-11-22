"""
Models for CNN-LSTM Hybrid Bearing Fault Diagnosis

This package provides configurable hybrid architectures combining:
- CNN models (from Milestone 1): Spatial feature extraction
- LSTM models (from Milestone 2): Temporal dependency modeling

Available CNN backbones:
- cnn1d: Basic 1D CNN
- resnet18, resnet34, resnet50: ResNet variants
- efficientnet_b0, efficientnet_b2, efficientnet_b4: EfficientNet

Available LSTM types:
- lstm: Vanilla LSTM (unidirectional)
- bilstm: Bidirectional LSTM

Author: Bearing Fault Diagnosis Team
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .hybrid import (
    HybridCNNLSTM,
    create_hybrid_model,
    create_recommended_hybrid_1,
    create_recommended_hybrid_2,
    create_recommended_hybrid_3,
)

__all__ = [
    'HybridCNNLSTM',
    'create_hybrid_model',
    'create_recommended_hybrid_1',
    'create_recommended_hybrid_2',
    'create_recommended_hybrid_3',
    'create_model',
    'list_available_models',
    'list_available_cnn_backbones',
    'list_available_lstm_types',
]


def create_model(
    model_name: str = 'recommended_1',
    num_classes: int = NUM_CLASSES,
    **kwargs
):
    """
    Factory function to create hybrid models by name.

    Args:
        model_name: Name of model configuration
            - 'recommended_1': ResNet34 + BiLSTM
            - 'recommended_2': EfficientNet-B2 + BiLSTM
            - 'recommended_3': ResNet18 + LSTM
            - 'custom': Create custom hybrid (requires cnn_type and lstm_type)
        num_classes: Number of output classes (default: 11)
        **kwargs: Additional model-specific arguments
            - cnn_type: CNN backbone type (for custom)
            - lstm_type: LSTM type (for custom)
            - lstm_hidden_size: LSTM hidden size
            - lstm_num_layers: Number of LSTM layers
            - pooling_method: Temporal pooling method
            - freeze_cnn: Whether to freeze CNN weights

    Returns:
        Initialized hybrid model

    Examples:
        >>> # Use recommended configuration
        >>> model = create_model('recommended_1', num_classes=11)

        >>> # Create custom hybrid
        >>> model = create_model(
        ...     'custom',
        ...     cnn_type='resnet34',
        ...     lstm_type='bilstm',
        ...     lstm_hidden_size=256
        ... )
    """
    model_map = {
        'recommended_1': create_recommended_hybrid_1,
        'recommended_hybrid_1': create_recommended_hybrid_1,
        'resnet34_bilstm': create_recommended_hybrid_1,

        'recommended_2': create_recommended_hybrid_2,
        'recommended_hybrid_2': create_recommended_hybrid_2,
        'efficientnet_bilstm': create_recommended_hybrid_2,

        'recommended_3': create_recommended_hybrid_3,
        'recommended_hybrid_3': create_recommended_hybrid_3,
        'resnet18_lstm': create_recommended_hybrid_3,
    }

    model_name_lower = model_name.lower()

    if model_name_lower in model_map:
        return model_map[model_name_lower](num_classes=num_classes, **kwargs)
    elif model_name_lower == 'custom':
        # Custom hybrid - requires cnn_type and lstm_type
        if 'cnn_type' not in kwargs or 'lstm_type' not in kwargs:
            raise ValueError("Custom hybrid requires 'cnn_type' and 'lstm_type' parameters")
        return create_hybrid_model(num_classes=num_classes, **kwargs)
    else:
        available = ', '.join(model_map.keys())
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {available}, or use 'custom' with cnn_type and lstm_type"
        )


def list_available_models():
    """List all available hybrid model configurations."""
    return [
        'recommended_1',  # ResNet34 + BiLSTM
        'recommended_2',  # EfficientNet-B2 + BiLSTM
        'recommended_3',  # ResNet18 + LSTM
        'custom',  # Custom combination (requires cnn_type and lstm_type)
    ]


def list_available_cnn_backbones():
    """List all available CNN backbones for custom hybrids."""
    return [
        'cnn1d',
        'resnet18',
        'resnet34',
        'resnet50',
        'efficientnet_b0',
        'efficientnet_b2',
        'efficientnet_b4',
    ]


def list_available_lstm_types():
    """List all available LSTM types for custom hybrids."""
    return [
        'lstm',  # Vanilla LSTM (unidirectional)
        'bilstm',  # Bidirectional LSTM
    ]
