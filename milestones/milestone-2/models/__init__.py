"""
Models for LSTM-Based Bearing Fault Diagnosis

This package provides LSTM architectures for bearing fault diagnosis.

Available models:
- vanilla_lstm: Unidirectional LSTM
- bilstm: Bidirectional LSTM

Author: Bearing Fault Diagnosis Team
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .lstm import VanillaLSTM, BiLSTM, create_lstm_model

__all__ = [
    'VanillaLSTM',
    'BiLSTM',
    'create_lstm_model',
]


def create_model(model_name: str, num_classes: int = NUM_CLASSES, **kwargs):
    """
    Factory function to create LSTM models by name.

    Args:
        model_name: Name of the model ('vanilla_lstm', 'bilstm')
        num_classes: Number of output classes (default: 11)
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized LSTM model

    Example:
        >>> model = create_model('bilstm', num_classes=11, hidden_size=256)
    """
    return create_lstm_model(
        model_type=model_name,
        num_classes=num_classes,
        **kwargs
    )


def list_available_models():
    """List all available LSTM model architectures."""
    return [
        'vanilla_lstm',
        'lstm',
        'bilstm',
        'bi_lstm',
        'bidirectional_lstm',
    ]
