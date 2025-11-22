"""
LSTM Models for Bearing Fault Diagnosis

This package provides LSTM architectures for time-series fault classification:
- VanillaLSTM: Unidirectional LSTM
- BiLSTM: Bidirectional LSTM for better context modeling

Author: Bearing Fault Diagnosis Team
"""

from .lstm_models import (
    VanillaLSTM,
    BiLSTM,
    create_lstm_model
)

__all__ = [
    'VanillaLSTM',
    'BiLSTM',
    'create_lstm_model',
]
