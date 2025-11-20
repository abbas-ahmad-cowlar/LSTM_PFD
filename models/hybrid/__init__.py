"""
Hybrid architectures combining multiple paradigms.

This package provides hybrid models that combine:
- CNN + LSTM: Convolutional features with temporal modeling
- CNN + TCN: Convolutional features with temporal convolutional networks
- Multi-scale CNN: Parallel processing at multiple scales
"""

from .cnn_lstm import CNNLSTM, create_cnn_lstm

__all__ = [
    'CNNLSTM',
    'create_cnn_lstm',
]
