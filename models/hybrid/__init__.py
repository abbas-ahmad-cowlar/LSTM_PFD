"""
Hybrid architectures combining multiple paradigms.

This package provides hybrid models that combine:
- CNN + LSTM: Convolutional features with temporal modeling
- CNN + TCN: Convolutional features with temporal convolutional networks
- Multi-scale CNN: Parallel processing at multiple scales
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .cnn_lstm import CNNLSTM, create_cnn_lstm
from .cnn_tcn import CNNTCN, create_cnn_tcn
from .multiscale_cnn import MultiScaleCNN, create_multiscale_cnn

__all__ = [
    'CNNLSTM',
    'create_cnn_lstm',
    'CNNTCN',
    'create_cnn_tcn',
    'MultiScaleCNN',
    'create_multiscale_cnn',
]
