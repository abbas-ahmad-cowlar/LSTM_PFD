"""
Hybrid models.

- CNNLSTM: CNN feature extractor + LSTM temporal modeling (Tier 1 —
  the recurrent family member of the benchmark zoo).
"""

from .cnn_lstm import CNNLSTM, create_cnn_lstm

__all__ = [
    'CNNLSTM',
    'create_cnn_lstm',
]
