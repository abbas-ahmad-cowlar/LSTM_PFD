"""
Hybrid Models combining CNN and Transformer architectures

Provides hybrid models that combine the strengths of CNNs (local pattern extraction)
with Transformers (long-range dependency modeling).
"""

from .cnn_transformer import (
    CNNTransformerHybrid,
    create_cnn_transformer_hybrid
)

__all__ = [
    'CNNTransformerHybrid',
    'create_cnn_transformer_hybrid',
]
