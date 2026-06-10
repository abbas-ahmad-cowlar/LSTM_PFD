"""
Transformer Models for 1D Signal Classification

- PatchTST: Patch Time Series Transformer (Nie et al., 2023) — Tier 1
- SignalTransformer: standard transformer with exposed attention — Tier 2
"""

from .signal_transformer import (
    SignalTransformer,
    create_transformer
)
from .patchtst import PatchTST

__all__ = [
    'SignalTransformer',
    'create_transformer',
    'PatchTST',
]
