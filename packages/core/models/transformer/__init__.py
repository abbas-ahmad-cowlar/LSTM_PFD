"""
Transformer Models for 1D Signal Classification

This module provides transformer-based architectures for bearing fault diagnosis:
- SignalTransformer: Standard transformer with global average pooling
- VisionTransformer1D: ViT-style transformer with [CLS] token
- PatchTST: Patch Time Series Transformer (Nie et al., 2023)
- TSMixer: Time Series Mixer - all-MLP architecture (Chen et al., 2023)

Usage:
    from models.transformer import SignalTransformer, create_transformer
    from models.transformer import VisionTransformer1D, create_vit_1d
    from models.transformer import PatchTST, TSMixer
"""

# Original SignalTransformer implementation
from .signal_transformer import (
    SignalTransformer,
    create_transformer
)

# Vision Transformer 1D with CLS token
from .vision_transformer_1d import (
    VisionTransformer1D,
    create_vit_1d,
    vit_tiny_1d,
    vit_small_1d,
    vit_base_1d
)

# PatchTST (Nie et al., 2023)
from .patchtst import PatchTST

# TSMixer (Chen et al., 2023)
from .tsmixer import TSMixer

__all__ = [
    # Original SignalTransformer
    'SignalTransformer',
    'create_transformer',

    # VisionTransformer1D variants
    'VisionTransformer1D',
    'create_vit_1d',
    'vit_tiny_1d',
    'vit_small_1d',
    'vit_base_1d',
    
    # SOTA Transformer baselines
    'PatchTST',
    'TSMixer',
]

