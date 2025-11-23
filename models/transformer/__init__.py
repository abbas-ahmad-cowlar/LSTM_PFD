"""
Transformer Models for 1D Signal Classification

This module provides transformer-based architectures for bearing fault diagnosis:
- SignalTransformer: Standard transformer with global average pooling
- VisionTransformer1D: ViT-style transformer with [CLS] token

Import from parent models module:
    from models.transformer import SignalTransformer, create_transformer
    from models.transformer.vision_transformer_1d import VisionTransformer1D, create_vit_1d
"""

from .vision_transformer_1d import (
    VisionTransformer1D,
    create_vit_1d,
    vit_tiny_1d,
    vit_small_1d,
    vit_base_1d
)

__all__ = [
    'VisionTransformer1D',
    'create_vit_1d',
    'vit_tiny_1d',
    'vit_small_1d',
    'vit_base_1d',
]
