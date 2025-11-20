"""
Transformer Architectures Module

This module provides comprehensive implementations of transformer architectures
for both computer vision and natural language processing.

Organization:
    - Basic transformer: Standard encoder-decoder architecture
    - Advanced transformers: State-of-the-art models (ViT, BERT, GPT, T5, Swin, etc.)

Quick Start:
    # Basic Transformer
    >>> from transformers import Transformer
    >>> model = Transformer(d_model=512, nhead=8)

    # Advanced Models
    >>> from transformers.advanced import vit_base_patch16_224, bert_base, gpt2_small
    >>> vit_model = vit_base_patch16_224(num_classes=1000)
    >>> bert_model = bert_base()
    >>> gpt_model = gpt2_small()

For detailed documentation, see transformers/README.md
"""

# Basic Transformer
from .transformer import Transformer

# Advanced architectures are available through the advanced submodule
# Import them explicitly: from transformers.advanced import ...

__version__ = "1.0.0"

__all__ = [
    'Transformer',
]

# Provide easy access to advanced module
from . import advanced

__all__.append('advanced')
