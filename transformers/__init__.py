"""
Advanced Transformer Architectures Module

This module provides comprehensive implementations of state-of-the-art transformer
architectures for both computer vision and natural language processing.

Includes:
    - Vision Transformers: ViT, Swin Transformer
    - Language Models: BERT, GPT/GPT-2, T5
    - Attention Mechanisms: Linear, Cross, MQA, GQA, Sliding Window, RoPE

Quick Start:
    # Vision Models
    >>> from transformers.advanced import vit_base_patch16_224, swin_tiny_patch4_window7_224
    >>> vit_model = vit_base_patch16_224(num_classes=1000)
    >>> swin_model = swin_tiny_patch4_window7_224(num_classes=1000)

    # Language Models
    >>> from transformers.advanced import bert_base, gpt2_small, t5_base
    >>> bert_model = bert_base()
    >>> gpt_model = gpt2_small()
    >>> t5_model = t5_base()

    # Attention Mechanisms
    >>> from transformers.advanced import MultiQueryAttention, GroupedQueryAttention
    >>> mqa = MultiQueryAttention(dim=512, num_heads=8)
    >>> gqa = GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2)

For detailed documentation, see transformers/README.md

Note: A basic transformer implementation is available in models/transformer.py
"""

# Advanced architectures are available through the advanced submodule
from . import advanced

__version__ = "1.0.0"

__all__ = ['advanced']
