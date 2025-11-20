"""
Advanced Transformer Architectures

This module contains state-of-the-art transformer architectures including:
- Vision Transformer (ViT): Transformer for computer vision
- BERT: Bidirectional encoder for NLP
- GPT: Autoregressive decoder for language modeling
- T5: Text-to-text transformer for seq2seq tasks
- Swin Transformer: Hierarchical vision transformer
"""

from .vision_transformer import (
    VisionTransformer,
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_base_patch32_224,
    vit_large_patch16_224,
    vit_huge_patch14_224,
    PatchEmbedding
)

from .bert import (
    BERTModel,
    BERTForMaskedLM,
    BERTForSequenceClassification,
    bert_base,
    bert_large
)

from .gpt import (
    GPTModel,
    GPTConfig,
    gpt_small,
    gpt2_small,
    gpt2_medium,
    gpt2_large,
    gpt2_xl
)

from .t5 import (
    T5Model,
    T5Config,
    t5_small,
    t5_base,
    t5_large,
    t5_xl
)

__all__ = [
    # Vision Transformer
    'VisionTransformer',
    'vit_tiny_patch16_224',
    'vit_small_patch16_224',
    'vit_base_patch16_224',
    'vit_base_patch32_224',
    'vit_large_patch16_224',
    'vit_huge_patch14_224',
    'PatchEmbedding',
    # BERT
    'BERTModel',
    'BERTForMaskedLM',
    'BERTForSequenceClassification',
    'bert_base',
    'bert_large',
    # GPT
    'GPTModel',
    'GPTConfig',
    'gpt_small',
    'gpt2_small',
    'gpt2_medium',
    'gpt2_large',
    'gpt2_xl',
    # T5
    'T5Model',
    'T5Config',
    't5_small',
    't5_base',
    't5_large',
    't5_xl',
]
