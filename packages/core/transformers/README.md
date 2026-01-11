# Transformer Architectures

This directory contains comprehensive implementations of transformer architectures for both computer vision and natural language processing tasks.

## Directory Structure

```
transformers/
├── transformer.py              # Basic transformer implementation
└── advanced/                   # Advanced transformer architectures
    ├── __init__.py
    ├── vision_transformer.py   # Vision Transformer (ViT)
    ├── bert.py                 # BERT
    ├── gpt.py                  # GPT/GPT-2
    ├── t5.py                   # T5
    ├── swin_transformer.py     # Swin Transformer
    └── attention_mechanisms.py # Advanced attention variants
```

## Quick Start

### Basic Transformer

The basic transformer implementation provides a standard encoder-decoder architecture:

```python
from transformers.transformer import Transformer

# Create a basic transformer
model = Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

# Forward pass
src = torch.randn(10, 32, 512)  # (seq_len, batch_size, d_model)
tgt = torch.randn(20, 32, 512)
output = model(src, tgt)
```

### Advanced Architectures

Import from the `advanced` module:

```python
from transformers.advanced import (
    VisionTransformer, vit_base_patch16_224,
    BERTModel, bert_base,
    GPTModel, gpt2_small,
    T5Model, t5_base,
    SwinTransformer, swin_tiny_patch4_window7_224
)
```

## Models

### Vision Transformer (ViT)

Vision Transformer applies the transformer architecture to image classification by treating images as sequences of patches.

**Paper:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

**Usage:**

```python
from transformers.advanced import vit_base_patch16_224

# Create ViT-Base model
model = vit_base_patch16_224(num_classes=1000)

# Forward pass
images = torch.randn(4, 3, 224, 224)
logits = model(images)  # (4, 1000)

# Extract features
features = model(images, return_features=True)  # (4, 768)

# Get attention maps for visualization
attn_maps = model.get_attention_maps(images, block_idx=-1)
```

**Available variants:**
- `vit_tiny_patch16_224` - 5M params
- `vit_small_patch16_224` - 22M params
- `vit_base_patch16_224` - 86M params (recommended)
- `vit_base_patch32_224` - 88M params
- `vit_large_patch16_224` - 304M params
- `vit_huge_patch14_224` - 632M params

### BERT

BERT is a bidirectional transformer encoder for language understanding tasks.

**Paper:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

**Usage:**

```python
from transformers.advanced import bert_base, BERTForMaskedLM, BERTForSequenceClassification

# Base BERT model
model = bert_base()

# Forward pass
input_ids = torch.randint(0, 30522, (4, 128))
sequence_output, pooled_output, attn_probs = model(input_ids)

# Masked Language Modeling
mlm_model = BERTForMaskedLM()
outputs = mlm_model(input_ids, masked_lm_labels=labels)
loss = outputs['loss']
logits = outputs['logits']

# Sequence Classification
classifier = BERTForSequenceClassification(num_labels=2)
outputs = classifier(input_ids, labels=labels)
```

**Available variants:**
- `bert_base` - 110M params (12 layers, 768 hidden, 12 heads)
- `bert_large` - 340M params (24 layers, 1024 hidden, 16 heads)

### GPT

GPT is an autoregressive decoder-only transformer for language modeling and generation.

**Papers:**
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

**Usage:**

```python
from transformers.advanced import gpt2_small

# Create GPT-2 Small model
model = gpt2_small()

# Training
input_ids = torch.randint(0, 50257, (4, 128))
labels = input_ids.clone()
outputs = model(input_ids, labels=labels)
loss = outputs['loss']

# Text Generation
prompt = torch.randint(0, 50257, (1, 10))
generated = model.generate(
    prompt,
    max_length=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    do_sample=True
)
```

**Available variants:**
- `gpt_small` - 117M params (GPT-1 style)
- `gpt2_small` - 117M params
- `gpt2_medium` - 345M params
- `gpt2_large` - 774M params
- `gpt2_xl` - 1.5B params

### T5

T5 treats every NLP task as a text-to-text problem using an encoder-decoder architecture.

**Paper:** [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

**Usage:**

```python
from transformers.advanced import t5_base

# Create T5-Base model
model = t5_base()

# Forward pass
input_ids = torch.randint(0, 32128, (4, 64))
decoder_input_ids = torch.randint(0, 32128, (4, 32))
outputs = model(input_ids, decoder_input_ids)
logits = outputs['logits']

# With labels for training
outputs = model(input_ids, decoder_input_ids, labels=target_ids)
loss = outputs['loss']
```

**Available variants:**
- `t5_small` - 60M params
- `t5_base` - 220M params (recommended)
- `t5_large` - 770M params
- `t5_xl` - 3B params

### Swin Transformer

Swin Transformer is a hierarchical vision transformer using shifted windows for efficient computation.

**Paper:** [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

**Usage:**

```python
from transformers.advanced import swin_tiny_patch4_window7_224

# Create Swin-Tiny model
model = swin_tiny_patch4_window7_224(num_classes=1000)

# Forward pass
images = torch.randn(4, 3, 224, 224)
logits = model(images)  # (4, 1000)
```

**Available variants:**
- `swin_tiny_patch4_window7_224` - 28M params
- `swin_small_patch4_window7_224` - 50M params
- `swin_base_patch4_window7_224` - 88M params
- `swin_large_patch4_window7_224` - 197M params

## Advanced Attention Mechanisms

The `attention_mechanisms.py` module provides various efficient attention implementations:

### Linear Attention

Reduces attention complexity from O(N²) to O(N) using kernel feature maps.

```python
from transformers.advanced.attention_mechanisms import LinearAttention

attn = LinearAttention(dim=512, num_heads=8)
output = attn(x)
```

### Cross Attention

Attention from one sequence to another (used in encoder-decoder models).

```python
from transformers.advanced.attention_mechanisms import CrossAttention

cross_attn = CrossAttention(dim=512, context_dim=768, num_heads=8)
output = cross_attn(query, context)
```

### Multi-Query Attention (MQA)

Uses a single key-value head shared across all query heads for faster inference.

```python
from transformers.advanced.attention_mechanisms import MultiQueryAttention

mqa = MultiQueryAttention(dim=512, num_heads=8)
output = mqa(x)
```

### Grouped-Query Attention (GQA)

Middle ground between MHA and MQA, grouping multiple query heads per key-value head.

```python
from transformers.advanced.attention_mechanisms import GroupedQueryAttention

gqa = GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2)
output = gqa(x)
```

### Sliding Window Attention

Only attends to a fixed-size window, reducing complexity to O(N·W).

```python
from transformers.advanced.attention_mechanisms import SlidingWindowAttention

sliding_attn = SlidingWindowAttention(dim=512, num_heads=8, window_size=512)
output = sliding_attn(x)
```

### Rotary Position Embedding (RoPE)

Applies rotary position embeddings to queries and keys.

```python
from transformers.advanced.attention_mechanisms import (
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb
)

rope = RotaryPositionalEmbedding(dim=64, max_seq_len=2048)
cos, sin = rope(x, seq_len=128)
q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
```

## Training Tips

### Vision Transformers (ViT, Swin)

1. **Data Augmentation:** Use strong augmentation (RandAugment, Mixup, CutMix)
2. **Learning Rate:** Use warmup (typically 5-20 epochs) followed by cosine decay
3. **Optimizer:** AdamW with weight decay (0.05-0.1)
4. **Batch Size:** Large batch sizes (1024-4096) work well
5. **Pre-training:** Pre-train on ImageNet-21k or JFT-300M for best results

### Language Models (BERT, GPT, T5)

1. **Learning Rate:** 1e-4 to 5e-4 with linear warmup
2. **Optimizer:** AdamW with β1=0.9, β2=0.999, ε=1e-6
3. **Gradient Clipping:** Clip gradients to norm of 1.0
4. **Sequence Length:** Start with shorter sequences and increase gradually
5. **Mixed Precision:** Use fp16 or bf16 for faster training

## Performance Benchmarks

### ImageNet-1K Classification (Top-1 Accuracy)

| Model | Params | Accuracy |
|-------|--------|----------|
| ViT-Base/16 | 86M | 84.5% |
| ViT-Large/16 | 304M | 87.1% |
| Swin-Tiny | 28M | 81.2% |
| Swin-Base | 88M | 85.2% |

### NLP Tasks

| Model | Task | Performance |
|-------|------|-------------|
| BERT-Base | GLUE | 84.4 avg |
| BERT-Large | GLUE | 86.4 avg |
| GPT-2 (1.5B) | WikiText-103 PPL | 17.5 |
| T5-Base | SuperGLUE | 75.1 avg |

## Citation

If you use these implementations, please cite the original papers:

```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and others},
  journal={ICLR},
  year={2021}
}

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and others},
  journal={NAACL},
  year={2019}
}

@article{radford2019gpt2,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and others},
  year={2019}
}

@article{raffel2020t5,
  title={Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author={Raffel, Colin and others},
  journal={JMLR},
  year={2020}
}

@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and others},
  booktitle={ICCV},
  year={2021}
}
```

## License

These implementations are provided for educational and research purposes. Please refer to the original papers and their licenses for commercial use.
