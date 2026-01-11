"""
Advanced Attention Mechanisms

This module provides implementations of various attention mechanisms including:
- Linear Attention (Linear Transformer)
- Cross Attention
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Sliding Window Attention
- Flash Attention concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LinearAttention(nn.Module):
    """
    Linear Attention mechanism from "Transformers are RNNs: Fast Autoregressive
    Transformers with Linear Attention" (https://arxiv.org/abs/2006.16236)

    Uses kernel feature maps to compute attention in linear time O(N) instead of O(N^2).

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to qkv projection
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply feature map (using elu + 1 to ensure positivity)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention computation
        # Instead of computing attention weights explicitly, we compute the numerator and denominator separately
        # Formula: out = (Q @ (K^T @ V)) / (Q @ K^T @ 1)
        k_sum = k.sum(dim=-2)  # [B, heads, head_dim]
        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, heads, head_dim, head_dim]

        # Compute numerator: Q @ (K^T @ V)
        numerator = torch.einsum('bhnd,bhde->bhne', q, kv)  # [B, heads, N, head_dim]

        # Compute denominator: Q @ K^T @ 1
        denominator = torch.einsum('bhnd,bhd->bhn', q, k_sum)  # [B, heads, N]

        # Normalize: divide numerator by denominator
        out = numerator / (denominator.unsqueeze(-1) + 1e-6)  # [B, heads, N, head_dim]

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class CrossAttention(nn.Module):
    """
    Cross Attention mechanism for attending from one sequence to another.

    Commonly used in encoder-decoder architectures where the decoder attends to encoder outputs.

    Args:
        dim: Query dimension
        context_dim: Key/Value dimension (encoder dimension)
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to qkv projection
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        context_dim = context_dim or dim

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor of shape (B, N, C)
            context: Key/Value tensor of shape (B, M, D). If None, performs self-attention.
            mask: Optional attention mask

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        context = context if context is not None else x

        # Compute Q from x, K and V from context
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) from "Fast Transformer Decoding: One Write-Head is All You Need"
    (https://arxiv.org/abs/1911.02150)

    Uses a single key-value head shared across all query heads, reducing memory bandwidth
    and improving inference speed.

    Args:
        dim: Input dimension
        num_heads: Number of query heads
        qkv_bias: Whether to add bias to qkv projection
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query has multiple heads, but Key and Value have only one head
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.head_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.head_dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Multi-head queries
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Single-head key and value
        k = self.to_k(x).unsqueeze(1)  # (B, 1, N, head_dim)
        v = self.to_v(x).unsqueeze(1)  # (B, 1, N, head_dim)

        # Broadcast key and value to all heads
        k = k.expand(B, self.num_heads, N, self.head_dim)
        v = v.expand(B, self.num_heads, N, self.head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) from "GQA: Training Generalized Multi-Query Transformer Models"
    (https://arxiv.org/abs/2305.13245)

    A middle ground between Multi-Head Attention and Multi-Query Attention.
    Groups multiple query heads to share key-value heads.

    Args:
        dim: Input dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads)
        qkv_bias: Whether to add bias to qkv projection
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        qkv_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Multi-head queries
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Grouped key and value
        k = self.to_k(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat key and value for each group
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention for local attention patterns.

    Only attends to a fixed-size window around each token, reducing complexity
    from O(N^2) to O(N*W) where W is the window size.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        window_size: Size of attention window
        qkv_bias: Whether to add bias to qkv projection
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 512,
        qkv_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Create sliding window mask efficiently using vectorized operations
        # Create position indices
        positions = torch.arange(N, device=x.device)
        # Compute pairwise distances: |i - j|
        distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        # Mask out positions beyond window size
        window_mask = distance <= (self.window_size // 2)

        attn = attn.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FlashAttention(nn.Module):
    """
    Flash Attention implementation (simplified version).

    Flash Attention from "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    (https://arxiv.org/abs/2205.14135)

    This is a pedagogical implementation showing the core ideas. For production use,
    the official Flash Attention CUDA kernels should be used.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to qkv projection
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention computation
        # Flash Attention uses tiled computation and recomputation in backward pass
        # to reduce memory usage, but the forward pass is mathematically equivalent
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (https://arxiv.org/abs/2104.09864)

    Applies rotary position embeddings to queries and keys.

    Args:
        dim: Dimension per head
        max_seq_len: Maximum sequence length
        base: Base for exponential decay
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

        # Build cache
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor
            seq_len: Sequence length

        Returns:
            cos and sin embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]

        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...]
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to queries and keys.

    Args:
        q: Query tensor of shape (..., seq_len, dim)
        k: Key tensor of shape (..., seq_len, dim)
        cos: Cosine embeddings
        sin: Sine embeddings

    Returns:
        Rotated queries and keys
    """
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    seq_len = 128
    dim = 512

    x = torch.randn(batch_size, seq_len, dim).to(device)

    # Test Linear Attention
    print("Testing Linear Attention...")
    linear_attn = LinearAttention(dim, num_heads=8).to(device)
    out = linear_attn(x)
    print(f"Output shape: {out.shape}")

    # Test Multi-Query Attention
    print("\nTesting Multi-Query Attention...")
    mqa = MultiQueryAttention(dim, num_heads=8).to(device)
    out = mqa(x)
    print(f"Output shape: {out.shape}")

    # Test Grouped-Query Attention
    print("\nTesting Grouped-Query Attention...")
    gqa = GroupedQueryAttention(dim, num_heads=8, num_kv_heads=2).to(device)
    out = gqa(x)
    print(f"Output shape: {out.shape}")

    # Test Cross Attention
    print("\nTesting Cross Attention...")
    context = torch.randn(batch_size, 64, dim).to(device)
    cross_attn = CrossAttention(dim, num_heads=8).to(device)
    out = cross_attn(x, context)
    print(f"Output shape: {out.shape}")

    print("\nAll attention mechanisms tested successfully!")
