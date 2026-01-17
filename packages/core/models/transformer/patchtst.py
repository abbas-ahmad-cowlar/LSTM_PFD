#!/usr/bin/env python3
"""
PatchTST - Patch Time Series Transformer for Bearing Fault Diagnosis

Implementation of PatchTST from "A Time Series is Worth 64 Words: 
Long-term Forecasting with Transformers" (Nie et al., 2023).

Key Features:
- Patching: Divides time series into patches to reduce token count
- Channel-independence: Each channel processed independently
- Standard Transformer encoder backbone
- Efficient attention on patches vs raw samples

Usage:
    from packages.core.models.transformer.patchtst import PatchTST
    
    model = PatchTST(
        num_classes=11,
        input_length=102400,
        patch_size=1024,
        d_model=128,
        n_heads=4,
        n_layers=3
    )
    
    # Input: [batch, channels, sequence_length]
    output = model(x)  # [batch, num_classes]

Author: Critical Deficiency Fix #9 (Priority: 80)
Date: 2026-01-18

Reference:
    Nie, Y., et al. "A Time Series is Worth 64 Words: Long-term Forecasting 
    with Transformers." ICLR 2023.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """Convert time series to patch embeddings."""
    
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        in_channels: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # Linear projection of patches
        self.projection = nn.Linear(patch_size * in_channels, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: [batch, channels, seq_len]
        
        Returns:
            patches: [batch, num_patches, d_model]
            num_patches: number of patches
        """
        batch_size, channels, seq_len = x.shape
        
        # Calculate number of patches
        num_patches = seq_len // self.patch_size
        
        # Reshape to patches: [batch, num_patches, patch_size * channels]
        x = x[:, :, :num_patches * self.patch_size]  # Trim to exact multiple
        x = x.reshape(batch_size, channels, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # [batch, num_patches, channels, patch_size]
        x = x.reshape(batch_size, num_patches, -1)  # [batch, num_patches, channels * patch_size]
        
        # Project to d_model
        patches = self.projection(x)
        patches = self.dropout(patches)
        
        return patches, num_patches


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm architecture."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Pre-norm feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out
        
        return x


class PatchTST(nn.Module):
    """
    Patch Time Series Transformer for classification.
    
    Architecture:
        Input: [B, C, L] where L = sequence length
        ├─ PatchEmbedding: [B, N, D] where N = L / patch_size
        ├─ PositionalEncoding: [B, N, D]
        ├─ TransformerEncoder × n_layers: [B, N, D]
        ├─ GlobalAveragePooling: [B, D]
        └─ ClassificationHead: [B, num_classes]
    
    Args:
        num_classes: Number of output classes
        input_length: Length of input sequence
        patch_size: Size of each patch
        in_channels: Number of input channels
        d_model: Transformer dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension (default: 4 * d_model)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int = 11,
        input_length: int = 102400,
        patch_size: int = 1024,
        in_channels: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_length = input_length
        self.patch_size = patch_size
        self.d_model = d_model
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Number of patches
        self.num_patches = input_length // patch_size
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            in_channels=in_channels,
            dropout=dropout
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=self.num_patches + 1,
            dropout=dropout
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len] or [batch, seq_len]
        
        Returns:
            logits: [batch, num_classes]
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Patch embedding
        patches, _ = self.patch_embedding(x)  # [B, N, D]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)  # [B, N+1, D]
        
        # Add positional encoding
        patches = self.pos_encoding(patches)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            patches = layer(patches)
        
        # Final normalization
        patches = self.norm(patches)
        
        # Use CLS token for classification
        cls_output = patches[:, 0, :]  # [B, D]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention maps for visualization."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Patch embedding
        patches, _ = self.patch_embedding(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        patches = self.pos_encoding(patches)
        
        # Get attention from first layer
        x_norm = self.encoder_layers[0].norm1(patches)
        _, attn_weights = self.encoder_layers[0].attn(
            x_norm, x_norm, x_norm, need_weights=True
        )
        
        return attn_weights


def test_patchtst():
    """Test PatchTST model."""
    print("=" * 60)
    print("PatchTST TEST")
    print("=" * 60)
    
    # Create model
    model = PatchTST(
        num_classes=11,
        input_length=102400,
        patch_size=1024,
        d_model=128,
        n_heads=4,
        n_layers=3
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"Patches: {model.num_patches}")
    
    # Test forward pass
    x = torch.randn(4, 1, 102400)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (4, 11), f"Expected (4, 11), got {output.shape}"
    
    # Test attention extraction
    attn = model.get_attention_maps(x[:1])
    print(f"Attention shape: {attn.shape}")
    
    print("\n✅ PatchTST test passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_patchtst()
