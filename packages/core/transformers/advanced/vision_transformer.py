"""
Vision Transformer (ViT) Implementation
Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
arXiv:2010.11929

This module provides a complete implementation of the Vision Transformer architecture,
which applies the Transformer model directly to sequences of image patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.

    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch (assumed square)
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Use a convolutional layer to extract patches and project them
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Patch embeddings of shape (batch_size, n_patches, embed_dim)
        """
        # x: (B, C, H, W) -> (B, embed_dim, n_patches_h, n_patches_w)
        x = self.projection(x)

        # Flatten spatial dimensions: (B, embed_dim, n_patches_h, n_patches_w) -> (B, embed_dim, n_patches)
        x = x.flatten(2)

        # Transpose: (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for Vision Transformer.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections for all heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2)  # (B, seq_len, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)

        # Final projection
        out = self.proj(out)
        out = self.dropout(out)

        return out


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) for Vision Transformer.

    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        dropout: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block for Vision Transformer.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        dropout: Dropout probability
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    This implementation follows the original ViT paper and supports various
    configurations (ViT-Base, ViT-Large, ViT-Huge).

    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch (assumed square)
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        dropout: Dropout probability
        use_cls_token: Whether to use a class token (default: True)

    Examples:
        # ViT-Base/16
        >>> model = VisionTransformer(
        ...     img_size=224, patch_size=16, num_classes=1000,
        ...     embed_dim=768, depth=12, num_heads=12
        ... )

        # ViT-Large/16
        >>> model = VisionTransformer(
        ...     img_size=224, patch_size=16, num_classes=1000,
        ...     embed_dim=1024, depth=24, num_heads=16
        ... )

        # ViT-Huge/14
        >>> model = VisionTransformer(
        ...     img_size=224, patch_size=14, num_classes=1000,
        ...     embed_dim=1280, depth=32, num_heads=16
        ... )
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # Class token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = num_patches + 1
        else:
            num_tokens = num_patches

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT paper."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize classification head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            return_features: If True, return features before classification head

        Returns:
            If return_features=False: Class logits of shape (batch_size, num_classes)
            If return_features=True: Features of shape (batch_size, embed_dim)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Extract features (use class token or global average pooling)
        if self.use_cls_token:
            features = x[:, 0]  # (B, embed_dim)
        else:
            features = x.mean(dim=1)  # (B, embed_dim)

        if return_features:
            return features

        # Classification
        logits = self.head(features)
        return logits

    def get_attention_maps(
        self,
        x: torch.Tensor,
        block_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract attention maps from the model for visualization.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            block_idx: Index of transformer block to extract attention from.
                      If None, returns attention from the last block.

        Returns:
            Attention maps of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Choose which block to extract attention from
        if block_idx is None:
            block_idx = len(self.blocks) - 1

        # Forward through blocks and extract attention
        for idx, block in enumerate(self.blocks):
            if idx == block_idx:
                # Extract attention from this block
                normed_x = block.norm1(x)
                qkv = block.attn.qkv(normed_x)
                batch_size, seq_len, _ = normed_x.shape

                qkv = qkv.reshape(batch_size, seq_len, 3, block.attn.num_heads, block.attn.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                scores = torch.matmul(q, k.transpose(-2, -1)) / (block.attn.head_dim ** 0.5)
                attn = F.softmax(scores, dim=-1)

                return attn

            x = block(x)

        return None


def vit_tiny_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Tiny with 16x16 patches for 224x224 images."""
    return VisionTransformer(
        img_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, **kwargs
    )


def vit_small_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Small with 16x16 patches for 224x224 images."""
    return VisionTransformer(
        img_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, **kwargs
    )


def vit_base_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Base with 16x16 patches for 224x224 images."""
    return VisionTransformer(
        img_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12, **kwargs
    )


def vit_base_patch32_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Base with 32x32 patches for 224x224 images."""
    return VisionTransformer(
        img_size=224, patch_size=32, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12, **kwargs
    )


def vit_large_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Large with 16x16 patches for 224x224 images."""
    return VisionTransformer(
        img_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=1024, depth=24, num_heads=16, **kwargs
    )


def vit_huge_patch14_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Huge with 14x14 patches for 224x224 images."""
    return VisionTransformer(
        img_size=224, patch_size=14, num_classes=num_classes,
        embed_dim=1280, depth=32, num_heads=16, **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create ViT-Base model
    model = vit_base_patch16_224(num_classes=1000).to(device)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    # Get predictions
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (4, 1000)

    # Get features
    features = model(x, return_features=True)
    print(f"Features shape: {features.shape}")  # Should be (4, 768)

    # Get attention maps
    attn_maps = model.get_attention_maps(x, block_idx=-1)
    print(f"Attention maps shape: {attn_maps.shape}")  # (4, 12, seq_len, seq_len)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
