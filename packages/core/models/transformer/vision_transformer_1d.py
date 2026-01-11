"""
Vision Transformer (ViT) Adapted for 1D Signals

Implements ViT-style architecture for 1D time-series classification with
a learnable [CLS] token instead of global average pooling.

Reference:
- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Adapted for 1D signal processing

Key differences from standard transformer:
- Uses [CLS] token for classification (prepended to patch sequence)
- [CLS] token output is used instead of global average pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import NUM_CLASSES
from packages.core.models.base_model import BaseModel


class PositionalEncoding1D(nn.Module):
    """Positional encoding for ViT1D (learnable)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoderBlock1D(nn.Module):
    """Transformer encoder block for ViT1D."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        # Pre-norm architecture (more stable for vision transformers)
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # Feedforward with residual
        x = x + self.ff(self.norm2(x))

        if return_attention:
            return x, attn_weights
        return x


class VisionTransformer1D(BaseModel):
    """
    Vision Transformer adapted for 1D signal classification.

    Uses a learnable [CLS] token prepended to the patch sequence,
    and classifies based on the [CLS] token's output representation.

    Architecture:
        - Patch embedding: Convert signal to sequence of patches
        - [CLS] token: Prepend learnable classification token
        - Positional encoding: Add position information
        - N transformer encoder blocks
        - Extract [CLS] token output
        - Classification head

    Args:
        num_classes: Number of output classes (default: 11)
        input_channels: Number of input channels (default: 1)
        patch_size: Size of each patch (default: 512)
        d_model: Dimension of embeddings (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer blocks (default: 6)
        d_ff: Dimension of feedforward network (default: 1024)
        dropout: Dropout probability (default: 0.1)
        max_len: Maximum sequence length (default: 5000)

    Example:
        >>> model = VisionTransformer1D(num_classes=11, patch_size=512)
        >>> x = torch.randn(4, 1, 102400)  # 4 signals
        >>> output = model(x)  # [4, 11]
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        patch_size: int = 512,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Patch embedding: Conv1d with stride=patch_size
        self.patch_embedding = nn.Conv1d(
            input_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # [CLS] token - learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding (includes position for [CLS] token)
        self.pos_encoding = PositionalEncoding1D(
            d_model,
            max_len=max_len + 1,  # +1 for [CLS] token
            dropout=dropout
        )

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock1D(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
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

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following ViT paper."""
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, T]

        Returns:
            logits: Output tensor of shape [B, num_classes]
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embedding(x)  # [B, d_model, L] where L = T // patch_size

        # Transpose for transformer: [B, L, d_model]
        x = x.transpose(1, 2)

        # Prepend [CLS] token to sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, L+1, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        # Extract [CLS] token output (first token)
        cls_output = x[:, 0]  # [B, d_model]

        # Classification
        logits = self.classifier(cls_output)

        return logits

    def get_attention_weights(
        self,
        x: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract attention weights from a specific transformer layer.

        Args:
            x: Input tensor of shape [B, C, T]
            layer_idx: Index of layer to extract attention from (-1 for last layer)

        Returns:
            Attention weights of shape [B, num_heads, L+1, L+1] where L is number of patches
            (L+1 includes the [CLS] token)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embedding(x)
        x = x.transpose(1, 2)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Handle negative indexing
        if layer_idx < 0:
            layer_idx = len(self.transformer_blocks) + layer_idx

        # Pass through transformer blocks up to target layer
        for idx, block in enumerate(self.transformer_blocks):
            if idx == layer_idx:
                _, attn_weights = block(x, return_attention=True)
                return attn_weights
            else:
                x = block(x, return_attention=False)

        raise IndexError(f"Layer index {layer_idx} out of range for {len(self.transformer_blocks)} layers")

    def get_all_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights from all transformer layers.

        Args:
            x: Input tensor of shape [B, C, T]

        Returns:
            List of attention weights, one per layer
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embedding(x)
        x = x.transpose(1, 2)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Collect attention weights from all layers
        all_attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, return_attention=True)
            all_attention_weights.append(attn_weights)

        return all_attention_weights

    def get_cls_token_attention(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention weights FROM the [CLS] token to all other tokens.

        This shows which patches the [CLS] token is attending to for classification.

        Args:
            x: Input tensor of shape [B, C, T]
            layer_idx: Layer to extract from

        Returns:
            Attention weights from [CLS] token of shape [B, num_heads, L]
            where L is number of patches
        """
        attn_weights = self.get_attention_weights(x, layer_idx)
        # Get attention from [CLS] token (first token) to all others
        cls_attention = attn_weights[:, :, 0, 1:]  # [B, num_heads, L] (exclude self-attention to CLS)
        return cls_attention

    def freeze_backbone(self):
        """Freeze feature extraction layers for transfer learning."""
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.pos_encoding.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        for block in self.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze feature extraction layers."""
        for param in self.patch_embedding.parameters():
            param.requires_grad = True
        for param in self.pos_encoding.parameters():
            param.requires_grad = True
        self.cls_token.requires_grad = True
        for block in self.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = True

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'VisionTransformer1D',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'patch_size': self.patch_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'num_parameters': self.get_num_params(),
            'uses_cls_token': True
        }


def create_vit_1d(
    num_classes: int = NUM_CLASSES,
    patch_size: int = 512,
    **kwargs
) -> VisionTransformer1D:
    """
    Factory function to create Vision Transformer 1D model.

    Args:
        num_classes: Number of output classes
        patch_size: Size of each patch
        **kwargs: Additional arguments passed to VisionTransformer1D

    Returns:
        VisionTransformer1D model instance

    Example:
        >>> model = create_vit_1d(num_classes=11, patch_size=512, d_model=256)
    """
    return VisionTransformer1D(num_classes=num_classes, patch_size=patch_size, **kwargs)


# Preset configurations
def vit_tiny_1d(num_classes: int = NUM_CLASSES, **kwargs) -> VisionTransformer1D:
    """ViT-Tiny for 1D signals (lightweight)."""
    return VisionTransformer1D(
        num_classes=num_classes,
        patch_size=512,
        d_model=192,
        num_heads=3,
        num_layers=12,
        d_ff=768,
        **kwargs
    )


def vit_small_1d(num_classes: int = NUM_CLASSES, **kwargs) -> VisionTransformer1D:
    """ViT-Small for 1D signals."""
    return VisionTransformer1D(
        num_classes=num_classes,
        patch_size=512,
        d_model=384,
        num_heads=6,
        num_layers=12,
        d_ff=1536,
        **kwargs
    )


def vit_base_1d(num_classes: int = NUM_CLASSES, **kwargs) -> VisionTransformer1D:
    """ViT-Base for 1D signals (recommended for most use cases)."""
    return VisionTransformer1D(
        num_classes=num_classes,
        patch_size=512,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        **kwargs
    )


if __name__ == '__main__':
    # Test the implementation
    print("Testing VisionTransformer1D...")

    # Create model
    model = create_vit_1d(num_classes=11, patch_size=512, d_model=256, num_heads=8, num_layers=6)

    # Test forward pass
    batch_size = 2
    signal_length = 102400
    x = torch.randn(batch_size, 1, signal_length)

    print(f"\nInput shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 11), f"Expected (2, 11), got {output.shape}"

    # Test attention extraction
    attn = model.get_attention_weights(x, layer_idx=-1)
    n_patches = signal_length // 512
    expected_attn_shape = (batch_size, 8, n_patches + 1, n_patches + 1)  # +1 for CLS token
    print(f"Attention shape: {attn.shape}")
    assert attn.shape == expected_attn_shape, f"Expected {expected_attn_shape}, got {attn.shape}"

    # Test CLS token attention
    cls_attn = model.get_cls_token_attention(x, layer_idx=-1)
    expected_cls_shape = (batch_size, 8, n_patches)
    print(f"CLS token attention shape: {cls_attn.shape}")
    assert cls_attn.shape == expected_cls_shape, f"Expected {expected_cls_shape}, got {cls_attn.shape}"

    # Test config
    config = model.get_config()
    print(f"\nModel config: {config}")
    print(f"Uses CLS token: {config['uses_cls_token']}")

    print("\n✅ All tests passed!")
