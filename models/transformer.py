"""
Transformer Encoder for Time-Series Classification

Implements Transformer architecture for bearing fault diagnosis using
self-attention mechanisms to capture long-range temporal dependencies.

Reference:
- Vaswani et al. (2017). "Attention Is All You Need"
- Adapted for 1D time-series classification

Input: [B, 1, T] where T is signal length
Output: [B, 11] for 11 fault classes
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_model import BaseModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.

    Can use either:
    - Learnable embeddings
    - Fixed sinusoidal embeddings (original Transformer paper)

    Args:
        d_model: Dimension of embeddings
        max_len: Maximum sequence length
        learnable: If True, use learnable embeddings; else use sinusoidal
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        learnable: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable

        if learnable:
            # Learnable positional embeddings
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        else:
            # Fixed sinusoidal positional encodings
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]

            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor with positional encoding added [B, L, D]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block.

    Structure:
        x -> MultiHeadAttention -> LayerNorm -> FFN -> LayerNorm -> out
        |______________________|           |_________|
             (residual)                    (residual)

    Args:
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        d_ff: Dimension of feedforward network
        dropout: Dropout probability
        activation: Activation function ('relu' or 'gelu')
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> tuple:
        """
        Args:
            x: Input tensor [B, L, D]
            attn_mask: Attention mask [L, L] or [B*num_heads, L, L]
            return_attention: If True, return attention weights

        Returns:
            If return_attention=False: Output tensor [B, L, D]
            If return_attention=True: (output tensor [B, L, D], attention weights [B, num_heads, L, L])
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        if return_attention:
            return x, attn_weights
        return x


class SignalTransformer(BaseModel):
    """
    Transformer model for time-series classification.

    Architecture:
        - Patch embedding: Convert signal to sequence of patches
        - Positional encoding
        - N transformer encoder blocks
        - Global average pooling
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
        learnable_pe: Use learnable positional encodings (default: True)
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
        max_len: int = 5000,
        learnable_pe: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.d_model = d_model

        # Patch embedding: Conv1d with stride=patch_size
        self.patch_embedding = nn.Conv1d(
            input_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model,
            max_len=max_len,
            learnable=learnable_pe,
            dropout=dropout
        )

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
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

        # Patch embedding
        x = self.patch_embedding(x)  # [B, d_model, L] where L = T // patch_size

        # Transpose for transformer: [B, L, d_model]
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [B, d_model]

        # Classification
        logits = self.classifier(x)

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
            Attention weights of shape [B, num_heads, L, L] where L is sequence length
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        # Patch embedding
        x = self.patch_embedding(x)  # [B, d_model, L]

        # Transpose for transformer: [B, L, d_model]
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Handle negative indexing
        if layer_idx < 0:
            layer_idx = len(self.transformer_blocks) + layer_idx

        # Pass through transformer blocks up to target layer
        for idx, block in enumerate(self.transformer_blocks):
            if idx == layer_idx:
                # Extract attention weights from this layer
                _, attn_weights = block(x, return_attention=True)
                return attn_weights
            else:
                x = block(x, return_attention=False)

        # If we get here, layer_idx was out of range
        raise IndexError(f"Layer index {layer_idx} out of range for {len(self.transformer_blocks)} layers")

    def get_all_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights from all transformer layers.

        Args:
            x: Input tensor of shape [B, C, T]

        Returns:
            List of attention weights, one per layer, each of shape [B, num_heads, L, L]
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        # Patch embedding
        x = self.patch_embedding(x)  # [B, d_model, L]

        # Transpose for transformer: [B, L, d_model]
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Collect attention weights from all layers
        all_attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, return_attention=True)
            all_attention_weights.append(attn_weights)

        return all_attention_weights

    def get_feature_extractor(self) -> nn.Module:
        """
        Return the feature extraction backbone.

        Returns:
            Feature extractor module
        """
        return nn.Sequential(
            self.patch_embedding,
            self.pos_encoding,
            *self.transformer_blocks,
            self.norm
        )

    def freeze_backbone(self):
        """Freeze feature extraction layers for transfer learning."""
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.pos_encoding.parameters():
            param.requires_grad = False
        for block in self.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze feature extraction layers."""
        for param in self.patch_embedding.parameters():
            param.requires_grad = True
        for param in self.pos_encoding.parameters():
            param.requires_grad = True
        for block in self.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = True

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'SignalTransformer',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'patch_size': self.patch_size,
            'd_model': self.d_model,
            'num_parameters': self.get_num_params()
        }


def create_transformer(num_classes: int = NUM_CLASSES, **kwargs) -> SignalTransformer:
    """
    Factory function to create Transformer model.

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to SignalTransformer

    Returns:
        SignalTransformer model instance
    """
    return SignalTransformer(num_classes=num_classes, **kwargs)
