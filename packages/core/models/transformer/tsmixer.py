#!/usr/bin/env python3
"""
TSMixer - Time Series Mixer for Bearing Fault Diagnosis

Implementation of TSMixer from "TSMixer: An All-MLP Architecture 
for Time Series Forecasting" (Chen et al., 2023 @ Google).

Key Features:
- All-MLP architecture (no attention mechanism)
- Temporal mixing with shared MLPs
- Feature mixing across channels
- Efficient computation: O(L) vs O(L²) for Transformers

Usage:
    from packages.core.models.transformer.tsmixer import TSMixer
    
    model = TSMixer(
        num_classes=11,
        input_length=102400,
        n_features=1,
        n_blocks=3,
        d_model=256
    )
    
    # Input: [batch, channels, sequence_length]
    output = model(x)  # [batch, num_classes]

Author: Critical Deficiency Fix #9 (Priority: 80)
Date: 2026-01-18

Reference:
    Chen, S., et al. "TSMixer: An All-MLP Architecture for Time Series 
    Forecasting." arXiv preprint arXiv:2303.06053 (2023).
"""

import torch
import torch.nn as nn
from typing import Optional


class TemporalMixingBlock(nn.Module):
    """
    Temporal mixing with shared MLP across all features.
    Mixes information along the time dimension.
    """
    
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, features, seq_len]
        Returns:
            [batch, features, seq_len]
        """
        # Residual connection
        residual = x
        
        # Normalize along time dimension
        x = self.norm(x)
        
        # MLP mixing across time (shared across all features)
        x = self.mlp(x)
        
        return x + residual


class FeatureMixingBlock(nn.Module):
    """
    Feature mixing with shared MLP across all time steps.
    Mixes information across feature channels.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(n_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, features, seq_len]
        Returns:
            [batch, features, seq_len]
        """
        # Residual connection
        residual = x
        
        # Transpose to [batch, seq_len, features]
        x = x.transpose(1, 2)
        
        # Normalize along feature dimension
        x = self.norm(x)
        
        # MLP mixing across features
        x = self.mlp(x)
        
        # Transpose back
        x = x.transpose(1, 2)
        
        return x + residual


class TSMixerBlock(nn.Module):
    """
    Single TSMixer block with temporal and feature mixing.
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        temporal_hidden: int,
        feature_hidden: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.temporal_mix = TemporalMixingBlock(seq_len, temporal_hidden, dropout)
        self.feature_mix = FeatureMixingBlock(n_features, feature_hidden, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, features, seq_len]
        Returns:
            [batch, features, seq_len]
        """
        x = self.temporal_mix(x)
        x = self.feature_mix(x)
        return x


class TSMixer(nn.Module):
    """
    TSMixer model for time-series classification.
    
    Architecture:
        Input: [B, C, L] where L = sequence length
        ├─ InputProjection: [B, D, L'] where L' = L // stride
        ├─ TSMixerBlock × n_blocks: [B, D, L']
        ├─ GlobalAveragePooling: [B, D]
        └─ ClassificationHead: [B, num_classes]
    
    Args:
        num_classes: Number of output classes
        input_length: Length of input sequence
        n_features: Number of input features/channels
        n_blocks: Number of TSMixer blocks
        d_model: Internal feature dimension
        temporal_hidden: Hidden dim for temporal mixing
        feature_hidden: Hidden dim for feature mixing
        stride: Downsampling stride for input
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int = 11,
        input_length: int = 102400,
        n_features: int = 1,
        n_blocks: int = 4,
        d_model: int = 256,
        temporal_hidden: Optional[int] = None,
        feature_hidden: Optional[int] = None,
        stride: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_length = input_length
        self.n_features = n_features
        self.d_model = d_model
        
        if temporal_hidden is None:
            temporal_hidden = d_model * 2
        if feature_hidden is None:
            feature_hidden = d_model * 2
        
        # Reduce sequence length with strided convolution
        self.seq_len = input_length // stride
        
        # Input projection: strided conv to reduce length and project to d_model
        self.input_proj = nn.Conv1d(
            n_features, d_model, 
            kernel_size=stride, stride=stride
        )
        
        # TSMixer blocks
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=self.seq_len,
                n_features=d_model,
                temporal_hidden=temporal_hidden,
                feature_hidden=feature_hidden,
                dropout=dropout
            )
            for _ in range(n_blocks)
        ])
        
        # Final normalization
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
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
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
        
        # Input projection and downsampling
        x = self.input_proj(x)  # [B, d_model, seq_len // stride]
        
        # TSMixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # [B, d_model]
        
        # Final normalization
        x = self.norm(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = x.mean(dim=2)
        x = self.norm(x)
        
        return x


def test_tsmixer():
    """Test TSMixer model."""
    print("=" * 60)
    print("TSMixer TEST")
    print("=" * 60)
    
    # Create model
    model = TSMixer(
        num_classes=11,
        input_length=102400,
        n_features=1,
        n_blocks=4,
        d_model=256,
        stride=100
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"Reduced sequence length: {model.seq_len}")
    
    # Test forward pass
    x = torch.randn(4, 1, 102400)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (4, 11), f"Expected (4, 11), got {output.shape}"
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
    
    print("\n✅ TSMixer test passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_tsmixer()
