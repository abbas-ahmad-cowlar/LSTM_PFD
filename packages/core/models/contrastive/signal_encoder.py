"""
Signal Encoder for Contrastive Learning

CNN-based 1D signal encoder that produces embeddings for contrastive learning.
Architecture inspired by SimCLR projection heads.

Extracted from: scripts/research/contrastive_physics.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalEncoder(nn.Module):
    """
    CNN-based signal encoder that produces embeddings for contrastive learning.
    Architecture inspired by SimCLR projection heads.

    Args:
        in_channels: Number of input channels (default: 1)
        embedding_dim: Dimension of output embeddings (default: 128)
        hidden_dim: Hidden dimension for projection head (default: 256)
        num_classes: Unused — accepted for factory compatibility
    """

    def __init__(self,
                 in_channels: int = 1,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_classes: int = None,
                 **kwargs):
        super().__init__()

        # Backbone CNN
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Projection head (for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input signals (B, C, L)
            return_features: If True, return backbone features instead of projections

        Returns:
            embeddings: (B, embedding_dim) or (B, 256) if return_features
        """
        features = self.backbone(x).squeeze(-1)  # (B, 256)

        if return_features:
            return features

        embeddings = self.projection(features)  # (B, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize

        return embeddings

    def get_config(self):
        """Return model configuration."""
        return {
            'model_type': 'SignalEncoder',
            'embedding_dim': self.embedding_dim,
        }

    def get_num_params(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
