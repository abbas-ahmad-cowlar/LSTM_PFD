"""
Contrastive Encoder

Generic encoder with projection head for contrastive learning.
Wraps any base encoder (e.g., ResNet-2D) with a ProjectionHead.

Extracted from: data/contrast_learning_tfr.py
"""

import torch
import torch.nn as nn
from typing import Tuple

from .projection import ProjectionHead


class ContrastiveEncoder(nn.Module):
    """
    Encoder with projection head for contrastive learning.

    Args:
        encoder: Base encoder (e.g., ResNet-2D)
        projection_dim: Output projection dimension (default: 128)
        hidden_dim: Hidden dimension in projection head (default: 2048)
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048
    ):
        super().__init__()

        self.encoder = encoder

        # Get encoder output dimension via dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 129, 400)
            encoder_output = encoder(dummy_input)
            encoder_dim = encoder_output.shape[1]

        # Projection head
        self.projection = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input spectrogram [B, 1, H, W]

        Returns:
            Tuple of (representations, projections)
            - representations: Encoder output [B, D]
            - projections: Projected embeddings [B, projection_dim]
        """
        representations = self.encoder(x)
        projections = self.projection(representations)

        return representations, projections
