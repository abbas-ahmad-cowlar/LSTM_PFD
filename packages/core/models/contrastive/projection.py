"""
Projection Head for Contrastive Learning

MLP projection head that maps representations to lower-dimensional space
for contrastive loss computation.

Extracted from: data/contrast_learning_tfr.py
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps representations to lower-dimensional space for contrastive loss.

    Args:
        input_dim: Input feature dimension (e.g., 512 from ResNet)
        hidden_dim: Hidden dimension (default: 2048)
        output_dim: Output embedding dimension (default: 128)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to embedding space."""
        return self.projection(x)
