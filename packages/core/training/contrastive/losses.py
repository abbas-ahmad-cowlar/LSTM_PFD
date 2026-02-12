"""
Contrastive Loss Functions

Canonical implementations for contrastive learning:
- PhysicsInfoNCELoss: InfoNCE loss for physics-based triplets (anchor, pos, neg)
- NTXentLoss: Normalized Temperature-scaled Cross-Entropy (SimCLR)
- SimCLRLoss: Alternative SimCLR loss using cosine similarity matrix

Extracted from:
- scripts/research/contrastive_physics.py (PhysicsInfoNCELoss, NTXentLoss)
- data/contrast_learning_tfr.py (SimCLRLoss — deduplicated NTXentLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss adapted for physics-based pairs.

    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

    where z_j is the physics-similar positive and z_k includes negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: (B, D) anchor embeddings
            positive: (B, D) positive embeddings (physics-similar)
            negatives: (B, N, D) negative embeddings

        Returns:
            loss: scalar contrastive loss
        """
        batch_size = anchor.size(0)

        # Positive similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)  # (B,)
        pos_sim = pos_sim / self.temperature

        # Negative similarities
        # anchor: (B, 1, D), negatives: (B, N, D)
        neg_sim = F.cosine_similarity(
            anchor.unsqueeze(1), negatives, dim=2
        )  # (B, N)
        neg_sim = neg_sim / self.temperature

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)

        return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    Canonical implementation supporting both 1D signal and 2D spectrogram
    contrastive learning. Uses in-batch negatives.

    Args:
        temperature: Temperature parameter for scaling (default: 0.5)
        normalize: Whether to L2-normalize embeddings (default: True)
    """

    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (B, D) embeddings from view 1
            z_j: (B, D) embeddings from view 2

        Returns:
            loss: NT-Xent loss
        """
        batch_size = z_i.size(0)

        # L2 normalize embeddings
        if self.normalize:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        # Combine all embeddings
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2B, 2B)
        sim = sim / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -float('inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)

        return loss


class SimCLRLoss(nn.Module):
    """
    Alternative implementation of SimCLR loss using cosine similarity matrix.

    Args:
        temperature: Temperature parameter (default: 0.5)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: Embeddings of view 1 [B, D]
            z2: Embeddings of view 2 [B, D]

        Returns:
            Scalar loss
        """
        B = z1.shape[0]

        # L2 normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity
        representations = torch.cat([z1, z2], dim=0)  # [2B, D]
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # [2B, 2B]

        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z1.device)

        # Mask out self-similarity
        mask = torch.eye(2*B, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Compute loss
        loss = self.criterion(similarity_matrix, labels)

        return loss
