"""
Contrastive Learning for Time-Frequency Representations

Implements self-supervised contrastive learning (SimCLR-style) for spectrograms.
Useful for pretraining when labeled data is limited.

Core classes and losses are now in packages/core/ — this module re-exports
them for backward compatibility and provides a CLI test harness.

Reference:
    Chen et al., "A Simple Framework for Contrastive Learning of Visual
    Representations" (SimCLR), 2020.

Usage:
    from data.contrast_learning_tfr import (
        ContrastiveSpectrogramDataset,
        NTXentLoss,
        SimCLRLoss,
        ProjectionHead,
        ContrastiveEncoder,
        pretrain_contrastive,
    )
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Re-export from canonical packages (classes were extracted from this file)
# ---------------------------------------------------------------------------
from packages.core.training.contrastive import (
    NTXentLoss,
    SimCLRLoss,
    ContrastiveSpectrogramDataset,
    pretrain_contrastive,
)
from packages.core.models.contrastive import (
    ProjectionHead,
    ContrastiveEncoder,
)

# Expose all public symbols for backward-compatible imports
__all__ = [
    'ContrastiveSpectrogramDataset',
    'NTXentLoss',
    'SimCLRLoss',
    'ProjectionHead',
    'ContrastiveEncoder',
    'pretrain_contrastive',
]


if __name__ == '__main__':
    # Test contrastive learning
    print("Testing Contrastive Learning...")

    # Create dummy spectrograms
    spectrograms = np.random.randn(100, 129, 400).astype(np.float32)

    # Create contrastive dataset
    dataset = ContrastiveSpectrogramDataset(spectrograms)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    # Test loss
    z1 = torch.randn(16, 128)
    z2 = torch.randn(16, 128)

    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(z1, z2)

    print(f"NT-Xent Loss: {loss.item():.4f}")

    # Test SimCLR loss
    criterion2 = SimCLRLoss(temperature=0.5)
    loss2 = criterion2(z1, z2)

    print(f"SimCLR Loss: {loss2.item():.4f}")

    # Test projection head
    proj_head = ProjectionHead(input_dim=512, output_dim=128)
    features = torch.randn(16, 512)
    projections = proj_head(features)

    print(f"Projection shape: {projections.shape}")
