"""
Contrastive Learning for Time-Frequency Representations

Implements self-supervised contrastive learning (SimCLR-style) for spectrograms.
Useful for pretraining when labeled data is limited.

Reference:
- Chen et al. (2020). "A Simple Framework for Contrastive Learning (SimCLR)"
- Adapted for time-frequency representations

Workflow:
1. Generate two augmented views of each spectrogram
2. Encode both views with same encoder
3. Maximize agreement between augmented views (positive pairs)
4. Minimize agreement with other samples (negative pairs)

Usage:
    from data.contrast_learning_tfr import ContrastiveSpectrogramDataset, NTXentLoss

    # Create contrastive dataset
    dataset = ContrastiveSpectrogramDataset(
        spectrograms=train_specs,
        augmentation=get_default_spec_augment('medium')
    )

    # Create encoder and train with contrastive loss
    encoder = resnet18_2d(num_classes=128)  # 128-dim embeddings
    criterion = NTXentLoss(temperature=0.5)

    # Training loop
    for spec1, spec2 in dataloader:
        z1 = encoder(spec1)
        z2 = encoder(spec2)
        loss = criterion(z1, z2)
        loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Callable, Tuple


class ContrastiveSpectrogramDataset(Dataset):
    """
    Dataset for contrastive learning on spectrograms.

    Generates pairs of augmented views from each spectrogram.

    Args:
        spectrograms: Array of spectrograms [N, H, W]
        augmentation: Augmentation function that takes spectrogram and returns augmented version
        return_original: Whether to also return original spectrogram (default: False)
    """

    def __init__(
        self,
        spectrograms: np.ndarray,
        augmentation: Optional[Callable] = None,
        return_original: bool = False
    ):
        self.spectrograms = spectrograms
        self.augmentation = augmentation
        self.return_original = return_original

        if augmentation is None:
            # Use default augmentation
            from data.spectrogram_augmentation import get_default_spec_augment
            self.augmentation = get_default_spec_augment('medium')

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get two augmented views of a spectrogram.

        Returns:
            Tuple of (view1, view2) where both are augmented versions of the same spectrogram
        """
        spec = self.spectrograms[idx]  # [H, W]

        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)  # [1, H, W]

        # Generate two augmented views
        view1 = self.augmentation(spec_tensor)
        view2 = self.augmentation(spec_tensor)

        if self.return_original:
            return view1, view2, spec_tensor
        else:
            return view1, view2


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    The contrastive loss used in SimCLR.

    Args:
        temperature: Temperature parameter for scaling (default: 0.5)
        normalize: Whether to L2-normalize embeddings (default: True)
    """

    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Args:
            z1: Embeddings of view 1 [B, D]
            z2: Embeddings of view 2 [B, D]

        Returns:
            Scalar loss
        """
        B = z1.shape[0]

        # L2 normalize embeddings
        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature  # [2B, 2B]

        # Create positive pair mask
        # For each sample i, its positive pair is at i+B (or i-B)
        positive_mask = torch.zeros(2*B, 2*B, dtype=torch.bool, device=z.device)
        for i in range(B):
            positive_mask[i, i+B] = True
            positive_mask[i+B, i] = True

        # Create negative pair mask (exclude self and positive)
        negative_mask = torch.ones(2*B, 2*B, dtype=torch.bool, device=z.device)
        negative_mask.fill_diagonal_(False)
        negative_mask = negative_mask & ~positive_mask

        # Compute loss
        loss = 0
        for i in range(2*B):
            # Positive similarity
            pos_sim = sim_matrix[i, positive_mask[i]]  # [1]

            # Negative similarities
            neg_sim = sim_matrix[i, negative_mask[i]]  # [2B-2]

            # LogSumExp for numerical stability
            logits = torch.cat([pos_sim, neg_sim])
            labels = torch.zeros(1, dtype=torch.long, device=z.device)  # Positive is at index 0

            loss += F.cross_entropy(logits.unsqueeze(0), labels)

        loss = loss / (2*B)

        return loss


class SimCLRLoss(nn.Module):
    """
    Alternative implementation of SimCLR loss using cosine similarity.

    Args:
        temperature: Temperature parameter (default: 0.5)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute SimCLR loss.

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

        # Get encoder output dimension
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


def pretrain_contrastive(
    encoder: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    temperature: float = 0.5,
    device: str = 'cuda'
) -> nn.Module:
    """
    Pretrain encoder using contrastive learning.

    Args:
        encoder: Base encoder to pretrain
        train_loader: DataLoader of ContrastiveSpectrogramDataset
        num_epochs: Number of pretraining epochs
        lr: Learning rate
        temperature: Temperature for contrastive loss
        device: Device to train on

    Returns:
        Pretrained encoder
    """
    # Wrap encoder with projection head
    model = ContrastiveEncoder(encoder).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss
    criterion = NTXentLoss(temperature=temperature)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for view1, view2 in train_loader:
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward pass
            _, z1 = model(view1)
            _, z2 = model(view2)

            # Compute loss
            loss = criterion(z1, z2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] Contrastive Loss: {avg_loss:.4f}")

    # Return pretrained encoder (without projection head)
    return model.encoder


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
