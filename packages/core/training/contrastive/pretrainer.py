"""
Contrastive Pretraining and Fine-tuning

Training loops for contrastive learning:
- ContrastivePretrainer: Physics-based contrastive pretraining (1D signals)
- ContrastiveFineTuner: Fine-tuning pretrained encoder for classification
- pretrain_contrastive: Function for spectrogram contrastive pretraining (2D)

Extracted from:
- scripts/research/contrastive_physics.py (ContrastivePretrainer, ContrastiveFineTuner)
- data/contrast_learning_tfr.py (pretrain_contrastive)
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from .losses import PhysicsInfoNCELoss, NTXentLoss

logger = logging.getLogger(__name__)


class ContrastivePretrainer:
    """Handles contrastive pretraining with physics-based pairs."""

    def __init__(self,
                 encoder: nn.Module,
                 learning_rate: float = 0.001,
                 temperature: float = 0.07,
                 device: str = None):
        self.encoder = encoder
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)

        self.optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.criterion = PhysicsInfoNCELoss(temperature=temperature)

        self.history = {'loss': [], 'epoch': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.encoder.train()
        total_loss = 0

        for anchor, positive, negatives in dataloader:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negatives = negatives.to(self.device)

            # Get embeddings
            z_anchor = self.encoder(anchor)
            z_positive = self.encoder(positive)

            # Process negatives: (B, N, C, L) -> (B, N, D)
            B, N, C, L = negatives.shape
            negatives_flat = negatives.view(B * N, C, L)
            z_negatives = self.encoder(negatives_flat).view(B, N, -1)

            # Compute loss
            loss = self.criterion(z_anchor, z_positive, z_negatives)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(dataloader)

    def pretrain(self,
                 dataloader: DataLoader,
                 epochs: int = 100,
                 save_path: Optional[Path] = None) -> Dict[str, List[float]]:
        """Run full pretraining."""
        logger.info(f"Starting contrastive pretraining for {epochs} epochs")

        for epoch in range(epochs):
            loss = self.train_epoch(dataloader)
            self.history['loss'].append(loss)
            self.history['epoch'].append(epoch)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history
            }, save_path)
            logger.info(f"Saved pretrained model to {save_path}")

        return self.history


class ContrastiveFineTuner:
    """Handles fine-tuning pretrained encoder for classification."""

    def __init__(self,
                 encoder: nn.Module,
                 num_classes: int,
                 freeze_encoder: bool = False,
                 learning_rate: float = 0.0001,
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Import here to avoid circular dependency
        from packages.core.models.contrastive.classifier import ContrastiveClassifier

        self.model = ContrastiveClassifier(
            encoder, num_classes, freeze_encoder
        ).to(self.device)

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

        self.history = {'train_loss': [], 'val_acc': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for signals, labels in train_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(signals)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, accuracy  # simplified: return acc as both metrics

    def finetune(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 epochs: int = 50) -> Dict[str, List[float]]:
        """Run fine-tuning."""
        logger.info(f"Starting fine-tuning for {epochs} epochs")

        best_acc = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc, _ = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Loss: {train_loss:.4f}, Acc: {val_acc:.4f}")

        logger.info(f"Best validation accuracy: {best_acc:.4f}")
        return self.history


def pretrain_contrastive(
    encoder: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    temperature: float = 0.5,
    device: str = 'cuda'
) -> nn.Module:
    """
    Pretrain encoder using contrastive learning (spectrogram/2D).

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
    # Import here to avoid circular dependency
    from packages.core.models.contrastive.encoder import ContrastiveEncoder

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
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Contrastive Loss: {avg_loss:.4f}")

    # Return pretrained encoder (without projection head)
    return model.encoder
