"""
Late Fusion for Multi-Modal Bearing Fault Diagnosis

Combine final predictions from multiple models (similar to voting ensemble).

Fusion strategies:
- Weighted Average: Average probability distributions with learned/fixed weights
- Max: Take most confident prediction
- Product: Product rule (multiply probabilities)
- Borda Count: Rank-based voting

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import sys
sys.path.append('/home/user/LSTM_PFD')
from models.base_model import BaseModel


class LateFusion(BaseModel):
    """
    Late fusion model that combines predictions from multiple models.

    Args:
        models: List of trained models
        fusion_method: Fusion strategy ('weighted_average', 'max', 'product', 'borda')
        weights: Optional weights for each model (for weighted_average)
        num_classes: Number of output classes (default: 11)
        learnable_weights: Whether to learn fusion weights (default: False)

    Example:
        >>> models = [cnn_model, resnet_model, transformer_model]
        >>> fusion = LateFusion(models, fusion_method='weighted_average')
        >>> predictions = fusion(x)
    """
    def __init__(
        self,
        models: List[nn.Module],
        fusion_method: str = 'weighted_average',
        weights: Optional[List[float]] = None,
        num_classes: int = 11,
        learnable_weights: bool = False
    ):
        super().__init__()

        if not models:
            raise ValueError("Must provide at least one model")

        self.num_models = len(models)
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.learnable_weights = learnable_weights

        # Store models
        self.models = nn.ModuleList(models)

        # Initialize weights
        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models
        else:
            if len(weights) != self.num_models:
                raise ValueError("Number of weights must match number of models")
            total = sum(weights)
            weights = [w / total for w in weights]

        if learnable_weights:
            self.weights = nn.Parameter(torch.tensor(weights))
        else:
            self.register_buffer('weights', torch.tensor(weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with late fusion.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Fused predictions [B, num_classes]
        """
        # Get predictions from all models
        all_probs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # Apply fusion strategy
        if self.fusion_method == 'weighted_average':
            fused_probs = self._weighted_average(all_probs)

        elif self.fusion_method == 'max':
            fused_probs = self._max_fusion(all_probs)

        elif self.fusion_method == 'product':
            fused_probs = self._product_fusion(all_probs)

        elif self.fusion_method == 'borda':
            fused_probs = self._borda_count(all_probs)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Convert back to logits
        logits = torch.log(fused_probs + 1e-10)

        return logits

    def _weighted_average(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """Weighted average of probabilities."""
        all_probs = torch.stack(all_probs, dim=0)  # [num_models, B, num_classes]

        # Normalize weights
        if self.learnable_weights:
            weights = F.softmax(self.weights, dim=0)
        else:
            weights = self.weights

        weights = weights.view(-1, 1, 1)  # [num_models, 1, 1]

        weighted_probs = (all_probs * weights).sum(dim=0)  # [B, num_classes]

        return weighted_probs

    def _max_fusion(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """Max fusion: Take maximum probability for each class."""
        all_probs = torch.stack(all_probs, dim=0)  # [num_models, B, num_classes]
        max_probs, _ = all_probs.max(dim=0)  # [B, num_classes]

        # Renormalize
        max_probs = max_probs / max_probs.sum(dim=1, keepdim=True)

        return max_probs

    def _product_fusion(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """Product fusion: Multiply probabilities (geometric mean)."""
        all_probs = torch.stack(all_probs, dim=0)  # [num_models, B, num_classes]

        # Product rule
        product_probs = all_probs.prod(dim=0)  # [B, num_classes]

        # Renormalize
        product_probs = product_probs / product_probs.sum(dim=1, keepdim=True)

        return product_probs

    def _borda_count(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """Borda count: Rank-based voting."""
        batch_size = all_probs[0].size(0)
        scores = torch.zeros(batch_size, self.num_classes, device=all_probs[0].device)

        for probs in all_probs:
            # Get rankings (higher prob = higher rank)
            _, rankings = torch.sort(probs, dim=1, descending=True)

            # Assign scores based on rank
            for i in range(self.num_classes):
                score = self.num_classes - i
                class_indices = rankings[:, i]
                scores[torch.arange(batch_size), class_indices] += score

        # Normalize to probabilities
        scores = scores / scores.sum(dim=1, keepdim=True)

        return scores

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'LateFusion',
            'num_models': self.num_models,
            'num_classes': self.num_classes,
            'fusion_method': self.fusion_method,
            'learnable_weights': self.learnable_weights,
            'weights': self.weights.tolist() if not self.learnable_weights else self.weights.data.tolist(),
            'num_parameters': self.get_num_params()
        }


def late_fusion_weighted_average(
    model_predictions: List[np.ndarray],
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Late fusion using weighted average of probability distributions.

    Args:
        model_predictions: List of probability predictions [N, num_classes] from each model
        weights: Optional weights for each model (default: equal weights)

    Returns:
        fused_pred: Predicted classes [N]
        fused_probs: Fused probabilities [N, num_classes]

    Example:
        >>> pred1 = model1.predict_proba(X)  # [100, 11]
        >>> pred2 = model2.predict_proba(X)  # [100, 11]
        >>> pred3 = model3.predict_proba(X)  # [100, 11]
        >>> fused_pred, fused_probs = late_fusion_weighted_average([pred1, pred2, pred3])
    """
    num_models = len(model_predictions)

    if weights is None:
        weights = np.ones(num_models) / num_models
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

    # Weighted average
    fused_probs = np.average(model_predictions, axis=0, weights=weights)
    fused_pred = np.argmax(fused_probs, axis=-1)

    return fused_pred, fused_probs


def late_fusion_max(
    model_predictions: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Late fusion using max (take most confident prediction for each class).

    Args:
        model_predictions: List of probability predictions [N, num_classes] from each model

    Returns:
        fused_pred: Predicted classes [N]
        fused_probs: Fused probabilities [N, num_classes]

    Example:
        >>> pred1 = model1.predict_proba(X)
        >>> pred2 = model2.predict_proba(X)
        >>> fused_pred, fused_probs = late_fusion_max([pred1, pred2])
    """
    # Take maximum probability for each class
    fused_probs = np.maximum.reduce(model_predictions)

    # Renormalize
    fused_probs = fused_probs / fused_probs.sum(axis=1, keepdims=True)

    fused_pred = np.argmax(fused_probs, axis=-1)

    return fused_pred, fused_probs


def late_fusion_product(
    model_predictions: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Late fusion using product rule (multiply probabilities).

    Args:
        model_predictions: List of probability predictions [N, num_classes] from each model

    Returns:
        fused_pred: Predicted classes [N]
        fused_probs: Fused probabilities [N, num_classes]

    Example:
        >>> pred1 = model1.predict_proba(X)
        >>> pred2 = model2.predict_proba(X)
        >>> fused_pred, fused_probs = late_fusion_product([pred1, pred2])
    """
    # Product rule
    fused_probs = np.prod(model_predictions, axis=0)

    # Renormalize
    fused_probs = fused_probs / fused_probs.sum(axis=1, keepdims=True)

    fused_pred = np.argmax(fused_probs, axis=-1)

    return fused_pred, fused_probs


def late_fusion_borda_count(
    model_predictions: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Late fusion using Borda count (rank-based voting).

    Args:
        model_predictions: List of probability predictions [N, num_classes] from each model

    Returns:
        fused_pred: Predicted classes [N]
        scores: Borda count scores [N, num_classes]

    Example:
        >>> pred1 = model1.predict_proba(X)
        >>> pred2 = model2.predict_proba(X)
        >>> fused_pred, scores = late_fusion_borda_count([pred1, pred2])
    """
    n_samples = model_predictions[0].shape[0]
    n_classes = model_predictions[0].shape[1]

    scores = np.zeros((n_samples, n_classes))

    for probs in model_predictions:
        # Get rankings (higher prob = higher rank)
        rankings = np.argsort(probs, axis=1)[:, ::-1]  # Descending order

        # Assign scores based on rank
        for i in range(n_classes):
            score = n_classes - i
            class_indices = rankings[:, i]
            scores[np.arange(n_samples), class_indices] += score

    # Normalize
    scores = scores / scores.sum(axis=1, keepdims=True)

    fused_pred = np.argmax(scores, axis=-1)

    return fused_pred, scores


def create_late_fusion(
    models: List[nn.Module],
    fusion_method: str = 'weighted_average',
    weights: Optional[List[float]] = None,
    num_classes: int = 11,
    learnable_weights: bool = False
) -> LateFusion:
    """
    Factory function to create late fusion model.

    Args:
        models: List of trained models
        fusion_method: Fusion strategy ('weighted_average', 'max', 'product', 'borda')
        weights: Optional weights for each model
        num_classes: Number of output classes
        learnable_weights: Whether to learn fusion weights

    Returns:
        LateFusion instance

    Example:
        >>> models = [cnn_model, resnet_model, transformer_model]
        >>> fusion = create_late_fusion(models, fusion_method='weighted_average')
    """
    return LateFusion(
        models=models,
        fusion_method=fusion_method,
        weights=weights,
        num_classes=num_classes,
        learnable_weights=learnable_weights
    )


def train_late_fusion_weights(
    models: List[nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_classes: int = 11,
    num_epochs: int = 10,
    lr: float = 0.01,
    device: str = 'cuda',
    verbose: bool = True
) -> LateFusion:
    """
    Train learnable fusion weights for late fusion.

    Args:
        models: List of trained models (frozen)
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        num_epochs: Training epochs
        lr: Learning rate
        device: Device to use
        verbose: Print progress

    Returns:
        LateFusion model with learned weights

    Example:
        >>> models = [cnn_model, resnet_model, transformer_model]
        >>> fusion = train_late_fusion_weights(models, train_loader, val_loader)
    """
    # Create late fusion with learnable weights
    fusion = LateFusion(
        models=models,
        fusion_method='weighted_average',
        num_classes=num_classes,
        learnable_weights=True
    )
    fusion = fusion.to(device)

    # Freeze base models
    for model in fusion.models:
        for param in model.parameters():
            param.requires_grad = False

    # Only optimize fusion weights
    optimizer = torch.optim.Adam([fusion.weights], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        fusion.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = fusion(x)
            loss = criterion(logits, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validate
        fusion.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                logits = fusion(x)
                _, predicted = logits.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_acc = 100. * val_correct / val_total

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%")
            print(f"Learned weights: {F.softmax(fusion.weights, dim=0).detach().cpu().numpy()}")

    return fusion
