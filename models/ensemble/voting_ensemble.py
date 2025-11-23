"""
Voting Ensemble for Bearing Fault Diagnosis

Combines predictions from multiple models using voting strategies:
- Soft voting: Average of predicted probabilities
- Hard voting: Majority class vote
- Weighted voting: Models have different importance

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import accuracy_score
from itertools import product
import sys
sys.path.append('/home/user/LSTM_PFD')
from models.base_model import BaseModel


class VotingEnsemble(BaseModel):
    """
    Voting ensemble that combines predictions from multiple models.

    Supports:
    - Hard voting: Majority class vote
    - Soft voting: Average of predicted probabilities
    - Weighted voting: Different model importance

    Args:
        models: List of trained models
        weights: Optional weights for each model (for weighted voting)
        voting_type: 'hard' or 'soft' (default: 'soft')
        num_classes: Number of output classes (default: 11)

    Example:
        >>> models = [cnn_model, resnet_model, transformer_model]
        >>> ensemble = VotingEnsemble(models, voting_type='soft')
        >>> predictions = ensemble(x)
    """
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        voting_type: str = 'soft',
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()

        if not models:
            raise ValueError("Must provide at least one model")

        self.num_models = len(models)
        self.num_classes = num_classes
        self.voting_type = voting_type

        # Store models as ModuleList for proper registration
        self.models = nn.ModuleList(models)

        # Normalize weights
        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models
        else:
            if len(weights) != self.num_models:
                raise ValueError("Number of weights must match number of models")
            total = sum(weights)
            weights = [w / total for w in weights]

        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and aggregate predictions.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Aggregated predictions [B, num_classes]
        """
        if self.voting_type == 'soft':
            # Soft voting: weighted average of probabilities
            all_probs = []

            for model in self.models:
                model.eval()  # Ensure eval mode for consistency
                with torch.no_grad():
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    all_probs.append(probs)

            # Stack and weighted average
            all_probs = torch.stack(all_probs, dim=0)  # [num_models, B, num_classes]
            weights = self.weights.view(-1, 1, 1)  # [num_models, 1, 1]

            weighted_probs = (all_probs * weights).sum(dim=0)  # [B, num_classes]

            # Convert back to logits for consistency
            logits = torch.log(weighted_probs + 1e-10)

            return logits

        else:  # hard voting
            # Hard voting: majority class vote
            all_preds = []

            for model in self.models:
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    all_preds.append(preds)

            all_preds = torch.stack(all_preds, dim=0)  # [num_models, B]

            # Count votes for each class
            batch_size = x.size(0)
            votes = torch.zeros(batch_size, self.num_classes, device=x.device)

            for i, preds in enumerate(all_preds):
                weight = self.weights[i]
                votes.scatter_add_(1, preds.unsqueeze(1), torch.full_like(preds.unsqueeze(1).float(), weight))

            # Return votes as logits
            return votes

    def predict_proba(self, dataloader: torch.utils.data.DataLoader, device: str = 'cuda') -> np.ndarray:
        """
        Get probability predictions for entire dataset.

        Args:
            dataloader: Data loader
            device: Device to use

        Returns:
            Probability predictions [N, num_classes]
        """
        self.eval()
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(device)
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'VotingEnsemble',
            'num_models': self.num_models,
            'num_classes': self.num_classes,
            'voting_type': self.voting_type,
            'weights': self.weights.tolist(),
            'num_parameters': self.get_num_params()
        }


def soft_voting(predictions_list: List[np.ndarray], weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft voting: Average probability distributions.

    Args:
        predictions_list: List of probability predictions [N, num_classes] from each model
        weights: Optional weights for each model (default: equal weights)

    Returns:
        ensemble_pred: Predicted classes [N]
        ensemble_probs: Ensemble probabilities [N, num_classes]

    Example:
        >>> pred1 = model1.predict_proba(X)  # [100, 11]
        >>> pred2 = model2.predict_proba(X)  # [100, 11]
        >>> ensemble_pred, ensemble_probs = soft_voting([pred1, pred2])
    """
    num_models = len(predictions_list)

    if weights is None:
        weights = np.ones(num_models) / num_models
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

    # Weighted average of probabilities
    ensemble_probs = np.average(predictions_list, axis=0, weights=weights)
    ensemble_pred = np.argmax(ensemble_probs, axis=-1)

    return ensemble_pred, ensemble_probs


def hard_voting(predictions_list: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Hard voting: Majority vote across models.

    Args:
        predictions_list: List of class predictions [N] from each model
        weights: Optional weights for each model (default: equal weights)

    Returns:
        ensemble_pred: Predicted classes [N]

    Example:
        >>> pred1 = model1.predict(X)  # [100]
        >>> pred2 = model2.predict(X)  # [100]
        >>> ensemble_pred = hard_voting([pred1, pred2])
    """
    num_models = len(predictions_list)
    num_samples = len(predictions_list[0])

    if weights is None:
        weights = np.ones(num_models)
    else:
        weights = np.array(weights)

    # Count weighted votes for each class
    num_classes = int(max([pred.max() for pred in predictions_list])) + 1
    votes = np.zeros((num_samples, num_classes))

    for i, preds in enumerate(predictions_list):
        for j, pred in enumerate(preds):
            votes[j, pred] += weights[i]

    # Return class with most votes
    ensemble_pred = np.argmax(votes, axis=1)

    return ensemble_pred


def optimize_ensemble_weights(
    models: List[nn.Module],
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    search_resolution: int = 10
) -> np.ndarray:
    """
    Find optimal weights for ensemble via grid search on validation set.

    Args:
        models: List of models
        val_loader: Validation data loader
        device: Device to use
        search_resolution: Number of weight values to try per model (default: 10)

    Returns:
        optimal_weights: Best weights [num_models]

    Example:
        >>> models = [cnn_model, resnet_model, transformer_model]
        >>> weights = optimize_ensemble_weights(models, val_loader)
        >>> ensemble = VotingEnsemble(models, weights=weights)
    """
    num_models = len(models)

    # Get predictions from all models
    print("Generating predictions from all models...")
    all_predictions = []
    true_labels = []

    for model in models:
        model.eval()
        model_preds = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x = batch
                    y = None

                x = x.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                model_preds.append(probs.cpu().numpy())

                if y is not None and len(true_labels) < len(val_loader.dataset):
                    true_labels.append(y.cpu().numpy())

        all_predictions.append(np.concatenate(model_preds, axis=0))

    true_labels = np.concatenate(true_labels, axis=0)

    # Grid search over weight space
    print(f"Searching weight space (resolution={search_resolution})...")
    weight_values = np.linspace(0.1, 1.0, search_resolution)

    best_accuracy = 0.0
    best_weights = None

    # Generate all weight combinations
    total_combinations = search_resolution ** num_models
    print(f"Testing {total_combinations} weight combinations...")

    for weights in product(weight_values, repeat=num_models):
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Soft voting with these weights
        _, ensemble_probs = soft_voting(all_predictions, weights)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        # Evaluate accuracy
        accuracy = accuracy_score(true_labels, ensemble_preds)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Optimal weights: {best_weights}")

    return best_weights


# Factory function for compatibility
def create_voting_ensemble(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
    voting_type: str = 'soft',
    num_classes: int = NUM_CLASSES
) -> VotingEnsemble:
    """
    Factory function to create voting ensemble.

    Args:
        models: List of trained models
        weights: Optional weights for each model
        voting_type: 'hard' or 'soft'
        num_classes: Number of output classes

    Returns:
        VotingEnsemble instance
    """
    return VotingEnsemble(
        models=models,
        weights=weights,
        voting_type=voting_type,
        num_classes=num_classes
    )
