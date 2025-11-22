"""
Ensemble Models for Bearing Fault Diagnosis

Combines multiple models to improve accuracy and robustness.
Supports multiple ensemble strategies:
- Voting (hard/soft)
- Stacking (meta-learner)
- Weighted averaging

Input: [B, 1, T] where T is signal length
Output: [B, 11] for 11 fault classes
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
from .base_model import BaseModel


class VotingEnsemble(BaseModel):
    """
    Voting ensemble that combines predictions from multiple models.

    Supports:
    - Hard voting: Majority class vote
    - Soft voting: Average of predicted probabilities

    Args:
        models: List of trained models
        weights: Optional weights for each model (for weighted voting)
        voting_type: 'hard' or 'soft' (default: 'soft')
        num_classes: Number of output classes (default: 11)
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

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'VotingEnsemble',
            'num_models': self.num_models,
            'num_classes': self.num_classes,
            'voting_type': self.voting_type,
            'num_parameters': self.get_num_params()
        }


class StackedEnsemble(BaseModel):
    """
    Stacked ensemble with meta-learner.

    Uses predictions from base models as features for a meta-learner.

    Args:
        base_models: List of trained base models
        meta_learner: Meta-learner model (simple FC network)
        num_classes: Number of output classes (default: 11)
    """
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: Optional[nn.Module] = None,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()

        if not base_models:
            raise ValueError("Must provide at least one base model")

        self.num_models = len(base_models)
        self.num_classes = num_classes

        # Store base models
        self.base_models = nn.ModuleList(base_models)

        # Create meta-learner if not provided
        if meta_learner is None:
            # Simple 2-layer FC network
            meta_input_dim = self.num_models * num_classes
            meta_learner = nn.Sequential(
                nn.Linear(meta_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )

        self.meta_learner = meta_learner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through base models and meta-learner.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Meta-learner predictions [B, num_classes]
        """
        # Get predictions from all base models
        all_logits = []

        for model in self.base_models:
            model.eval()  # Keep base models in eval mode
            with torch.no_grad():
                logits = model(x)
                all_logits.append(logits)

        # Concatenate all predictions
        meta_features = torch.cat(all_logits, dim=1)  # [B, num_models * num_classes]

        # Meta-learner prediction
        final_logits = self.meta_learner(meta_features)

        return final_logits

    def train_meta_learner(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 10,
        device: str = 'cuda'
    ):
        """
        Train only the meta-learner (base models are frozen).

        Args:
            dataloader: Training data loader
            optimizer: Optimizer for meta-learner
            criterion: Loss function
            num_epochs: Number of training epochs
            device: Device to use
        """
        self.meta_learner.train()

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch in dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                # Forward pass
                logits = self.forward(x)
                loss = criterion(logits, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'StackedEnsemble',
            'num_models': self.num_models,
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_params()
        }


class EnsembleModel(BaseModel):
    """
    Generic ensemble model that can switch between different strategies.

    Args:
        models: List of models to ensemble
        ensemble_type: Type of ensemble ('voting', 'stacking', 'weighted')
        weights: Optional weights for weighted ensemble
        num_classes: Number of output classes (default: 11)
    """
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_type: str = 'voting',
        weights: Optional[List[float]] = None,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()

        self.ensemble_type = ensemble_type
        self.num_classes = num_classes

        if ensemble_type == 'voting':
            self.ensemble = VotingEnsemble(
                models=models,
                weights=weights,
                voting_type='soft',
                num_classes=num_classes
            )

        elif ensemble_type == 'stacking':
            self.ensemble = StackedEnsemble(
                base_models=models,
                num_classes=num_classes
            )

        elif ensemble_type == 'weighted':
            self.ensemble = VotingEnsemble(
                models=models,
                weights=weights,
                voting_type='soft',
                num_classes=num_classes
            )

        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.ensemble(x)

    def add_model(self, model: nn.Module, weight: float = 1.0):
        """
        Add a new model to the ensemble.

        Args:
            model: Model to add
            weight: Weight for this model (for voting ensemble)
        """
        if self.ensemble_type in ['voting', 'weighted']:
            self.ensemble.models.append(model)
            current_weights = self.ensemble.weights.tolist()
            current_weights.append(weight)
            # Renormalize
            total = sum(current_weights)
            current_weights = [w / total for w in current_weights]
            self.ensemble.weights = torch.tensor(current_weights)
            self.ensemble.num_models += 1

        elif self.ensemble_type == 'stacking':
            self.ensemble.base_models.append(model)
            self.ensemble.num_models += 1
            # Note: Meta-learner needs retraining after adding models

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'EnsembleModel',
            'ensemble_type': self.ensemble_type,
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_params()
        }


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


def create_stacked_ensemble(
    base_models: List[nn.Module],
    meta_learner: Optional[nn.Module] = None,
    num_classes: int = NUM_CLASSES
) -> StackedEnsemble:
    """
    Factory function to create stacked ensemble.

    Args:
        base_models: List of trained base models
        meta_learner: Optional meta-learner model
        num_classes: Number of output classes

    Returns:
        StackedEnsemble instance
    """
    return StackedEnsemble(
        base_models=base_models,
        meta_learner=meta_learner,
        num_classes=num_classes
    )
