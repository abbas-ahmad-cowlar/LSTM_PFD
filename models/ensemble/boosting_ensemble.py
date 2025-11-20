"""
Boosting Ensemble for Bearing Fault Diagnosis

Sequential training where each model focuses on samples previous models got wrong.
Implements Adaptive Boosting (AdaBoost) for neural networks.

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from tqdm import tqdm
import copy
import sys
sys.path.append('/home/user/LSTM_PFD')
from models.base_model import BaseModel


class AdaptiveBoosting:
    """
    Adaptive Boosting for neural network models.

    Train models sequentially, with each model focusing on samples
    that previous models misclassified.

    Algorithm:
        1. Initialize sample weights uniformly
        2. For iteration t = 1 to T:
            a. Train model on weighted samples
            b. Compute model weight based on error
            c. Increase weights for misclassified samples
        3. Final prediction: weighted vote

    Args:
        base_model_fn: Function that creates a new model instance
        n_estimators: Number of models to train (default: 5)
        learning_rate: Weight update rate (default: 1.0)
        num_classes: Number of classes (default: 11)

    Example:
        >>> def create_model():
        ...     return CNN1D(num_classes=11)
        >>> boosting = AdaptiveBoosting(create_model, n_estimators=5)
        >>> boosting.fit(train_loader, val_loader, device='cuda')
        >>> predictions = boosting.predict(test_loader)
    """
    def __init__(
        self,
        base_model_fn: Callable,
        n_estimators: int = 5,
        learning_rate: float = 1.0,
        num_classes: int = 11
    ):
        self.base_model_fn = base_model_fn
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.models = []
        self.model_weights = []

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs_per_model: int = 20,
        lr: float = 0.001,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """
        Train boosting ensemble.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs_per_model: Epochs to train each model
            lr: Learning rate for each model
            device: Device to use
            verbose: Print progress
        """
        # Initialize sample weights
        dataset_size = len(train_loader.dataset)
        sample_weights = np.ones(dataset_size) / dataset_size

        for iteration in range(self.n_estimators):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Training Model {iteration+1}/{self.n_estimators}")
                print(f"{'='*60}")

            # Create new model
            model = self.base_model_fn()
            model = model.to(device)

            # Create weighted sampler
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            # Create weighted dataloader
            weighted_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=sampler,
                num_workers=0
            )

            # Train model
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            self._train_model(
                model, weighted_loader, optimizer, criterion,
                num_epochs_per_model, device, verbose
            )

            # Evaluate model on full training set
            train_predictions, train_labels = self._predict_dataset(
                model, train_loader, device
            )

            # Compute error
            errors = (train_predictions != train_labels).astype(float)
            weighted_error = np.sum(sample_weights * errors)

            if weighted_error > 0.5:
                if verbose:
                    print(f"Model {iteration+1} error > 0.5 ({weighted_error:.4f}), stopping boosting")
                break

            # Compute model weight
            epsilon = 1e-10  # Avoid division by zero
            model_weight = self.learning_rate * 0.5 * np.log((1 - weighted_error + epsilon) / (weighted_error + epsilon))

            # Update sample weights
            sample_weights *= np.exp(model_weight * (2 * errors - 1))
            sample_weights /= np.sum(sample_weights)  # Normalize

            # Store model and weight
            self.models.append(model)
            self.model_weights.append(model_weight)

            if verbose:
                print(f"Model {iteration+1} weighted error: {weighted_error:.4f}")
                print(f"Model {iteration+1} weight: {model_weight:.4f}")

                if val_loader is not None:
                    val_acc = self._evaluate(model, val_loader, device)
                    print(f"Model {iteration+1} validation accuracy: {val_acc:.2f}%")

                    # Evaluate ensemble so far
                    ensemble_acc = self.evaluate(val_loader, device)
                    print(f"Ensemble validation accuracy: {ensemble_acc:.2f}%")

    def _train_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        device: str,
        verbose: bool
    ):
        """Train a single model."""
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else dataloader

            for batch in pbar:
                x, y = batch
                x, y = x.to(device), y.to(device)

                # Forward pass
                logits = model(x)
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

                if verbose:
                    pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    def _predict_dataset(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for entire dataset."""
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(device)

                logits = model(x)
                _, predicted = logits.max(1)

                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(y.numpy())

        return np.concatenate(all_predictions), np.concatenate(all_labels)

    def _evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ) -> float:
        """Evaluate a single model."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                logits = model(x)
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        return 100. * correct / total

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ) -> np.ndarray:
        """
        Predict using weighted ensemble.

        Args:
            dataloader: Data loader
            device: Device to use

        Returns:
            Predictions [N]
        """
        all_predictions = []

        # Get predictions from all models
        for model in self.models:
            predictions, _ = self._predict_dataset(model, dataloader, device)
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions)  # [n_estimators, N]

        # Weighted vote
        batch_size = all_predictions.shape[1]
        votes = np.zeros((batch_size, self.num_classes))

        for i, predictions in enumerate(all_predictions):
            weight = self.model_weights[i]
            for j, pred in enumerate(predictions):
                votes[j, pred] += weight

        # Return class with most votes
        ensemble_predictions = np.argmax(votes, axis=1)

        return ensemble_predictions

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ) -> float:
        """
        Evaluate ensemble accuracy.

        Args:
            dataloader: Data loader
            device: Device to use

        Returns:
            Accuracy (%)
        """
        predictions = self.predict(dataloader, device)

        # Get true labels
        all_labels = []
        for batch in dataloader:
            _, y = batch
            all_labels.append(y.numpy())
        labels = np.concatenate(all_labels)

        correct = (predictions == labels).sum()
        total = len(labels)

        return 100. * correct / total


class BoostingEnsemble(BaseModel):
    """
    PyTorch wrapper for AdaptiveBoosting to maintain compatibility with BaseModel.

    Args:
        models: List of trained models
        model_weights: Weights for each model
        num_classes: Number of classes (default: 11)
    """
    def __init__(
        self,
        models: List[nn.Module],
        model_weights: List[float],
        num_classes: int = 11
    ):
        super().__init__()

        self.num_models = len(models)
        self.num_classes = num_classes

        self.models = nn.ModuleList(models)
        self.register_buffer('model_weights', torch.tensor(model_weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weighted voting.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Weighted votes as logits [B, num_classes]
        """
        batch_size = x.size(0)
        votes = torch.zeros(batch_size, self.num_classes, device=x.device)

        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                logits = model(x)
                preds = logits.argmax(dim=1)

                weight = self.model_weights[i]
                votes.scatter_add_(1, preds.unsqueeze(1), torch.full_like(preds.unsqueeze(1).float(), weight))

        return votes

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'BoostingEnsemble',
            'num_models': self.num_models,
            'num_classes': self.num_classes,
            'model_weights': self.model_weights.tolist(),
            'num_parameters': self.get_num_params()
        }


def train_boosting(
    base_model_class: Callable,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    n_estimators: int = 5,
    num_epochs_per_model: int = 20,
    lr: float = 0.001,
    device: str = 'cuda',
    verbose: bool = True
) -> BoostingEnsemble:
    """
    Train a boosting ensemble end-to-end.

    Args:
        base_model_class: Function that creates a new model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_estimators: Number of models to train
        num_epochs_per_model: Epochs per model
        lr: Learning rate
        device: Device to use
        verbose: Print progress

    Returns:
        Trained BoostingEnsemble

    Example:
        >>> def create_cnn():
        ...     return CNN1D(num_classes=11)
        >>> boosting = train_boosting(create_cnn, train_loader, val_loader, n_estimators=5)
    """
    # Create adaptive boosting trainer
    adaptive_boosting = AdaptiveBoosting(
        base_model_fn=base_model_class,
        n_estimators=n_estimators
    )

    # Train boosting ensemble
    adaptive_boosting.fit(
        train_loader, val_loader,
        num_epochs_per_model, lr, device, verbose
    )

    # Convert to BoostingEnsemble
    boosting_ensemble = BoostingEnsemble(
        models=adaptive_boosting.models,
        model_weights=adaptive_boosting.model_weights
    )

    return boosting_ensemble
