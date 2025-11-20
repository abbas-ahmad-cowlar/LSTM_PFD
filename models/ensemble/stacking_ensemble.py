"""
Stacking Ensemble for Bearing Fault Diagnosis

Meta-learner trained on base model predictions.
Architecture:
  Base Models → Meta-Features → Meta-Learner → Final Prediction

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import sys
sys.path.append('/home/user/LSTM_PFD')
from models.base_model import BaseModel


class StackingEnsemble(BaseModel):
    """
    Stacked ensemble with meta-learner.

    Uses predictions from base models as features for a meta-learner.

    Architecture:
        Base Models: [ResNet-1D, Transformer, PINN, ResNet-2D, Random Forest]
          ↓ Generate predictions on validation set
        Meta-Features: [B, 5 × 11] = [B, 55]  # 5 models × 11 class probabilities
          ↓ Meta-Learner (Logistic Regression or MLP)
        Final Prediction: [B, 11]

    Args:
        base_models: List of trained base models
        meta_learner: Meta-learner model (simple FC network)
        num_classes: Number of output classes (default: 11)

    Example:
        >>> base_models = [cnn_model, resnet_model, transformer_model]
        >>> meta_learner = nn.Sequential(nn.Linear(33, 64), nn.ReLU(), nn.Linear(64, 11))
        >>> stacking = StackingEnsemble(base_models, meta_learner)
        >>> predictions = stacking(x)
    """
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: Optional[nn.Module] = None,
        num_classes: int = 11
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
        device: str = 'cuda',
        verbose: bool = True
    ) -> Dict:
        """
        Train only the meta-learner (base models are frozen).

        Args:
            dataloader: Training data loader
            optimizer: Optimizer for meta-learner
            criterion: Loss function
            num_epochs: Number of training epochs
            device: Device to use
            verbose: Print training progress

        Returns:
            Training history dict
        """
        self.meta_learner.train()
        history = {'loss': [], 'accuracy': []}

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else dataloader

            for batch in pbar:
                x, y = batch
                x, y = x.to(device), y.to(device)

                # Forward pass
                logits = self.forward(x)
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

            avg_loss = total_loss / len(dataloader)
            accuracy = 100. * correct / total

            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return history

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'StackingEnsemble',
            'num_models': self.num_models,
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_params()
        }


def create_meta_features(
    base_models: List[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    return_probabilities: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate meta-features from base model predictions.

    Args:
        base_models: List of base models
        dataloader: Data loader
        device: Device to use
        return_probabilities: Return probabilities instead of logits

    Returns:
        meta_features: Meta-features [N, num_models * num_classes]
        labels: True labels [N]

    Example:
        >>> base_models = [cnn_model, resnet_model]
        >>> meta_features, labels = create_meta_features(base_models, train_loader)
        >>> # meta_features: [1000, 22] for 2 models × 11 classes
    """
    all_features = []
    all_labels = []

    for model in base_models:
        model.eval()
        model_outputs = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x = batch
                    y = None

                x = x.to(device)
                logits = model(x)

                if return_probabilities:
                    outputs = F.softmax(logits, dim=1)
                else:
                    outputs = logits

                model_outputs.append(outputs.cpu().numpy())

                # Collect labels only once
                if len(all_labels) == 0 and y is not None:
                    all_labels.append(y.cpu().numpy())

        all_features.append(np.concatenate(model_outputs, axis=0))

    # Concatenate features from all models
    meta_features = np.concatenate(all_features, axis=1)

    # Concatenate labels
    if all_labels:
        labels = np.concatenate(all_labels, axis=0)
    else:
        labels = None

    return meta_features, labels


def train_stacking(
    base_models: List[nn.Module],
    meta_learner: Union[nn.Module, str],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_classes: int = 11,
    num_epochs: int = 20,
    lr: float = 0.001,
    device: str = 'cuda',
    verbose: bool = True
) -> StackingEnsemble:
    """
    Train a stacking ensemble end-to-end.

    Step 1: Generate meta-features from base models on training set
    Step 2: Train meta-learner on meta-features
    Step 3: Validate on validation set

    Args:
        base_models: List of trained base models
        meta_learner: Meta-learner model or 'mlp'/'logistic'
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        num_epochs: Training epochs for meta-learner
        lr: Learning rate
        device: Device to use
        verbose: Print progress

    Returns:
        Trained StackingEnsemble

    Example:
        >>> base_models = [cnn_model, resnet_model, transformer_model]
        >>> stacking = train_stacking(base_models, 'mlp', train_loader, val_loader)
        >>> test_acc = evaluate(stacking, test_loader)
    """
    if verbose:
        print("Step 1: Generating meta-features from base models...")

    # Generate meta-features on training set
    meta_features_train, labels_train = create_meta_features(
        base_models, train_loader, device, return_probabilities=True
    )

    # Generate meta-features on validation set
    meta_features_val, labels_val = create_meta_features(
        base_models, val_loader, device, return_probabilities=True
    )

    if verbose:
        print(f"Meta-features shape: {meta_features_train.shape}")

    # Create meta-learner if string is provided
    if isinstance(meta_learner, str):
        meta_input_dim = len(base_models) * num_classes

        if meta_learner == 'mlp':
            meta_learner = nn.Sequential(
                nn.Linear(meta_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        elif meta_learner == 'logistic':
            meta_learner = nn.Linear(meta_input_dim, num_classes)
        else:
            raise ValueError(f"Unknown meta-learner type: {meta_learner}")

    # Create stacking ensemble
    stacking = StackingEnsemble(base_models, meta_learner, num_classes)
    stacking = stacking.to(device)

    # Prepare optimizer and criterion
    optimizer = torch.optim.Adam(stacking.meta_learner.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        print("\nStep 2: Training meta-learner...")

    # Train meta-learner
    history = stacking.train_meta_learner(
        train_loader, optimizer, criterion, num_epochs, device, verbose
    )

    # Validate
    if verbose:
        print("\nStep 3: Validating ensemble...")
        stacking.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                logits = stacking(x)
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        val_accuracy = 100. * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

    return stacking


# Factory function for compatibility
def create_stacked_ensemble(
    base_models: List[nn.Module],
    meta_learner: Optional[nn.Module] = None,
    num_classes: int = 11
) -> StackingEnsemble:
    """
    Factory function to create stacked ensemble.

    Args:
        base_models: List of trained base models
        meta_learner: Optional meta-learner model
        num_classes: Number of output classes

    Returns:
        StackingEnsemble instance
    """
    return StackingEnsemble(
        base_models=base_models,
        meta_learner=meta_learner,
        num_classes=num_classes
    )
