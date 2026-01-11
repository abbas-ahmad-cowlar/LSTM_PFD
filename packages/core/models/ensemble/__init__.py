"""
Ensemble Methods for Bearing Fault Diagnosis

This module contains various ensemble learning techniques:
- Voting: Soft/hard voting across models
- Stacking: Meta-learner on base predictions
- Boosting: Sequential boosting for hard examples
- Mixture of Experts: Gating network selects specialized models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .voting_ensemble import (
    VotingEnsemble,
    optimize_ensemble_weights,
    soft_voting,
    hard_voting
)
from .stacking_ensemble import (
    StackingEnsemble,
    train_stacking,
    create_meta_features
)
from .boosting_ensemble import (
    BoostingEnsemble,
    AdaptiveBoosting,
    train_boosting
)
from .mixture_of_experts import (
    MixtureOfExperts,
    GatingNetwork,
    ExpertModel
)
from .model_selector import (
    DiversityBasedSelector,
    select_diverse_models
)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> float:
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use

    Returns:
        Accuracy as percentage (0-100)

    Example:
        >>> accuracy = evaluate(ensemble, test_loader)
        >>> print(f"Test Accuracy: {accuracy:.2f}%")
    """
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                raise ValueError("Dataloader must return (x, y) tuples")

            x, y = x.to(device), y.to(device)

            logits = model(x)
            _, predicted = logits.max(1)

            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    accuracy = 100. * correct / total
    return accuracy


__all__ = [
    # Voting
    'VotingEnsemble',
    'optimize_ensemble_weights',
    'soft_voting',
    'hard_voting',

    # Stacking
    'StackingEnsemble',
    'train_stacking',
    'create_meta_features',

    # Boosting
    'BoostingEnsemble',
    'AdaptiveBoosting',
    'train_boosting',

    # Mixture of Experts
    'MixtureOfExperts',
    'GatingNetwork',
    'ExpertModel',

    # Model Selection
    'DiversityBasedSelector',
    'select_diverse_models',

    # Helper functions
    'evaluate',
]
