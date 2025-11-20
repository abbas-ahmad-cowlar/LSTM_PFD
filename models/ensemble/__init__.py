"""
Ensemble Methods for Bearing Fault Diagnosis

This module contains various ensemble learning techniques:
- Voting: Soft/hard voting across models
- Stacking: Meta-learner on base predictions
- Boosting: Sequential boosting for hard examples
- Mixture of Experts: Gating network selects specialized models
"""

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
]
