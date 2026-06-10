"""
Ensemble methods.

Soft/hard voting is the single kept ensemble strategy
(stacking/boosting/MoE were pruned in the 2026-06 convergence).
"""

from .voting_ensemble import (
    VotingEnsemble,
    create_voting_ensemble,
    optimize_ensemble_weights,
    soft_voting,
    hard_voting
)

__all__ = [
    'VotingEnsemble',
    'create_voting_ensemble',
    'optimize_ensemble_weights',
    'soft_voting',
    'hard_voting',
]
