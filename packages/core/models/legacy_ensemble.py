"""
Legacy Ensemble — backward-compatible re-export shim.

Canonical implementations live in ``ensemble/`` sub-package.
This module re-exports under the old names so that existing imports
(``from .legacy_ensemble import ...``) keep working.
"""

from .ensemble.voting_ensemble import VotingEnsemble  # noqa: F401
from .ensemble.stacking_ensemble import StackingEnsemble  # noqa: F401


# Backward-compat aliases
StackedEnsemble = StackingEnsemble
EnsembleModel = VotingEnsemble  # simplified — legacy EnsembleModel was a wrapper


def create_voting_ensemble(models, weights=None, voting_type='soft', num_classes=11):
    """Backward-compat factory — delegates to VotingEnsemble."""
    return VotingEnsemble(models=models, num_classes=num_classes)


def create_stacked_ensemble(base_models, meta_learner=None, num_classes=11):
    """Backward-compat factory — delegates to StackingEnsemble."""
    return StackingEnsemble(base_models=base_models, num_classes=num_classes)
