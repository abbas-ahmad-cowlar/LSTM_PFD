"""Quarantine guards for the inert / non-authoritative physics paths.

P6 remediation Step 3 (2026-06-14). These tests PIN the known-bad state of the
quarantined physics machinery so it cannot be silently used as if it worked:

  * `FrequencyConsistencyLoss` / `PhysicalConstraintLoss` are non-differentiable
    (argmax) → zero gradient (external audit Finding 5). We assert they stay inert.
  * `scripts/research/pinn_ablation.py::run_ablation_study` is broken/misleading
    (builds HybridPINN with non-existent args, trains CE-only) → we assert it
    refuses to run.

WHEN THE BAND-ENERGY LOSS LANDS (Step 4): the first two tests must be REPLACED
with their opposites — assert `requires_grad is True`, nonzero parameter grads,
and per-class (tonal / broadband / mixed) behavior. Leaving these green after
Step 4 would mean the inert loss was not actually replaced.
"""
import pytest
import torch

from packages.core.training.physics_loss_functions import (
    FrequencyConsistencyLoss,
    PhysicalConstraintLoss,
)


def _inputs():
    torch.manual_seed(0)
    signal = torch.randn(4, 1, 10240)
    logits = torch.randn(4, 11, requires_grad=True)  # a model WOULD make these
    return signal, logits


def test_frequency_consistency_loss_is_non_differentiable():
    """Inert: argmax severs the graph → no grad to logits (audit Finding 5)."""
    signal, logits = _inputs()
    loss = FrequencyConsistencyLoss(sample_rate=20480)(signal, logits)
    assert loss.requires_grad is False, (
        "FrequencyConsistencyLoss became differentiable — if you fixed it, replace "
        "this quarantine guard with a requires_grad=True + nonzero-grad assertion."
    )
    assert loss.grad_fn is None
    with pytest.raises(RuntimeError):
        loss.backward()  # 'does not require grad and does not have a grad_fn'


def test_physical_constraint_loss_is_inert():
    """The combined loss inherits the inert frequency term → no gradient."""
    signal, logits = _inputs()
    total, _ = PhysicalConstraintLoss()(signal, logits)
    assert total.requires_grad is False
    # and it carries no gradient back to the logits a model produced
    assert logits.grad is None


def test_pinn_ablation_script_is_quarantined():
    """The stale research ablation refuses to run (non-authoritative, broken)."""
    from scripts.research.pinn_ablation import run_ablation_study
    with pytest.raises(RuntimeError, match="QUARANTINED"):
        run_ablation_study([])
