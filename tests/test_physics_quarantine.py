"""Quarantine guards for the inert / non-authoritative physics paths.

P6 remediation Step 3 (2026-06-14), HARDENED after the 2026-06-16 external audit
(Finding 10 / Rec 5). The generic PINN physics path was never replaced by the
band-energy loss — that landed in the *model method*
`PhysicsConstrainedCNN.compute_physics_loss` (differentiability tested in
`tests/test_physics_band_energy_loss.py`). Because the generic path stays
non-differentiable (argmax), leaving it *callable* is a footgun: a future user
could wire it into training and believe it does physics. So it is now
HARD-BLOCKED — these tests assert it RAISES instead of silently doing nothing:

  * `FrequencyConsistencyLoss` / `PhysicalConstraintLoss` `.forward()` raise.
  * `PINNTrainer(lambda_physics>0)` raises at construction; the CE-only path
    (`lambda_physics=0`) still constructs and is valid.
  * `scripts/research/pinn_ablation.py::run_ablation_study` refuses to run.
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


def test_frequency_consistency_loss_is_hard_blocked():
    """Inert (argmax) → now RAISES rather than contributing zero gradient."""
    signal, logits = _inputs()
    with pytest.raises(RuntimeError, match="quarantined and hard-blocked"):
        FrequencyConsistencyLoss(sample_rate=20480)(signal, logits)


def test_physical_constraint_loss_is_hard_blocked():
    """The combined inert loss RAISES rather than return a no-grad tensor."""
    signal, logits = _inputs()
    with pytest.raises(RuntimeError, match="quarantined and hard-blocked"):
        PhysicalConstraintLoss()(signal, logits)


def test_pinn_trainer_physics_mode_is_hard_blocked():
    """PINNTrainer(lambda_physics>0) refuses to construct; CE-only still works."""
    from packages.core.training.pinn_trainer import PINNTrainer
    from packages.core.models.cnn.cnn_1d import CNN1D

    model = CNN1D(num_classes=11, input_channels=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(RuntimeError, match="quarantined and hard-blocked"):
        PINNTrainer(model=model, optimizer=opt, lambda_physics=0.5)

    # The CE-only path (lambda_physics=0) is unaffected and still constructs.
    trainer = PINNTrainer(model=model, optimizer=opt, lambda_physics=0.0)
    assert trainer.lambda_physics == 0.0


def test_pinn_ablation_script_is_quarantined():
    """The stale research ablation refuses to run (non-authoritative, broken)."""
    from scripts.research.pinn_ablation import run_ablation_study
    with pytest.raises(RuntimeError, match="QUARANTINED"):
        run_ablation_study([])
