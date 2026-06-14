"""Gradient + per-class tests for the ratified band-energy physics loss.

P6 remediation Step 4 (2026-06-14). Pins the contract of
`PhysicsConstrainedCNN.compute_physics_loss` (band-energy consistency,
PROTOCOL §7): it is differentiable (the inert/tonal-only versions were the
defects in Findings 5-6), it constrains tonal AND broadband classes, it uses
per-sample rpm, and the healthy class carries no penalty. Signals are
synthetic with known spectral content so the expected ordering is deterministic.
"""
import math

import pytest
import torch

from packages.core.models.pinn.physics_constrained_cnn import PhysicsConstrainedCNN
from utils.constants import FAULT_TYPES

FS = 20480          # model default sample_rate
N = 2048            # = n_fft → no truncation; 10 Hz/bin
NAMES = list(FAULT_TYPES)
I_SAIN = NAMES.index('sain')
I_IMBAL = NAMES.index('desequilibre')   # pure 1X tone
I_CAV = NAMES.index('cavitation')       # 1.4–2.6 kHz broadband band


@pytest.fixture(scope="module")
def model():
    return PhysicsConstrainedCNN(backbone='cnn1d', sample_rate=FS)


def tone(freq_hz, b=1, n=N):
    t = torch.arange(n, dtype=torch.float32) / FS
    return torch.sin(2 * math.pi * freq_hz * t).unsqueeze(0).repeat(b, 1)  # [b, n]


def _onehot(c, b=1, num=len(NAMES)):
    z = torch.full((b, num), -10.0)
    z[:, c] = 10.0
    return z  # softmax ≈ indicator(c) → loss ≈ pen[:, c].mean()


def loss_for_class(model, signal, c, metadata=None):
    loss, _ = model.compute_physics_loss(signal, _onehot(c, b=signal.shape[0]), metadata)
    return float(loss)


def test_loss_is_differentiable_with_grad_to_logits(model):
    """The defect (Findings 5-6) was no gradient. Assert grad flows to logits."""
    signal = tone(60.0, b=2)
    logits = torch.randn(2, len(NAMES), requires_grad=True)
    loss, d = model.compute_physics_loss(signal, logits)
    assert loss.requires_grad and loss.grad_fn is not None
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert 'band_energy_consistency' in d


def test_healthy_class_has_no_band_penalty(model):
    """'sain' has no expected bands → zero penalty regardless of signal."""
    assert loss_for_class(model, tone(60.0), I_SAIN) == pytest.approx(0.0, abs=1e-6)


def test_tonal_class_present_vs_absent(model):
    """A 1X (60 Hz) tone is consistent with imbalance, not with cavitation."""
    sig = tone(60.0)
    l_imbal = loss_for_class(model, sig, I_IMBAL)   # 1X band lit → low
    l_cav = loss_for_class(model, sig, I_CAV)       # HF band empty → high
    assert l_imbal < 0.1
    assert l_cav > 0.5
    assert l_imbal < l_cav


def test_broadband_class_present_vs_absent(model):
    """A 2 kHz tone lands in the cavitation band, not the 1X imbalance band."""
    sig = tone(2000.0)
    l_cav = loss_for_class(model, sig, I_CAV)        # HF band lit → low
    l_imbal = loss_for_class(model, sig, I_IMBAL)    # 1X band empty → high
    assert l_cav < 0.1
    assert l_imbal > 0.5
    assert l_cav < l_imbal


def test_per_sample_rpm_shifts_the_bands(model):
    """Same 60 Hz tone: consistent with imbalance at 3600 rpm (1X=60 Hz), not at
    5400 rpm (1X=90 Hz). Proves the loss uses per-sample rpm, not a fixed 3600."""
    sig = tone(60.0)
    l_3600 = loss_for_class(model, sig, I_IMBAL, {'rpm': torch.tensor([3600.0])})
    l_5400 = loss_for_class(model, sig, I_IMBAL, {'rpm': torch.tensor([5400.0])})
    assert l_3600 < 0.1
    assert l_5400 > 0.5
    assert l_3600 < l_5400
