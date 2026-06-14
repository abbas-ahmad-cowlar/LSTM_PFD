"""Gradient + per-class tests for the band-energy physics loss vs the HEALTHY
reference (P6 remediation Step 4, owner-ratified 2026-06-14).

Pins the contract of `PhysicsConstrainedCNN.compute_physics_loss`: differentiable
(the inert/tonal-only versions were Findings 5-6), constrains tonal AND broadband
classes, uses per-sample rpm, healthy carries no penalty, and judges "signature
present" as energy ABOVE the healthy baseline. Tests inject a SYNTHETIC healthy
reference so they are deterministic and independent of the dataset; the real
frozen reference (`healthy_reference.json`) is exercised by the before/after
penalty audit and `tests/test_signature_db_consistency.py`.
"""
import math

import pytest
import torch

from packages.core.models.pinn.physics_constrained_cnn import PhysicsConstrainedCNN
from utils.constants import FAULT_TYPES

FS = 20480          # model default sample_rate
N = 20480           # full 1-s window → 1 Hz/bin (resolves low-freq bands)
NAMES = list(FAULT_TYPES)
I_SAIN = NAMES.index('sain')
I_IMBAL = NAMES.index('desequilibre')   # pure 1X tone
I_CAV = NAMES.index('cavitation')       # 1.4–2.6 kHz broadband band
H = 0.01            # synthetic healthy-reference fraction per band


@pytest.fixture(scope="module")
def model():
    m = PhysicsConstrainedCNN(backbone='cnn1d', sample_rate=FS)
    # deterministic synthetic reference: every band's healthy level = H
    m.healthy_reference = {
        name: {'tonal': [H] * len(sig.tonal), 'bands_hz': [H] * len(sig.bands_hz)}
        for name, sig in m.signature_db.signatures.items()
    }
    return m


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


def test_missing_reference_raises():
    """Without the frozen reference the loss refuses to run (no flat fallback)."""
    m = PhysicsConstrainedCNN(backbone='cnn1d', sample_rate=FS)
    m.healthy_reference = None
    with pytest.raises(RuntimeError, match="healthy reference"):
        m.compute_physics_loss(tone(60.0), _onehot(I_IMBAL))


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
    l_imbal = loss_for_class(model, sig, I_IMBAL)   # 1X band lit above healthy → low
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
