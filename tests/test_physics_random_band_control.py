"""Contract tests for the §8.8 RANDOM-BAND control (pre-registered + owner-ratified
2026-06-23): the matched-structure NON-PHYSICS control for the noise-robustness
result.

§8.7 (F9 scramble) used WRONG REAL bands; §8.8 uses RANDOM NON-FAULT bands of
identical count + width. These tests pin: the frozen artifact is structure-matched
to the real DB, the random bands are provably non-physical (no overlap with any real
characteristic band, at any rpm), and the loss with the control active stays
differentiable and actually changes vs the validated loss. Default (control off) is
byte-identical to the validated path — asserted here and by the band-energy tests.
"""
import pytest
import torch

from packages.core.models.pinn.physics_constrained_cnn import PhysicsConstrainedCNN
from packages.core.models.physics.fault_signatures import (
    FaultSignatureDatabase,
    load_random_reference,
)
from utils.constants import FAULT_TYPES

FS = 20480
NAMES = list(FAULT_TYPES)
I_IMBAL = NAMES.index('desequilibre')   # real signature = pure 1X tone
OMEGA_MIN, OMEGA_MAX = 3240.0 / 60.0, 3960.0 / 60.0   # dataset rpm range / 60


def _overlaps(lo, hi, intervals):
    return any(not (hi <= a or lo >= b) for a, b in intervals)


@pytest.fixture(scope="module")
def random_ref():
    sigs, ref = load_random_reference()
    if sigs is None:
        pytest.skip("random_reference.json not generated "
                    "(run scripts/compute_random_reference.py)")
    return sigs, ref


def tone(freq_hz, b=1, n=FS):
    import math
    t = torch.arange(n, dtype=torch.float32) / FS
    return torch.sin(2 * math.pi * freq_hz * t).unsqueeze(0).repeat(b, 1)


def _onehot(c, b=1, num=len(NAMES)):
    z = torch.full((b, num), -10.0)
    z[:, c] = 10.0
    return z


def test_count_matched_per_class(random_ref):
    """Every class has the SAME number of random tonal + absolute bands as real."""
    sigs, ref = random_ref
    db = FaultSignatureDatabase()
    assert set(sigs) == set(NAMES)
    for nm in NAMES:
        assert len(sigs[nm].tonal) == len(db.signatures[nm].tonal), nm
        assert len(sigs[nm].bands_hz) == len(db.signatures[nm].bands_hz), nm
        # reference vector lengths match the band counts (loss zips them)
        assert len(ref[nm]['tonal']) == len(sigs[nm].tonal), nm
        assert len(ref[nm]['bands_hz']) == len(sigs[nm].bands_hz), nm
    assert sigs['sain'].tonal == [] and sigs['sain'].bands_hz == []


def test_random_bands_are_non_physical(random_ref):
    """No random band overlaps ANY real characteristic band — tonal in multiplier
    space (holds at every rpm), absolute in Hz incl. real tonal footprints."""
    sigs, _ = random_ref
    db = FaultSignatureDatabase()
    real_mult = [(m * (1 - hw), m * (1 + hw))
                 for nm in NAMES for m, hw in db.signatures[nm].tonal]
    real_hz = [(lo, hi) for nm in NAMES for (lo, hi) in db.signatures[nm].bands_hz]
    real_tonal_hz = [(m * OMEGA_MIN * (1 - hw), m * OMEGA_MAX * (1 + hw))
                     for nm in NAMES for m, hw in db.signatures[nm].tonal]
    for nm in NAMES:
        for (m_r, hw) in sigs[nm].tonal:
            assert not _overlaps(m_r * (1 - hw), m_r * (1 + hw), real_mult), (nm, m_r)
        for (lo, hi) in sigs[nm].bands_hz:
            assert not _overlaps(lo, hi, real_hz + real_tonal_hz), (nm, lo, hi)


def test_structure_preserving_remap(random_ref):
    """A shared real band maps to the SAME random band wherever it appears (1X is
    shared by desequilibre/usure/mixed_*; LOW by lubrification/mixed_wear_lube)."""
    sigs, _ = random_ref
    # desequilibre's only tonal is real 1X; usure's first tonal is also real 1X.
    assert sigs['desequilibre'].tonal[0] == sigs['usure'].tonal[0]
    assert sigs['mixed_misalign_imbalance'].tonal[0] == sigs['usure'].tonal[0]
    # lubrification's band (real LOW) is shared by mixed_wear_lube.
    assert sigs['lubrification'].bands_hz[0] == sigs['mixed_wear_lube'].bands_hz[0]


def test_default_model_control_off():
    """A fresh model has the control OFF (validated physics path)."""
    m = PhysicsConstrainedCNN(backbone='cnn1d', sample_rate=FS)
    assert m.random_signature is None and m.random_reference is None


def test_control_changes_loss_and_stays_differentiable(random_ref):
    """With the control active a class is judged against its RANDOM band, not its
    real one: a 60 Hz tone is consistent with imbalance (real 1X) under the validated
    loss but NOT under the random band (~150 Hz) → higher penalty; gradient flows."""
    sigs, ref = random_ref
    m = PhysicsConstrainedCNN(backbone='cnn1d', sample_rate=FS)  # real healthy ref loaded
    sig = tone(60.0)
    l_validated, _ = m.compute_physics_loss(sig, _onehot(I_IMBAL))

    m.random_signature, m.random_reference = sigs, ref
    logits = torch.zeros(1, len(NAMES), requires_grad=True)
    loss, d = m.compute_physics_loss(sig, logits)
    assert loss.requires_grad and loss.grad_fn is not None
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert 'band_energy_consistency' in d

    l_random = float(m.compute_physics_loss(sig, _onehot(I_IMBAL))[0])
    assert float(l_validated) < 0.1      # 60 Hz lights the real 1X band
    assert l_random > 0.5                # but not the random ~150 Hz band
    assert l_random > float(l_validated)
