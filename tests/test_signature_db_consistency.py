"""DB <-> data consistency, aligned to the FROZEN HEALTHY REFERENCE (the systemic guard).

Asserts that, judged against the SAME healthy-class reference the band-energy
physics loss uses (`healthy_reference.json`), each fault class's signals carry
MORE energy in that class's expected bands than a healthy bearing does — so the
loss assigns each real fault a LOWER band-energy penalty for its own class than it
assigns to healthy signals. This locks the physics the models assume
(FaultSignatureDatabase + the healthy baseline) to the physics the generator
produces (PHYSICS.md §4), so the rolling-element / wrong-bearing-type drift found
2026-06-14 cannot silently recur, AND keeps the loss and this CI test on one
reference (owner directive, 2026-06-14). The penalty here mirrors the loss exactly
(per band: relu(1 - frac/H_ref), averaged over the class's bands).
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from packages.core.models.physics.fault_signatures import (
    FaultSignatureDatabase, load_healthy_reference)
from utils.constants import FAULT_TYPES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data/generated/dataset_v2.h5'
FS, WINDOW = 20480, 20480
NAMES = list(FAULT_TYPES)
REF = load_healthy_reference()

pytestmark = pytest.mark.skipif(
    not DATA.exists() or REF is None,
    reason="dataset_v2.h5 or healthy_reference.json not present")


def _class_penalty(psd, freqs, omega, name, db):
    """Band-energy penalty for `name` on one window — mirrors compute_physics_loss."""
    sig = db.signatures[name]
    rc = REF[name]
    hrefs = list(rc['tonal']) + list(rc['bands_hz'])
    bands = [(m * omega * (1 - hw), m * omega * (1 + hw)) for m, hw in sig.tonal]
    bands += [(lo, hi) for lo, hi in sig.bands_hz]
    if not bands:
        return 0.0
    e_total = psd.sum() + 1e-12
    terms = []
    for (lo, hi), href in zip(bands, hrefs):
        mask = (freqs >= lo) & (freqs <= hi)
        frac = psd[mask].sum() / e_total
        terms.append(max(0.0, 1.0 - frac / (href + 1e-12)))
    return float(np.mean(terms))


@pytest.fixture(scope="module")
def mean_self_vs_healthy():
    """Per fault class: (mean own-class penalty on its signals, on healthy signals)."""
    db = FaultSignatureDatabase()
    freqs = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    h_idx = NAMES.index('sain')
    with h5py.File(DATA, 'r') as f:
        g = f['test']
        labels = g['labels'][:]
        rpms = np.array([float(json.loads(m)['speed_rpm']) for m in g['metadata'][:]])
        signals = g['signals']
        wpr = signals.shape[1] // WINDOW

        def pen_over(records, name):
            acc, n = 0.0, 0
            for rec in records:
                omega = rpms[rec] / 60.0
                sigfull = signals[rec]
                for wi in range(wpr):
                    w = sigfull[wi * WINDOW:(wi + 1) * WINDOW]
                    psd = np.abs(np.fft.rfft(w)) ** 2
                    acc += _class_penalty(psd, freqs, omega, name, db)
                    n += 1
            return acc / n

        healthy_recs = np.where(labels == h_idx)[0][:15]
        out = {}
        for ci, name in enumerate(NAMES):
            if name == 'sain':
                continue
            recs = np.where(labels == ci)[0][:15]
            out[name] = (pen_over(recs, name), pen_over(healthy_recs, name))
    return out


def test_reference_covers_all_classes():
    """Frozen reference exists for every class with bands parallel to the DB."""
    db = FaultSignatureDatabase()
    for name, sig in db.signatures.items():
        assert name in REF, f"{name}: missing from healthy_reference.json"
        assert len(REF[name]['tonal']) == len(sig.tonal)
        assert len(REF[name]['bands_hz']) == len(sig.bands_hz)


def test_all_classes_have_a_signature():
    """Every non-healthy class resolves to bands (no KeyError, no empty)."""
    db = FaultSignatureDatabase()
    for name, sig in db.signatures.items():
        if name == 'sain':
            assert not sig.tonal and not sig.bands_hz
        else:
            assert sig.tonal or sig.bands_hz, f"{name}: no DB signature"


@pytest.mark.parametrize("name", [n for n in NAMES if n != 'sain'])
def test_fault_penalty_below_healthy(mean_self_vs_healthy, name):
    """Each fault's own-class band-energy penalty is LOWER on its real signals than
    on healthy signals — i.e. its signature is genuinely present above the healthy
    baseline (the premise the band-energy loss relies on)."""
    self_pen, healthy_pen = mean_self_vs_healthy[name]
    assert self_pen < healthy_pen, (
        f"{name}: own-class penalty {self_pen:.3f} not below healthy {healthy_pen:.3f} "
        f"— signature not elevated above the frozen healthy reference")
