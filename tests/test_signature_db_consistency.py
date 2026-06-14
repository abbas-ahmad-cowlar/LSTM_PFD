"""DB <-> data consistency (the systemic guard).

Asserts that each fault class's DB-expected spectral bands carry MORE relative
energy in that fault's signals than in healthy signals. This locks the physics
the models assume (FaultSignatureDatabase) to the physics the generator actually
produces (PHYSICS.md §4) — so the rolling-element / wrong-bearing-type drift
found on 2026-06-14 (audit_reports/PHYSICS_LOSS_AUDIT_2026-06-14.md) cannot
silently recur.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from packages.core.models.physics.fault_signatures import FaultSignatureDatabase
from utils.constants import FAULT_TYPES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data/generated/dataset_v2.h5'
FS = 20480

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="dataset_v2.h5 not present")


@pytest.fixture(scope="module")
def spectra():
    db = FaultSignatureDatabase()
    with h5py.File(DATA, 'r') as f:
        g = f['test']
        labels = g['labels'][:]
        rpms = np.array([float(json.loads(m)['speed_rpm']) for m in g['metadata'][:]])
        sig = g['signals']
        freqs = np.fft.rfftfreq(sig.shape[1], d=1.0 / FS)
        spec, rpm_mean = {}, {}
        for c in range(len(FAULT_TYPES)):
            idx = np.where(labels == c)[0][:20]
            acc = None
            for i in idx:
                P = np.abs(np.fft.rfft(sig[i])) ** 2
                acc = P if acc is None else acc + P
            spec[c] = acc / len(idx)
            rpm_mean[c] = float(rpms[idx].mean())
    return db, freqs, spec, rpm_mean


def _band_frac(spec, freqs, bands):
    if not bands:
        return 0.0
    mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in bands:
        mask |= (freqs >= lo) & (freqs <= hi)
    return float(spec[mask].sum() / spec.sum())


def test_all_classes_have_a_signature():
    """Every class (incl. the 3 mixed) must resolve — no KeyError, no empty."""
    db = FaultSignatureDatabase()
    for c, name in enumerate(FAULT_TYPES):
        bands, broadband = db.get_expected_bands(c, 3600.0)
        if name == 'sain':
            assert not bands and not broadband
        else:
            assert bands or broadband, f"{name}: no DB signature (would give zero physics constraint)"


@pytest.mark.parametrize("c", [i for i in range(len(FAULT_TYPES)) if FAULT_TYPES[i] != 'sain'])
def test_fault_bands_elevated_vs_healthy(spectra, c):
    """The fault's expected bands carry more relative energy than in healthy."""
    db, freqs, spec, rpm_mean = spectra
    bands, broadband = db.get_expected_bands(c, rpm_mean[c])
    f_fault = _band_frac(spec[c], freqs, bands)
    # healthy energy in the SAME (rpm-matched) bands
    healthy_bands, _ = db.get_expected_bands(c, rpm_mean[0])
    f_healthy = _band_frac(spec[0], freqs, healthy_bands)
    # broadband faults also raise the overall floor; tonal faults concentrate energy
    elevated = f_fault > f_healthy
    if broadband:
        # wear: total high-frequency energy elevated vs healthy as a fallback signal
        hf = (freqs > 200)
        elevated = elevated or (spec[c][hf].sum() / spec[c].sum() >
                                spec[0][hf].sum() / spec[0].sum())
    assert elevated, (f"{FAULT_TYPES[c]}: expected-band energy not elevated vs healthy "
                      f"({f_fault:.4f} vs {f_healthy:.4f}) — DB/generator mismatch")
