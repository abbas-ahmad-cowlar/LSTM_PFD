"""Before/after penalty audit for the band-energy physics loss (P6 Step 4).

Shows, on REAL dataset_v2 test windows, the per-class penalty the loss assigns
under the OLD flat/uniform baseline vs the NEW frozen healthy-class reference.
Demonstrates the masquerade fix the owner flagged: under a flat baseline,
healthy-shared energy (60 Hz EMI, low-frequency pink noise) makes HEALTHY signals
look like 1X-imbalance / lubrication faults; the healthy reference removes that.

Usage: python scripts/audit_physics_penalties.py
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.core.models.physics.fault_signatures import (  # noqa: E402
    FaultSignatureDatabase, load_healthy_reference)
from utils.constants import FAULT_TYPES  # noqa: E402

DATA = PROJECT_ROOT / 'data/generated/dataset_v2.h5'
FS, WINDOW = 20480, 20480
NAMES = list(FAULT_TYPES)
db = FaultSignatureDatabase()
REF = load_healthy_reference()


def class_bands(name, omega):
    sig = db.signatures[name]
    tonal = [(m * omega * (1 - hw), m * omega * (1 + hw)) for m, hw in sig.tonal]
    return tonal + [(lo, hi) for lo, hi in sig.bands_hz]


def penalties(psd, freqs, omega):
    """Return (flat_pen[C], healthy_pen[C]) for one window."""
    e_total = psd.sum() + 1e-12
    F = len(freqs)
    flat = np.zeros(len(NAMES))
    healthy = np.zeros(len(NAMES))
    for ci, name in enumerate(NAMES):
        bands = class_bands(name, omega)
        if not bands:
            continue
        # FLAT (old): union mask, concentration vs uniform spectrum
        union = np.zeros(F, dtype=bool)
        for lo, hi in bands:
            union |= (freqs >= lo) & (freqs <= hi)
        nb = union.sum()
        if nb > 0:
            conc = (psd[union].sum() / e_total) * (F / nb)
            flat[ci] = max(0.0, 1.0 - conc)
        # HEALTHY (new): per-band frac / H_ref
        rc = REF[name]
        hrefs = list(rc['tonal']) + list(rc['bands_hz'])
        terms = []
        for (lo, hi), href in zip(bands, hrefs):
            m = (freqs >= lo) & (freqs <= hi)
            frac = psd[m].sum() / e_total
            terms.append(max(0.0, 1.0 - frac / (href + 1e-12)))
        healthy[ci] = float(np.mean(terms)) if terms else 0.0
    return flat, healthy


def mean_penalties_for_class(g, rpms, label_idx, k=25):
    idx = np.where(g['labels'][:] == label_idx)[0][:k]
    freqs = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    flat_acc = np.zeros(len(NAMES))
    healthy_acc = np.zeros(len(NAMES))
    n = 0
    for rec in idx:
        omega = rpms[rec] / 60.0
        sigfull = g['signals'][rec]
        for wi in range(g['signals'].shape[1] // WINDOW):
            w = sigfull[wi * WINDOW:(wi + 1) * WINDOW]
            psd = np.abs(np.fft.rfft(w)) ** 2
            fl, he = penalties(psd, freqs, omega)
            flat_acc += fl
            healthy_acc += he
            n += 1
    return flat_acc / n, healthy_acc / n


def main():
    with h5py.File(DATA, 'r') as f:
        g = f['test']
        rpms = np.array([float(json.loads(m)['speed_rpm']) for m in g['metadata'][:]])
        i_sain = NAMES.index('sain')

        print("=" * 78)
        print("TABLE 1 — penalty the loss assigns to each fault's OWN class")
        print("(for that fault's real signals; lower = 'signature present'.)")
        print(f"{'true fault class':<26} {'FLAT pen':>10} {'HEALTHY pen':>12}")
        print("-" * 78)
        for ci, name in enumerate(NAMES):
            if name == 'sain':
                continue
            fl, he = mean_penalties_for_class(g, rpms, ci)
            print(f"{name:<26} {fl[ci]:>10.3f} {he[ci]:>12.3f}")

        print()
        print("=" * 78)
        print("TABLE 2 — penalty assigned to each FAULT class for HEALTHY signals")
        print("(masquerade check: FLAT near 0 = healthy wrongly looks like that")
        print(" fault; HEALTHY ref should push these UP, away from 0.)")
        fl_h, he_h = mean_penalties_for_class(g, rpms, i_sain)
        print(f"{'fault class (on healthy)':<26} {'FLAT pen':>10} {'HEALTHY pen':>12}")
        print("-" * 78)
        for ci, name in enumerate(NAMES):
            if name == 'sain':
                continue
            flag = '  <- masquerade' if fl_h[ci] < 0.25 else ''
            print(f"{name:<26} {fl_h[ci]:>10.3f} {he_h[ci]:>12.3f}{flag}")


if __name__ == '__main__':
    main()
