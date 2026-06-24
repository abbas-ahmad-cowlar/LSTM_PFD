"""
Precompute the FROZEN RANDOM-band control reference (§8.8, pre-registered + owner-
ratified 2026-06-23). The NON-PHYSICS matched-structure control for the noise-
robustness result.

§8.7 (F9 scramble) showed that judging each fault against a DIFFERENT fault's REAL
bands reproduces the 5 dB robustness -> the correct per-class mapping is not
necessary. §8.8 asks the next question: does judging each class against RANDOM
NON-FAULT bands (same loss form, same per-class band count + bandwidth, but band
locations that are provably NOT any fault's characteristic frequency) also
reproduce it? If yes, the effect is a generic high-weight SPECTRAL-band regularizer
(band locations don't matter at all); if no, it needs real fault-frequency bands.

This script (a) deterministically generates the random band layout: for every real
band of every class it draws a replacement of identical KIND and WIDTH placed at a
random location that does NOT overlap ANY class's real bands (tonal checked in
Omega-multiplier space, so non-overlap holds at every rpm; absolute checked in Hz
over the dataset rpm range), and (b) measures the healthy-energy fraction in each
random band exactly as scripts/compute_healthy_reference.py does (tonal rpm-matched
per record, absolute fixed). The loss = relu(1 - frac_b / H_rand[c][b]) is then
structurally identical to the validated loss; only the bands are non-physical.

Frozen artifact: packages/core/models/physics/random_reference.json
Representation MUST match the loss + the healthy reference: 1-second windows
(20480 samples), full-window rfft (1 Hz resolution).

Usage:
    python scripts/compute_random_reference.py
Deterministic (fixed RNG seed, recorded). Regenerate ONLY if the signature DB bands
or the dataset change (then re-run the §8.8 tests + re-pre-register).
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.core.models.physics.fault_signatures import FaultSignatureDatabase  # noqa: E402
from utils.constants import FAULT_TYPES  # noqa: E402

DATA = PROJECT_ROOT / 'data/generated/dataset_v2.h5'
OUT = PROJECT_ROOT / 'packages/core/models/physics/random_reference.json'
FS = 20480
WINDOW = 20480
NYQUIST = FS / 2.0
HEALTHY = 'sain'

# Dataset operating range (speed 3600 rpm +/-10%; metadata 3240-3959) -> Omega=rpm/60
RPM_MIN, RPM_MAX = 3240.0, 3960.0
OMEGA_MIN, OMEGA_MAX = RPM_MIN / 60.0, RPM_MAX / 60.0   # 54 .. 66 Hz
M_LO, M_HI = 0.2, 9.0          # random tonal multiplier range (stays << Nyquist)
MARGIN_M = 0.05                # multiplier-space safety margin
MARGIN_HZ = 8.0                # Hz safety margin
SEED = 20260623                # fixed; recorded in provenance


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def _overlaps(lo, hi, intervals, margin):
    return any(not (hi + margin <= a or lo - margin >= b) for a, b in intervals)


def generate_random_layout(seed: int = SEED):
    """Deterministic random non-fault band layout, matched per class to the real DB.

    Returns (layout, audit) where layout[name] = {'tonal': [(m,hw)...],
    'bands_hz': [(lo,hi)...]} and audit holds the forbidden/placed intervals for the
    non-overlap assertion."""
    db = FaultSignatureDatabase()
    rng = np.random.default_rng(seed)

    # Structure-preserving remap: the real DB SHARES bands across classes (1X is the
    # same band wherever it appears; LOW likewise). Draw ONE random replacement per
    # DISTINCT real band and reuse it consistently -> matched count per class + the
    # same sharing structure, non-physical locations. Order is deterministic (first
    # appearance in FAULT_TYPES order).
    distinct_tonal, distinct_bands = [], []
    for nm in FAULT_TYPES:
        for t in db.signatures[nm].tonal:
            if tuple(t) not in distinct_tonal:
                distinct_tonal.append(tuple(t))
        for b in db.signatures[nm].bands_hz:
            if tuple(b) not in distinct_bands:
                distinct_bands.append(tuple(b))

    # Real tonal multiplier intervals — non-overlap in Omega-multiplier space holds
    # at EVERY rpm (both real and random tonal scale with Omega).
    forbidden_mult = [(m * (1 - hw), m * (1 + hw)) for (m, hw) in distinct_tonal]
    tonal_map, placed_mult = {}, []
    for (m, hw) in distinct_tonal:
        for _ in range(100000):
            m_r = float(rng.uniform(M_LO, M_HI))
            lo_m, hi_m = m_r * (1 - hw), m_r * (1 + hw)
            if _overlaps(lo_m, hi_m, forbidden_mult + placed_mult, MARGIN_M):
                continue
            tonal_map[(m, hw)] = (round(m_r, 4), hw)
            placed_mult.append((lo_m, hi_m))
            break
        else:
            raise RuntimeError(f"could not place random tonal for {(m, hw)}")

    # Occupied Hz for absolute placement: real absolute bands + ALL tonal Hz
    # footprints (real + random) over the rpm range + placed random absolutes.
    occupied_hz = [(lo, hi) for (lo, hi) in distinct_bands]
    for (m, hw) in distinct_tonal:
        occupied_hz.append((m * OMEGA_MIN * (1 - hw), m * OMEGA_MAX * (1 + hw)))
    for (m_r, hw) in tonal_map.values():
        occupied_hz.append((m_r * OMEGA_MIN * (1 - hw), m_r * OMEGA_MAX * (1 + hw)))

    bands_map = {}
    for (lo, hi) in distinct_bands:
        w = hi - lo
        for _ in range(100000):
            center = float(rng.uniform(w / 2 + MARGIN_HZ, NYQUIST - w / 2 - MARGIN_HZ))
            blo, bhi = center - w / 2, center + w / 2
            if _overlaps(blo, bhi, occupied_hz, MARGIN_HZ):
                continue
            bands_map[(lo, hi)] = (round(blo, 2), round(bhi, 2))
            occupied_hz.append((blo, bhi))
            break
        else:
            raise RuntimeError(f"could not place random band for {(lo, hi)}")

    layout = {nm: {'tonal': [tonal_map[tuple(t)] for t in db.signatures[nm].tonal],
                   'bands_hz': [bands_map[tuple(b)] for b in db.signatures[nm].bands_hz]}
              for nm in FAULT_TYPES}
    audit = {'tonal_map': tonal_map, 'bands_map': bands_map}
    return layout, audit


def assert_non_overlap(layout):
    """Sanity gate: no random band coincides with any REAL characteristic band."""
    db = FaultSignatureDatabase()
    real_mult = [(m * (1 - hw), m * (1 + hw))
                 for nm in FAULT_TYPES for m, hw in db.signatures[nm].tonal]
    real_hz = [(lo, hi) for nm in FAULT_TYPES for (lo, hi) in db.signatures[nm].bands_hz]
    real_tonal_hz = [(m * OMEGA_MIN * (1 - hw), m * OMEGA_MAX * (1 + hw))
                     for nm in FAULT_TYPES for m, hw in db.signatures[nm].tonal]
    for nm in FAULT_TYPES:
        for (m_r, hw) in layout[nm]['tonal']:
            assert not _overlaps(m_r * (1 - hw), m_r * (1 + hw), real_mult, 0.0), \
                f"random tonal {m_r} of {nm} overlaps a real tonal multiplier"
        for (lo, hi) in layout[nm]['bands_hz']:
            assert not _overlaps(lo, hi, real_hz + real_tonal_hz, 0.0), \
                f"random band ({lo},{hi}) of {nm} overlaps a real band"


def band_fraction(psd, freqs, lo, hi, e_total):
    mask = (freqs >= lo) & (freqs <= hi)
    return float(psd[mask].sum() / e_total)


def main():
    layout, _audit = generate_random_layout(SEED)
    assert_non_overlap(layout)
    names = list(FAULT_TYPES)
    h_idx = names.index(HEALTHY)

    with h5py.File(DATA, 'r') as f:
        g = f['train']
        labels = g['labels'][:]
        rpms = np.array([float(json.loads(m)['speed_rpm']) for m in g['metadata'][:]])
        signals = g['signals']
        rec_len = signals.shape[1]
        wpr = rec_len // WINDOW
        freqs = np.fft.rfftfreq(WINDOW, d=1.0 / FS)

        healthy_records = np.where(labels == h_idx)[0]
        acc = {nm: {'tonal': [0.0] * len(layout[nm]['tonal']),
                    'bands_hz': [0.0] * len(layout[nm]['bands_hz'])}
               for nm in names}
        n_windows = 0
        for rec in healthy_records:
            sig_full = signals[rec]
            omega = rpms[rec] / 60.0
            for wi in range(wpr):
                w = sig_full[wi * WINDOW:(wi + 1) * WINDOW]
                psd = np.abs(np.fft.rfft(w)) ** 2
                e_total = psd.sum() + 1e-12
                n_windows += 1
                for nm in names:
                    for bi, (m, hw) in enumerate(layout[nm]['tonal']):
                        lo, hi = m * omega * (1 - hw), m * omega * (1 + hw)
                        acc[nm]['tonal'][bi] += band_fraction(psd, freqs, lo, hi, e_total)
                    for bi, (lo, hi) in enumerate(layout[nm]['bands_hz']):
                        acc[nm]['bands_hz'][bi] += band_fraction(psd, freqs, lo, hi, e_total)

    per_class = {nm: {'tonal': [v / n_windows for v in acc[nm]['tonal']],
                      'bands_hz': [v / n_windows for v in acc[nm]['bands_hz']]}
                 for nm in names}

    out = {
        '_doc': ('Frozen RANDOM-band control reference (§8.8 non-physics control). '
                 'Per class, RANDOM non-fault bands matched in count + width to the '
                 'real bands (tonal in Omega-multiplier space, absolute in Hz), '
                 'provably non-overlapping with ANY real characteristic band, plus '
                 'the mean fraction of total 1-s-window energy HEALTHY training '
                 'signals carry in each random band. Drives the §8.8 random-band '
                 'arm: relu(1 - frac_b / H_rand[c][b]) -- identical loss form to the '
                 'validated physics loss, non-physical band locations.'),
        '_provenance': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'git_sha': git_sha(),
            'dataset': DATA.name,
            'split': 'train', 'healthy_class': HEALTHY,
            'n_healthy_windows': int(n_windows),
            'fs': FS, 'window': WINDOW, 'n_fft': WINDOW,
            'rng_seed': SEED, 'rpm_range': [RPM_MIN, RPM_MAX],
            'non_overlap_verified': True,
            'source': 'scripts/compute_random_reference.py',
        },
        'band_layout': {nm: {'tonal': [list(t) for t in layout[nm]['tonal']],
                             'bands_hz': [list(b) for b in layout[nm]['bands_hz']]}
                        for nm in names},
        'per_class': per_class,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}  ({n_windows} healthy windows)")
    print(f"{'class':<26} random band layout (tonal m | bands Hz)")
    print('-' * 70)
    for nm in names:
        t = ', '.join(f'{m:g}x' for m, _ in layout[nm]['tonal'])
        b = ', '.join(f'{lo:g}-{hi:g}Hz' for lo, hi in layout[nm]['bands_hz'])
        print(f'{nm:<26} {t or "-":<28} {b or "-"}')


if __name__ == '__main__':
    main()
