"""
Precompute the FROZEN healthy-class spectral reference (P6 Step 4, owner-ratified
2026-06-14: use the actual healthy baseline, not a flat/uniform spectrum).

For every fault class's expected bands (FaultSignatureDatabase — tonal harmonics
scaled by per-record rpm, plus absolute Hz bands), we measure how much spectral
energy a HEALTHY bearing carries there, as a fraction of total window energy,
averaged over the training healthy windows (each window's tonal bands placed at
that record's own rpm — "rpm-matched"). This per-band fraction is the healthy
baseline H_ref against which the band-energy physics loss and the DB↔data CI test
both judge whether a fault's signature is actually present (energy ABOVE healthy),
so healthy-shared energy (60 Hz EMI, low-frequency pink noise) can no longer
masquerade as a fault signature.

Frozen artifact: packages/core/models/physics/healthy_reference.json
Representation MUST match the loss: 1-second windows (20480 samples), full-window
rfft (1 Hz resolution — needed to resolve the 1-6 Hz lubrication band).

Usage:
    python scripts/compute_healthy_reference.py
Regenerate ONLY if the dataset or the signature DB bands change (then re-run the
gradient/CI tests and re-pre-register in PROTOCOL §8.0-quater).
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
OUT = PROJECT_ROOT / 'packages/core/models/physics/healthy_reference.json'
FS = 20480
WINDOW = 20480           # 1-second window (matches the loss / WindowedView)
HEALTHY = 'sain'


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def band_fraction(psd, freqs, lo, hi, e_total):
    mask = (freqs >= lo) & (freqs <= hi)
    return float(psd[mask].sum() / e_total)


def main():
    db = FaultSignatureDatabase()
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
        F_bins = freqs.shape[0]

        healthy_records = np.where(labels == h_idx)[0]
        # accumulators: per class -> per band running sum + count
        acc = {nm: {'tonal': [0.0] * len(db.signatures[nm].tonal),
                    'bands_hz': [0.0] * len(db.signatures[nm].bands_hz)}
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
                    s = db.signatures[nm]
                    for bi, (m, hw) in enumerate(s.tonal):
                        lo, hi = m * omega * (1 - hw), m * omega * (1 + hw)
                        acc[nm]['tonal'][bi] += band_fraction(psd, freqs, lo, hi, e_total)
                    for bi, (lo, hi) in enumerate(s.bands_hz):
                        acc[nm]['bands_hz'][bi] += band_fraction(psd, freqs, lo, hi, e_total)

    per_class = {}
    for nm in names:
        per_class[nm] = {
            'tonal': [v / n_windows for v in acc[nm]['tonal']],
            'bands_hz': [v / n_windows for v in acc[nm]['bands_hz']],
        }

    out = {
        '_doc': ('Frozen healthy-class spectral reference (P6 Step 4). Per fault '
                 'class, the mean fraction of total 1-s-window energy that HEALTHY '
                 'training signals carry in each of that class\'s expected bands '
                 '(tonal bands rpm-matched per record). The band-energy physics '
                 'loss penalizes a class whose bands are NOT elevated above these '
                 'values; the DB-consistency CI test asserts faults exceed them. '
                 'Aligns loss + CI to the same baseline.'),
        '_provenance': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'git_sha': git_sha(),
            'dataset': DATA.name,
            'split': 'train', 'healthy_class': HEALTHY,
            'n_healthy_windows': int(n_windows),
            'fs': FS, 'window': WINDOW, 'n_fft': WINDOW,
            'freq_resolution_hz': float(FS / WINDOW),
            'source': 'scripts/compute_healthy_reference.py',
        },
        'band_layout': {nm: {'tonal': [list(t) for t in db.signatures[nm].tonal],
                             'bands_hz': [list(b) for b in db.signatures[nm].bands_hz]}
                        for nm in names},
        'per_class': per_class,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}  ({n_windows} healthy windows)")
    print(f"{'class':<26} healthy band fractions (H_ref)")
    print('-' * 70)
    for nm in names:
        vals = per_class[nm]['tonal'] + per_class[nm]['bands_hz']
        pretty = ', '.join(f'{v:.4f}' for v in vals) if vals else '(no bands)'
        print(f'{nm:<26} {pretty}')


if __name__ == '__main__':
    main()
