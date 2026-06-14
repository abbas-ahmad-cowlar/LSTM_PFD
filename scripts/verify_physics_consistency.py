"""Does the generated data actually contain the fault-characteristic frequencies
that the physics loss / signature DB expect? Closes the loop:
  PHYSICS.md spec  ->  FaultSignatureDatabase (what the physics loss uses)
                   ->  vs the ACTUAL spectra of dataset_v2 signals.
If they match, the physics is self-consistent (the signatures ARE in the data);
the negative result is then 'physics prior redundant', not 'physics broken'.
"""
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from packages.core.models.physics.fault_signatures import FaultSignatureDatabase
from utils.constants import FAULT_TYPES

FS = 20480
DATA = ROOT / 'data/generated/dataset_v2.h5'
db = FaultSignatureDatabase()

with h5py.File(DATA, 'r') as f:
    g = f['test']
    labels = g['labels'][:]
    rpms = np.array([float(json.loads(m)['speed_rpm']) for m in g['metadata'][:]])
    signals = g['signals']
    sig_len = signals.shape[1]
    freqs = np.fft.rfftfreq(sig_len, d=1.0 / FS)

    def mean_spectrum(idx):
        acc = None
        for i in idx:
            X = np.abs(np.fft.rfft(signals[i]))
            acc = X if acc is None else acc + X
        return acc / len(idx)

    print(f"signal length {sig_len} samples ({sig_len/FS:.1f}s); freq res {freqs[1]:.2f} Hz\n")
    print(f"{'class':<26} {'rpm':>6} {'expected Hz (loss)':<28} {'matched in spectrum?'}")
    print("-" * 92)
    hits = total = 0
    for c in range(len(FAULT_TYPES)):
        idx = np.where(labels == c)[0][:25]
        if len(idx) == 0:
            continue
        rpm = float(np.mean(rpms[idx]))
        try:
            exp = np.asarray(db.get_expected_frequencies(c, rpm, top_k=5), dtype=float)
            exp = exp[(exp > 0) & (exp < FS / 2)]
        except Exception as e:
            print(f"{FAULT_TYPES[c]:<26} {rpm:>6.0f} (lookup failed: {e})")
            continue
        if exp.size == 0:
            print(f"{FAULT_TYPES[c]:<26} {rpm:>6.0f} (no characteristic freqs — healthy/none)")
            continue
        spec = mean_spectrum(idx)
        # dominant peaks: top-20 spectral bins above 5 Hz, with the broadband median as baseline
        band = freqs > 5
        med = np.median(spec[band])
        order = np.argsort(spec)[::-1]
        top_freqs = [freqs[i] for i in order if freqs[i] > 5][:20]
        # is each expected freq near a dominant peak AND elevated above baseline?
        marks = []
        for fe in exp:
            near = any(abs(tf - fe) <= 0.15 * fe for tf in top_freqs)
            bin_i = int(round(fe / freqs[1]))
            elevated = bin_i < len(spec) and spec[bin_i] > 3 * med
            ok = near or elevated
            marks.append(f"{fe:.0f}{'OK' if ok else 'x'}")
            hits += ok; total += 1
        print(f"{FAULT_TYPES[c]:<26} {rpm:>6.0f} {','.join(f'{e:.0f}' for e in exp):<28} {' '.join(marks)}")
    print("-" * 92)
    print(f"\nExpected-frequency match rate: {hits}/{total} = {100*hits/max(total,1):.0f}%")
    print("(OK = expected freq is a dominant peak or >3x broadband baseline in the actual data)")
