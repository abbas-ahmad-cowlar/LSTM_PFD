"""
Generate Dataset v2 per experiments/DATASET_V2.md (Convergence Plan P3.4).

Key differences from the generator's built-in save path:
- Deterministic severity stratification: 80 records per class per severity
  level (320/class), via severity_override cycling.
- Record-level splits stratified by (class × severity), with PER-SPLIT
  metadata stored alongside signals (v1 stored metadata in generation order,
  making it unusable after split shuffling — the defect this script fixes).
- SNR-variant copies of the test split (20/10/5 dB AWGN vs clean RMS) stored
  as test_snr20/test_snr10/test_snr5 groups for noise-robustness curves
  (P5.3) without retraining.
- Hard validation gates: exact class/severity balance, leakage check by
  record hash, NaN/Inf checks. The script FAILS rather than writing a bad file.

Usage:
    python scripts/generate_dataset_v2.py                  # full v2 (~3520 records)
    python scripts/generate_dataset_v2.py --records-per-class 8 --output data/generated/dataset_v2_smoke.h5
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.data_config import DataConfig  # noqa: E402
from data.signal_generation import SignalGenerator  # noqa: E402
from data.signal_validation import validate_signal  # noqa: E402
from utils.constants import FAULT_TYPES  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

SEVERITY_LEVELS = ['incipient', 'mild', 'moderate', 'severe']
SNR_VARIANTS_DB = [20, 10, 5]
SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED = 42


def record_hash(signal: np.ndarray) -> str:
    return hashlib.sha1(signal.tobytes()).hexdigest()


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at the given SNR relative to the signal RMS."""
    rms = np.sqrt(np.mean(signal ** 2))
    noise_rms = rms / (10 ** (snr_db / 20))
    return (signal + rng.normal(0.0, noise_rms, signal.shape)).astype(np.float32)


def stratified_record_split(labels, severities, rng):
    """Record-level split stratified by (class, severity): 70/15/15."""
    split_of = np.empty(len(labels), dtype=object)
    strata = {}
    for i, (lab, sev) in enumerate(zip(labels, severities)):
        strata.setdefault((lab, sev), []).append(i)
    for key, idxs in sorted(strata.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, round(n * SPLITS['val'])) if n >= 3 else 0
        n_test = max(1, round(n * SPLITS['test'])) if n >= 3 else max(0, n - 1)
        n_train = n - n_val - n_test
        assert n_train > 0, f"stratum {key} too small to split ({n} records)"
        split_of[idxs[:n_train]] = 'train'
        split_of[idxs[n_train:n_train + n_val]] = 'val'
        split_of[idxs[n_train + n_val:]] = 'test'
    return split_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-per-class', type=int, default=320,
                        help='Must be divisible by 4 severity levels (default 320)')
    parser.add_argument('--output', default='data/generated/dataset_v2.h5')
    args = parser.parse_args()

    if args.records_per_class % len(SEVERITY_LEVELS) != 0:
        sys.exit(f"records-per-class must be divisible by {len(SEVERITY_LEVELS)}")
    per_severity = args.records_per_class // len(SEVERITY_LEVELS)

    config = DataConfig()
    config.rng_seed = SEED
    config.augmentation.enabled = False  # v2: no generation-time augmentation
    generator = SignalGenerator(config)

    t0 = time.time()
    signals, labels, severities, metadata = [], [], [], []

    print(f"Generating v2: {len(FAULT_TYPES)} classes x {args.records_per_class} records "
          f"({per_severity}/severity level), seed {SEED}")
    counter = 0
    for fault in FAULT_TYPES:
        for sev_idx, severity in enumerate(SEVERITY_LEVELS):
            for n in range(per_severity):
                set_seed(SEED + counter)  # per-record reproducibility
                override = None if fault == 'sain' else severity
                x, meta = generator.generate_single_signal(
                    fault, is_augmented=False, severity_override=override
                )
                errs = validate_signal(x, expected_length=config.signal.N,
                                       label=f"{fault}_{severity}_{n}",
                                       raise_on_error=False)
                if errs:
                    sys.exit(f"GATE FAIL signal validation: {errs}")
                signals.append(x.astype(np.float32))
                labels.append(fault)
                # 'sain' has no physical severity; keep the slot label so the
                # (class x severity) strata stay balanced for splitting.
                severities.append(severity)
                metadata.append(asdict(meta))
                counter += 1
        print(f"  [{fault}] {args.records_per_class} records")

    signals = np.stack(signals)
    label_to_idx = {f: i for i, f in enumerate(FAULT_TYPES)}
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    # ---- Gates: balance ----
    class_counts = Counter(labels)
    assert all(c == args.records_per_class for c in class_counts.values()), class_counts
    sev_counts = Counter(zip(labels, severities))
    assert all(c == per_severity for c in sev_counts.values()), "severity imbalance"

    # ---- Record-level stratified split ----
    rng = np.random.default_rng(SEED)
    split_of = stratified_record_split(labels, severities, rng)

    # ---- Gate: leakage (record hashes unique across splits) ----
    hashes = [record_hash(s) for s in signals]
    assert len(set(hashes)) == len(hashes), "GATE FAIL: duplicate records detected"

    # ---- Write HDF5 ----
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    noise_rng = np.random.default_rng(SEED + 999_983)

    with h5py.File(out_path, 'w') as f:
        for split in ('train', 'val', 'test'):
            idx = np.where(split_of == split)[0]
            g = f.create_group(split)
            g.create_dataset('signals', data=signals[idx], dtype='float32',
                             compression='gzip', compression_opts=1)
            g.create_dataset('labels', data=y[idx])
            g.create_dataset('severities',
                             data=np.array([severities[i] for i in idx], dtype='S12'))
            g.create_dataset('record_hashes',
                             data=np.array([hashes[i] for i in idx], dtype='S40'))
            g.create_dataset('metadata', data=np.array(
                [json.dumps(metadata[i], default=str) for i in idx], dtype=object),
                dtype=h5py.string_dtype())
            print(f"  split {split}: {len(idx)} records")

        # SNR variants of the test split (P5.3 noise-robustness, no retraining)
        test_idx = np.where(split_of == 'test')[0]
        for snr in SNR_VARIANTS_DB:
            g = f.create_group(f'test_snr{snr}')
            noisy = np.stack([add_awgn(signals[i], snr, noise_rng) for i in test_idx])
            g.create_dataset('signals', data=noisy, dtype='float32',
                             compression='gzip', compression_opts=1)
            g.create_dataset('labels', data=y[test_idx])
            g.create_dataset('severities',
                             data=np.array([severities[i] for i in test_idx], dtype='S12'))
            print(f"  split test_snr{snr}: {len(test_idx)} records")

        f.attrs.update({
            'dataset_version': 'v2',
            'design_doc': 'experiments/DATASET_V2.md',
            'generation_date': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'rng_seed': SEED,
            'num_classes': len(FAULT_TYPES),
            'records_per_class': args.records_per_class,
            'records_per_class_per_severity': per_severity,
            'sampling_rate': config.signal.fs,
            'signal_length': config.signal.N,
            'split_ratios': list(SPLITS.values()),
            'snr_variants_db': SNR_VARIANTS_DB,
            'severity_levels': SEVERITY_LEVELS,
        })

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_path} ({size_mb:.0f} MB)")
    print("All gates passed: balance exact, no duplicate records, signals valid.")


if __name__ == '__main__':
    main()
