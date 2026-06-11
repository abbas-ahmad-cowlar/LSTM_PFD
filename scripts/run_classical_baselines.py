"""
Classical baselines — Phase 4 (PROTOCOL.md §2: RF, SVM, GradientBoosting).

Extracts the 36 hand-crafted features per 1 s window (cached to .npz so
reruns are instant), trains each classical model at seeds 0/1/2, and writes
metrics.json artifacts in the same schema as the deep matrix
(results/benchmark/classical/<model>/seed<k>/metrics.json).

Usage:
    python scripts/run_classical_baselines.py
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset, WindowedView  # noqa: E402
from packages.core.features.feature_extractor import FeatureExtractor  # noqa: E402
from utils.constants import FAULT_TYPES, FAULT_TYPE_DISPLAY_NAMES, NUM_CLASSES, SAMPLING_RATE  # noqa: E402

WINDOW_LENGTH = 20480
SEEDS = [0, 1, 2]
MODELS = ['random_forest', 'svm', 'gradient_boosting']


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def extract_split_features(h5_path, split, fx, cache_dir):
    cache = cache_dir / f'features_{split}.npz'
    if cache.exists():
        z = np.load(cache)
        return z['X'], z['y']
    base = BearingFaultDataset.from_hdf5(h5_path, split=split)
    windows = WindowedView(base, WINDOW_LENGTH)
    n = len(windows)
    X = np.empty((n, 36), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    t0 = time.time()
    for i in range(n):
        sig, lab = windows[i]
        X[i] = fx.extract_features(np.asarray(sig, dtype=np.float64))
        y[i] = lab
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f'  {split}: {i+1}/{n} ({rate:.0f} win/s, ETA {((n-i-1)/rate)/60:.0f} min)')
    np.savez_compressed(cache, X=X, y=y)
    print(f'  {split}: {n} windows featurized in {(time.time()-t0)/60:.1f} min -> {cache.name}')
    return X, y


def build_model(name, seed):
    """sklearn models with PROTOCOL defaults; seed via random_state."""
    from sklearn.ensemble import (GradientBoostingClassifier as SkGB,
                                  RandomForestClassifier as SkRF)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == 'random_forest':
        return SkRF(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == 'svm':
        # SVC is deterministic given data; seed affects nothing material —
        # recorded anyway so the artifact schema stays uniform.
        return make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10.0,
                                                   random_state=seed))
    if name == 'gradient_boosting':
        return SkGB(n_estimators=200, random_state=seed)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/generated/dataset_v2.h5')
    parser.add_argument('--out-root', default='results/benchmark/classical')
    args = parser.parse_args()

    out_root = PROJECT_ROOT / args.out_root
    cache_dir = out_root / '_feature_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    h5 = PROJECT_ROOT / args.data

    fx = FeatureExtractor(fs=SAMPLING_RATE)
    print('Featurizing splits (cached after first run)...')
    X_train, y_train = extract_split_features(h5, 'train', fx, cache_dir)
    X_val, y_val = extract_split_features(h5, 'val', fx, cache_dir)
    X_test, y_test = extract_split_features(h5, 'test', fx, cache_dir)

    from sklearn.metrics import confusion_matrix, f1_score
    display_names = [FAULT_TYPE_DISPLAY_NAMES[ft] for ft in FAULT_TYPES]

    for model_name in MODELS:
        for seed in SEEDS:
            run_dir = out_root / model_name / f'seed{seed}'
            if (run_dir / 'metrics.json').exists():
                print(f'[skip] {model_name} seed {seed} already complete')
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            model = build_model(model_name, seed)
            model.fit(X_train, y_train)
            val_acc = float((model.predict(X_val) == y_val).mean() * 100)
            preds = model.predict(X_test)
            artifact = {
                'model': model_name, 'seed': seed,
                'accuracy': float((preds == y_test).mean() * 100),
                'f1_macro': float(f1_score(y_test, preds, average='macro')),
                'per_class_accuracy': {
                    display_names[c]: float((preds[y_test == c] == c).mean() * 100)
                    for c in range(NUM_CLASSES)},
                'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
                'class_names': display_names,
                'num_test_windows': int(len(y_test)),
                'val_accuracy': val_acc,
                'provenance': {
                    'protocol': 'experiments/PROTOCOL.md',
                    'data': str(args.data), 'window_length': WINDOW_LENGTH,
                    'features': 36, 'host': platform.node(),
                    'git_sha': git_sha(),
                    'wall_time_s': round(time.time() - t0, 1),
                    'finished_at': datetime.now(timezone.utc).isoformat(),
                },
            }
            (run_dir / 'metrics.json').write_text(json.dumps(artifact, indent=2))
            print(f'[done] {model_name} seed {seed}: '
                  f"test {artifact['accuracy']:.2f}% f1 {artifact['f1_macro']:.4f} "
                  f"({artifact['provenance']['wall_time_s']:.0f}s)")

    print('Classical tier complete.')


if __name__ == '__main__':
    main()
