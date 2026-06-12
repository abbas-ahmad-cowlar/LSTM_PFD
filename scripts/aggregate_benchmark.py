"""
Ensemble row + benchmark aggregation — Phase 4 (P4.5 + P4.6).

1. Soft-voting ensemble of the top-3 deep models by val accuracy
   (PROTOCOL.md §2), evaluated once on the clean test split.
2. Aggregation across all tiers: mean ± std table, Wilcoxon (per-seed
   accuracies, best-physics vs best-vanilla), McNemar (per-window paired
   predictions of the top models), significance-annotated figure.

Outputs: results/benchmark/ensemble/voting/metrics.json,
         results/benchmark/summary.{json,md,png}

Usage:
    python scripts/aggregate_benchmark.py
"""
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset, WindowedView  # noqa: E402
from packages.core.models.model_factory import create_model  # noqa: E402
from utils.constants import FAULT_TYPES, FAULT_TYPE_DISPLAY_NAMES, NUM_CLASSES  # noqa: E402

WINDOW_LENGTH = 20480
DATA = PROJECT_ROOT / 'data/generated/dataset_v2.h5'
DEEP = PROJECT_ROOT / 'results/benchmark/deep'
CLASSICAL = PROJECT_ROOT / 'results/benchmark/classical'
ENSEMBLE_DIR = PROJECT_ROOT / 'results/benchmark/ensemble/voting'

MODEL_ORDER = ['random_forest', 'svm', 'gradient_boosting',
               'cnn1d', 'attention_cnn', 'cnn_lstm', 'resnet18', 'patchtst',
               'hybrid_pinn', 'physics_constrained_cnn', 'multitask_pinn',
               'voting_ensemble']
PHYSICS_MODELS = {'hybrid_pinn', 'physics_constrained_cnn', 'multitask_pinn'}


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def load_runs(root: Path) -> dict:
    runs = {}
    for mj in sorted(root.glob('*/seed*/metrics.json')):
        m = json.loads(mj.read_text())
        runs.setdefault(m['model'], []).append(m)
    return runs


def model_forward(model, model_key, signals, device='cpu'):
    if signals.dim() == 2:
        signals = signals.unsqueeze(1)
    if model_key == 'hybrid_pinn':
        b = signals.shape[0]
        metadata = {'rpm': torch.full((b,), 3600.0),
                    'load': torch.full((b,), 500.0),
                    'viscosity': torch.full((b,), 0.03)}
        out = model(signals, metadata)
    else:
        out = model(signals)
    if isinstance(out, dict):
        for key in ('fault_logits', 'logits', 'fault'):
            if key in out:
                return out[key]
        return next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def evaluate_checkpoint(model_key: str, ckpt_path: Path, loader) -> tuple:
    """Return (softmax probs [N,C], preds [N]) on the test loader (CPU)."""
    model = create_model(model_key, num_classes=NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    probs = []
    with torch.no_grad():
        for signals, _ in loader:
            logits = model_forward(model, model_key, signals)
            probs.append(torch.softmax(logits, dim=1).numpy())
    probs = np.concatenate(probs)
    return probs, probs.argmax(1)


def mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Exact McNemar (binomial) on discordant pairs."""
    from scipy.stats import binomtest
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def main() -> None:
    t0 = time.time()
    deep = load_runs(DEEP)
    classical = load_runs(CLASSICAL)
    all_runs = {**classical, **deep}

    # ---- P4.5: top-3 by mean best_val_acc → soft-voting ensemble ----------
    mean_val = {m: np.mean([r['provenance']['best_val_acc'] for r in rs])
                for m, rs in deep.items()}
    top3 = sorted(mean_val, key=mean_val.get, reverse=True)[:3]
    print(f"Top-3 by mean val acc: {top3}")

    base = BearingFaultDataset.from_hdf5(DATA, split='test')
    windows = WindowedView(base, WINDOW_LENGTH)
    loader = DataLoader(windows, batch_size=64, shuffle=False, num_workers=0)
    targets = np.array([windows[i][1] for i in range(len(windows))])

    member_probs, member_preds = {}, {}
    for mk in top3:
        best = max(deep[mk], key=lambda r: r['provenance']['best_val_acc'])
        ckpt = DEEP / mk / f"seed{best['seed']}" / 'best_model.pth'
        print(f"  evaluating {mk} (seed {best['seed']}) on test ...")
        p, yhat = evaluate_checkpoint(mk, ckpt, loader)
        member_probs[mk], member_preds[mk] = p, yhat
        acc = (yhat == targets).mean() * 100
        print(f"    member test acc: {acc:.2f}%")

    ens_probs = np.mean([member_probs[m] for m in top3], axis=0)
    ens_preds = ens_probs.argmax(1)

    from sklearn.metrics import confusion_matrix, f1_score
    display_names = [FAULT_TYPE_DISPLAY_NAMES[ft] for ft in FAULT_TYPES]
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    ens_artifact = {
        'model': 'voting_ensemble', 'members': [
            {'model': m, 'seed': max(deep[m], key=lambda r: r['provenance']['best_val_acc'])['seed']}
            for m in top3],
        'accuracy': float((ens_preds == targets).mean() * 100),
        'f1_macro': float(f1_score(targets, ens_preds, average='macro')),
        'per_class_accuracy': {
            display_names[c]: float((ens_preds[targets == c] == c).mean() * 100)
            for c in range(NUM_CLASSES)},
        'confusion_matrix': confusion_matrix(targets, ens_preds).tolist(),
        'num_test_windows': int(len(targets)),
        'provenance': {'protocol': 'experiments/PROTOCOL.md', 'data': str(DATA.name),
                       'host': platform.node(), 'git_sha': git_sha(),
                       'finished_at': datetime.now(timezone.utc).isoformat()},
    }
    (ENSEMBLE_DIR / 'metrics.json').write_text(json.dumps(ens_artifact, indent=2))
    print(f"Ensemble: {ens_artifact['accuracy']:.2f}% / f1 {ens_artifact['f1_macro']:.4f}")

    # ---- P4.6: aggregation table -------------------------------------------
    table = {}
    for m, rs in all_runs.items():
        accs = [r['accuracy'] for r in rs]
        table[m] = {'mean_acc': float(np.mean(accs)), 'std_acc': float(np.std(accs)),
                    'mean_f1': float(np.mean([r['f1_macro'] for r in rs])),
                    'seeds': len(rs), 'per_seed_acc': sorted(accs)}
    table['voting_ensemble'] = {'mean_acc': ens_artifact['accuracy'], 'std_acc': 0.0,
                                'mean_f1': ens_artifact['f1_macro'], 'seeds': 1,
                                'per_seed_acc': [ens_artifact['accuracy']]}

    # Statistics (PROTOCOL §5)
    deep_only = {m: v for m, v in table.items() if m in deep}
    best_physics = max((m for m in deep_only if m in PHYSICS_MODELS),
                       key=lambda m: table[m]['mean_acc'])
    best_vanilla = max((m for m in deep_only if m not in PHYSICS_MODELS),
                       key=lambda m: table[m]['mean_acc'])

    from scipy.stats import wilcoxon
    a = [r['accuracy'] for r in deep[best_physics]]
    b = [r['accuracy'] for r in deep[best_vanilla]]
    try:
        wstat, wp = wilcoxon(a, b)
    except ValueError:
        wstat, wp = float('nan'), 1.0

    stats = {
        'best_physics': best_physics, 'best_vanilla': best_vanilla,
        'wilcoxon_p': float(wp),
        'wilcoxon_note': 'n=3 seeds; minimum attainable two-sided p is 0.25 — '
                         'reported for completeness, McNemar is the powered test',
        'mcnemar': {},
    }
    correct = {m: (member_preds[m] == targets) for m in top3}
    correct['voting_ensemble'] = (ens_preds == targets)
    pairs = [(top3[0], top3[1]), (top3[0], top3[2]), (top3[1], top3[2])]
    if best_physics in correct and best_vanilla in correct:
        pairs.append((best_physics, best_vanilla))
    for x, y in dict.fromkeys(pairs):
        stats['mcnemar'][f'{x}_vs_{y}'] = mcnemar_p(correct[x], correct[y])

    summary = {'table': table, 'statistics': stats,
               'classical_bar': table['random_forest']['mean_acc'],
               'generated_at': datetime.now(timezone.utc).isoformat(),
               'git_sha': git_sha()}
    out = PROJECT_ROOT / 'results/benchmark'
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    # Markdown table
    lines = ['# Benchmark Summary — Dataset v2, frozen protocol',
             '', f'Generated {summary["generated_at"]} @ {git_sha()[:8]}', '',
             '| Model | Test acc (mean ± std) | Macro-F1 | Seeds |',
             '|---|---|---|---|']
    for m in MODEL_ORDER:
        if m not in table:
            continue
        t = table[m]
        flag = ' **(physics)**' if m in PHYSICS_MODELS else ''
        lines.append(f"| {m}{flag} | {t['mean_acc']:.2f} ± {t['std_acc']:.2f} | "
                     f"{t['mean_f1']:.4f} | {t['seeds']} |")
    lines += ['', f"Classical bar (RF): {summary['classical_bar']:.2f}%",
              f"Best physics: {best_physics} | Best vanilla: {best_vanilla} | "
              f"Wilcoxon p={wp:.3f} (n=3, see note)", 'McNemar (paired, exact):']
    for k, v in stats['mcnemar'].items():
        lines.append(f"- {k}: p = {v:.4g}")
    (out / 'summary.md').write_text('\n'.join(lines))

    # Figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = [m for m in MODEL_ORDER if m in table]
    means = [table[m]['mean_acc'] for m in models]
    stds = [table[m]['std_acc'] for m in models]
    colors = ['#6baed6' if m in ('random_forest', 'svm', 'gradient_boosting')
              else '#fd8d3c' if m in PHYSICS_MODELS
              else '#74c476' if m == 'voting_ensemble'
              else '#9e9ac8' for m in models]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(models)), means, yerr=stds, capsize=4, color=colors)
    ax.axhline(summary['classical_bar'], ls='--', c='gray', lw=1,
               label=f"classical bar (RF {summary['classical_bar']:.1f}%)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(80, 100)
    ax.set_title('11-class fault diagnosis — dataset v2, 1 s windows, frozen protocol')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'summary.png', dpi=200)

    print(f"\nDone in {time.time()-t0:.0f}s -> results/benchmark/summary.{{json,md,png}}")


if __name__ == '__main__':
    main()
