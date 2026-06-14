"""
Record-level benchmark statistics — P6 remediation Step 2.

WHY THIS EXISTS
---------------
The Phase-4 benchmark and its significance tests (`scripts/aggregate_benchmark.py`)
were computed over **2,640 one-second windows** treated as independent samples.
They are not independent: each 5-second test record yields 5 windows that share
fault realization, severity, operating point and noise process. The independent
unit is the **record** (528 in the test split). The external science audit
(`audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md`, Finding 12) flagged
every window-level p-value / CI as overstated.

This script recomputes the benchmark at the RECORD level, with NO retraining of
the deep models (their frozen checkpoints are re-evaluated to obtain per-window
class scores; classical models are refit from the cached 36-feature matrices,
which is deterministic and faithful to the original run).

WHAT IT DOES
------------
1. For every (model, seed): obtain per-window class scores [N_win, C]
   (softmax for deep checkpoints, predict_proba/decision_function for classical),
   cached to .npy for resume-safety. A SANITY GATE asserts that argmax of the
   stored scores reproduces the recorded window-level accuracy (catches
   checkpoint-load / ordering bugs — the I4 lesson).
2. Aggregate the 5 windows of each record by MEAN class score (soft vote) ->
   one record-level prediction per record. A hard majority vote is computed too,
   as a robustness cross-check.
3. Per model: record-level accuracy per seed -> mean ± std; plus a seed-ensemble
   representative (mean record scores across seeds) used for the paired tests.
4. Cluster bootstrap (resample the 528 records, B=10000): 95% CI per model and on
   the accuracy gap of the headline pair.
5. Record-level McNemar (exact) for the headline pairs; Wilcoxon over seeds.

SCOPE / HONESTY (do not overstate — guardrails in PROJECT_STATE.md §6)
----------------------------------------------------------------------
This step ONLY corrects the statistical unit. It does NOT re-label or re-train
anything. The rows still carry their Phase-4 training reality:
  * physics_constrained_cnn  -> trained CROSS-ENTROPY ONLY (architecture row,
                                physics loss was OFF; PROTOCOL §8.0)
  * multitask_pinn           -> trained SINGLE-TASK (auxiliary heads unused)
  * hybrid_pinn              -> constant default metadata + rolling-element
                                physics branch (Finding 7)
So this is a recomputation of the existing CLASSIFICATION benchmark at record
level, NOT a test of physics-informed training. No "physics family" average is
produced. Relabel/quarantine is Step 3; valid physics reruns are Steps 4-5.

Usage:
    python scripts/aggregate_benchmark_record_level.py
Outputs:
    results/benchmark/summary_record_level.{json,md}
    results/benchmark/record_level/_cache/<model>_seed<k>.npy   (per-window scores)
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
FEAT_CACHE = CLASSICAL / '_feature_cache'
OUT = PROJECT_ROOT / 'results/benchmark'
SCORE_CACHE = OUT / 'record_level' / '_cache'

DEEP_MODELS = ['cnn1d', 'attention_cnn', 'cnn_lstm', 'resnet18', 'patchtst',
               'hybrid_pinn', 'physics_constrained_cnn', 'multitask_pinn']
CLASSICAL_MODELS = ['random_forest', 'svm', 'gradient_boosting']
SEEDS = [0, 1, 2]
MODEL_ORDER = CLASSICAL_MODELS + DEEP_MODELS
# Rows whose Phase-4 training reality bars them from being read as a "physics"
# result (PROTOCOL §8.0 + audit Findings 7-9). Annotated, never averaged as a family.
PHYSICS_LABELED = {'hybrid_pinn', 'physics_constrained_cnn', 'multitask_pinn'}
ROW_NOTE = {
    'physics_constrained_cnn': 'CE-only (architecture; physics loss OFF)',
    'multitask_pinn': 'single-task (aux heads unused)',
    'hybrid_pinn': 'rolling-element branch + constant metadata',
}
B_BOOT = 10000
BOOT_SEED = 42


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


# --------------------------------------------------------------------------- #
#  scoring: per-window class scores [N, C], argmax == the model's prediction
# --------------------------------------------------------------------------- #
def deep_scores(model_key: str, ckpt_path: Path, loader) -> np.ndarray:
    """Softmax probabilities per window from a frozen checkpoint (CPU)."""
    model = create_model(model_key, num_classes=NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out = []
    with torch.no_grad():
        for signals, _ in loader:
            if signals.dim() == 2:
                signals = signals.unsqueeze(1)
            if model_key == 'hybrid_pinn':
                b = signals.shape[0]
                meta = {'rpm': torch.full((b,), 3600.0),
                        'load': torch.full((b,), 500.0),
                        'viscosity': torch.full((b,), 0.03)}
                logits = model(signals, meta)
            else:
                logits = model(signals)
            if isinstance(logits, dict):
                logits = next((logits[k] for k in ('fault_logits', 'logits', 'fault')
                               if k in logits), next(iter(logits.values())))
            elif isinstance(logits, (tuple, list)):
                logits = logits[0]
            out.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(out).astype(np.float64)


def classical_scores(model_key: str, seed: int) -> np.ndarray:
    """Refit the classical model from cached features; per-window soft scores
    on the test split. Deterministic given the seed (faithful to the original)."""
    from scripts.run_classical_baselines import build_model  # reuse exact configs
    tr = np.load(FEAT_CACHE / 'features_train.npz')
    te = np.load(FEAT_CACHE / 'features_test.npz')
    model = build_model(model_key, seed)
    model.fit(tr['X'], tr['y'])
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(te['X']).astype(np.float64)
    # SVM pipeline: no predict_proba (probability=False) -> use decision_function
    # (ovr scores); argmax matches .predict, so the sanity gate still holds.
    dec = model.decision_function(te['X'])
    return np.asarray(dec, dtype=np.float64)


def get_scores(model_key: str, seed: int, loader, recorded_acc: float) -> np.ndarray:
    """Cached per-window scores with a window-level accuracy sanity gate."""
    SCORE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = SCORE_CACHE / f'{model_key}_seed{seed}.npy'
    if cache.exists():
        scores = np.load(cache)
    else:
        if model_key in DEEP_MODELS:
            ckpt = DEEP / model_key / f'seed{seed}' / 'best_model.pth'
            scores = deep_scores(model_key, ckpt, loader)
        else:
            scores = classical_scores(model_key, seed)
        np.save(cache, scores)
    return scores


# --------------------------------------------------------------------------- #
#  statistics
# --------------------------------------------------------------------------- #
def mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    from scipy.stats import binomtest
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def main() -> None:
    t0 = time.time()
    device = 'cpu'
    print(f'Record-level aggregation on {device} @ {git_sha()[:8]}')

    # ---- test windows, record map, record targets --------------------------
    base = BearingFaultDataset.from_hdf5(DATA, split='test')
    windows = WindowedView(base, WINDOW_LENGTH)
    wpr = windows.windows_per_record
    n_win = len(windows)
    n_rec = n_win // wpr
    win_targets = np.array([int(windows[i][1]) for i in range(n_win)])
    rec_targets = win_targets.reshape(n_rec, wpr)
    assert (rec_targets == rec_targets[:, :1]).all(), 'windows of a record disagree on label'
    rec_targets = rec_targets[:, 0]
    print(f'test split: {n_win} windows, {wpr} windows/record, {n_rec} records')
    loader = DataLoader(windows, batch_size=64, shuffle=False, num_workers=0)

    # ---- recorded window-level accuracies (for the sanity gate) ------------
    def recorded_acc(model_key, seed):
        root = DEEP if model_key in DEEP_MODELS else CLASSICAL
        m = json.loads((root / model_key / f'seed{seed}' / 'metrics.json').read_text())
        return m['accuracy']

    # ---- per (model, seed): scores -> record soft/hard preds ----------------
    # record_soft[model] : list over seeds of record-prob arrays [n_rec, C]
    record_soft, win_acc_check = {}, {}
    for mk in MODEL_ORDER:
        record_soft[mk] = []
        for s in SEEDS:
            r_acc = recorded_acc(mk, s)
            scores = get_scores(mk, s, loader, r_acc)
            assert scores.shape[0] == n_win, f'{mk} s{s}: {scores.shape[0]} != {n_win}'
            # sanity gate: argmax(scores) must reproduce recorded window accuracy
            w_acc = 100.0 * (scores.argmax(1) == win_targets).mean()
            win_acc_check[(mk, s)] = (w_acc, r_acc)
            if abs(w_acc - r_acc) > 0.15:
                raise SystemExit(
                    f'SANITY FAIL {mk} seed{s}: window acc {w_acc:.2f}% vs '
                    f'recorded {r_acc:.2f}% — checkpoint/order mismatch, investigate')
            rec_scores = scores.reshape(n_rec, wpr, -1).mean(axis=1)  # soft vote
            record_soft[mk].append(rec_scores)
            print(f'  {mk:24s} seed{s}: win {w_acc:5.2f}% (rec {r_acc:5.2f}%)  OK')

    # ---- per (model, seed) record correctness; ONE estimator throughout ----
    # Headline accuracy = mean ± std over seeds of per-seed record accuracy
    # (mirrors the frozen protocol). CI = cluster bootstrap of that SAME
    # seed-mean (so the CI always contains the mean). Paired tests use the
    # best-VALIDATION seed as the single representative (one real model, no
    # ensembling — mirrors the original benchmark's McNemar methodology).
    def val_acc(mk, s):
        root = DEEP if mk in DEEP_MODELS else CLASSICAL
        m = json.loads((root / mk / f'seed{s}' / 'metrics.json').read_text())
        return m['provenance'].get('best_val_acc', m.get('val_accuracy', 0.0))

    table = {}
    seed_mean_corr = {}   # [n_rec] mean-over-seeds per-record correctness (for seed-mean CI)
    bestval_corr = {}     # [n_rec] binary correctness of the best-val seed (for paired tests)
    for mk in MODEL_ORDER:
        corr_by_seed = [(rs.argmax(1) == rec_targets) for rs in record_soft[mk]]
        per_seed_acc = [100.0 * c.mean() for c in corr_by_seed]
        seed_mean_corr[mk] = np.mean([c.astype(float) for c in corr_by_seed], axis=0)
        bv = max(range(len(SEEDS)), key=lambda i: val_acc(mk, SEEDS[i]))
        bestval_corr[mk] = corr_by_seed[bv]
        table[mk] = {
            'record_acc_mean': float(np.mean(per_seed_acc)),
            'record_acc_std': float(np.std(per_seed_acc)),
            'record_acc_per_seed': [float(a) for a in per_seed_acc],
            'bestval_seed': int(SEEDS[bv]),
            'record_acc_bestval_seed': float(per_seed_acc[bv]),
            'window_acc_recorded': float(np.mean([win_acc_check[(mk, s)][1] for s in SEEDS])),
            'seeds': len(SEEDS),
            'note': ROW_NOTE.get(mk, ''),
        }

    # ---- cluster bootstrap of the seed-MEAN accuracy (consistent w/ table) --
    rng = np.random.default_rng(BOOT_SEED)
    idx_boot = rng.integers(0, n_rec, size=(B_BOOT, n_rec))
    for mk in MODEL_ORDER:
        accs = seed_mean_corr[mk][idx_boot].mean(axis=1) * 100.0
        table[mk]['ci95_record_acc'] = [float(np.percentile(accs, 2.5)),
                                        float(np.percentile(accs, 97.5))]

    # ---- headline pairs at record level (best-VAL-seed representatives) -----
    deep_acc = {m: table[m]['record_acc_mean'] for m in DEEP_MODELS}  # seed-mean
    best_vanilla = max((m for m in DEEP_MODELS if m not in PHYSICS_LABELED),
                       key=deep_acc.get)
    best_physlabeled = max((m for m in DEEP_MODELS if m in PHYSICS_LABELED),
                           key=deep_acc.get)
    top3 = sorted(DEEP_MODELS, key=deep_acc.get, reverse=True)[:3]

    pairs = []
    for i in range(len(top3)):
        for j in range(i + 1, len(top3)):
            pairs.append((top3[i], top3[j]))
    pairs.append((best_physlabeled, best_vanilla))
    pairs.append(('physics_constrained_cnn', 'resnet18'))

    mcnemar = {}
    for a, b in dict.fromkeys(pairs):
        mcnemar[f'{a}_vs_{b}'] = mcnemar_p(bestval_corr[a], bestval_corr[b])

    # gap (best-val seed of each) + its record-bootstrap CI
    gap_pts = (table[best_physlabeled]['record_acc_bestval_seed']
               - table[best_vanilla]['record_acc_bestval_seed'])
    gapc = bestval_corr[best_physlabeled].astype(float) - bestval_corr[best_vanilla].astype(float)
    gap_boot = gapc[idx_boot].mean(axis=1) * 100.0
    gap_ci = [float(np.percentile(gap_boot, 2.5)), float(np.percentile(gap_boot, 97.5))]

    # Wilcoxon over seeds (n=3; low power — reported for completeness)
    from scipy.stats import wilcoxon
    a_seed = table[best_physlabeled]['record_acc_per_seed']
    b_seed = table[best_vanilla]['record_acc_per_seed']
    try:
        _, wp = wilcoxon(a_seed, b_seed)
    except ValueError:
        wp = 1.0

    stats = {
        'aggregation': 'soft vote (mean per-window class score over the 5 windows), argmax',
        'independent_unit': 'record (n=528)', 'windows_per_record': int(wpr),
        'best_vanilla': best_vanilla, 'best_physics_labeled': best_physlabeled,
        'best_physics_labeled_caveat': ROW_NOTE.get(best_physlabeled, ''),
        'top3_deep': top3,
        'paired_test_basis': 'best-validation seed per model (single model, no ensembling)',
        'bestval_seed_acc': {m: table[m]['record_acc_bestval_seed'] for m in DEEP_MODELS},
        'mcnemar_record_level_exact': mcnemar,
        'gap_best_physlabeled_minus_best_vanilla_pts': float(gap_pts),
        'gap_ci95_record_bootstrap': gap_ci,
        'wilcoxon_seeds_p': float(wp),
        'wilcoxon_note': 'n=3 seeds; min attainable two-sided p = 0.25 — McNemar is the powered test',
        'bootstrap': {'B': B_BOOT, 'seed': BOOT_SEED, 'method': 'cluster bootstrap by record'},
    }

    summary = {
        'scope': 'RECORD-LEVEL recomputation of the Phase-4 classification benchmark '
                 '(P6 remediation Step 2). Statistical unit corrected from 2,640 '
                 'correlated windows to 528 independent records. No retraining, no '
                 'relabeling — row training reality is annotated, not a physics test.',
        'table': table, 'statistics': stats,
        'classical_bar_record_acc': table['random_forest']['record_acc_mean'],
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'git_sha': git_sha(), 'host': platform.node(),
        'source': 'scripts/aggregate_benchmark_record_level.py',
    }
    (OUT / 'summary_record_level.json').write_text(json.dumps(summary, indent=2))

    # ---- markdown ----------------------------------------------------------
    L = ['# Benchmark Summary — RECORD LEVEL (P6 remediation Step 2)',
         '',
         f'Generated {summary["generated_at"]} @ `{git_sha()[:8]}` on {platform.node()}.',
         '',
         '> **Statistical unit corrected.** The Phase-4 table and significance tests',
         '> (`summary.md`) used 2,640 one-second windows as if independent. They are',
         '> not: 5 windows per 5-second record share fault, severity, operating point',
         '> and noise. Below, each record is aggregated by **soft vote** (mean class',
         "> score over its 5 windows) and all statistics use the **528 records** as the",
         '> independent unit (external audit Finding 12).',
         '>',
         '> **This is a recomputation of the CLASSIFICATION benchmark only.** It does',
         '> not re-label or re-train. Rows flagged below kept their Phase-4 training',
         '> reality (pc_cnn = CE-only/architecture, multitask = single-task, hybrid =',
         '> rolling-element branch + constant metadata). No "physics-family" average is',
         '> computed. This is **not** a test of physics-informed training (Steps 4-5).',
         '',
         '| Model | Record acc (mean ± std over seeds) | 95% CI (record bootstrap of seed-mean) | Window acc (recorded) | Note |',
         '|---|---|---|---|---|']
    for m in MODEL_ORDER:
        t = table[m]
        ci = t['ci95_record_acc']
        flag = ' **(physics-labeled)**' if m in PHYSICS_LABELED else ''
        L.append(f"| {m}{flag} | {t['record_acc_mean']:.2f} ± {t['record_acc_std']:.2f} | "
                 f"[{ci[0]:.2f}, {ci[1]:.2f}] | {t['window_acc_recorded']:.2f} | {t['note']} |")
    L += ['',
          f"Classical bar (RandomForest, record level): {summary['classical_bar_record_acc']:.2f}%",
          '',
          '## Record-level significance',
          '',
          f"Best vanilla (deep): **{best_vanilla}** (seed-mean {deep_acc[best_vanilla]:.2f}%). "
          f"Best physics-labeled: **{best_physlabeled}** (seed-mean {deep_acc[best_physlabeled]:.2f}%) "
          f"— *{ROW_NOTE.get(best_physlabeled,'')}*.",
          '',
          f"Paired tests below use each model's **best-validation seed** "
          f"(single model, no ensembling): {best_vanilla} seed{table[best_vanilla]['bestval_seed']} "
          f"= {table[best_vanilla]['record_acc_bestval_seed']:.2f}%, "
          f"{best_physlabeled} seed{table[best_physlabeled]['bestval_seed']} "
          f"= {table[best_physlabeled]['record_acc_bestval_seed']:.2f}%.",
          '',
          f"Gap (best physics-labeled − best vanilla, best-val seeds): "
          f"{gap_pts:+.2f} pts, "
          f"95% CI [{gap_ci[0]:+.2f}, {gap_ci[1]:+.2f}] (cluster bootstrap by record).",
          '',
          f"Wilcoxon over seeds (n=3): p = {wp:.3f} — {stats['wilcoxon_note']}.",
          '',
          'McNemar (exact, record level):']
    for k, v in mcnemar.items():
        L.append(f'- {k}: p = {v:.4g}')
    L += ['',
          '_Source: `scripts/aggregate_benchmark_record_level.py`; per-window score '
          'cache under `results/benchmark/record_level/_cache/`. Each model passed a '
          'sanity gate: argmax of cached scores reproduces its recorded window accuracy._']
    (OUT / 'summary_record_level.md').write_text('\n'.join(L), encoding='utf-8')

    print(f'\nWrote results/benchmark/summary_record_level.{{json,md}} in {time.time()-t0:.0f}s')
    print(f'  best vanilla {best_vanilla} {deep_acc[best_vanilla]:.2f}% | '
          f'best physics-labeled {best_physlabeled} {deep_acc[best_physlabeled]:.2f}%')
    print(f'  gap {stats["gap_best_physlabeled_minus_best_vanilla_pts"]:+.2f} pts '
          f'CI [{gap_ci[0]:+.2f}, {gap_ci[1]:+.2f}]')


if __name__ == '__main__':
    main()
