"""
Noise robustness (P5.3, pre-registered PROTOCOL.md §8.1).

Evaluates ALL frozen Phase-4 checkpoints on the v2 SNR test variants
(test_snr20/10/5) — no retraining. Clean-test accuracies come from the
Phase-4 metrics.json (test set is never re-evaluated, PROTOCOL §1).

Resume-safe: per (model, seed) an output metrics.json marks completion.

Outputs: results/noise_robustness/<model>/seed<k>/metrics.json
         results/noise_robustness/summary.{json,md,png}

Usage:
    python scripts/run_noise_robustness.py            # evaluate (hours, CPU)
    python scripts/run_noise_robustness.py --summarize-only
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
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset, WindowedView  # noqa: E402
from packages.core.models.model_factory import create_model  # noqa: E402
from utils.constants import NUM_CLASSES  # noqa: E402

WINDOW = 20480
DATA = PROJECT_ROOT / 'data/generated/dataset_v2.h5'
DEEP = PROJECT_ROOT / 'results/benchmark/deep'
OUT = PROJECT_ROOT / 'results/noise_robustness'
SNR_SPLITS = ['test_snr20', 'test_snr10', 'test_snr5']
PHYSICS = {'hybrid_pinn', 'physics_constrained_cnn', 'multitask_pinn'}


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def model_forward(model, model_key, signals):
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


def accuracy_on(model, model_key, loader) -> float:
    correct, n = 0, 0
    with torch.no_grad():
        for signals, labels in loader:
            preds = model_forward(model, model_key, signals).argmax(1)
            correct += (preds == labels).sum().item()
            n += labels.numel()
    return 100.0 * correct / n


def evaluate_all() -> None:
    loaders = {}
    for split in SNR_SPLITS:
        base = BearingFaultDataset.from_hdf5(DATA, split=split)
        loaders[split] = DataLoader(WindowedView(base, WINDOW), batch_size=64,
                                    shuffle=False, num_workers=0)

    jobs = sorted(DEEP.glob('*/seed*/metrics.json'))
    for i, mj in enumerate(jobs, 1):
        bench = json.loads(mj.read_text())
        model_key, seed = bench['model'], bench['seed']
        run_out = OUT / model_key / f'seed{seed}'
        if (run_out / 'metrics.json').exists():
            print(f'[{i}/{len(jobs)}] {model_key} seed {seed} — done, skip')
            continue
        t0 = time.time()
        try:
            model = create_model(model_key, num_classes=NUM_CLASSES)
            ckpt = torch.load(mj.parent / 'best_model.pth', map_location='cpu',
                              weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            accs = {'clean': bench['accuracy']}  # from Phase 4, never re-evaluated
            for split in SNR_SPLITS:
                accs[split.replace('test_', '')] = accuracy_on(model, model_key,
                                                               loaders[split])
            run_out.mkdir(parents=True, exist_ok=True)
            (run_out / 'metrics.json').write_text(json.dumps({
            'model': model_key, 'seed': seed, 'accuracy_by_snr': accs,
            'degradation_clean_to_5db': accs['clean'] - accs['snr5'],
            'provenance': {'preregistration': 'PROTOCOL.md §8.1',
                           'checkpoint': str(mj.parent / 'best_model.pth'),
                           'host': platform.node(), 'git_sha': git_sha(),
                           'wall_time_s': round(time.time() - t0, 1),
                           'finished_at': datetime.now(timezone.utc).isoformat()},
            }, indent=2))
            print(f'[{i}/{len(jobs)}] {model_key} seed {seed}: '
                  + ' | '.join(f'{k} {v:.2f}' for k, v in accs.items())
                  + f'  ({time.time()-t0:.0f}s)')
        except Exception as e:  # one bad checkpoint must not kill the queue
            print(f'[{i}/{len(jobs)}] {model_key} seed {seed} FAILED: '
                  f'{type(e).__name__}: {e}')


def summarize() -> None:
    rows = {}
    for mj in sorted(OUT.glob('*/seed*/metrics.json')):
        m = json.loads(mj.read_text())
        rows.setdefault(m['model'], []).append(m)

    levels = ['clean', 'snr20', 'snr10', 'snr5']
    summary = {}
    for model, ms in rows.items():
        summary[model] = {
            lv: {'mean': float(np.mean([x['accuracy_by_snr'][lv] for x in ms])),
                 'std': float(np.std([x['accuracy_by_snr'][lv] for x in ms]))}
            for lv in levels}
        summary[model]['degradation'] = {
            'mean': float(np.mean([x['degradation_clean_to_5db'] for x in ms])),
            'std': float(np.std([x['degradation_clean_to_5db'] for x in ms]))}

    fam = {
        'physics_family_mean_degradation': float(np.mean(
            [summary[m]['degradation']['mean'] for m in summary if m in PHYSICS])),
        'vanilla_family_mean_degradation': float(np.mean(
            [summary[m]['degradation']['mean'] for m in summary if m not in PHYSICS])),
    }
    out = {'preregistration': 'PROTOCOL.md §8.1', 'per_model': summary,
           'family_comparison': fam, 'git_sha': git_sha(),
           'generated_at': datetime.now(timezone.utc).isoformat()}
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2))

    lines = ['# Noise robustness (P5.3, prereg §8.1)', '',
             '| Model | clean | 20 dB | 10 dB | 5 dB | Δ(clean→5dB) |',
             '|---|---|---|---|---|---|']
    for model in sorted(summary, key=lambda m: summary[m]['degradation']['mean']):
        s = summary[model]
        flag = ' **(physics)**' if model in PHYSICS else ''
        lines.append(f"| {model}{flag} | " + " | ".join(
            f"{s[lv]['mean']:.2f}±{s[lv]['std']:.2f}" for lv in levels)
            + f" | {s['degradation']['mean']:.2f}±{s['degradation']['std']:.2f} |")
    lines += ['', f"Family mean degradation — physics: "
                  f"{fam['physics_family_mean_degradation']:.2f} | vanilla: "
                  f"{fam['vanilla_family_mean_degradation']:.2f}"]
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = [0, 1, 2, 3]
    for model, s in summary.items():
        style = '-o' if model in PHYSICS else '--s'
        ax.errorbar(x, [s[lv]['mean'] for lv in levels],
                    yerr=[s[lv]['std'] for lv in levels],
                    fmt=style, capsize=3, label=model)
    ax.set_xticks(x); ax.set_xticklabels(['clean', '20 dB', '10 dB', '5 dB'])
    ax.set_ylabel('Test accuracy (%)'); ax.set_xlabel('Test SNR')
    ax.set_title('Noise robustness — frozen Phase-4 checkpoints (3 seeds)')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'summary.png', dpi=200)
    print('\nSummary -> results/noise_robustness/summary.{json,md,png}')
    print(f"Family degradation: physics {fam['physics_family_mean_degradation']:.2f} "
          f"vs vanilla {fam['vanilla_family_mean_degradation']:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if not args.summarize_only:
        evaluate_all()
    summarize()
