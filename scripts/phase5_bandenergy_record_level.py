"""
Record-level confirmation of the band-energy physics reruns (P6 Step 5).

The §8.2/§8.3/§8.4 reruns in results/phase5_bandenergy/ are reported window-level.
Significance must be judged at the RECORD level (the independent unit is the 5-s
record, not the 5 correlated 1-s windows — external audit Finding 12). This
re-evaluates the retained checkpoints (no retraining), soft-votes the windows of
each record, and recomputes accuracy + cluster-bootstrap CIs + exact McNemar at
the record level — the same method as scripts/aggregate_benchmark_record_level.py.

A SANITY GATE asserts the window-level accuracy from each re-eval reproduces the
value recorded in that run's metrics.json (catches checkpoint-load / eval-set
reconstruction bugs — the I4 lesson). Per-checkpoint record-probs are cached.

Outputs: results/phase5_bandenergy/summary_record_level.{json,md}

Usage: python scripts/phase5_bandenergy_record_level.py
"""
import json
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset, WindowedView  # noqa: E402
from packages.core.models.model_factory import create_model  # noqa: E402
from utils.constants import NUM_CLASSES  # noqa: E402

DATA = PROJECT_ROOT / 'data/generated/dataset_v2.h5'
BE = PROJECT_ROOT / 'results/phase5_bandenergy'
BENCH = PROJECT_ROOT / 'results/benchmark/deep'
CACHE = BE / '_record_cache'
WINDOW = 20480
SEEDS = [0, 1, 2]
B_BOOT, BOOT_SEED = 10000, 42


def load_split(split):
    with h5py.File(DATA, 'r') as f:
        g = f[split]
        sig = g['signals'][:]
        lab = g['labels'][:]
        sev = np.array([s.decode() for s in g['severities'][:]]) if 'severities' in g else None
    return sig, lab, sev


def record_probs(model_key, ckpt_path: Path, signals, labels, expected_acc, split_tag):
    """Per-record soft-vote probs [n_rec, C] from a checkpoint, cached, sanity-gated.

    split_tag distinguishes the eval set (test / snr5 / ood_*) so the cache key
    never collides between splits of the same checkpoint+record-count.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r'[^A-Za-z0-9]+', '_', str(ckpt_path.relative_to(PROJECT_ROOT))) + f'_{split_tag}_{len(labels)}'
    cache = CACHE / f'{key}.npy'
    base = BearingFaultDataset(signals, labels)
    win = WindowedView(base, WINDOW)
    wpr = win.windows_per_record
    n_rec = len(win) // wpr
    win_targets = np.repeat(np.asarray(labels), wpr)
    if cache.exists():
        probs = np.load(cache)
    else:
        loader = DataLoader(win, batch_size=64, shuffle=False, num_workers=0)
        model = create_model(model_key, num_classes=NUM_CLASSES)
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        out = []
        with torch.no_grad():
            for sig, _ in loader:
                if sig.dim() == 2:
                    sig = sig.unsqueeze(1)
                o = model(sig)
                if isinstance(o, dict):
                    o = next((o[k] for k in ('fault_logits', 'logits', 'fault') if k in o),
                             next(iter(o.values())))
                elif isinstance(o, (tuple, list)):
                    o = o[0]
                out.append(torch.softmax(o, dim=1).numpy())
        probs = np.concatenate(out).astype(np.float64)
        np.save(cache, probs)
    # sanity gate: window-level acc must match the recorded eval accuracy
    w_acc = 100.0 * (probs.argmax(1) == win_targets).mean()
    if expected_acc is not None and abs(w_acc - expected_acc) > 0.2:
        raise SystemExit(f'SANITY FAIL {ckpt_path}: window acc {w_acc:.2f} vs recorded {expected_acc:.2f}')
    rec_probs = probs.reshape(n_rec, wpr, -1).mean(axis=1)  # soft vote
    return rec_probs, np.asarray(labels)


def acc(rec_probs, targets):
    return 100.0 * (rec_probs.argmax(1) == targets).mean()


def mcnemar_p(a_correct, b_correct):
    from scipy.stats import binomtest
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


def metrics_for(run_dir: Path):
    return json.loads((run_dir / 'metrics.json').read_text())


def ev_acc(m, key):
    v = m['evals'][key]
    return v['accuracy'] if isinstance(v, dict) else v


def best_val_seed(run_dirs):
    """Pick the seed whose run has the highest val accuracy (for paired tests)."""
    def vacc(rd):
        m = metrics_for(rd)
        h = m.get('history', {})
        return max(h.get('val_acc', [0])) if h.get('val_acc') else m['provenance'].get('best_val', 0)
    return max(run_dirs, key=vacc)


def main():
    t0 = time.time()
    rng = np.random.default_rng(BOOT_SEED)
    te_sig, te_lab, te_sev = load_split('test')
    s5_sig, s5_lab, _ = load_split('test_snr5')
    n_rec = len(te_lab)
    idx_boot = rng.integers(0, n_rec, size=(B_BOOT, n_rec))
    print(f'test records: {n_rec}')
    out = {'scope': 'record-level (528-record) recompute of the band-energy reruns '
                    '§8.2/§8.3/§8.4; soft-vote per record; cluster-bootstrap + exact McNemar.'}

    # ===== §8.4 ablation: pc_cnn w-sweep on test + test_snr5 ==================
    print('\n§8.4 ablation ...')
    abl = {}  # w -> {'clean':[acc per seed], 'snr5':[...], 'snr5_rep_correct':vec}
    for w, dirs in [(0.0, [BENCH / 'physics_constrained_cnn' / f'seed{s}' for s in SEEDS])] + \
                   [(w, [BE / 'pinn_ablation' / f'w{w}' / f'seed{s}' for s in SEEDS]) for w in (0.1, 0.3, 1.0)]:
        clean_acc, snr5_acc, snr5_correct = [], [], {}
        for rd in dirs:
            m = metrics_for(rd)
            # w=0 (benchmark) has only 'accuracy' (clean) in its schema; eval snr5 fresh
            exp_clean = ev_acc(m, 'test') if 'evals' in m else m['accuracy']
            rp, tg = record_probs('physics_constrained_cnn', rd / 'best_model.pth', te_sig, te_lab, exp_clean, 'test')
            clean_acc.append(acc(rp, tg))
            exp_snr5 = ev_acc(m, 'test_snr5') if ('evals' in m and 'test_snr5' in m['evals']) else None
            rp5, tg5 = record_probs('physics_constrained_cnn', rd / 'best_model.pth', s5_sig, s5_lab, exp_snr5, 'snr5')
            snr5_acc.append(acc(rp5, tg5))
            snr5_correct[rd] = (rp5.argmax(1) == tg5)
        rep = best_val_seed(dirs)
        abl[w] = {'clean': clean_acc, 'snr5': snr5_acc, 'snr5_rep': snr5_correct[rep]}
        print(f'  w={w}: clean {np.mean(clean_acc):.2f}±{np.std(clean_acc):.2f} | '
              f'5dB {np.mean(snr5_acc):.2f}±{np.std(snr5_acc):.2f}')

    # vanilla resnet18 @5dB (same-backbone reference)
    rn_dirs = [BENCH / 'resnet18' / f'seed{s}' for s in SEEDS]
    rn5_acc, rn5_correct = [], {}
    for rd in rn_dirs:
        rp5, tg5 = record_probs('resnet18', rd / 'best_model.pth', s5_sig, s5_lab, None, 'snr5')
        rn5_acc.append(acc(rp5, tg5)); rn5_correct[rd] = (rp5.argmax(1) == tg5)
    rn_rep = best_val_seed(rn_dirs)
    print(f'  resnet18 5dB {np.mean(rn5_acc):.2f}±{np.std(rn5_acc):.2f}')

    def gap_ci(a_vec, b_vec):
        g = (a_vec.astype(float) - b_vec.astype(float))
        gb = g[idx_boot].mean(1) * 100
        return [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]

    out['ablation_8_4'] = {
        'table': {str(w): {'clean': [round(float(np.mean(v['clean'])), 2), round(float(np.std(v['clean'])), 2)],
                           'snr5': [round(float(np.mean(v['snr5'])), 2), round(float(np.std(v['snr5'])), 2)]}
                  for w, v in abl.items()},
        'mcnemar_5dB_w0_vs_w1.0': mcnemar_p(abl[0.0]['snr5_rep'], abl[1.0]['snr5_rep']),
        'gap_5dB_w1.0_minus_w0_pts': round(float(np.mean(abl[1.0]['snr5']) - np.mean(abl[0.0]['snr5'])), 2),
        'gap_5dB_w1.0_minus_w0_ci95': gap_ci(abl[1.0]['snr5_rep'], abl[0.0]['snr5_rep']),
        'mcnemar_5dB_pccnn_w1.0_vs_resnet18': mcnemar_p(abl[1.0]['snr5_rep'], rn5_correct[rn_rep]),
        'gap_5dB_pccnn_w1.0_minus_resnet18_pts': round(float(np.mean(abl[1.0]['snr5']) - np.mean(rn5_acc)), 2),
        'gap_5dB_pccnn_w1.0_minus_resnet18_ci95': gap_ci(abl[1.0]['snr5_rep'], rn5_correct[rn_rep]),
        'resnet18_5dB': [round(float(np.mean(rn5_acc)), 2), round(float(np.std(rn5_acc)), 2)],
    }

    # ===== §8.2 data efficiency: record-level test acc =======================
    print('\n§8.2 data efficiency ...')
    de = {}
    for mk, sub in [('physics_constrained_cnn', 'physics_constrained_cnn_w0.3'), ('resnet18', 'resnet18_w0.0')]:
        for frac in (10, 25, 50, 100):
            dirs = ([BENCH / 'resnet18' / f'seed{s}' for s in SEEDS] if (mk == 'resnet18' and frac == 100)
                    else [BE / 'data_efficiency' / sub / f'frac{frac}' / f'seed{s}' for s in SEEDS])
            if not all((d / 'best_model.pth').exists() for d in dirs):
                continue
            accs = []
            for rd in dirs:
                m = metrics_for(rd)
                exp = ev_acc(m, 'test') if 'evals' in m else m['accuracy']
                rp, tg = record_probs(mk, rd / 'best_model.pth', te_sig, te_lab, exp, 'test')
                accs.append(acc(rp, tg))
            de[(mk, frac)] = [round(float(np.mean(accs)), 2), round(float(np.std(accs)), 2)]
            print(f'  {mk} {frac}%: {np.mean(accs):.2f}±{np.std(accs):.2f}')
    out['data_efficiency_8_2'] = {f'{mk}_{frac}': v for (mk, frac), v in de.items()}

    # ===== §8.3 severity OOD: record-level on the held-out severity slice =====
    print('\n§8.3 severity-OOD ...')
    DIRS = {'A_train_low_test_severe': ['severe'], 'B_train_high_test_incipient': ['incipient']}
    so = {}
    for direction, test_sevs in DIRS.items():
        ood_idx = np.where(np.isin(te_sev, test_sevs))[0]
        o_sig, o_lab = te_sig[ood_idx], te_lab[ood_idx]
        n_ood = len(ood_idx)
        boot_o = rng.integers(0, n_ood, size=(B_BOOT, n_ood))
        rep_correct = {}
        for mk, sub in [('physics_constrained_cnn', 'physics_constrained_cnn_w0.3'), ('resnet18', 'resnet18_w0.0')]:
            dirs = [BE / 'severity_ood' / sub / direction / f'seed{s}' for s in SEEDS]
            accs = []
            for rd in dirs:
                m = metrics_for(rd)
                exp = ev_acc(m, 'ood_test') if 'evals' in m else None
                rp, tg = record_probs(mk, rd / 'best_model.pth', o_sig, o_lab, exp, f'ood_{direction}')
                accs.append(acc(rp, tg))
                rep_correct.setdefault(mk, {})[rd] = (rp.argmax(1) == tg)
            so[(direction, mk)] = [round(float(np.mean(accs)), 2), round(float(np.std(accs)), 2)]
            print(f'  {direction} {mk}: {np.mean(accs):.2f}±{np.std(accs):.2f}')
        pc_rep = rep_correct['physics_constrained_cnn'][best_val_seed(
            [BE / 'severity_ood' / 'physics_constrained_cnn_w0.3' / direction / f'seed{s}' for s in SEEDS])]
        rn_rep_o = rep_correct['resnet18'][best_val_seed(
            [BE / 'severity_ood' / 'resnet18_w0.0' / direction / f'seed{s}' for s in SEEDS])]
        g = (pc_rep.astype(float) - rn_rep_o.astype(float))
        so[(direction, 'mcnemar_pc_vs_rn')] = mcnemar_p(pc_rep, rn_rep_o)
        so[(direction, 'gap_ci95')] = [float(np.percentile(g[boot_o].mean(1) * 100, 2.5)),
                                       float(np.percentile(g[boot_o].mean(1) * 100, 97.5))]
    out['severity_ood_8_3'] = {f'{d}__{k}': v for (d, k), v in so.items()}

    (BE / 'summary_record_level.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nWrote results/phase5_bandenergy/summary_record_level.json in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
