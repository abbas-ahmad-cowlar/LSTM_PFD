"""INDEPENDENT auditor recomputation of the §8.8 n=12 strengthen grid.

Does NOT use the maintainers' cache or their analysis script. Loads each of the
48 checkpoints fresh, forward-passes test + test_snr5, soft-votes windows -> record
(528) predictions, computes clean->5dB degradation per seed, seed-level Wilcoxon vs
CE-only (n=12), and robust-seed counts. Also runs a sanity gate: the window-level
accuracy from this fresh pass must match each metrics.json recorded eval accuracy.

Auditor: Claude/Opus  2026-06-24
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch.set_num_threads(8)  # use all logical processors for this single process
from packages.core.models.model_factory import create_model  # noqa: E402

DATA = ROOT / 'data/generated/dataset_v2.h5'
P7 = ROOT / 'results/p7_strengthen'
WINDOW = 20480
SEEDS = list(range(12))
TAU = 1.0
ARMS = {
    'w0_CEonly':     P7 / 'pinn_ablation' / 'w0.0',
    'w1.0_correct':  P7 / 'pinn_ablation' / 'w1.0',
    'w1.0_scramble': P7 / 'pinn_ablation_scramble' / 'w1.0',
    'w1.0_random':   P7 / 'pinn_ablation_random' / 'w1.0',
}


def load_split(split):
    with h5py.File(DATA, 'r') as f:
        g = f[split]
        return g['signals'][:].astype(np.float32), g['labels'][:].astype(np.int64)


def record_softvote(ckpt, sig, lab):
    """Fresh forward pass -> (window_acc, record_acc, record_correct_vector)."""
    n_rec, L = sig.shape
    wpr = L // WINDOW
    wins = sig.reshape(n_rec, wpr, WINDOW).reshape(n_rec * wpr, WINDOW)  # [N, 20480]
    win_targets = np.repeat(lab, wpr)
    model = create_model('physics_constrained_cnn', num_classes=11)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    probs = np.empty((wins.shape[0], 11), dtype=np.float64)
    BS = 256
    with torch.inference_mode():
        for i in range(0, wins.shape[0], BS):
            x = torch.from_numpy(wins[i:i + BS]).unsqueeze(1)  # [b,1,20480]
            o = model(x)
            probs[i:i + BS] = torch.softmax(o, dim=1).numpy()
    win_acc = 100.0 * (probs.argmax(1) == win_targets).mean()
    rec_probs = probs.reshape(n_rec, wpr, 11).mean(axis=1)  # soft vote
    rec_pred = rec_probs.argmax(1)
    rec_acc = 100.0 * (rec_pred == lab).mean()
    return win_acc, rec_acc, (rec_pred == lab)


def main():
    te_sig, te_lab = load_split('test')
    s5_sig, s5_lab = load_split('test_snr5')
    print(f'records: test={len(te_lab)} snr5={len(s5_lab)}  seeds={len(SEEDS)}\n')
    data = {}
    sanity_fail = []
    for name, root in ARMS.items():
        clean, snr5, degr = [], [], []
        rep_correct_5 = {}
        for s in SEEDS:
            rd = root / f'seed{s}'
            m = json.loads((rd / 'metrics.json').read_text())
            ev = m['evals']
            exp_clean = ev['test']['accuracy']
            exp_s5 = ev['test_snr5']['accuracy']
            wc, rc, _ = record_softvote(rd / 'best_model.pth', te_sig, te_lab)
            wc5, rc5, corr5 = record_softvote(rd / 'best_model.pth', s5_sig, s5_lab)
            # sanity gate vs recorded window-level eval
            if abs(wc - exp_clean) > 0.2:
                sanity_fail.append((str(rd), 'test', wc, exp_clean))
            if abs(wc5 - exp_s5) > 0.2:
                sanity_fail.append((str(rd), 'snr5', wc5, exp_s5))
            clean.append(rc); snr5.append(rc5); degr.append(rc - rc5)
            rep_correct_5[s] = corr5
        clean = np.array(clean); snr5 = np.array(snr5); degr = np.array(degr)
        n_robust = int(np.sum(degr < TAU))
        data[name] = dict(clean=clean, snr5=snr5, degr=degr, n_robust=n_robust,
                          rep5=rep_correct_5)
        print(f'  {name:14s}: clean {clean.mean():.2f}±{clean.std():.2f} | '
              f'5dB {snr5.mean():.2f}±{snr5.std():.2f} | '
              f'degr mean {degr.mean():.2f} median {np.median(degr):.2f} '
              f'min {degr.min():.2f} max {degr.max():.2f} | robust {n_robust}/12')
        print(f'                  degr/seed: {[round(float(x),2) for x in degr]}')

    print('\n  SANITY GATE (window acc vs metrics.json, tol 0.2):',
          'ALL PASS' if not sanity_fail else f'{len(sanity_fail)} FAILURES')
    for f in sanity_fail:
        print('    FAIL', f)

    print('\n  seed-level Wilcoxon signed-rank vs CE-only (paired per-seed degradation, n=12):')
    ce = data['w0_CEonly']['degr']
    for name in ('w1.0_correct', 'w1.0_scramble', 'w1.0_random'):
        d = data[name]['degr']
        diff = ce - d  # >0 => arm more robust than CE-only
        try:
            p_two = float(wilcoxon(ce, d, zero_method='wilcox').pvalue)
            p_less = float(wilcoxon(ce, d, alternative='greater', zero_method='wilcox').pvalue)
        except ValueError:
            p_two = p_less = 1.0
        print(f'    {name:14s}: median Δdegr {np.median(diff):+.2f} mean {np.mean(diff):+.2f} | '
              f'p(2-sided) {p_two:.4g} | p(more-robust) {p_less:.4g} | '
              f'{int(np.sum(diff>0))}/12 beat CE | robust {data[name]["n_robust"]}/12')

    print('\n  robust-seed counts (degr<1.0): ' +
          ' | '.join(f'{k} {v["n_robust"]}/12' for k, v in data.items()))


if __name__ == '__main__':
    main()
