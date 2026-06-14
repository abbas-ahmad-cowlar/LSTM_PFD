"""
Phase-5 §8.6 — XAI alignment (8.6a) + MC-dropout calibration (8.6b).

Laptop/CPU. Pre-registered in PROTOCOL.md §8.6.

8.6a — Physics-consistent XAI (contribution C4):
  Integrated-Gradients attributions on test windows for best-vanilla
  (resnet18) and best-physics (fixed pc_cnn). FFT each attribution; measure
  the fraction of attribution *energy* that falls inside the true class's
  PHYSICS.md characteristic-frequency bands vs equally-wide CONTROL bands.
  Hypothesis: the physics-trained model concentrates more attribution energy
  in physically-meaningful bands.

8.6b — Uncertainty & calibration (C5-adjacent):
  MC-dropout (n passes) on the same two models, clean + 5 dB. Report ECE,
  a reliability table, and an accuracy-vs-coverage "reject" curve.
  Hypothesis: physics model better calibrated under noise.

Outputs (json + md + png), provenance-stamped:
  results/xai_alignment/   results/uncertainty/

Usage:
  python scripts/run_xai_calibration.py --smoke         # tiny, ~2 min
  python scripts/run_xai_calibration.py                 # full
  python scripts/run_xai_calibration.py --pc-cnn-ckpt <path.pth>
"""
import argparse
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset, WindowedView  # noqa: E402
from packages.core.models.model_factory import create_model  # noqa: E402
from packages.core.models.physics.fault_signatures import FaultSignatureDatabase  # noqa: E402
from packages.core.explainability.integrated_gradients import IntegratedGradientsExplainer  # noqa: E402
from utils.constants import NUM_CLASSES  # noqa: E402

WINDOW, FS = 20480, 20480
DATA = PROJECT_ROOT / 'data/generated/dataset_v2.h5'
# best-vanilla and best-physics checkpoints (override pc_cnn via --pc-cnn-ckpt)
VANILLA_CKPT = PROJECT_ROOT / 'results/benchmark/deep/resnet18/seed0/best_model.pth'
PCCNN_CKPT_DEFAULT = Path(r'D:\Libraries\results_phase5_fixed_full-20260613T221621Z-3-001'
                          r'\results_phase5_fixed_full\pinn_ablation\w0.3\seed1\best_model.pth')
TOL = 0.15  # ±15% band half-width (matches the physics-loss tolerance)


def git_sha():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def load_model(key, ckpt_path, device):
    model = create_model(key, num_classes=NUM_CLASSES).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    model.load_state_dict(state)
    model.eval()
    return model


def load_test(group, with_rpm=False):
    with h5py.File(DATA, 'r') as f:
        g = f[group]
        signals = g['signals'][:]
        labels = g['labels'][:]
        rpm = None
        if with_rpm and 'metadata' in g:
            rpm = np.array([float(json.loads(m)['speed_rpm']) for m in g['metadata'][:]])
    return signals, labels, rpm


def stratified_window_idx(windows, labels_per_window, per_class, seed=0):
    rng = np.random.default_rng(seed)
    by_c = defaultdict(list)
    for i, c in enumerate(labels_per_window):
        by_c[int(c)].append(i)
    keep = []
    for c, idxs in sorted(by_c.items()):
        idxs = np.array(idxs); rng.shuffle(idxs)
        keep.extend(idxs[:per_class].tolist())
    return sorted(keep)


# --------------------------------------------------------------------------
# 8.6a — attribution-energy alignment with physics frequency bands
# --------------------------------------------------------------------------

def band_energy(power, freqs_hz, centers, tol):
    """Sum of spectral power within ±tol*center of each center frequency."""
    if len(centers) == 0:
        return 0.0
    mask = np.zeros_like(power, dtype=bool)
    for fc in centers:
        if fc <= 0:
            continue
        mask |= np.abs(freqs_hz - fc) <= tol * fc
    return float(power[mask].sum())


def xai_alignment(model, windows, win_idx, rpm_per_window, sigdb, steps, device):
    ig = IntegratedGradientsExplainer(model, device=device)
    freqs_hz = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    per_class_in, per_class_ctrl = defaultdict(list), defaultdict(list)
    for i in win_idx:
        sig, label = windows[i]
        label = int(label)
        if label == 0:  # 'sain' / healthy — no characteristic fault freqs
            continue
        rpm = float(rpm_per_window[i])
        try:
            expected = np.asarray(sigdb.get_expected_frequencies(label, rpm, top_k=5), dtype=float)
            expected = expected[expected > 0]
        except Exception:
            continue
        if expected.size == 0:
            continue
        # control bands: same count/width, shifted off the harmonic grid (×1.37)
        control = expected * 1.37
        control = control[control < FS / 2]
        sig_t = sig if torch.is_tensor(sig) else torch.as_tensor(sig, dtype=torch.float32)
        if sig_t.dim() == 1:
            sig_t = sig_t.unsqueeze(0)  # [1, T]
        attr = ig.explain(sig_t, target_class=label, steps=steps).squeeze().detach().cpu().numpy()
        power = np.abs(np.fft.rfft(attr)) ** 2
        total = power.sum()
        if total <= 0:
            continue
        per_class_in[label].append(band_energy(power, freqs_hz, expected, TOL) / total)
        per_class_ctrl[label].append(band_energy(power, freqs_hz, control, TOL) / total)
    return per_class_in, per_class_ctrl


# --------------------------------------------------------------------------
# 8.6b — MC-dropout calibration
# --------------------------------------------------------------------------

def ece(confidences, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(confidences)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.sum() == 0:
            continue
        acc = correct[m].mean(); conf = confidences[m].mean(); w = m.sum() / n
        e += w * abs(acc - conf)
        rows.append({'lo': float(lo), 'hi': float(hi), 'n': int(m.sum()),
                     'acc': float(acc), 'conf': float(conf)})
    return float(e), rows


def reject_curve(confidences, correct, points=11):
    """accuracy vs coverage: keep the most-confident fraction, report its accuracy."""
    order = np.argsort(-confidences)
    correct_sorted = correct[order]
    out = []
    for cov in np.linspace(0.1, 1.0, points):
        k = max(1, int(cov * len(correct_sorted)))
        out.append({'coverage': float(cov), 'accuracy': float(correct_sorted[:k].mean())})
    return out


def _logits(out):
    if isinstance(out, dict):
        for k in ('fault_logits', 'logits', 'fault'):
            if k in out:
                return out[k]
        return next(iter(out.values()))
    return out[0] if isinstance(out, (tuple, list)) else out


def enable_mc_dropout(model):
    """Correct MC-dropout: dropout layers ON, everything else (incl. BatchNorm)
    stays in eval. (The stock UncertaintyQuantifier uses model.train(), which
    wrecks BatchNorm on small batches -> ~chance accuracy.)"""
    model.eval()
    for m in model.modules():
        if 'Dropout' in m.__class__.__name__:
            m.train()


def _stack_windows(windows, win_idx, device):
    sigs = torch.stack([torch.as_tensor(windows[i][0], dtype=torch.float32) for i in win_idx])
    labs = np.array([int(windows[i][1]) for i in win_idx])
    if sigs.dim() == 2:
        sigs = sigs.unsqueeze(1)  # [N, 1, T]
    return sigs, labs


def calibration(model, windows, win_idx, n_samples, device, batch=64):
    sigs, labs = _stack_windows(windows, win_idx, device)
    # plain-eval accuracy (sanity: should match the ~96% benchmark numbers)
    model.eval()
    ev = []
    with torch.no_grad():
        for s in range(0, len(sigs), batch):
            ev.append(_logits(model(sigs[s:s + batch].to(device))).argmax(1).cpu().numpy())
    eval_acc = float((np.concatenate(ev) == labs).mean())
    # MC-dropout: mean softmax over n_samples passes (dropout on, BN eval)
    probs_sum = None
    for _ in range(n_samples):
        enable_mc_dropout(model)
        batch_probs = []
        with torch.no_grad():
            for s in range(0, len(sigs), batch):
                batch_probs.append(torch.softmax(_logits(model(sigs[s:s + batch].to(device))), 1).cpu().numpy())
        p = np.concatenate(batch_probs)
        probs_sum = p if probs_sum is None else probs_sum + p
    model.eval()
    mc = probs_sum / n_samples
    confs = mc.max(1); preds = mc.argmax(1); corrects = (preds == labs).astype(int)
    e, rows = ece(confs, corrects)
    return {'n': len(labs), 'eval_accuracy': eval_acc, 'mc_accuracy': float(corrects.mean()),
            'ece': e, 'reliability': rows, 'reject_curve': reject_curve(confs, corrects)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--pc-cnn-ckpt', default=str(PCCNN_CKPT_DEFAULT))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    per_class = 3 if args.smoke else 20
    steps = 8 if args.smoke else 32
    n_mc = 5 if args.smoke else 30
    sigdb = FaultSignatureDatabase()

    models = {}
    if VANILLA_CKPT.exists():
        models['resnet18_vanilla'] = load_model('resnet18', VANILLA_CKPT, device)
    else:
        print(f'WARN vanilla ckpt missing: {VANILLA_CKPT}')
    pc = Path(args.pc_cnn_ckpt)
    if pc.exists():
        models['pc_cnn_physics'] = load_model('physics_constrained_cnn', pc, device)
    else:
        print(f'WARN pc_cnn ckpt missing: {pc}')
    if not models:
        sys.exit('No checkpoints found — cannot run §8.6.')

    # clean test windows (+ per-window rpm) for XAI; clean+5dB for calibration
    sig_c, lab_c, rpm_c = load_test('test', with_rpm=True)
    win_c = WindowedView(BearingFaultDataset(sig_c, lab_c), WINDOW)
    lab_per_win = np.array([int(win_c[i][1]) for i in range(len(win_c))])
    rpm_per_win = np.array([rpm_c[win_c.record_index(i)] for i in range(len(win_c))])
    xai_idx = stratified_window_idx(win_c, lab_per_win, per_class)

    prov = {'git_sha': git_sha(), 'host': platform.node(), 'device': device,
            'data': DATA.name, 'finished_at': datetime.now(timezone.utc).isoformat(),
            'vanilla_ckpt': str(VANILLA_CKPT), 'pc_cnn_ckpt': str(pc),
            'per_class': per_class, 'ig_steps': steps, 'mc_samples': n_mc,
            'smoke': args.smoke}

    # ---- 8.6a XAI alignment ----
    print(f'[8.6a] XAI alignment on {len(xai_idx)} windows x {len(models)} models ...')
    xai = {}
    for name, model in models.items():
        pin, pctrl = xai_alignment(model, win_c, xai_idx, rpm_per_win, sigdb, steps, device)
        in_all = [v for vs in pin.values() for v in vs]
        ctrl_all = [v for vs in pctrl.values() for v in vs]
        xai[name] = {
            'mean_in_band_frac': float(np.mean(in_all)) if in_all else None,
            'mean_control_frac': float(np.mean(ctrl_all)) if ctrl_all else None,
            'specificity_ratio': (float(np.mean(in_all) / np.mean(ctrl_all))
                                  if in_all and np.mean(ctrl_all) > 0 else None),
            'n_windows': len(in_all),
            'per_class_in_band': {int(k): float(np.mean(v)) for k, v in pin.items()},
        }
        print(f'  {name}: in-band {xai[name]["mean_in_band_frac"]}, '
              f'control {xai[name]["mean_control_frac"]}, ratio {xai[name]["specificity_ratio"]}')

    # ---- 8.6b calibration (clean + 5 dB) ----
    print(f'[8.6b] MC-dropout calibration (n={n_mc}) ...')
    cal = {}
    cal_idx = xai_idx  # same stratified subsample
    sig5, lab5, _ = load_test('test_snr5')
    win5 = WindowedView(BearingFaultDataset(sig5, lab5), WINDOW)
    for name, model in models.items():
        cal[name] = {
            'clean': calibration(model, win_c, cal_idx, n_mc, device),
            'snr5': calibration(model, win5, cal_idx, n_mc, device),
        }
        cc, c5 = cal[name]['clean'], cal[name]['snr5']
        print(f'  {name}: clean eval/mc {cc["eval_accuracy"]:.3f}/{cc["mc_accuracy"]:.3f} ECE {cc["ece"]:.4f}'
              f' | 5dB eval/mc {c5["eval_accuracy"]:.3f}/{c5["mc_accuracy"]:.3f} ECE {c5["ece"]:.4f}')

    # ---- write artifacts ----
    suffix = '_smoke' if args.smoke else ''
    xdir = PROJECT_ROOT / f'results/xai_alignment{suffix}'
    udir = PROJECT_ROOT / f'results/uncertainty{suffix}'
    xdir.mkdir(parents=True, exist_ok=True); udir.mkdir(parents=True, exist_ok=True)
    (xdir / 'alignment.json').write_text(json.dumps({'provenance': prov, 'results': xai}, indent=2),
                                         encoding='utf-8')
    (udir / 'calibration.json').write_text(json.dumps({'provenance': prov, 'results': cal}, indent=2),
                                           encoding='utf-8')
    print(f'wrote {xdir/"alignment.json"} and {udir/"calibration.json"}')


if __name__ == '__main__':
    main()
