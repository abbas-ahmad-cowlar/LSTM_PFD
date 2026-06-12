"""
Phase-5 GPU experiment queue (pre-registered PROTOCOL.md §8.2–8.5).

One resume-safe queue covering:
  data_efficiency (§8.2): pc_cnn(w=0.3) & resnet18 × {10,25,50,100}% × seeds
  severity_ood   (§8.3): pc_cnn(w=0.3) & resnet18 × {A: train inc+mild+mod →
                          test severe, B: train mild+mod+sev → test incipient}
  pinn_ablation  (§8.4): pc_cnn × w ∈ {0.1, 0.3, 1.0} × seeds (w=0 = Phase-4
                          runs, reused); every arm also evaluated at 5 dB SNR
  true_metadata  (§8.5): hybrid_pinn × seeds with TRUE per-record operating
                          conditions (vs Phase-4's constant defaults)

A run is COMPLETE when its metrics.json exists; rerunning skips it and
resumes the interrupted run from checkpoint. Training budget identical to
the frozen protocol (Adam 1e-3, batch 64, ≤60 epochs, patience 10, AMP).

Metadata mapping for §8.5 / physics rpm (documented decision):
  rpm       = metadata.speed_rpm
  load [N]  = 1000 * load_percent / 100        (rated load 1 kN)
  viscosity = 0.03 * exp(-0.03*(temperature_C - 60))   (PHYSICS.md §3 law)

Usage (Colab T4 — see experiments/COLAB_PHASE5_RUNBOOK.md):
    python scripts/run_phase5_gpu.py                 # everything (~45 runs)
    python scripts/run_phase5_gpu.py --only data_efficiency
    python scripts/run_phase5_gpu.py --smoke         # 2-epoch pipeline sanity
"""
import argparse
import json
import logging
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset, WindowedView  # noqa: E402
from packages.core.models.model_factory import create_model  # noqa: E402
from utils.constants import FAULT_TYPES, FAULT_TYPE_DISPLAY_NAMES, NUM_CLASSES  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

# Frozen protocol budget
WINDOW, BATCH, MAX_EPOCHS, PATIENCE, LR = 20480, 64, 60, 10, 1e-3
SEEDS = [0, 1, 2]
DATA = PROJECT_ROOT / 'data/generated/dataset_v2.h5'
OUT = PROJECT_ROOT / 'results/phase5'
ABLATION_W = [0.1, 0.3, 1.0]  # w=0 arm = Phase-4 runs (PROTOCOL §8.0)
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
SEVERITY_DIRECTIONS = {
    'A_train_low_test_severe': (['incipient', 'mild', 'moderate'], ['severe']),
    'B_train_high_test_incipient': (['mild', 'moderate', 'severe'], ['incipient']),
}

log = logging.getLogger('phase5')


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


# --------------------------------------------------------------------------
# Data plumbing: filtered/subset record sets, optional per-record metadata
# --------------------------------------------------------------------------

def load_split(split: str, with_metadata: bool = False):
    """Load (signals, labels, severities[, ops]) arrays for a split."""
    with h5py.File(DATA, 'r') as f:
        g = f[split]
        signals = g['signals'][:]
        labels = g['labels'][:]
        severities = np.array([s.decode() for s in g['severities'][:]])
        ops = None
        if with_metadata:
            ops = np.zeros((len(labels), 3), dtype=np.float32)  # rpm, load_N, visc
            for i, raw in enumerate(g['metadata'][:]):
                m = json.loads(raw)
                rpm = float(m['speed_rpm'])
                load_n = 1000.0 * float(m['load_percent']) / 100.0
                visc = 0.03 * math.exp(-0.03 * (float(m['temperature_C']) - 60.0))
                ops[i] = (rpm, load_n, visc)
    return signals, labels, severities, ops


def stratified_fraction_idx(labels, severities, fraction, seed):
    """Record indices: `fraction` of each (class, severity) stratum."""
    if fraction >= 0.999:
        return np.arange(len(labels))
    rng = np.random.default_rng(10_000 + seed)
    keep = []
    strata = {}
    for i, key in enumerate(zip(labels.tolist(), severities.tolist())):
        strata.setdefault(key, []).append(i)
    for key, idxs in sorted(strata.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        keep.extend(idxs[:max(1, round(fraction * len(idxs)))].tolist())
    return np.array(sorted(keep))


class OpsWindowedView(torch.utils.data.Dataset):
    """WindowedView that also yields per-record operating conditions."""

    def __init__(self, base_windows: WindowedView, ops: np.ndarray):
        self.w = base_windows
        self.ops = ops

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        signal, label = self.w[idx]
        rpm, load, visc = self.ops[self.w.record_index(idx)]
        return signal, label, rpm, load, visc


def make_loader(signals, labels, ops=None, shuffle=False, smoke=False):
    base = BearingFaultDataset(signals, labels)
    windows = WindowedView(base, WINDOW)
    ds = OpsWindowedView(windows, ops) if ops is not None else windows
    if smoke:
        gen = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=gen)[:200].tolist()
        ds = torch.utils.data.Subset(ds, idx)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


# --------------------------------------------------------------------------
# Training with optional physics loss / true metadata
# --------------------------------------------------------------------------

def unpack_batch(batch, device):
    if len(batch) == 5:  # ops-aware
        signals, labels, rpm, load, visc = batch
        metadata = {'rpm': rpm.float().to(device), 'load': load.float().to(device),
                    'viscosity': visc.float().to(device)}
    else:
        signals, labels = batch
        metadata = None
    if signals.dim() == 2:
        signals = signals.unsqueeze(1)
    return signals.to(device, non_blocking=True), labels.to(device, non_blocking=True), metadata


def forward_logits(model, model_key, signals, metadata, device):
    if model_key == 'hybrid_pinn':
        if metadata is None:  # constant defaults (Phase-4 convention)
            b = signals.shape[0]
            metadata = {'rpm': torch.full((b,), 3600.0, device=device),
                        'load': torch.full((b,), 500.0, device=device),
                        'viscosity': torch.full((b,), 0.03, device=device)}
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


def run_epoch(model, model_key, loader, criterion, device,
              optimizer=None, scaler=None, physics_w=0.0):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss = tot_correct = tot_n = 0
    tot_phys = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            signals, labels, metadata = unpack_batch(batch, device)
            if training:
                optimizer.zero_grad()

            def compute():
                logits = forward_logits(model, model_key, signals, metadata, device)
                loss = criterion(logits, labels)
                phys = torch.zeros((), device=device)
                if physics_w > 0 and hasattr(model, 'compute_physics_loss'):
                    phys, _ = model.compute_physics_loss(signals, logits, metadata)
                    loss = loss + physics_w * phys
                return logits, loss, phys

            if scaler is not None and training:
                with torch.autocast(device_type='cuda'):
                    logits, loss, phys = compute()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss, phys = compute()
                if training:
                    loss.backward()
                    optimizer.step()
            b = labels.size(0)
            tot_loss += loss.item() * b
            tot_phys += float(phys) * b
            tot_correct += (logits.argmax(1) == labels).sum().item()
            tot_n += b
    return tot_loss / tot_n, 100.0 * tot_correct / tot_n, tot_phys / tot_n


def evaluate(model, model_key, loader, device, ops_aware=False):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for batch in loader:
            signals, labels, metadata = unpack_batch(batch, device)
            if not ops_aware:
                metadata = None
            logits = forward_logits(model, model_key, signals, metadata, device)
            preds.append(logits.argmax(1).cpu().numpy())
            targs.append(labels.cpu().numpy())
    preds, targs = np.concatenate(preds), np.concatenate(targs)
    from sklearn.metrics import confusion_matrix, f1_score
    return {'accuracy': float((preds == targs).mean() * 100),
            'f1_macro': float(f1_score(targs, preds, average='macro')),
            'confusion_matrix': confusion_matrix(
                targs, preds, labels=list(range(NUM_CLASSES))).tolist(),
            'n': int(len(targs))}


def train_run(run_dir: Path, model_key: str, seed: int, loaders: dict,
              physics_w: float, ops_aware: bool, max_epochs: int,
              patience: int, extra_eval: dict, tags: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / 'best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(seed)
    model = create_model(model_key, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    best_val, best_epoch, bad, start = 0.0, 0, 0, 1
    history = {'train_acc': [], 'val_acc': [], 'physics_loss': []}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        best_val, best_epoch = ck['best_val_acc'], ck['epoch']
        history = ck['history']
        start = best_epoch + 1
        log.info("    RESUMED from epoch %d (val %.2f%%)", best_epoch, best_val)

    t0 = time.time()
    for epoch in range(start, max_epochs + 1):
        te = time.time()
        _, tr_acc, tr_phys = run_epoch(model, model_key, loaders['train'],
                                       criterion, device, optimizer, scaler,
                                       physics_w)
        _, va_acc, _ = run_epoch(model, model_key, loaders['val'], criterion,
                                 device, physics_w=0.0)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['physics_loss'].append(tr_phys)
        log.info("    epoch %d/%d (%.0fs) train %.2f%% val %.2f%% phys %.4f",
                 epoch, max_epochs, time.time() - te, tr_acc, va_acc, tr_phys)
        if va_acc > best_val:
            best_val, best_epoch, bad = va_acc, epoch, 0
            torch.save({'epoch': epoch, 'best_val_acc': best_val,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'history': history, 'git_sha': git_sha()}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                log.info("    early stop at %d (best %.2f%% @ %d)",
                         epoch, best_val, best_epoch)
                break

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])

    evals = {name: evaluate(model, model_key, loader, device, ops_aware)
             for name, loader in extra_eval.items()}
    artifact = {
        'model': model_key, 'seed': seed, **tags,
        'physics_weight': physics_w, 'ops_aware': ops_aware,
        'evals': evals, 'history': history,
        'provenance': {'budget': {'lr': LR, 'batch': BATCH,
                                  'max_epochs': max_epochs, 'patience': patience},
                       'best_epoch': best_epoch, 'best_val_acc': best_val,
                       'data': DATA.name, 'device': device,
                       'host': platform.node(), 'git_sha': git_sha(),
                       'wall_time_s': round(time.time() - t0, 1),
                       'finished_at': datetime.now(timezone.utc).isoformat()},
    }
    (run_dir / 'metrics.json').write_text(json.dumps(artifact, indent=2))
    headline = ' | '.join(f"{k} {v['accuracy']:.2f}%" for k, v in evals.items())
    log.info("    DONE: %s", headline)


# --------------------------------------------------------------------------
# Experiment queue assembly
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['data_efficiency', 'severity_ood',
                                           'pinn_ablation', 'true_metadata'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(PROJECT_ROOT / 'logs' / 'phase5_gpu.log'),
                                  logging.StreamHandler()])
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    patience = 99 if args.smoke else PATIENCE
    out_root = OUT.parent / (OUT.name + '_smoke') if args.smoke else OUT

    log.info("Loading splits ...")
    tr_sig, tr_lab, tr_sev, tr_ops = load_split('train', with_metadata=True)
    va_sig, va_lab, va_sev, va_ops = load_split('val', with_metadata=True)
    te_sig, te_lab, te_sev, te_ops = load_split('test', with_metadata=True)
    s5_sig, s5_lab, s5_sev, _ = load_split('test_snr5')

    test_loader = make_loader(te_sig, te_lab, smoke=args.smoke)
    snr5_loader = make_loader(s5_sig, s5_lab, smoke=args.smoke)

    queue = []  # (run_dir, callable)

    def std_loaders(tr_idx, va_idx=None, ops=False):
        va_index = va_idx if va_idx is not None else np.arange(len(va_lab))
        return {
            'train': make_loader(tr_sig[tr_idx], tr_lab[tr_idx],
                                 tr_ops[tr_idx] if ops else None,
                                 shuffle=True, smoke=args.smoke),
            'val': make_loader(va_sig[va_index], va_lab[va_index],
                               va_ops[va_index] if ops else None,
                               smoke=args.smoke),
        }

    # §8.2 data efficiency — pc_cnn(w=0.3) & resnet18 × fractions × seeds
    if args.only in (None, 'data_efficiency'):
        for model_key, w in [('physics_constrained_cnn', 0.3), ('resnet18', 0.0)]:
            for frac in FRACTIONS:
                if frac == 1.0 and model_key == 'resnet18':
                    continue  # = Phase-4 runs, reused in analysis
                for seed in args.seeds:
                    rd = out_root / 'data_efficiency' / f'{model_key}_w{w}' / f'frac{int(frac*100)}' / f'seed{seed}'
                    idx = stratified_fraction_idx(tr_lab, tr_sev, frac, seed)
                    queue.append((rd, lambda rd=rd, mk=model_key, sd=seed, ix=idx, ww=w: train_run(
                        rd, mk, sd, std_loaders(ix), ww, False, max_epochs, patience,
                        {'test': test_loader},
                        {'experiment': 'data_efficiency', 'prereg': '§8.2',
                         'fraction': frac})))

    # §8.3 severity OOD
    if args.only in (None, 'severity_ood'):
        for model_key, w in [('physics_constrained_cnn', 0.3), ('resnet18', 0.0)]:
            for direction, (train_sevs, test_sevs) in SEVERITY_DIRECTIONS.items():
                tr_idx = np.where(np.isin(tr_sev, train_sevs))[0]
                va_idx = np.where(np.isin(va_sev, train_sevs))[0]
                ood_idx = np.where(np.isin(te_sev, test_sevs))[0]
                ood_loader = make_loader(te_sig[ood_idx], te_lab[ood_idx],
                                         smoke=args.smoke)
                for seed in args.seeds:
                    rd = out_root / 'severity_ood' / f'{model_key}_w{w}' / direction / f'seed{seed}'
                    queue.append((rd, lambda rd=rd, mk=model_key, sd=seed,
                                  ti=tr_idx, vi=va_idx, ol=ood_loader, ww=w,
                                  dn=direction: train_run(
                        rd, mk, sd, std_loaders(ti, vi), ww, False, max_epochs,
                        patience, {'ood_test': ol, 'full_test': test_loader},
                        {'experiment': 'severity_ood', 'prereg': '§8.3',
                         'direction': dn})))

    # §8.4 PINN ablation — pc_cnn × w sweep (w=0 = Phase-4), eval clean + 5dB
    if args.only in (None, 'pinn_ablation'):
        full_idx = np.arange(len(tr_lab))
        for w in ABLATION_W:
            for seed in args.seeds:
                rd = out_root / 'pinn_ablation' / f'w{w}' / f'seed{seed}'
                queue.append((rd, lambda rd=rd, sd=seed, ww=w: train_run(
                    rd, 'physics_constrained_cnn', sd, std_loaders(full_idx),
                    ww, False, max_epochs, patience,
                    {'test': test_loader, 'test_snr5': snr5_loader},
                    {'experiment': 'pinn_ablation', 'prereg': '§8.4'})))

    # §8.5 hybrid_pinn with TRUE per-record metadata
    if args.only in (None, 'true_metadata'):
        full_idx = np.arange(len(tr_lab))
        ops_test = make_loader(te_sig, te_lab, te_ops, smoke=args.smoke)
        for seed in args.seeds:
            rd = out_root / 'true_metadata' / 'hybrid_pinn_ops' / f'seed{seed}'
            queue.append((rd, lambda rd=rd, sd=seed, ot=ops_test: train_run(
                rd, 'hybrid_pinn', sd, std_loaders(full_idx, ops=True), 0.0,
                True, max_epochs, patience, {'test_ops': ot},
                {'experiment': 'true_metadata', 'prereg': '§8.5'})))

    log.info("Queue: %d runs (%s)", len(queue), 'SMOKE' if args.smoke else 'FROZEN budget')
    for i, (rd, job) in enumerate(queue, 1):
        rel = rd.relative_to(out_root)
        if (rd / 'metrics.json').exists():
            log.info("[%d/%d] %s — complete, skip", i, len(queue), rel)
            continue
        log.info("[%d/%d] %s — starting", i, len(queue), rel)
        try:
            job()
        except Exception:
            log.exception("[%d/%d] %s FAILED — queue continues", i, len(queue), rel)

    done = sum(1 for rd, _ in queue if (rd / 'metrics.json').exists())
    log.info("Phase-5 GPU queue finished: %d/%d complete.", done, len(queue))


if __name__ == '__main__':
    main()
