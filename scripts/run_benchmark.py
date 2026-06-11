"""
Benchmark queue runner — Phase 4 (Convergence Plan P4.2).

Sequential, resume-safe execution of the Tier-1 deep matrix per the frozen
experiments/PROTOCOL.md. A run is COMPLETE when its metrics.json exists;
re-invoking the queue skips completed runs, and an interrupted run resumes
from its own checkpoint. Safe to kill at any time.

Usage (office GPU PC, detached — see experiments/OFFICE_PC_RUNBOOK.md):
    python scripts/run_benchmark.py                       # full T1 matrix, seeds 0 1 2
    python scripts/run_benchmark.py --models cnn1d resnet18 --seeds 0
    python scripts/run_benchmark.py --smoke               # 2-epoch tiny sanity queue
"""
import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset, WindowedView  # noqa: E402
from packages.core.models.model_factory import create_model  # noqa: E402
from utils.constants import FAULT_TYPES, FAULT_TYPE_DISPLAY_NAMES, NUM_CLASSES  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

# ---- PROTOCOL.md §2/§3 (frozen) -------------------------------------------
TIER1_MODELS = [
    'cnn1d', 'attention_cnn', 'cnn_lstm', 'resnet18', 'patchtst',
    'hybrid_pinn', 'physics_constrained_cnn', 'multitask_pinn',
]
SEEDS = [0, 1, 2]
WINDOW_LENGTH = 20480
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 10
LR = 1e-3
# ----------------------------------------------------------------------------

log = logging.getLogger('benchmark')


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def normalize_logits(out):
    """Multi-output models (e.g. multitask_pinn) -> fault logits."""
    if isinstance(out, dict):
        for key in ('fault_logits', 'logits', 'fault'):
            if key in out:
                return out[key]
        return next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def model_forward(model, model_key, signals, device):
    if signals.dim() == 2:
        signals = signals.unsqueeze(1)
    signals = signals.to(device, non_blocking=True)
    if model_key == 'hybrid_pinn':
        b = signals.shape[0]
        metadata = {  # PROTOCOL.md §3: identical defaults for all PINNs
            'rpm': torch.full((b,), 3600.0, device=device),
            'load': torch.full((b,), 500.0, device=device),
            'viscosity': torch.full((b,), 0.03, device=device),
        }
        return normalize_logits(model(signals, metadata))
    return normalize_logits(model(signals))


def run_epoch(model, model_key, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for signals, labels in loader:
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            if scaler is not None and training:
                with torch.autocast(device_type='cuda'):
                    logits = model_forward(model, model_key, signals, device)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model_forward(model, model_key, signals, device)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    optimizer.step()
            b = labels.size(0)
            total_loss += loss.item() * b
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += b
    return total_loss / total_n, 100.0 * total_correct / total_n


def train_one_run(model_key, seed, loaders, run_dir, device,
                  max_epochs, patience, subset_note=None):
    """Train one (model, seed) cell to completion; write metrics.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / 'best_model.pth'
    train_loader, val_loader, test_loader = loaders

    set_seed(seed)
    model = create_model(model_key, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    best_val_acc, best_epoch, bad_epochs, start_epoch = 0.0, 0, 0, 1
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if ckpt_path.exists():  # resume an interrupted run
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_val_acc = ckpt['best_val_acc']
        best_epoch = ckpt['epoch']
        start_epoch = best_epoch + 1
        history = ckpt['history']
        log.info("  [%s seed %d] RESUMED from epoch %d (val %.2f%%)",
                 model_key, seed, best_epoch, best_val_acc)

    t0 = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        te = time.time()
        tr_loss, tr_acc = run_epoch(model, model_key, train_loader, criterion,
                                    device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, model_key, val_loader, criterion, device)
        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss); history['val_acc'].append(va_acc)
        log.info("  [%s seed %d] epoch %d/%d (%.0fs) train %.2f%% val %.2f%%",
                 model_key, seed, epoch, max_epochs, time.time() - te, tr_acc, va_acc)

        if va_acc > best_val_acc:
            best_val_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            torch.save({
                'version': 3, 'epoch': epoch, 'best_val_acc': best_val_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model.get_config(), 'history': history,
                'window_length': WINDOW_LENGTH, 'git_sha': git_sha(),
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("  [%s seed %d] early stop at %d (best %.2f%% @ %d)",
                         model_key, seed, epoch, best_val_acc, best_epoch)
                break

    # Final test evaluation with best checkpoint (PROTOCOL.md §1: once)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            logits = model_forward(model, model_key, signals, device)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds); targets = np.concatenate(all_labels)

    from sklearn.metrics import confusion_matrix, f1_score
    display_names = [FAULT_TYPE_DISPLAY_NAMES[ft] for ft in FAULT_TYPES]
    artifact = {
        'model': model_key, 'seed': seed,
        'accuracy': float((preds == targets).mean() * 100),
        'f1_macro': float(f1_score(targets, preds, average='macro')),
        'per_class_accuracy': {
            display_names[c]: float((preds[targets == c] == c).mean() * 100)
            for c in range(NUM_CLASSES)},
        'confusion_matrix': confusion_matrix(targets, preds).tolist(),
        'class_names': display_names,
        'num_test_windows': int(len(targets)),
        'history': history,
        'provenance': {
            'protocol': 'experiments/PROTOCOL.md',
            'data': 'data/generated/dataset_v2.h5',
            'window_length': WINDOW_LENGTH, 'batch_size': BATCH_SIZE,
            'lr': LR, 'max_epochs': max_epochs, 'patience': patience,
            'best_epoch': best_epoch, 'best_val_acc': best_val_acc,
            'epochs_run': len(history['train_loss']),
            'device': device, 'host': platform.node(),
            'git_sha': git_sha(), 'wall_time_s': round(time.time() - t0, 1),
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'subset_note': subset_note,
        },
    }
    (run_dir / 'metrics.json').write_text(json.dumps(artifact, indent=2))
    log.info("  [%s seed %d] DONE: test %.2f%% (f1 %.4f) -> %s",
             model_key, seed, artifact['accuracy'], artifact['f1_macro'], run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/generated/dataset_v2.h5')
    parser.add_argument('--models', nargs='+', default=TIER1_MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--out-root', default='results/benchmark/deep')
    parser.add_argument('--smoke', action='store_true',
                        help='2 epochs, 200-window subset — pipeline sanity only')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(PROJECT_ROOT / 'logs' / 'benchmark_queue.log'),
                  logging.StreamHandler()])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h5 = PROJECT_ROOT / args.data
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    patience = 99 if args.smoke else PATIENCE
    out_root = PROJECT_ROOT / (args.out_root + ('_smoke' if args.smoke else ''))

    # Shared loaders (windows; subset only in smoke mode)
    def windowed(split, shuffle):
        base = BearingFaultDataset.from_hdf5(h5, split=split)
        ds = WindowedView(base, WINDOW_LENGTH)
        if args.smoke:
            gen = torch.Generator().manual_seed(0)
            ds = Subset(ds, torch.randperm(len(ds), generator=gen)[:200].tolist())
        workers = 2 if device == 'cuda' else 0
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=workers, pin_memory=(device == 'cuda'))

    loaders = (windowed('train', True), windowed('val', False), windowed('test', False))
    queue = [(m, s) for m in args.models for s in args.seeds]
    log.info("Queue: %d runs on %s (%s) — protocol %s", len(queue), device,
             platform.node(), 'SMOKE' if args.smoke else 'FROZEN')

    for i, (model_key, seed) in enumerate(queue, 1):
        run_dir = out_root / model_key / f'seed{seed}'
        if (run_dir / 'metrics.json').exists():
            log.info("[%d/%d] %s seed %d — already complete, skipping",
                     i, len(queue), model_key, seed)
            continue
        log.info("[%d/%d] %s seed %d — starting", i, len(queue), model_key, seed)
        try:
            train_one_run(model_key, seed, loaders, run_dir, device, max_epochs,
                          patience, subset_note='SMOKE 200 windows' if args.smoke else None)
        except Exception:
            log.exception("[%d/%d] %s seed %d FAILED — queue continues",
                          i, len(queue), model_key, seed)

    done = sum(1 for m, s in queue
               if (out_root / m / f'seed{s}' / 'metrics.json').exists())
    log.info("Queue finished: %d/%d runs complete.", done, len(queue))


if __name__ == '__main__':
    main()
