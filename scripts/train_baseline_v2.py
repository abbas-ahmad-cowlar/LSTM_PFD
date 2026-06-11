"""
CNN1D re-baseline on Dataset v2 with 1 s windows (Convergence Plan P3.5).

Trains CNN1D on WindowedView(train) with early stopping, evaluates on the
windowed test split, and writes a full results artifact (metrics.json with
provenance + confusion matrix PNG) to results/cnn1d_v2_baseline/.

This number becomes the reference for the Phase 4 benchmark matrix.

Usage:
    python scripts/train_baseline_v2.py                 # full run (overnight CPU)
    python scripts/train_baseline_v2.py --epochs 2 --subset 200   # smoke
"""
import argparse
import json
import logging
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

WINDOW_LENGTH = 20480  # 1 s @ 20480 Hz (DATASET_V2.md §A)


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def make_loader(h5_path, split, batch_size, shuffle, subset=None, seed=42):
    base = BearingFaultDataset.from_hdf5(h5_path, split=split)
    windows = WindowedView(base, WINDOW_LENGTH)
    ds = windows
    if subset is not None and subset < len(windows):
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(windows), generator=gen)[:subset].tolist()
        ds = Subset(windows, idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, len(ds)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for signals, labels in loader:
            if signals.dim() == 2:
                signals = signals.unsqueeze(1)
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch = labels.size(0)
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += batch
    return total_loss / total_n, 100.0 * total_correct / total_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/generated/dataset_v2.h5')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--subset', type=int, default=None,
                        help='Limit train windows (smoke runs)')
    parser.add_argument('--output', default='results/cnn1d_v2_baseline')
    parser.add_argument('--checkpoint', default='checkpoints/cnn_v2/best_model.pth')
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = PROJECT_ROOT / args.checkpoint
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(PROJECT_ROOT / 'logs' / 'train_baseline_v2.log'),
                  logging.StreamHandler()])
    log = logging.getLogger('baseline_v2')

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h5 = PROJECT_ROOT / args.data

    train_loader, n_train = make_loader(h5, 'train', args.batch_size, True, args.subset)
    val_loader, n_val = make_loader(h5, 'val', args.batch_size, False, args.subset)
    test_loader, n_test = make_loader(h5, 'test', args.batch_size, False, None)
    log.info("Windows — train: %d, val: %d, test: %d (window=%d)",
             n_train, n_val, n_test, WINDOW_LENGTH)

    model = create_model('cnn1d', num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc, best_epoch, bad_epochs = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        te = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss); history['val_acc'].append(va_acc)
        log.info("Epoch %d/%d (%.0fs) - train %.4f/%.2f%% - val %.4f/%.2f%%",
                 epoch, args.epochs, time.time() - te, tr_loss, tr_acc, va_loss, va_acc)

        if va_acc > best_val_acc:
            best_val_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            torch.save({
                'version': 3, 'epoch': epoch, 'best_val_acc': best_val_acc,
                'model_state_dict': model.state_dict(),
                'model_config': model.get_config(),
                'window_length': WINDOW_LENGTH,
                'data': str(args.data), 'git_sha': git_sha(),
            }, ckpt_path)
            log.info("  checkpoint saved (val_acc %.2f%%)", best_val_acc)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("Early stopping at epoch %d (best %.2f%% @ %d)",
                         epoch, best_val_acc, best_epoch)
                break

    # ---- Test evaluation with the best checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            if signals.dim() == 2:
                signals = signals.unsqueeze(1)
            logits = model(signals.to(device))
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds); targets = np.concatenate(all_labels)

    from sklearn.metrics import confusion_matrix, f1_score
    acc = float((preds == targets).mean() * 100)
    cm = confusion_matrix(targets, preds)
    display_names = [FAULT_TYPE_DISPLAY_NAMES[ft] for ft in FAULT_TYPES]

    artifact = {
        'accuracy': acc,
        'f1_macro': float(f1_score(targets, preds, average='macro')),
        'per_class_accuracy': {
            display_names[c]: float((preds[targets == c] == c).mean() * 100)
            for c in range(NUM_CLASSES)
        },
        'confusion_matrix': cm.tolist(),
        'class_names': display_names,
        'num_test_windows': int(len(targets)),
        'history': history,
        'provenance': {
            'model': 'cnn1d', 'window_length': WINDOW_LENGTH,
            'data': str(args.data), 'checkpoint': str(args.checkpoint),
            'best_epoch': best_epoch, 'best_val_acc': best_val_acc,
            'epochs_run': len(history['train_loss']),
            'train_windows': n_train, 'seed': 42, 'device': device,
            'git_sha': git_sha(),
            'wall_time_s': round(time.time() - t0, 1),
            'finished_at': datetime.now(timezone.utc).isoformat(),
        },
    }
    (out_dir / 'metrics.json').write_text(json.dumps(artifact, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'CNN1D v2 baseline — test acc {acc:.2f}% ({len(targets)} windows)')
    thr = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > thr else 'black')
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrix.png', dpi=150)

    log.info("TEST: acc %.2f%%, f1_macro %.4f on %d windows -> %s",
             acc, artifact['f1_macro'], len(targets), out_dir)


if __name__ == '__main__':
    main()
