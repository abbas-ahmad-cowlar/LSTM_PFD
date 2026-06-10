"""
HybridPINN 2-epoch CPU sanity training (Convergence Plan P1.1 DoD).

Trains HybridPINN (cnn1d backbone) on a small subset of the real dataset
for 2 epochs. Pass criteria: completes without error, loss decreases,
no NaN. This is a trainability proof, not a benchmark — full PINN training
happens in Phase 4 on GPU.

Usage:
    python scripts/pinn_sanity_train.py [--subset 256] [--epochs 2]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset  # noqa: E402
from packages.core.models.model_factory import create_model  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/generated/dataset.h5')
    parser.add_argument('--model', default='hybrid_pinn',
                        help='Any key from the curated MODEL_REGISTRY')
    parser.add_argument('--subset', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    log_path = PROJECT_ROOT / 'logs' / f'{args.model}_sanity.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger('pinn_sanity')

    set_seed(42)
    device = 'cpu'

    dataset = BearingFaultDataset.from_hdf5(PROJECT_ROOT / args.data, split='train')
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator)[: args.subset].tolist()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=True,
        num_workers=0,
    )
    log.info("Subset: %d samples, batch_size=%d", len(indices), args.batch_size)

    extra = {'backbone': 'cnn1d'} if args.model == 'hybrid_pinn' else {}
    model = create_model(args.model, num_classes=11, **extra).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s: %s params", args.model, f"{n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_losses = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            batch = signals.shape[0]

            optimizer.zero_grad()
            if args.model == 'hybrid_pinn':
                metadata = {
                    'rpm': torch.full((batch,), 3600.0),
                    'load': torch.full((batch,), 500.0),
                    'viscosity': torch.full((batch,), 0.03),
                }
                logits = model(signals, metadata)
            else:
                logits = model(signals)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            elif isinstance(logits, dict):
                logits = next(iter(logits.values()))
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                log.error("NON-FINITE LOSS at epoch %d — SANITY FAIL", epoch)
                sys.exit(1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += batch

        epoch_loss = total_loss / total_n
        epoch_losses.append(epoch_loss)
        log.info(
            "Epoch %d/%d (%.1fs) - loss: %.4f, acc: %.2f%%",
            epoch, args.epochs, time.time() - t0,
            epoch_loss, 100.0 * total_correct / total_n,
        )

    if epoch_losses[-1] < epoch_losses[0]:
        log.info("SANITY PASS: loss decreased %.4f -> %.4f, no NaN",
                 epoch_losses[0], epoch_losses[-1])
    else:
        log.error("SANITY FAIL: loss did not decrease (%.4f -> %.4f)",
                  epoch_losses[0], epoch_losses[-1])
        sys.exit(1)


if __name__ == '__main__':
    main()
