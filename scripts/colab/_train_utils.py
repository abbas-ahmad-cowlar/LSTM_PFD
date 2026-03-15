#!/usr/bin/env python3
"""
Shared training utilities for Colab batch training scripts.

Provides a reusable `train_model_batch()` function that:
- Loads the shared HDF5 dataset
- Creates models via model_factory
- Trains each model with early stopping
- Saves checkpoints + JSON results
- Prints summary tables

Used by: 03_train_batch1_cnn.py through 08_train_batch6_advanced.py
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.reproducibility import set_seed
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

# ── Logging ──────────────────────────────────────────────────────────────
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        ),
    ],
)
logger = logging.getLogger("colab_training")

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "generated"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_device() -> torch.device:
    """Detect best available device and log info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        logger.info("WARNING: No GPU detected! Training will be very slow.")
    return device


def load_datasets(
    hdf5_path: Path = None,
    batch_size: int = 32,
) -> tuple:
    """Load train/val/test datasets from HDF5."""
    from data.dataset import BearingFaultDataset

    if hdf5_path is None:
        hdf5_path = DATA_DIR / "dataset.h5"

    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {hdf5_path}\n"
            f"Run 02_generate_data.py first."
        )

    logger.info(f"Loading datasets from {hdf5_path}")

    train_ds = BearingFaultDataset.from_hdf5(hdf5_path, split="train")
    val_ds = BearingFaultDataset.from_hdf5(hdf5_path, split="val")
    test_ds = BearingFaultDataset.from_hdf5(hdf5_path, split="test")

    logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_single_model(
    model_key: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
) -> dict:
    """
    Train a single model end-to-end.

    Args:
        model_key: Registry key from model_factory (e.g. 'cnn1d', 'resnet18')
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        device: torch device
        epochs: Max training epochs
        lr: Learning rate
        patience: Early stopping patience

    Returns:
        Dict with model name, accuracy, training time, checkpoint path
    """
    from packages.core.models.model_factory import create_model

    logger.info("=" * 60)
    logger.info(f"TRAINING: {model_key}")
    logger.info("=" * 60)

    # Build model
    try:
        model = create_model(model_key, num_classes=NUM_CLASSES).to(device)
    except Exception as e:
        logger.error(f"Failed to create model '{model_key}': {e}")
        return {"model": model_key, "error": f"Creation failed: {e}"}

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {num_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    # Checkpoint dir
    ckpt_dir = CHECKPOINT_DIR / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    train_start = time.time()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            signals, labels = batch
            signals = signals.to(device)
            labels = labels.to(device)

            if signals.dim() == 2:
                signals = signals.unsqueeze(1)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                signals, labels = batch
                signals = signals.to(device)
                labels = labels.to(device)

                if signals.dim() == 2:
                    signals = signals.unsqueeze(1)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(signals)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(signals)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Scheduler step
        scheduler.step(val_loss)

        # Logging (every 10 epochs + first + last)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e}"
            )

        # Early stopping + best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "model_config": (
                        model.get_config() if hasattr(model, "get_config") else {}
                    ),
                },
                ckpt_dir / f"{model_key}_best.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    train_time = time.time() - train_start

    # ── Test evaluation ───────────────────────────────────────────────
    # Load best model for final test
    best_ckpt = ckpt_dir / f"{model_key}_best.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            signals, labels = batch
            signals = signals.to(device)
            labels = labels.to(device)

            if signals.dim() == 2:
                signals = signals.unsqueeze(1)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(signals)
            else:
                outputs = model(signals)

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total if test_total > 0 else 0.0

    # ── Save results ──────────────────────────────────────────────────
    result = {
        "model": model_key,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "training_time_s": train_time,
        "epochs_trained": epoch,
        "params": num_params,
        "checkpoint": str(best_ckpt),
        "timestamp": datetime.now().isoformat(),
    }

    # Save per-model result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{model_key}_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"  [DONE] {model_key}: test_acc={test_acc:.4f}, "
        f"val_acc={best_val_acc:.4f}, "
        f"time={train_time:.0f}s, epochs={epoch}"
    )

    # Free GPU memory
    del model, optimizer, scaler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


def train_model_batch(
    batch_name: str,
    model_keys: list,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,
    seed: int = 42,
):
    """
    Train a batch of models sequentially.

    Args:
        batch_name: Human-readable batch name for logging
        model_keys: List of model registry keys to train
        epochs: Max epochs per model
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        seed: Random seed
    """
    set_seed(seed)
    device = get_device()

    logger.info("=" * 70)
    logger.info(f"BATCH: {batch_name}")
    logger.info(f"Models: {', '.join(model_keys)}")
    logger.info(f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    logger.info("=" * 70)

    # Load data once for the whole batch
    train_loader, val_loader, test_loader = load_datasets(batch_size=batch_size)

    # Train each model
    all_results = []
    batch_start = time.time()

    for i, model_key in enumerate(model_keys, 1):
        logger.info(f"\n[{i}/{len(model_keys)}] Starting {model_key}")
        try:
            result = train_single_model(
                model_key=model_key,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                patience=patience,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"FAILED: {model_key} -- {e}", exc_info=True)
            all_results.append({"model": model_key, "error": str(e)})

    batch_time = time.time() - batch_start

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"BATCH COMPLETE: {batch_name}")
    logger.info(f"Total time: {batch_time:.0f}s ({batch_time / 60:.1f} min)")
    logger.info("=" * 70)
    logger.info(
        f"{'Model':<30} {'Test Acc':>10} {'Val Acc':>10} "
        f"{'Params':>12} {'Time':>8} {'Epochs':>7}"
    )
    logger.info("-" * 80)

    for r in all_results:
        if "error" in r:
            logger.info(f"  {r['model']:<28} FAILED: {r['error'][:40]}")
        else:
            logger.info(
                f"  {r['model']:<28} {r['test_accuracy']:>10.4f} "
                f"{r['best_val_accuracy']:>10.4f} "
                f"{r['params']:>12,} {r['training_time_s']:>7.0f}s "
                f"{r['epochs_trained']:>6}ep"
            )

    # Save batch summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"batch_{batch_name.lower().replace(' ', '_')}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "batch_name": batch_name,
                "total_time_s": batch_time,
                "device": str(device),
                "seed": seed,
                "epochs": epochs,
                "results": all_results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    logger.info(f"\nBatch results saved: {summary_path}")

    return all_results
