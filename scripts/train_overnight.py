#!/usr/bin/env python
"""
Overnight Training Pipeline — Phase 3 of the Master Plan.

Generates synthetic data (if not already cached), then trains models
sequentially. Designed to run unattended overnight.

Usage:
    python scripts/train_overnight.py                     # full pipeline
    python scripts/train_overnight.py --skip-datagen      # reuse existing data
    python scripts/train_overnight.py --models cnn resnet # train specific models
    python scripts/train_overnight.py --epochs 50         # override epochs
    python scripts/train_overnight.py --quick             # quick smoke-test (10 samples, 3 epochs)

Author: Phase 3 — Data Generation & Baseline Training
Date: 2026-03-15
"""

import sys
import os
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.reproducibility import set_seed
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, FAULT_TYPES

# ── Logging ──────────────────────────────────────────────────────────────
# Force UTF-8 for Windows console
import io
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')),
        logging.FileHandler(PROJECT_ROOT / "logs" / "train_overnight.log"),
    ],
)
logger = logging.getLogger("overnight_training")

# ── Constants ────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "generated"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_NUM_SIGNALS = 200        # per fault type (200 × 11 = 2200 base + 30% aug ≈ 2860)
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_dataset(num_signals_per_fault: int = DEFAULT_NUM_SIGNALS, seed: int = DEFAULT_SEED) -> Path:
    """Generate HDF5 dataset using the physics-based signal generator."""
    from config.data_config import DataConfig
    from data.signal_generation.generator import SignalGenerator

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hdf5_path = DATA_DIR / "dataset.h5"

    if hdf5_path.exists():
        logger.info(f"Dataset already exists: {hdf5_path} ({hdf5_path.stat().st_size / 1e6:.1f} MB)")
        return hdf5_path

    logger.info("=" * 70)
    logger.info("STEP 1: GENERATING SYNTHETIC DATASET")
    logger.info("=" * 70)

    config = DataConfig(
        num_signals_per_fault=num_signals_per_fault,
        output_dir=str(DATA_DIR),
        rng_seed=seed,
    )

    total = config.get_total_signals()
    logger.info(f"  Fault types: {len(config.fault.get_fault_list())} ({', '.join(config.fault.get_fault_list())})")
    logger.info(f"  Signals per fault: {num_signals_per_fault} base + 30% augmented")
    logger.info(f"  Total signals: {total}")
    logger.info(f"  Signal length: {config.signal.N} samples ({config.signal.T}s @ {config.signal.fs} Hz)")

    gen_start = time.time()
    generator = SignalGenerator(config)
    dataset = generator.generate_dataset()
    hdf5_path = generator.save_dataset(dataset, output_dir=DATA_DIR, format='hdf5')['hdf5']
    gen_time = time.time() - gen_start

    logger.info(f"  [DONE] Dataset generated in {gen_time:.1f}s -> {hdf5_path}")
    logger.info(f"  File size: {hdf5_path.stat().st_size / 1e6:.1f} MB")

    return hdf5_path


# ═══════════════════════════════════════════════════════════════════════════
# 2. DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(hdf5_path: Path, batch_size: int = DEFAULT_BATCH_SIZE):
    """Load train/val/test datasets from HDF5 and create DataLoaders."""
    from data.dataset import BearingFaultDataset

    logger.info(f"Loading datasets from {hdf5_path}")

    train_ds = BearingFaultDataset.from_hdf5(hdf5_path, split='train')
    val_ds = BearingFaultDataset.from_hdf5(hdf5_path, split='val')
    test_ds = BearingFaultDataset.from_hdf5(hdf5_path, split='test')

    logger.info(f"  Train: {len(train_ds)} samples")
    logger.info(f"  Val:   {len(val_ds)} samples")
    logger.info(f"  Test:  {len(test_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_cnn(device):
    """Build CNN1D model."""
    from packages.core.models.cnn.cnn_1d import CNN1D
    model = CNN1D(num_classes=NUM_CLASSES).to(device)
    logger.info(f"  CNN1D: {sum(p.numel() for p in model.parameters()):,} params")
    return model


def build_resnet(device):
    """Build ResNet1D model."""
    from packages.core.models.resnet.resnet_1d import ResNet1D
    model = ResNet1D(num_classes=NUM_CLASSES).to(device)
    logger.info(f"  ResNet1D: {sum(p.numel() for p in model.parameters()):,} params")
    return model


def build_transformer(device):
    """Build SignalTransformer model."""
    from packages.core.models.transformer.signal_transformer import SignalTransformer
    model = SignalTransformer(
        signal_length=SIGNAL_LENGTH,
        num_classes=NUM_CLASSES,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)
    logger.info(f"  SignalTransformer: {sum(p.numel() for p in model.parameters()):,} params")
    return model


MODEL_BUILDERS = {
    'cnn': ('CNN1D', build_cnn),
    'resnet': ('ResNet1D', build_resnet),
    'transformer': ('SignalTransformer', build_transformer),
}


# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    model_key: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
) -> dict:
    """Train a single model end-to-end using CNNTrainer (BaseTrainer)."""
    from packages.core.training.cnn_trainer import CNNTrainer

    model_name, builder = MODEL_BUILDERS[model_key]
    logger.info("=" * 70)
    logger.info(f"TRAINING: {model_name}")
    logger.info("=" * 70)

    model = builder(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    ckpt_dir = CHECKPOINT_DIR / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = CNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        lr_scheduler=scheduler,
        max_grad_norm=1.0,
        mixed_precision=(device.type == 'cuda'),
        checkpoint_dir=ckpt_dir,
        num_classes=NUM_CLASSES,
    )

    # --- Train ---
    train_start = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
    )
    train_time = time.time() - train_start

    # --- Evaluate on test set ---
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            signals, labels = batch
            signals = signals.to(device)
            labels = labels.to(device)

            # Add channel dim if needed (B, T) → (B, 1, T)
            if signals.dim() == 2:
                signals = signals.unsqueeze(1)

            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct / total if total > 0 else 0.0

    # --- Save final checkpoint ---
    final_ckpt_path = ckpt_dir / f"{model_key}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model.get_config() if hasattr(model, 'get_config') else {},
        'test_accuracy': test_accuracy,
        'num_classes': NUM_CLASSES,
        'epochs_trained': epochs,
        'training_time_s': train_time,
        'timestamp': datetime.now().isoformat(),
    }, final_ckpt_path)

    result = {
        'model': model_name,
        'test_accuracy': test_accuracy,
        'training_time_s': train_time,
        'epochs': epochs,
        'checkpoint': str(final_ckpt_path),
        'params': sum(p.numel() for p in model.parameters()),
    }

    logger.info(f"  [OK] {model_name}: test_acc={test_accuracy:.4f}, time={train_time:.1f}s")
    logger.info(f"     Checkpoint saved: {final_ckpt_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Overnight training pipeline")
    parser.add_argument("--skip-datagen", action="store_true", help="Skip data generation (reuse existing)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_BUILDERS.keys()),
                        choices=list(MODEL_BUILDERS.keys()), help="Models to train")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--num-signals", type=int, default=DEFAULT_NUM_SIGNALS,
                        help="Signals per fault type for data generation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (10 signals, 3 epochs)")
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.num_signals = 10
        args.epochs = 3
        args.models = ['cnn']
        logger.info("[QUICK] QUICK MODE: 10 signals, 3 epochs, CNN only")

    # Setup
    set_seed(args.seed)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Step 1: Generate data
    hdf5_path = DATA_DIR / "dataset.h5"
    if not args.skip_datagen:
        hdf5_path = generate_dataset(num_signals_per_fault=args.num_signals, seed=args.seed)
    else:
        if not hdf5_path.exists():
            logger.error(f"Dataset not found: {hdf5_path}. Run without --skip-datagen first.")
            sys.exit(1)
        logger.info(f"Using existing dataset: {hdf5_path}")

    # Step 2: Load data
    train_loader, val_loader, test_loader = load_datasets(hdf5_path, batch_size=args.batch_size)

    # Step 3: Train models
    all_results = []
    for model_key in args.models:
        try:
            result = train_model(
                model_key=model_key,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {model_key}: {e}", exc_info=True)
            all_results.append({'model': model_key, 'error': str(e)})

    # Step 4: Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    for r in all_results:
        if 'error' in r:
            logger.info(f"  [FAIL] {r['model']}: FAILED -- {r['error']}")
        else:
            logger.info(f"  [OK] {r['model']}: acc={r['test_accuracy']:.4f}, "
                        f"time={r['training_time_s']:.0f}s, params={r['params']:,}")

    # Save results
    results_path = RESULTS_DIR / "overnight_training_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'results': all_results,
        }, f, indent=2)
    logger.info(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()
