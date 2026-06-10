#!/usr/bin/env python3
"""
Batch 2: Physics-informed models (Tier 1) + Tier 2 extension.

Models: HybridPINN, PhysicsConstrainedCNN, MultitaskPINN
Extension (only with --include-tier2): MultiScaleCNN1D, SE-ResNet18,
SignalTransformer

Usage:
    !python scripts/colab/04_train_batch2_physics.py
    !python scripts/colab/04_train_batch2_physics.py --include-tier2
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.colab._train_utils import train_model_batch

MODELS = [
    "hybrid_pinn",
    "physics_constrained_cnn",
    "multitask_pinn",
]

TIER2_MODELS = [
    "multi_scale_cnn",
    "se_resnet18",
    "signal_transformer",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--include-tier2", action="store_true",
                        help="Also train the Tier-2 extension zoo")
    args = parser.parse_args()

    models = MODELS + (TIER2_MODELS if args.include_tier2 else [])

    train_model_batch(
        batch_name="Physics-Informed (Tier 1)" + (
            " + Extension (Tier 2)" if args.include_tier2 else ""),
        model_keys=models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
