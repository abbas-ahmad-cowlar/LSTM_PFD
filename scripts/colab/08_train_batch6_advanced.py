#!/usr/bin/env python3
"""
Batch 6: Advanced Models (6 models)

Models: HybridPINN, PhysicsConstrainedCNN, DualStreamCNN,
        ResNet18-2D (spectrogram), EfficientNet2D-B0, ContrastiveClassifier

These models require special data handling:
- PINN models: metadata (rpm, load) from dataset
- Spectrogram models: 2D input (model converts internally or needs spectrogram)
- Contrastive: pair sampling

The training utility handles standard [B, 1, T] input; models that need
different inputs may require their forward() to handle reshaping internally.

Est. time on T4: ~25 min

Usage:
    !python scripts/colab/08_train_batch6_advanced.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.colab._train_utils import train_model_batch, logger

MODELS = [
    "hybrid_pinn",
    "physics_cnn",
    "dual_stream_cnn",
    "contrastive_classifier",
]

# These need spectrogram input — may fail with standard 1D loader.
# We try them and log errors gracefully.
SPECTROGRAM_MODELS = [
    "resnet18_2d",
    "efficientnet_2d_b0",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--skip-spectrogram", action="store_true",
        help="Skip 2D spectrogram models (need custom data pipeline)"
    )
    args = parser.parse_args()

    models = MODELS.copy()
    if not args.skip_spectrogram:
        models.extend(SPECTROGRAM_MODELS)
        logger.info(
            "NOTE: Spectrogram 2D models may fail with standard 1D data loader. "
            "Use --skip-spectrogram to skip them."
        )

    train_model_batch(
        batch_name="Advanced Models",
        model_keys=models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
