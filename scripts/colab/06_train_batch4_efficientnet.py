#!/usr/bin/env python3
"""
Batch 4: EfficientNet 1D Models (4 models)

Models: EfficientNet-B0, B1, B2, B3
(B4-B7 skipped — very large, likely OOM on T4 16GB)

Est. time on T4: ~20 min

Usage:
    !python scripts/colab/06_train_batch4_efficientnet.py
    !python scripts/colab/06_train_batch4_efficientnet.py --include-large  # add B4-B7
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.colab._train_utils import train_model_batch

MODELS_CORE = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
]

MODELS_LARGE = [
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--include-large", action="store_true",
        help="Include B4-B7 (may OOM on T4)"
    )
    args = parser.parse_args()

    models = MODELS_CORE + (MODELS_LARGE if args.include_large else [])

    train_model_batch(
        batch_name="EfficientNet Models",
        model_keys=models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
