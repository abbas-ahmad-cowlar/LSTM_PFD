#!/usr/bin/env python3
"""
Batch 2: ResNet Models (10 models)

Models: ResNet18/34/50, WideResNet 16-8/16-10/22-8/28-10,
        SE-ResNet 18/34/50

Est. time on T4: ~40 min

Usage:
    !python scripts/colab/04_train_batch2_resnet.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.colab._train_utils import train_model_batch

MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "wide_resnet16_8",
    "wide_resnet16_10",
    "wide_resnet22_8",
    "wide_resnet28_10",
    "se_resnet18",
    "se_resnet34",
    "se_resnet50",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model_batch(
        batch_name="ResNet Models",
        model_keys=MODELS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
