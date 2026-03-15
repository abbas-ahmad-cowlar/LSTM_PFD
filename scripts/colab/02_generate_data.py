#!/usr/bin/env python3
"""
Generate the shared HDF5 dataset for all model training.

Usage (in Colab):
    !python scripts/colab/02_generate_data.py
    !python scripts/colab/02_generate_data.py --num-signals 500   # larger dataset
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.colab._train_utils import logger, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument(
        "--num-signals", type=int, default=200,
        help="Signals per fault type (default: 200, total = 200 * 11 * 1.3 = ~2860)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hdf5_path = DATA_DIR / "dataset.h5"

    if hdf5_path.exists():
        size_mb = hdf5_path.stat().st_size / 1e6
        logger.info(f"Dataset already exists: {hdf5_path} ({size_mb:.1f} MB)")
        logger.info("Delete it manually if you want to regenerate.")
        return

    from config.data_config import DataConfig
    from data.signal_generation.generator import SignalGenerator
    from utils.reproducibility import set_seed

    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC DATASET")
    logger.info("=" * 60)

    config = DataConfig(
        num_signals_per_fault=args.num_signals,
        output_dir=str(DATA_DIR),
        rng_seed=args.seed,
    )

    total = config.get_total_signals()
    logger.info(f"  Fault types: {len(config.fault.get_fault_list())}")
    logger.info(f"  Signals per fault: {args.num_signals} base + 30% augmented")
    logger.info(f"  Total signals: ~{total}")

    gen_start = time.time()
    generator = SignalGenerator(config)
    dataset = generator.generate_dataset()
    paths = generator.save_dataset(dataset, output_dir=DATA_DIR, format="hdf5")
    hdf5_path = paths["hdf5"]
    gen_time = time.time() - gen_start

    size_mb = hdf5_path.stat().st_size / 1e6
    logger.info(f"  [DONE] Generated in {gen_time:.1f}s")
    logger.info(f"  File: {hdf5_path} ({size_mb:.1f} MB)")
    logger.info("Next: Run training batch scripts (03-08)")


if __name__ == "__main__":
    main()
