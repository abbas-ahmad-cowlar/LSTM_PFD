#!/usr/bin/env python3
"""
Generate the shared HDF5 dataset for all model training.

Supports versioned datasets with optional advanced physics.

Usage (in Colab):
    # V1: Basic physics (default)
    !python scripts/colab/02_generate_data.py --version-tag v1_basic

    # V2: All advanced physics enabled
    !python scripts/colab/02_generate_data.py --version-tag v2_advanced --advanced-physics

    # Custom: specific effects only
    !python scripts/colab/02_generate_data.py --version-tag v2_rotor \\
        --enable speed_fluctuation rotor_dynamics cross_coupling
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
        help="Signals per fault type (default: 200, total ~ 200*11*1.3 = 2860)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--version-tag", type=str, default=None,
        help="Version tag for this dataset (e.g. 'v1_basic', 'v2_advanced'). "
             "Creates a subdirectory under data/generated/"
    )
    parser.add_argument(
        "--advanced-physics", action="store_true",
        help="Enable ALL advanced physics effects (V2)"
    )
    parser.add_argument(
        "--enable", nargs="+", default=[],
        choices=[
            "speed_transients", "speed_fluctuation", "rotor_dynamics",
            "cross_coupling", "thermal_growth", "axial_vibration"
        ],
        help="Enable specific advanced physics effects"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if dataset already exists"
    )
    args = parser.parse_args()

    # Determine output directory
    if args.version_tag:
        output_dir = DATA_DIR / args.version_tag
    else:
        output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = output_dir / "dataset.h5"

    if hdf5_path.exists() and not args.force:
        size_mb = hdf5_path.stat().st_size / 1e6
        logger.info(f"Dataset already exists: {hdf5_path} ({size_mb:.1f} MB)")
        logger.info("Use --force to regenerate.")
        return

    from config.data_config import DataConfig, AdvancedPhysicsConfig
    from data.signal_generation.generator import SignalGenerator
    from utils.reproducibility import set_seed

    set_seed(args.seed)

    # Build config with advanced physics
    adv_physics = AdvancedPhysicsConfig()

    if args.advanced_physics:
        # Enable everything
        adv_physics = AdvancedPhysicsConfig(
            speed_transients=True,
            speed_fluctuation=True,
            rotor_dynamics=True,
            cross_coupling=True,
            thermal_growth=True,
            axial_vibration=True,
        )
    elif args.enable:
        # Enable specific effects
        kwargs = {effect: True for effect in args.enable}
        adv_physics = AdvancedPhysicsConfig(**kwargs)

    config = DataConfig(
        num_signals_per_fault=args.num_signals,
        output_dir=str(output_dir),
        rng_seed=args.seed,
        advanced_physics=adv_physics,
    )

    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC DATASET")
    logger.info("=" * 60)

    enabled_effects = adv_physics.get_enabled_effects()
    logger.info(f"  Version tag: {args.version_tag or 'auto'}")
    logger.info(f"  Advanced physics: {enabled_effects or 'none (basic)'}")
    logger.info(f"  Fault types: {len(config.fault.get_fault_list())}")
    logger.info(f"  Signals per fault: {args.num_signals} base + 30%% augmented")
    logger.info(f"  Output: {output_dir}")

    gen_start = time.time()
    generator = SignalGenerator(config)

    # Set version tag for embedding in HDF5
    if args.version_tag:
        generator._version_tag = args.version_tag

    dataset = generator.generate_dataset()
    paths = generator.save_dataset(dataset, output_dir=output_dir, format="hdf5")
    hdf5_path = paths["hdf5"]
    gen_time = time.time() - gen_start

    size_mb = hdf5_path.stat().st_size / 1e6
    logger.info(f"  [DONE] Generated in {gen_time:.1f}s")
    logger.info(f"  File: {hdf5_path} ({size_mb:.1f} MB)")
    logger.info(f"  Config saved: {output_dir / 'dataset_config.json'}")
    logger.info("Next: Run training batch scripts (03-08)")


if __name__ == "__main__":
    main()
