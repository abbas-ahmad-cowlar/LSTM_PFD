"""
Phase 0 Adapter – Data Generation & MAT Import.

Bridges dashboard Celery tasks to the core data generation layer:
  - SignalGenerator  (data/signal_generator.py)
  - MatlabImporter   (data/matlab_importer.py)

Used by:
  - tasks/data_generation_tasks.py  →  Phase0Adapter.generate_dataset()
  - tasks/mat_import_tasks.py       →  Phase0Adapter.import_mat_files()
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Phase0Adapter:
    """
    Static adapter that wraps the core data-generation and MAT-import
    utilities behind the same interface the dashboard Celery tasks expect.
    """

    # ------------------------------------------------------------------ #
    #  Dataset generation  (data_generation_tasks.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_dataset(
        config: dict,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate a synthetic vibration dataset.

        Args:
            config: Generation configuration from the dashboard.
                Expected keys:
                  - num_signals_per_fault (int): signals per fault type
                  - fault_types (list[str]): fault type names to generate
                  - severity_levels (list[str]): severities to include
                  - noise_layers (dict): per-layer enable flags
                  - augmentation (dict): augmentation settings
                  - output_format (str): 'hdf5', 'mat', or 'both'
                  - random_seed (int | None): RNG seed
                  - output_dir (str | None): output directory
            progress_callback: Optional ``callback(current, total, fault_type=None)``

        Returns:
            dict with keys:
              - success (bool)
              - total_signals (int)
              - num_faults (int)
              - output_path (str)
              - generation_time (float)   seconds
              - error (str)              only on failure
        """
        try:
            import sys, os
            # Ensure project root is on sys.path for absolute imports
            project_root = str(Path(__file__).resolve().parents[3])
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from data.signal_generator import SignalGenerator
            from config.data_config import DataConfig

            logger.info("Phase0Adapter: starting dataset generation")
            start = time.time()

            # ---- build DataConfig from dashboard dict ---- #
            data_config = DataConfig()

            if config.get("num_signals_per_fault"):
                data_config.num_signals_per_fault = int(config["num_signals_per_fault"])

            if config.get("random_seed") is not None:
                data_config.rng_seed = int(config["random_seed"])

            if config.get("augmentation"):
                aug = config["augmentation"]
                if isinstance(aug, dict):
                    data_config.augmentation.enabled = aug.get("enabled", False)
                    if "ratio" in aug:
                        data_config.augmentation.ratio = float(aug["ratio"])

            # TODO: map fault_types, severity_levels, noise_layers to DataConfig
            #       once those setters exist on DataConfig.

            # ---- run generation ---- #
            generator = SignalGenerator(data_config)
            dataset = generator.generate_dataset()

            # Determine output path
            output_dir = Path(config.get("output_dir", "data/processed"))
            output_dir.mkdir(parents=True, exist_ok=True)

            fmt = config.get("output_format", "hdf5")
            output_path = output_dir / "signals_cache.h5"

            # Save HDF5 (always, since the dashboard needs it)
            hdf5_path = generator._save_as_hdf5(dataset, output_path)

            elapsed = time.time() - start

            # Fire final progress
            total = dataset["statistics"]["total_signals"]
            if progress_callback:
                try:
                    progress_callback(total, total, fault_type="done")
                except Exception:
                    pass

            logger.info(
                f"Phase0Adapter: generation complete – "
                f"{total} signals in {elapsed:.1f}s"
            )

            result = {
                "success": True,
                "total_signals": total,
                "num_faults": dataset["statistics"]["num_faults"],
                "output_path": str(hdf5_path),
                "generation_time": elapsed,
            }

            # ---- register in ModelRegistry (optional, non-blocking) ---- #
            try:
                from integration.model_registry import ModelRegistry
                registry = ModelRegistry()
                registry.register_model(
                    model_name='dataset_generation',
                    phase='phase_0',
                    accuracy=0.0,
                    model_path=str(hdf5_path),
                    training_duration_s=elapsed,
                    notes=f"{total} signals, {dataset['statistics']['num_faults']} faults",
                )
            except Exception as reg_err:
                logger.debug(f"ModelRegistry registration skipped: {reg_err}")

            return result

        except Exception as exc:
            logger.error(f"Phase0Adapter.generate_dataset failed: {exc}")
            logger.debug(traceback.format_exc())
            return {
                "success": False,
                "error": str(exc),
                "total_signals": 0,
                "num_faults": 0,
                "output_path": "",
                "generation_time": 0.0,
            }

    # ------------------------------------------------------------------ #
    #  MAT file import  (mat_import_tasks.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    def import_mat_files(
        config: dict,
        mat_file_paths: List[str],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Import MATLAB .mat files and convert to HDF5.

        Args:
            config: Import configuration from the dashboard.
                Expected keys:
                  - output_dir (str): directory to write HDF5
                  - signal_length (int | None): target signal length
                  - validate (bool): run validation checks
                  - auto_normalize (bool): z-score normalize signals
                  - output_format (str): 'hdf5', 'mat', or 'both'
            mat_file_paths: list of absolute paths to uploaded .mat files
            progress_callback: Optional ``callback(current, total, filename=None)``

        Returns:
            dict with keys:
              - success (bool)
              - total_signals (int)
              - num_files (int)
              - output_path (str)
              - failed_files (list[str])
              - error (str)   only on failure
        """
        try:
            import sys
            project_root = str(Path(__file__).resolve().parents[3])
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from data.matlab_importer import MatlabImporter
            import numpy as np

            logger.info(
                f"Phase0Adapter: importing {len(mat_file_paths)} MAT files"
            )
            start = time.time()

            importer = MatlabImporter()
            all_signals: List[np.ndarray] = []
            all_labels: List[str] = []
            failed_files: List[str] = []

            total = len(mat_file_paths)
            for idx, mat_path in enumerate(mat_file_paths):
                try:
                    data = importer.load_mat_file(Path(mat_path))
                    all_signals.append(data.signal)
                    all_labels.append(data.label)
                except Exception as file_err:
                    logger.warning(f"Skipping {mat_path}: {file_err}")
                    failed_files.append(str(mat_path))

                if progress_callback:
                    try:
                        progress_callback(
                            idx + 1, total,
                            filename=Path(mat_path).name,
                        )
                    except Exception:
                        pass

            if not all_signals:
                return {
                    "success": False,
                    "error": "No signals could be loaded from the provided MAT files",
                    "total_signals": 0,
                    "num_files": 0,
                    "output_path": "",
                    "failed_files": failed_files,
                }

            # Stack signals into (N, L) array
            signals = np.stack(all_signals, axis=0).astype(np.float32)

            # Optional normalization
            if config.get("auto_normalize", False):
                mean = signals.mean(axis=1, keepdims=True)
                std = signals.std(axis=1, keepdims=True) + 1e-8
                signals = (signals - mean) / std

            # Save as HDF5
            output_dir = Path(config.get("output_dir", "data/processed"))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "imported_signals.h5"

            import h5py
            with h5py.File(output_path, "w") as f:
                f.create_dataset(
                    "signals", data=signals,
                    compression="gzip", compression_opts=4,
                )
                # Store labels as UTF-8 strings
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset("labels", data=all_labels, dtype=dt)
                f.attrs["num_signals"] = len(all_signals)
                f.attrs["num_files"] = total - len(failed_files)

            elapsed = time.time() - start
            logger.info(
                f"Phase0Adapter: import complete – "
                f"{len(all_signals)} signals from {total} files in {elapsed:.1f}s"
            )

            return {
                "success": True,
                "total_signals": len(all_signals),
                "num_files": total - len(failed_files),
                "output_path": str(output_path),
                "failed_files": failed_files,
            }

        except Exception as exc:
            logger.error(f"Phase0Adapter.import_mat_files failed: {exc}")
            logger.debug(traceback.format_exc())
            return {
                "success": False,
                "error": str(exc),
                "total_signals": 0,
                "num_files": 0,
                "output_path": "",
                "failed_files": [],
            }
