"""
Phase 0 integration adapter.
Wraps Phase 0 data generation functionality.
"""
import sys
from pathlib import Path

# Add parent directory to path to import Phase 0 modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.logger import setup_logger

# Add dash_app to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DASHBOARD_TO_PHASE0_FAULT_MAP

logger = setup_logger(__name__)


class Phase0Adapter:
    """Adapter for Phase 0 data generation."""

    @staticmethod
    def generate_dataset(config: dict, progress_callback=None) -> dict:
        """
        Generate dataset using Phase 0 logic via subprocess.

        Uses a standalone script to avoid Python import namespace conflicts
        between the dashboard (packages/dashboard/config.py) and the project's
        config/ package.

        Args:
            config: Dataset configuration
            progress_callback: Optional callback(current, total, fault_type)

        Returns:
            dict with success, output_path, total_signals, num_faults
        """
        import subprocess
        import json
        import tempfile
        import os

        try:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            script_path = project_root / "scripts" / "generate_dataset_standalone.py"

            if not script_path.exists():
                logger.error(f"Standalone generation script not found: {script_path}")
                return {
                    'success': False,
                    'error': f'Generation script not found: {script_path}',
                    'output_path': None,
                    'total_signals': 0,
                    'num_faults': 0,
                }

            # Write config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
                json.dump(config, config_file)
                config_path = config_file.name

            # Write output to temp file
            output_path = tempfile.mktemp(suffix='_result.json')

            try:
                # Set PYTHONPATH to project root only (exclude dashboard)
                env = os.environ.copy()
                env['PYTHONPATH'] = str(project_root)

                logger.info(f"Running standalone generation for '{config.get('name')}'...")

                result = subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        '--config-file', config_path,
                        '--output-json', output_path
                    ],
                    cwd=str(project_root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )

                if result.returncode != 0:
                    logger.error(f"Generation subprocess failed: {result.stderr}")
                    return {
                        'success': False,
                        'error': result.stderr or 'Subprocess failed',
                        'output_path': None,
                        'total_signals': 0,
                        'num_faults': 0,
                    }

                # Read result
                with open(output_path, 'r') as f:
                    gen_result = json.load(f)

                logger.info(f"Generation complete: {gen_result}")
                return gen_result

            finally:
                # Cleanup temp files
                if os.path.exists(config_path):
                    os.unlink(config_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)

        except subprocess.TimeoutExpired:
            logger.error("Dataset generation timed out after 1 hour")
            return {
                'success': False,
                'error': 'Generation timed out',
                'output_path': None,
                'total_signals': 0,
                'num_faults': 0,
            }
        except Exception as e:
            logger.error(f"Error generating dataset: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'output_path': None,
                'total_signals': 0,
                'num_faults': 0,
            }

    @staticmethod
    def load_existing_cache(cache_path: str):
        """Load existing Phase 0 cache file."""
        import h5py
        try:
            with h5py.File(cache_path, 'r') as f:
                return {
                    "num_signals": len(f.keys()),
                    "signal_ids": list(f.keys())[:10]  # Sample
                }
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            raise

    @staticmethod
    def import_mat_files(config: dict, mat_file_paths: list, progress_callback=None) -> dict:
        """
        Import MAT files using MatlabImporter.

        Args:
            config: Import configuration
                - name: Dataset name
                - output_dir: Output directory path
                - signal_length: Target signal length (samples)
                - validate: Enable validation
                - auto_normalize: Enable auto-normalization
                - output_format: 'mat', 'hdf5', or 'both'
            mat_file_paths: List of paths to MAT files to import
            progress_callback: Optional callback(current, total, filename)

        Returns:
            dict with:
                - success: bool
                - output_path: str (HDF5 file path or directory)
                - total_signals: int
                - num_files: int
                - import_time: float
                - failed_files: list of filenames that failed
        """
        try:
            from data.matlab_importer import MatlabImporter
            from pathlib import Path
            import h5py
            import numpy as np

            logger.info(f"Importing {len(mat_file_paths)} MAT files for dataset '{config['name']}'")

            # Initialize importer
            importer = MatlabImporter()

            # Load all MAT files
            all_signals = []
            all_metadata = []
            all_labels = []
            failed_files = []

            signal_length = config.get('signal_length', 102400)
            validate = config.get('validate', True)
            auto_normalize = config.get('auto_normalize', False)

            for i, mat_file_path in enumerate(mat_file_paths):
                try:
                    # Load MAT file
                    mat_data = importer.load_mat_file(Path(mat_file_path))

                    signal = mat_data.signal
                    label = mat_data.label
                    metadata = mat_data.metadata

                    # Validation
                    if validate:
                        # Check for zeros
                        if np.all(signal == 0):
                            logger.warning(f"Signal is all zeros: {Path(mat_file_path).name}")
                            failed_files.append(Path(mat_file_path).name)
                            continue

                        # Check for NaNs
                        if np.any(np.isnan(signal)):
                            logger.warning(f"Signal contains NaN: {Path(mat_file_path).name}")
                            failed_files.append(Path(mat_file_path).name)
                            continue

                        # Check minimum length
                        if len(signal) < 1000:
                            logger.warning(f"Signal too short ({len(signal)} samples): {Path(mat_file_path).name}")
                            failed_files.append(Path(mat_file_path).name)
                            continue

                    # Truncate or pad to standard length
                    if len(signal) > signal_length:
                        signal = signal[:signal_length]
                    elif len(signal) < signal_length:
                        signal = np.pad(signal, (0, signal_length - len(signal)))

                    # Auto-normalize
                    if auto_normalize:
                        signal_std = np.std(signal)
                        if signal_std > 0:
                            signal = signal / signal_std

                    all_signals.append(signal.astype(np.float32))
                    all_labels.append(label)
                    all_metadata.append({
                        'file': Path(mat_file_path).name,
                        'label': label,
                        'severity': mat_data.severity,
                        'original_length': len(mat_data.signal)
                    })

                    # Report progress
                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(i + 1, len(mat_file_paths), Path(mat_file_path).name)

                except Exception as e:
                    logger.error(f"Failed to import {Path(mat_file_path).name}: {e}")
                    failed_files.append(Path(mat_file_path).name)
                    continue

            if not all_signals:
                return {
                    'success': False,
                    'error': 'No valid signals imported',
                    'failed_files': failed_files,
                    'total_signals': 0,
                    'num_files': 0
                }

            # Convert to arrays
            all_signals = np.array(all_signals, dtype=np.float32)

            logger.info(f"Successfully imported {len(all_signals)} signals from {len(mat_file_paths) - len(failed_files)} files")

            # Save based on output format
            output_format = config.get('output_format', 'hdf5')
            output_dir = Path(config.get('output_dir', 'data/imported'))
            output_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = config.get('name', 'imported_dataset')

            # Save as HDF5 if requested
            hdf5_path = None
            if output_format in ['hdf5', 'both']:
                hdf5_path = output_dir / f"{dataset_name}.h5"
                logger.info(f"Saving HDF5 file to {hdf5_path}")

                with h5py.File(hdf5_path, 'w') as f:
                    # Store signals and labels
                    f.create_dataset('signals', data=all_signals, compression='gzip', compression_opts=4)
                    f.create_dataset('labels', data=np.array([hash(label) % 1000 for label in all_labels], dtype=np.int32))

                    # Store metadata as attributes
                    f.attrs['num_signals'] = len(all_signals)
                    f.attrs['signal_length'] = signal_length
                    f.attrs['dataset_name'] = dataset_name
                    f.attrs['num_files'] = len(mat_file_paths) - len(failed_files)

            # Copy original MAT files if requested
            mat_dir = None
            if output_format in ['mat', 'both']:
                mat_dir = output_dir / f"{dataset_name}_mat"
                mat_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Copying MAT files to {mat_dir}")

                for mat_path in mat_file_paths:
                    if Path(mat_path).name not in failed_files:
                        try:
                            import shutil
                            shutil.copy2(mat_path, mat_dir / Path(mat_path).name)
                        except Exception as e:
                            logger.error(f"Failed to copy {Path(mat_path).name}: {e}")

            # Final progress callback
            if progress_callback:
                progress_callback(len(mat_file_paths), len(mat_file_paths), "Complete")

            output_path = str(hdf5_path) if hdf5_path else str(mat_dir)

            logger.info(f"✓ MAT import complete: {output_path}")

            return {
                'success': True,
                'output_path': output_path,
                'total_signals': len(all_signals),
                'num_files': len(mat_file_paths) - len(failed_files),
                'failed_files': failed_files,
                'import_time': 0,  # Will be calculated by task
            }

        except Exception as e:
            logger.error(f"Error importing MAT files: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'output_path': None,
                'total_signals': 0,
                'num_files': 0,
                'failed_files': []
            }
