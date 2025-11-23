"""
Phase 0 integration adapter.
Wraps Phase 0 data generation functionality.
"""
import sys
from pathlib import Path

# Add parent directory to path to import Phase 0 modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class Phase0Adapter:
    """Adapter for Phase 0 data generation."""

    @staticmethod
    def generate_dataset(config: dict, progress_callback=None) -> dict:
        """
        Generate dataset using Phase 0 logic.

        Args:
            config: Dataset configuration
                - name: Dataset name
                - output_dir: Output directory path
                - num_signals_per_fault: Number of signals per fault type
                - fault_types: List of fault types to include
                - severity_levels: List of severity levels
                - temporal_evolution: Enable temporal evolution
                - noise_layers: Dict of noise layer enables
                - speed_variation: Speed variation percentage (0-1)
                - load_range: [min, max] load range (0-1)
                - temp_range: [min, max] temperature range (°C)
                - augmentation: Augmentation config dict
                - output_format: 'mat', 'hdf5', or 'both'
                - random_seed: Random seed for reproducibility
            progress_callback: Optional callback(current, total, fault_type)

        Returns:
            dict with:
                - success: bool
                - output_path: str (HDF5 file path)
                - total_signals: int
                - num_faults: int
                - generation_time: float
        """
        try:
            from data.signal_generator import SignalGenerator
            from config.data_config import DataConfig, SignalConfig, FaultConfig, \
                SeverityConfig, NoiseConfig, OperatingConfig, PhysicsConfig, \
                TransientConfig, AugmentationConfig
            from pathlib import Path
            import h5py
            import numpy as np
            from dataclasses import asdict

            logger.info(f"Generating dataset '{config['name']}' with config: {config}")

            # Convert dashboard config to DataConfig
            data_config = DataConfig(
                num_signals_per_fault=config.get('num_signals_per_fault', 100),
                output_dir=config.get('output_dir', 'data/generated'),
                rng_seed=config.get('random_seed', 42),
            )

            # Configure signal parameters (use defaults)
            data_config.signal = SignalConfig()

            # Configure fault types
            # Import mapping from config
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from config import DASHBOARD_TO_PHASE0_FAULT_MAP

            fault_config = FaultConfig()
            selected_faults = config.get('fault_types', [])

            # Disable all faults first
            fault_config.include_healthy = False
            for fault in fault_config.single_faults:
                fault_config.single_faults[fault] = False
            for fault in fault_config.mixed_faults:
                fault_config.mixed_faults[fault] = False

            # Enable selected faults (convert from dashboard names to Phase 0 names)
            for dashboard_fault in selected_faults:
                # Convert dashboard name to Phase 0 name
                phase0_fault = DASHBOARD_TO_PHASE0_FAULT_MAP.get(dashboard_fault, dashboard_fault)

                if phase0_fault == 'sain':
                    fault_config.include_healthy = True
                elif phase0_fault.startswith('mixed_'):
                    # Remove 'mixed_' prefix for the dict key
                    fault_key = phase0_fault.replace('mixed_', '')
                    if fault_key in fault_config.mixed_faults:
                        fault_config.include_single = True  # Need single faults for mixed
                        fault_config.include_mixed = True
                        fault_config.mixed_faults[fault_key] = True
                else:
                    if phase0_fault in fault_config.single_faults:
                        fault_config.include_single = True
                        fault_config.single_faults[phase0_fault] = True

            data_config.fault = fault_config

            # Configure severity
            severity_config = SeverityConfig()
            severity_config.enabled = True
            severity_config.levels = config.get('severity_levels', ['incipient', 'mild', 'moderate', 'severe'])
            severity_config.temporal_evolution = 0.30 if config.get('temporal_evolution', True) else 0.0
            data_config.severity = severity_config

            # Configure noise layers
            noise_config = NoiseConfig()
            noise_layers = config.get('noise_layers', {})
            noise_config.measurement = noise_layers.get('measurement', True)
            noise_config.emi = noise_layers.get('emi', True)
            noise_config.pink = noise_layers.get('pink', True)
            noise_config.drift = noise_layers.get('drift', True)
            noise_config.quantization = noise_layers.get('quantization', True)
            noise_config.sensor_drift = noise_layers.get('sensor_drift', True)
            noise_config.impulse = noise_layers.get('impulse', True)
            data_config.noise = noise_config

            # Configure operating conditions
            operating_config = OperatingConfig()
            operating_config.speed_variation = config.get('speed_variation', 0.10)
            load_range = config.get('load_range', [0.30, 1.00])
            operating_config.load_range = tuple(load_range)
            temp_range = config.get('temp_range', [40.0, 80.0])
            operating_config.temp_range = tuple(temp_range)
            data_config.operating = operating_config

            # Configure physics (use defaults)
            data_config.physics = PhysicsConfig()

            # Configure transients (use defaults)
            data_config.transient = TransientConfig()

            # Configure augmentation
            aug_config_dict = config.get('augmentation', {})
            aug_config = AugmentationConfig()
            aug_config.enabled = aug_config_dict.get('enabled', True)
            aug_config.ratio = aug_config_dict.get('ratio', 0.30)
            aug_config.methods = aug_config_dict.get('methods', ['time_shift', 'amplitude_scale', 'noise_injection'])
            data_config.augmentation = aug_config

            # Initialize generator
            generator = SignalGenerator(data_config)

            # Generate dataset with progress tracking
            fault_types = data_config.fault.get_fault_list()
            total_faults = len(fault_types)
            total_signals_target = data_config.get_total_signals()

            logger.info(f"Will generate {total_signals_target} signals for {total_faults} fault types")

            # Custom generation with progress
            all_signals = []
            all_metadata = []
            all_labels = []
            signal_count = 0

            for k, fault in enumerate(fault_types):
                num_base = data_config.num_signals_per_fault

                if data_config.augmentation.enabled:
                    num_augmented = int(num_base * data_config.augmentation.ratio)
                else:
                    num_augmented = 0

                num_total_for_fault = num_base + num_augmented

                for n in range(num_total_for_fault):
                    is_augmented = (n >= num_base)

                    # Generate signal
                    signal, metadata = generator.generate_single_signal(fault, is_augmented)

                    all_signals.append(signal)
                    all_metadata.append(metadata)
                    all_labels.append(fault)
                    signal_count += 1

                    # Report progress
                    if progress_callback and signal_count % 10 == 0:
                        progress_callback(signal_count, total_signals_target, fault)

                logger.info(f"Generated {num_total_for_fault} signals for fault: {fault}")

            # Save dataset based on output format
            output_format = config.get('output_format', 'both')
            output_dir = Path(config.get('output_dir', 'data/generated'))
            output_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = config.get('name', 'dataset')

            # Save as MAT files if requested
            if output_format in ['mat', 'both']:
                mat_dir = output_dir / f"{dataset_name}_mat"
                mat_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving MAT files to {mat_dir}")

                dataset_dict = {
                    'signals': all_signals,
                    'metadata': all_metadata,
                    'labels': all_labels,
                    'config': data_config,
                }
                generator.save_dataset(dataset_dict, mat_dir)

            # Save as HDF5 if requested
            hdf5_path = None
            if output_format in ['hdf5', 'both']:
                hdf5_path = output_dir / f"{dataset_name}.h5"
                logger.info(f"Saving HDF5 file to {hdf5_path}")

                with h5py.File(hdf5_path, 'w') as f:
                    # Store signals and labels
                    for i, (signal, metadata, label) in enumerate(zip(all_signals, all_metadata, all_labels)):
                        signal_id = f"signal_{i:06d}"
                        grp = f.create_group(signal_id)
                        grp.create_dataset('data', data=signal, compression='gzip')
                        grp.attrs['fault_type'] = label
                        grp.attrs['severity'] = metadata.severity
                        grp.attrs['is_augmented'] = metadata.is_augmented
                        grp.attrs['signal_rms'] = metadata.signal_rms
                        grp.attrs['signal_peak'] = metadata.signal_peak

                    # Store metadata
                    meta_grp = f.create_group('metadata')
                    meta_grp.attrs['total_signals'] = len(all_signals)
                    meta_grp.attrs['num_faults'] = len(fault_types)
                    meta_grp.attrs['sampling_rate'] = data_config.signal.fs
                    meta_grp.attrs['signal_duration'] = data_config.signal.T
                    meta_grp.attrs['dataset_name'] = dataset_name

            # Final progress callback
            if progress_callback:
                progress_callback(total_signals_target, total_signals_target, "Complete")

            output_path = str(hdf5_path) if hdf5_path else str(output_dir / f"{dataset_name}_mat")

            logger.info(f"✓ Dataset generation complete: {output_path}")

            return {
                'success': True,
                'output_path': output_path,
                'total_signals': len(all_signals),
                'num_faults': len(fault_types),
                'generation_time': 0,  # Will be calculated by task
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
