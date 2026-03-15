"""
Main signal generation orchestrator (Python port of generator.m).

Generates synthetic vibration signals with physics-based fault models,
7-layer noise, multi-severity progression, and data augmentation.

Extracted from: data/signal_generator.py
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import asdict
import time
from scipy.io import savemat
from sklearn.model_selection import train_test_split
import h5py
import json
from datetime import datetime

from config.data_config import DataConfig
from utils.reproducibility import set_seed
from utils.logging import get_logger
from utils.constants import FAULT_TYPES, NUM_CLASSES, SAMPLING_RATE

from .metadata import SignalMetadata
from .fault_modeler import FaultModeler
from .noise_generator import NoiseGenerator
from data.signal_validation import validate_signal

logger = get_logger(__name__)


class SignalGenerator:
    """
    Main signal generation orchestrator (Python port of generator.m).

    Generates synthetic vibration signals with physics-based fault models,
    7-layer noise, multi-severity progression, and data augmentation.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize signal generator.

        Args:
            config: Data generation configuration (uses defaults if None)
        """
        if config is None:
            config = DataConfig()

        self.config = config
        self.fault_modeler = FaultModeler(config)
        self.noise_generator = NoiseGenerator(config)

        # Set random seed for reproducibility
        if config.rng_seed is not None:
            set_seed(config.rng_seed)

        logger.info("Signal Generator initialized")
        logger.info(f"  Sampling rate: {config.signal.fs} Hz")
        logger.info(f"  Signal duration: {config.signal.T} s")
        logger.info(f"  Samples per signal: {config.signal.N}")

    def generate_dataset(self) -> Dict[str, Any]:
        """
        Generate complete dataset of fault signals.

        Returns:
            Dictionary containing:
                - signals: List of generated signals
                - metadata: List of metadata dictionaries
                - config: Configuration used
                - statistics: Generation statistics

        This is the main entry point matching MATLAB's generation loop.
        """
        fault_types = self.config.fault.get_fault_list()
        logger.info(f"Generating dataset for {len(fault_types)} fault types")

        all_signals = []
        all_metadata = []
        all_labels = []

        generation_start = time.time()
        total_signals = 0

        for k, fault in enumerate(fault_types):
            logger.info(f"[{k+1}/{len(fault_types)}] Generating fault: {fault}")

            num_base = self.config.num_signals_per_fault

            if self.config.augmentation.enabled:
                num_augmented = int(num_base * self.config.augmentation.ratio)
            else:
                num_augmented = 0

            num_total_for_fault = num_base + num_augmented

            for n in range(num_total_for_fault):
                # Per-signal seed variation
                if self.config.per_signal_seed_variation and self.config.rng_seed is not None:
                    set_seed(self.config.rng_seed + total_signals)

                is_augmented = (n >= num_base)

                # Generate single signal
                signal, metadata = self.generate_single_signal(fault, is_augmented)

                # Post-generation validation (warn, don't drop)
                val_errors = validate_signal(
                    signal,
                    expected_length=self.config.signal.N,
                    label=f"{fault}_{n:03d}",
                    raise_on_error=False,
                )
                if val_errors:
                    for err in val_errors:
                        logger.warning(f"Signal validation: {err}")

                all_signals.append(signal)
                all_metadata.append(metadata)
                all_labels.append(fault)
                total_signals += 1

            logger.info(f"  [OK] Generated {num_total_for_fault} signals ({num_base} base + {num_augmented} augmented)")

        generation_time = time.time() - generation_start

        logger.info("=" * 60)
        logger.info("[DONE] DATA GENERATION COMPLETE")
        logger.info(f"  Total signals: {total_signals}")
        logger.info(f"  Fault types: {len(fault_types)}")
        logger.info(f"  Generation time: {generation_time:.2f} s ({total_signals/generation_time:.2f} signals/s)")
        logger.info("=" * 60)

        return {
            'signals': all_signals,
            'metadata': all_metadata,
            'labels': all_labels,
            'config': self.config,
            'statistics': {
                'total_signals': total_signals,
                'num_faults': len(fault_types),
                'generation_time_s': generation_time,
                'signals_per_second': total_signals / generation_time
            }
        }

    def generate_single_signal(
        self,
        fault: str,
        is_augmented: bool = False
    ) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Generate a single fault signal with metadata.

        Args:
            fault: Fault type name
            is_augmented: Whether this is an augmented sample

        Returns:
            Tuple of (signal array, metadata object)
        """
        # Severity configuration
        severity, severity_factor, severity_curve, has_evolution = self._configure_severity(fault)

        # Operating conditions
        Omega, omega, speed_variation, load_factor, load_percent, temperature_C, temp_factor, operating_factor, amp_base = self._configure_operating_conditions()

        # Physics parameters
        sommerfeld, reynolds, clearance_ratio, physics_factor = self._configure_physics(Omega, temperature_C, load_factor)

        # Transients
        transient_modulation, transient_type, transient_params = self._configure_transients()

        # Initialize baseline signal
        x = amp_base * 0.05 * np.random.randn(self.config.signal.N)

        # Apply noise layers
        x, noise_sources_applied = self.noise_generator.apply_noise_layers(x)

        # Apply fault-specific signature
        x_fault = self.fault_modeler.generate_fault_signal(
            fault, severity_curve, transient_modulation, omega, Omega,
            load_factor, temp_factor, operating_factor, physics_factor,
            sommerfeld, speed_variation
        )
        x += x_fault

        # Apply advanced physics effects (V2 — only if any toggles are on)
        if self.config.advanced_physics.get_enabled_effects():
            x = self.fault_modeler.apply_advanced_physics(
                x, omega, Omega, temperature_C
            )

        # Data augmentation
        aug_params = self._apply_augmentation(x, is_augmented)

        # Assemble metadata
        metadata = SignalMetadata(
            fault=fault,
            severity=severity,
            severity_factor_initial=severity_factor,
            has_evolution=has_evolution,
            speed_rpm=Omega * 60,
            speed_variation_factor=speed_variation,
            load_percent=load_percent,
            temperature_C=temperature_C,
            operating_factor=operating_factor,
            sommerfeld_number=sommerfeld,
            reynolds_number=reynolds,
            clearance_ratio=clearance_ratio,
            physics_factor=physics_factor,
            sommerfeld_calculated=self.config.physics.calculate_sommerfeld,
            transient_type=transient_type,
            transient_params=transient_params,
            fs=self.config.signal.fs,
            duration_s=self.config.signal.T,
            num_samples=self.config.signal.N,
            signal_rms=float(np.sqrt(np.mean(x**2))),
            signal_peak=float(np.max(np.abs(x))),
            signal_crest_factor=float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-10)),
            is_augmented=is_augmented,
            augmentation=aug_params,
            noise_sources=noise_sources_applied,
            generation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            generator_version="Python_v2.0",
            rng_seed=self.config.rng_seed,
            is_overlapping_fault='mixed_' in fault
        )

        return x, metadata

    def _configure_severity(self, fault: str) -> Tuple[str, float, np.ndarray, bool]:
        """Configure severity level and temporal evolution."""
        if self.config.severity.enabled and fault != 'sain':
            severity = np.random.choice(self.config.severity.levels)
            severity_range = self.config.severity.ranges[severity]
            severity_factor = severity_range[0] + (severity_range[1] - severity_range[0]) * np.random.rand()
        else:
            severity = 'nominal'
            severity_factor = 1.0

        # Temporal evolution
        if self.config.severity.temporal_evolution > 0 and np.random.rand() < self.config.severity.temporal_evolution:
            evolution_end = min(1.0, severity_factor + 0.3)
            evolution_curve = np.linspace(severity_factor, evolution_end, self.config.signal.N)
            has_evolution = True
        else:
            evolution_curve = severity_factor * np.ones(self.config.signal.N)
            has_evolution = False

        return severity, severity_factor, evolution_curve, has_evolution

    def _configure_operating_conditions(self) -> Tuple:
        """Configure variable operating conditions."""
        # Speed variation
        speed_variation = 1.0 + (np.random.rand() - 0.5) * 2 * self.config.operating.speed_variation
        Omega = self.config.signal.Omega_base * speed_variation
        omega = 2 * np.pi * Omega

        # Load (30-100% of rated)
        load_percent = (self.config.operating.load_range[0] * 100 +
                       (self.config.operating.load_range[1] - self.config.operating.load_range[0]) * 100 * np.random.rand())
        load_factor = 0.3 + 0.7 * (load_percent / 100)

        # Temperature (40-80°C)
        temperature_C = (self.config.operating.temp_range[0] +
                        (self.config.operating.temp_range[1] - self.config.operating.temp_range[0]) * np.random.rand())
        temp_factor = 0.9 + 0.2 * ((temperature_C - self.config.operating.temp_range[0]) /
                                   (self.config.operating.temp_range[1] - self.config.operating.temp_range[0]))

        # Combined operating factor
        operating_factor = load_factor * temp_factor
        amp_base = (0.2 + 0.1 * np.random.rand()) * operating_factor

        return Omega, omega, speed_variation, load_factor, load_percent, temperature_C, temp_factor, operating_factor, amp_base

    def _configure_physics(self, Omega: float, temperature_C: float, load_factor: float) -> Tuple:
        """Configure physics-based parameters."""
        if self.config.physics.enabled and self.config.physics.calculate_sommerfeld:
            # Calculate Sommerfeld from operating conditions (PHYSICALLY CORRECT)
            viscosity_factor = np.exp(-0.03 * (temperature_C - 60))
            speed_factor = Omega / self.config.signal.Omega_base
            load_factor_somm = 1.0 / load_factor

            sommerfeld = (self.config.physics.sommerfeld_base * viscosity_factor *
                         speed_factor * load_factor_somm)
            sommerfeld = np.clip(sommerfeld, 0.05, 0.5)
        else:
            sommerfeld = self.config.physics.sommerfeld_base + (np.random.rand() - 0.5) * 0.2

        reynolds = (self.config.physics.reynolds_range[0] +
                   (self.config.physics.reynolds_range[1] - self.config.physics.reynolds_range[0]) * np.random.rand())

        clearance_ratio = (self.config.physics.clearance_ratio_range[0] +
                          (self.config.physics.clearance_ratio_range[1] - self.config.physics.clearance_ratio_range[0]) * np.random.rand())

        physics_factor = np.sqrt(sommerfeld / self.config.physics.sommerfeld_base)

        return sommerfeld, reynolds, clearance_ratio, physics_factor

    def _configure_transients(self) -> Tuple:
        """Configure non-stationary transient behavior."""
        transient_modulation = np.ones(self.config.signal.N)
        transient_type = 'none'
        transient_params = {}

        if self.config.transient.enabled and np.random.rand() < self.config.transient.probability:
            transient_type = np.random.choice(self.config.transient.types)

            if transient_type == 'speed_ramp':
                ramp_start_idx = int(0.2 * self.config.signal.N)
                ramp_end_idx = int(0.6 * self.config.signal.N)
                speed_mult = np.linspace(0.85, 1.15, ramp_end_idx - ramp_start_idx)
                transient_modulation[ramp_start_idx:ramp_end_idx] = speed_mult
                transient_params = {'start_idx': ramp_start_idx, 'end_idx': ramp_end_idx, 'speed_range': [0.85, 1.15]}

            elif transient_type == 'load_step':
                step_idx = int(0.4 * self.config.signal.N)
                transient_modulation[:step_idx] = 0.7
                transient_modulation[step_idx:] = 1.0
                transient_params = {'step_idx': step_idx, 'load_values': [0.7, 1.0]}

            elif transient_type == 'thermal_expansion':
                thermal_time_const = 0.3 * self.config.signal.N
                transient_modulation = 0.9 + 0.2 * (1 - np.exp(-np.arange(self.config.signal.N) / thermal_time_const))
                transient_params = {'time_constant': thermal_time_const}

        return transient_modulation, transient_type, transient_params

    def _apply_augmentation(self, x: np.ndarray, is_augmented: bool) -> Dict[str, Any]:
        """Apply data augmentation if enabled."""
        aug_params = {'method': 'none'}

        if is_augmented and self.config.augmentation.enabled:
            aug_method = np.random.choice(self.config.augmentation.methods)

            if aug_method == 'time_shift':
                shift_max = int(self.config.augmentation.time_shift_max * self.config.signal.N)
                shift_samples = np.random.randint(-shift_max, shift_max)
                x[:] = np.roll(x, shift_samples)
                aug_params = {'method': 'time_shift', 'shift_samples': int(shift_samples)}

            elif aug_method == 'amplitude_scale':
                scale_range = self.config.augmentation.amplitude_scale_range
                scale_factor = scale_range[0] + (scale_range[1] - scale_range[0]) * np.random.rand()
                x[:] = x * scale_factor
                aug_params = {'method': 'amplitude_scale', 'scale_factor': float(scale_factor)}

            elif aug_method == 'noise_injection':
                noise_range = self.config.augmentation.extra_noise_range
                extra_noise_level = noise_range[0] + (noise_range[1] - noise_range[0]) * np.random.rand()
                extra_noise = extra_noise_level * np.random.randn(self.config.signal.N)
                x[:] += extra_noise
                aug_params = {'method': 'noise_injection', 'noise_level': float(extra_noise_level)}

        return aug_params

    def save_dataset(
        self,
        dataset: Dict,
        output_dir: Optional[Path] = None,
        format: str = 'hdf5',  # DEFAULT='hdf5' (recommended for performance)
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Dict[str, Path]:
        """
        Save generated dataset to disk in .mat and/or HDF5 format.

        Args:
            dataset: Dataset from generate_dataset()
            output_dir: Output directory (uses config if None)
            format: Output format - 'mat', 'hdf5', or 'both' (default: 'hdf5')
                HDF5 is recommended: 25× faster loading, 30% smaller, single file
            train_val_test_split: Train/val/test split ratios for HDF5 format

        Returns:
            Dictionary with paths to saved files:
            - 'mat_dir': Path to directory with .mat files (if format='mat' or 'both')
            - 'hdf5': Path to HDF5 file (if format='hdf5' or 'both')

        Examples:
            >>> # Backward compatible - saves .mat files only (default behavior)
            >>> generator.save_dataset(dataset, output_dir='data/processed')

            >>> # Save as HDF5 only
            >>> paths = generator.save_dataset(dataset, format='hdf5')

            >>> # Save both formats
            >>> paths = generator.save_dataset(dataset, format='both')
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        # Save as .mat files (original format - for backward compatibility)
        if format in ['mat', 'both']:
            # For backward compatibility: save directly in output_dir if format='mat'
            # Otherwise save in subdirectory
            if format == 'mat':
                mat_dir = output_dir
            else:
                mat_dir = output_dir / 'mat_files'
                mat_dir.mkdir(exist_ok=True)

            logger.info(f"Saving .mat files to: {mat_dir}")

            for i, (signal, metadata, label) in enumerate(zip(
                dataset['signals'], dataset['metadata'], dataset['labels']
            )):
                # Determine filename
                signal_idx = (i % self.config.num_signals_per_fault) + 1
                if metadata.is_augmented:
                    filename = f"{label}_{signal_idx:03d}_aug.mat"
                else:
                    filename = f"{label}_{signal_idx:03d}.mat"

                filepath = mat_dir / filename

                # Prepare MATLAB-compatible structure
                mat_data = {
                    'x': signal,
                    'fs': metadata.fs,
                    'fault': label,
                    'metadata': self._metadata_to_matlab_struct(metadata)
                }

                # Save .mat file
                savemat(filepath, mat_data, do_compression=True)

            logger.info(f"✓ Saved {len(dataset['signals'])} .mat files")
            saved_paths['mat_dir'] = mat_dir

        # Save as HDF5 file (new format - faster, more efficient)
        if format in ['hdf5', 'both']:
            hdf5_path = output_dir / 'dataset.h5'
            logger.info(f"Saving HDF5 file to: {hdf5_path}")

            saved_paths['hdf5'] = self._save_as_hdf5(
                dataset,
                hdf5_path,
                train_val_test_split
            )

            logger.info(f"✓ Saved HDF5 file: {hdf5_path}")

        return saved_paths

    def _metadata_to_matlab_struct(self, metadata: SignalMetadata) -> Dict:
        """Convert metadata to MATLAB-compatible structure."""
        return asdict(metadata)

    def _save_as_hdf5(
        self,
        dataset: Dict,
        output_path: Path,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Path:
        """
        Save dataset as HDF5 file with train/val/test splits.

        This creates an HDF5 file compatible with dash_app and training scripts:
        - f['train']['signals'] and f['train']['labels']
        - f['val']['signals'] and f['val']['labels']
        - f['test']['signals'] and f['test']['labels']

        Args:
            dataset: Dataset dictionary from generate_dataset()
            output_path: Path to HDF5 output file
            split_ratios: (train, val, test) split ratios (default: 0.7, 0.15, 0.15)

        Returns:
            Path to created HDF5 file
        """
        signals = dataset['signals']
        labels_str = dataset['labels']
        metadata = dataset['metadata']

        # Convert signals to numpy array if it's a list
        if isinstance(signals, list):
            signals = np.array(signals, dtype=np.float32)
        elif not isinstance(signals, np.ndarray):
            raise TypeError(f"Signals must be list or numpy array, got {type(signals)}")

        # Validate signals array
        if len(signals) == 0:
            raise ValueError("Cannot save empty dataset - no signals generated")

        if signals.ndim != 2:
            raise ValueError(f"Signals must be 2D array (num_samples, signal_length), got shape {signals.shape}")

        # Convert string labels to integers using FAULT_TYPES mapping
        label_to_idx = {label: idx for idx, label in enumerate(FAULT_TYPES)}

        # Validate all labels are known
        unknown_labels = set(labels_str) - set(FAULT_TYPES)
        if unknown_labels:
            raise ValueError(f"Unknown fault types in dataset: {unknown_labels}. Valid types: {FAULT_TYPES}")

        labels = np.array([label_to_idx[label] for label in labels_str], dtype=np.int64)

        # Create stratified splits to ensure each split has all classes
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            signals, labels,
            test_size=test_ratio,
            stratify=labels,
            random_state=self.config.rng_seed
        )

        # Second split: separate train and val from temp
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.config.rng_seed
        )

        # Create HDF5 file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Config Versioning prep ────────────────────────────────────
        import hashlib
        config_dict = asdict(self.config)
        config_json_str = json.dumps(config_dict, default=str, sort_keys=True)
        config_hash = hashlib.md5(config_json_str.encode()).hexdigest()[:8]
        version_tag = getattr(
            self, '_version_tag',
            f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        with h5py.File(output_path, 'w') as f:
            # Store global attributes
            f.attrs['num_classes'] = NUM_CLASSES
            f.attrs['sampling_rate'] = SAMPLING_RATE
            f.attrs['signal_length'] = int(signals.shape[1])
            f.attrs['generation_date'] = datetime.now().isoformat()
            f.attrs['split_ratios'] = split_ratios
            f.attrs['rng_seed'] = self.config.rng_seed

            # Create train group
            train_grp = f.create_group('train')
            train_grp.create_dataset(
                'signals',
                data=X_train,
                compression='gzip',
                compression_opts=4
            )
            train_grp.create_dataset('labels', data=y_train)
            train_grp.attrs['num_samples'] = len(X_train)

            # Create val group
            val_grp = f.create_group('val')
            val_grp.create_dataset(
                'signals',
                data=X_val,
                compression='gzip',
                compression_opts=4
            )
            val_grp.create_dataset('labels', data=y_val)
            val_grp.attrs['num_samples'] = len(X_val)

            # Create test group
            test_grp = f.create_group('test')
            test_grp.create_dataset(
                'signals',
                data=X_test,
                compression='gzip',
                compression_opts=4
            )
            test_grp.create_dataset('labels', data=y_test)
            test_grp.attrs['num_samples'] = len(X_test)

            # Store metadata as JSON (optional, for reference)
            if metadata:
                metadata_json = [json.dumps(asdict(m)) for m in metadata]
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('metadata', data=metadata_json, dtype=dt)

            # ── Config Versioning (V2) ────────────────────────────────
            f.attrs['config_json'] = config_json_str
            f.attrs['config_hash'] = config_hash
            f.attrs['config_version'] = version_tag
            f.attrs['generator_version'] = 'Python_v2.0'
            f.attrs['advanced_physics_effects'] = json.dumps(
                self.config.advanced_physics.get_enabled_effects()
            )

        # Save standalone config JSON sidecar for easy inspection
        config_sidecar = output_path.parent / 'dataset_config.json'
        with open(config_sidecar, 'w') as cf:
            json.dump({
                'config': config_dict,
                'config_hash': config_hash,
                'version_tag': version_tag,
                'generation_timestamp': datetime.now().isoformat(),
                'generator_version': 'Python_v2.0',
                'advanced_physics_enabled': self.config.advanced_physics.get_enabled_effects(),
                'total_signals': len(signals),
                'split_sizes': {
                    'train': len(X_train),
                    'val': len(X_val),
                    'test': len(X_test),
                },
            }, cf, indent=2, default=str)
        logger.info(f"Config saved: {config_sidecar}")

        logger.info(
            f"Created HDF5 with splits - Train: {len(X_train)}, "
            f"Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return output_path
