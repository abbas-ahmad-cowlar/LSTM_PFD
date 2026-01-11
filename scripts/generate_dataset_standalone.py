#!/usr/bin/env python
"""
Standalone dataset generation script.
Called by dashboard via subprocess to avoid import namespace conflicts.
"""
import sys
import json
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_dataset(config: dict) -> dict:
    """Generate dataset using Phase 0 SignalGenerator."""
    try:
        from data.signal_generator import SignalGenerator
        from config.data_config import DataConfig, SignalConfig, FaultConfig, \
            SeverityConfig, NoiseConfig, OperatingConfig, PhysicsConfig, \
            TransientConfig, AugmentationConfig
        import h5py
        import numpy as np

        logger.info(f"Generating dataset '{config['name']}'...")

        # Build DataConfig
        data_config = DataConfig(
            num_signals_per_fault=config.get('num_signals_per_fault', 100),
            output_dir=config.get('output_dir', 'data/generated'),
            rng_seed=config.get('random_seed', 42),
        )

        # Configure signal parameters
        data_config.signal = SignalConfig()

        # Configure fault types
        fault_config = FaultConfig()
        selected_faults = config.get('fault_types', [])

        # Disable all faults first
        fault_config.include_healthy = False
        for fault in fault_config.single_faults:
            fault_config.single_faults[fault] = False
        for fault in fault_config.mixed_faults:
            fault_config.mixed_faults[fault] = False

        # Enable selected faults
        for dashboard_fault in selected_faults:
            if dashboard_fault == 'sain':
                fault_config.include_healthy = True
            elif dashboard_fault.startswith('mixed_'):
                fault_key = dashboard_fault.replace('mixed_', '')
                if fault_key in fault_config.mixed_faults:
                    fault_config.include_single = True
                    fault_config.include_mixed = True
                    fault_config.mixed_faults[fault_key] = True
            else:
                if dashboard_fault in fault_config.single_faults:
                    fault_config.include_single = True
                    fault_config.single_faults[dashboard_fault] = True

        data_config.fault = fault_config

        # Configure severity
        severity_config = SeverityConfig()
        severity_config.enabled = True
        severity_config.levels = config.get('severity_levels', ['incipient', 'mild', 'moderate', 'severe'])
        severity_config.temporal_evolution = 0.30 if config.get('temporal_evolution', True) else 0.0
        data_config.severity = severity_config

        # Configure noise
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

        data_config.physics = PhysicsConfig()
        data_config.transient = TransientConfig()

        # Configure augmentation
        aug_dict = config.get('augmentation', {})
        aug_config = AugmentationConfig()
        aug_config.enabled = aug_dict.get('enabled', True)
        aug_config.ratio = aug_dict.get('ratio', 0.30)
        aug_config.methods = aug_dict.get('methods', ['time_shift', 'amplitude_scale', 'noise_injection'])
        data_config.augmentation = aug_config

        # Generate signals
        generator = SignalGenerator(data_config)
        fault_types = data_config.fault.get_fault_list()
        logger.info(f"Generating signals for {len(fault_types)} fault types...")

        all_signals = []
        all_metadata = []
        all_labels = []

        for fault in fault_types:
            num_base = data_config.num_signals_per_fault
            num_augmented = int(num_base * data_config.augmentation.ratio) if data_config.augmentation.enabled else 0
            num_total = num_base + num_augmented

            for n in range(num_total):
                is_augmented = (n >= num_base)
                signal, metadata = generator.generate_single_signal(fault, is_augmented)
                all_signals.append(signal)
                all_metadata.append(metadata)
                all_labels.append(fault)

            logger.info(f"  Generated {num_total} signals for {fault}")

        # Save to HDF5
        output_dir = Path(config.get('output_dir', 'data/generated'))
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = config.get('name', 'dataset')
        hdf5_path = output_dir / f"{dataset_name}.h5"

        logger.info(f"Saving to {hdf5_path}...")
        with h5py.File(hdf5_path, 'w') as f:
            signals_array = np.array(all_signals)
            labels_array = np.array([fault_types.index(l) if l in fault_types else -1 for l in all_labels])

            # Split 70/15/15
            n = len(signals_array)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)

            indices = np.random.permutation(n)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            train_grp = f.create_group('train')
            train_grp.create_dataset('signals', data=signals_array[train_idx])
            train_grp.create_dataset('labels', data=labels_array[train_idx])

            val_grp = f.create_group('val')
            val_grp.create_dataset('signals', data=signals_array[val_idx])
            val_grp.create_dataset('labels', data=labels_array[val_idx])

            test_grp = f.create_group('test')
            test_grp.create_dataset('signals', data=signals_array[test_idx])
            test_grp.create_dataset('labels', data=labels_array[test_idx])

            # Metadata
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['total_signals'] = len(all_signals)
            meta_grp.attrs['num_faults'] = len(fault_types)
            meta_grp.attrs['dataset_name'] = dataset_name
            meta_grp.attrs['num_classes'] = len(fault_types)
            # Store fault class names
            dt = h5py.special_dtype(vlen=str)
            meta_grp.create_dataset('fault_classes', data=[l for l in all_labels], dtype=dt)

        logger.info(f"✓ Dataset generation complete: {hdf5_path}")

        return {
            'success': True,
            'output_path': str(hdf5_path),
            'total_signals': len(all_signals),
            'num_faults': len(fault_types),
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'output_path': None,
            'total_signals': 0,
            'num_faults': 0,
        }


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic vibration dataset')
    parser.add_argument('--config', type=str, help='JSON config string')
    parser.add_argument('--config-file', type=str, help='Path to JSON config file')
    parser.add_argument('--output-json', type=str, help='Output JSON file path for results')
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    elif args.config:
        config = json.loads(args.config)
    else:
        parser.error('Either --config or --config-file is required')

    result = generate_dataset(config)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result, f)
        print(f"Result saved to {args.output_json}")
    else:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
