#!/usr/bin/env python
"""
Standalone dataset generation script.
Called by dashboard via subprocess to avoid import namespace conflicts.

Uses STREAMING writes to HDF5 for memory efficiency - can generate
datasets of any size without running out of RAM.
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
    """
    Generate dataset using Phase 0 SignalGenerator.
    
    Uses STREAMING mode - writes each signal directly to disk as it's generated,
    avoiding the need to hold all signals in memory at once.
    """
    try:
        from data.signal_generator import SignalGenerator
        from config.data_config import DataConfig, SignalConfig, FaultConfig, \
            SeverityConfig, NoiseConfig, OperatingConfig, PhysicsConfig, \
            TransientConfig, AugmentationConfig
        import h5py
        import numpy as np

        logger.info(f"Generating dataset '{config['name']}' (streaming mode)...")

        # Build DataConfig
        data_config = DataConfig(
            num_signals_per_fault=config.get('num_signals_per_fault', 100),
            output_dir=config.get('output_dir', 'data/generated'),
            rng_seed=config.get('random_seed', 42),
        )

        # Configure signal parameters
        data_config.signal = SignalConfig()
        signal_length = data_config.signal.duration * data_config.signal.sample_rate

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

        # Generate signals with STREAMING to disk
        generator = SignalGenerator(data_config)
        fault_types = data_config.fault.get_fault_list()
        logger.info(f"Generating signals for {len(fault_types)} fault types...")

        # Calculate total signals for pre-allocation
        num_base = data_config.num_signals_per_fault
        num_augmented = int(num_base * data_config.augmentation.ratio) if data_config.augmentation.enabled else 0
        signals_per_fault = num_base + num_augmented
        total_signals = signals_per_fault * len(fault_types)

        logger.info(f"Total signals to generate: {total_signals} ({signals_per_fault} per fault type)")
        logger.info(f"Signal length: {signal_length} samples")
        logger.info(f"Estimated memory per signal: {signal_length * 4 / 1024 / 1024:.2f} MB (float32)")

        # Setup output path
        project_root = Path(__file__).resolve().parent.parent
        output_dir = config.get('output_dir', 'data/generated')
        if not Path(output_dir).is_absolute():
            output_dir = project_root / output_dir
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = config.get('name', 'dataset')
        hdf5_path = output_dir / f"{dataset_name}.h5"

        # Calculate split sizes (70/15/15)
        n_train = int(0.7 * total_signals)
        n_val = int(0.15 * total_signals)
        n_test = total_signals - n_train - n_val

        # Generate random assignment for each signal index
        np.random.seed(config.get('random_seed', 42))
        all_indices = np.random.permutation(total_signals)
        train_indices = set(all_indices[:n_train])
        val_indices = set(all_indices[n_train:n_train + n_val])
        # test_indices = remaining

        logger.info(f"Saving to {hdf5_path} (streaming mode)...")
        logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")

        # Create HDF5 file with pre-allocated, chunked datasets
        with h5py.File(hdf5_path, 'w') as f:
            # Create groups
            train_grp = f.create_group('train')
            val_grp = f.create_group('val')
            test_grp = f.create_group('test')

            # Pre-allocate datasets with chunking for efficient streaming writes
            # Use float32 instead of float64 to save memory (50% reduction)
            chunk_size = min(100, max(1, n_train // 10))  # ~10% of data per chunk
            
            train_signals = train_grp.create_dataset(
                'signals', shape=(n_train, int(signal_length)), dtype='float32',
                chunks=(min(chunk_size, n_train), int(signal_length)),
                compression='gzip', compression_opts=4
            )
            train_labels = train_grp.create_dataset(
                'labels', shape=(n_train,), dtype='int32'
            )

            val_signals = val_grp.create_dataset(
                'signals', shape=(n_val, int(signal_length)), dtype='float32',
                chunks=(min(chunk_size, n_val) if n_val > 0 else 1, int(signal_length)),
                compression='gzip', compression_opts=4
            )
            val_labels = val_grp.create_dataset(
                'labels', shape=(n_val,), dtype='int32'
            )

            test_signals = test_grp.create_dataset(
                'signals', shape=(n_test, int(signal_length)), dtype='float32',
                chunks=(min(chunk_size, n_test) if n_test > 0 else 1, int(signal_length)),
                compression='gzip', compression_opts=4
            )
            test_labels = test_grp.create_dataset(
                'labels', shape=(n_test,), dtype='int32'
            )

            # Track current write positions for each split
            train_idx = 0
            val_idx = 0
            test_idx = 0

            # Track all labels for metadata
            all_labels = []
            global_idx = 0

            # Generate and stream signals
            for fault_idx, fault in enumerate(fault_types):
                fault_label = fault_idx  # Use index as label
                
                for n in range(signals_per_fault):
                    is_augmented = (n >= num_base)
                    signal, metadata = generator.generate_single_signal(fault, is_augmented)
                    signal = signal.astype(np.float32)  # Convert to float32

                    # Determine which split this signal belongs to
                    if global_idx in train_indices:
                        train_signals[train_idx] = signal
                        train_labels[train_idx] = fault_label
                        train_idx += 1
                    elif global_idx in val_indices:
                        val_signals[val_idx] = signal
                        val_labels[val_idx] = fault_label
                        val_idx += 1
                    else:
                        test_signals[test_idx] = signal
                        test_labels[test_idx] = fault_label
                        test_idx += 1

                    all_labels.append(fault)
                    global_idx += 1

                    # Log progress every 500 signals
                    if global_idx % 500 == 0:
                        logger.info(f"  Progress: {global_idx}/{total_signals} signals ({100*global_idx/total_signals:.1f}%)")

                logger.info(f"  ✓ Generated {signals_per_fault} signals for '{fault}' ({fault_idx+1}/{len(fault_types)})")

            # Store metadata
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['total_signals'] = total_signals
            meta_grp.attrs['num_faults'] = len(fault_types)
            meta_grp.attrs['dataset_name'] = dataset_name
            meta_grp.attrs['num_classes'] = len(fault_types)
            meta_grp.attrs['signal_length'] = int(signal_length)
            meta_grp.attrs['train_size'] = n_train
            meta_grp.attrs['val_size'] = n_val
            meta_grp.attrs['test_size'] = n_test

            # Store fault type names
            dt = h5py.special_dtype(vlen=str)
            meta_grp.create_dataset('fault_types', data=fault_types, dtype=dt)

        logger.info(f"✓ Dataset generation complete: {hdf5_path}")
        logger.info(f"  Total signals: {total_signals}")
        logger.info(f"  File size: {hdf5_path.stat().st_size / 1024 / 1024:.2f} MB")

        return {
            'success': True,
            'output_path': str(hdf5_path),
            'total_signals': total_signals,
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
