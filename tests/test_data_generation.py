"""
Unit tests for data generation pipeline.

Purpose:
    Comprehensive tests for signal generation, validation, and data loading.
    Ensures correctness, reproducibility, and numerical accuracy.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

from config.data_config import DataConfig, SignalConfig, FaultConfig, SeverityConfig
from data.signal_generator import SignalGenerator, FaultModeler, NoiseGenerator, SignalMetadata
from utils.reproducibility import set_seed
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class TestSignalConfig(unittest.TestCase):
    """Test configuration validation and serialization."""

    def setUp(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Cleanup temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_signal_config_defaults(self):
        """Test SignalConfig default values match MATLAB."""
        config = SignalConfig()
        self.assertEqual(config.fs, 20480.0)
        self.assertEqual(config.T, 5.0)
        self.assertEqual(config.N, 102400)
        self.assertEqual(config.Omega_base, 60.0)

    def test_signal_config_validation(self):
        """Test configuration validation."""
        config = SignalConfig()
        self.assertTrue(config.validate())

        # Invalid config (negative fs)
        with self.assertRaises(ValueError):
            bad_config = SignalConfig(fs=-100)
            bad_config.validate()

    def test_config_yaml_serialization(self):
        """Test YAML save/load roundtrip."""
        config = DataConfig()
        yaml_path = self.temp_dir / 'test_config.yaml'

        # Save
        config.to_yaml(yaml_path)
        self.assertTrue(yaml_path.exists())

        # Load
        loaded_config = DataConfig.from_yaml(yaml_path)
        self.assertEqual(config.signal.fs, loaded_config.signal.fs)
        self.assertEqual(config.rng_seed, loaded_config.rng_seed)

    def test_fault_config_enables(self):
        """Test fault type enable/disable flags."""
        config = FaultConfig()

        # All enabled by default
        self.assertTrue(config.enable_sain)
        self.assertTrue(config.enable_desalignement)
        self.assertTrue(config.enable_mixed_misalign_imbalance)

        # Disable specific fault
        config.enable_desalignement = False
        self.assertFalse(config.enable_desalignement)


class TestReproducibility(unittest.TestCase):
    """Test deterministic behavior of signal generation."""

    def test_seed_reproducibility(self):
        """Test that same seed produces identical signals."""
        config = DataConfig(num_signals_per_fault=2, rng_seed=42)

        # Generate first dataset
        set_seed(42)
        gen1 = SignalGenerator(config)
        dataset1 = gen1.generate_dataset()
        signal1 = dataset1['signals'][0]

        # Generate second dataset with same seed
        set_seed(42)
        gen2 = SignalGenerator(config)
        dataset2 = gen2.generate_dataset()
        signal2 = dataset2['signals'][0]

        # Should be identical
        np.testing.assert_array_equal(signal1, signal2)

    def test_different_seeds_produce_different_signals(self):
        """Test that different seeds produce different signals."""
        config = DataConfig(num_signals_per_fault=1, rng_seed=42)

        # Seed 42
        set_seed(42)
        gen1 = SignalGenerator(config)
        dataset1 = gen1.generate_dataset()
        signal1 = dataset1['signals'][0]

        # Seed 123
        set_seed(123)
        gen2 = SignalGenerator(config)
        dataset2 = gen2.generate_dataset()
        signal2 = dataset2['signals'][0]

        # Should be different
        self.assertFalse(np.array_equal(signal1, signal2))


class TestFaultModeler(unittest.TestCase):
    """Test fault signature generation."""

    def setUp(self):
        """Setup test fixtures."""
        self.fs = 20480.0
        self.N = 102400
        self.omega = np.linspace(0, self.N - 1, self.N) / self.fs * 2 * np.pi

    def test_healthy_signal_generation(self):
        """Test healthy (sain) signal generation."""
        severity = np.ones(self.N) * 0.1
        transient = np.ones(self.N)

        signal = FaultModeler.generate_fault_signal(
            fault_type='sain',
            severity_curve=severity,
            transient_modulation=transient,
            omega=self.omega,
            Omega=60.0,
            load_factor=1.0,
            temp_factor=1.0,
            sommerfeld=0.5,
            fs=self.fs,
            N=self.N
        )

        # Healthy signal should be low amplitude
        self.assertEqual(len(signal), self.N)
        self.assertLess(np.max(np.abs(signal)), 0.5)

    def test_desalignement_harmonics(self):
        """Test desalignement produces 2X and 3X harmonics."""
        severity = np.ones(self.N) * 0.5
        transient = np.ones(self.N)

        signal = FaultModeler.generate_fault_signal(
            fault_type='desalignement',
            severity_curve=severity,
            transient_modulation=transient,
            omega=self.omega,
            Omega=60.0,
            load_factor=1.0,
            temp_factor=1.0,
            sommerfeld=0.5,
            fs=self.fs,
            N=self.N
        )

        # Compute FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(self.N, 1/self.fs)

        # Check for 2X harmonic (2 * 60Hz = 120Hz)
        idx_2x = np.argmin(np.abs(freqs - 120.0))
        power_2x = np.abs(fft[idx_2x])

        # Check for 3X harmonic (3 * 60Hz = 180Hz)
        idx_3x = np.argmin(np.abs(freqs - 180.0))
        power_3x = np.abs(fft[idx_3x])

        # Both should have significant power
        self.assertGreater(power_2x, np.mean(np.abs(fft)))
        self.assertGreater(power_3x, np.mean(np.abs(fft)))

    def test_desequilibre_speed_dependence(self):
        """Test desequilibre amplitude scales with speed squared."""
        severity = np.ones(self.N) * 0.5
        transient = np.ones(self.N)

        # Low speed
        signal_low = FaultModeler.generate_fault_signal(
            fault_type='desequilibre',
            severity_curve=severity,
            transient_modulation=transient,
            omega=self.omega,
            Omega=30.0,  # Low speed
            load_factor=1.0,
            temp_factor=1.0,
            sommerfeld=0.5,
            fs=self.fs,
            N=self.N
        )

        # High speed
        signal_high = FaultModeler.generate_fault_signal(
            fault_type='desequilibre',
            severity_curve=severity,
            transient_modulation=transient,
            omega=self.omega,
            Omega=120.0,  # High speed (4x)
            load_factor=1.0,
            temp_factor=1.0,
            sommerfeld=0.5,
            fs=self.fs,
            N=self.N
        )

        # High speed should have ~16x higher RMS (4^2)
        rms_low = np.sqrt(np.mean(signal_low**2))
        rms_high = np.sqrt(np.mean(signal_high**2))

        ratio = rms_high / (rms_low + 1e-10)
        self.assertGreater(ratio, 10.0)  # Should be significantly higher

    def test_all_fault_types_generate(self):
        """Test that all 11 fault types can be generated without errors."""
        fault_types = [
            'sain', 'desalignement', 'desequilibre', 'jeu', 'lubrification',
            'cavitation', 'usure', 'oilwhirl', 'mixed_misalign_imbalance',
            'mixed_wear_lube', 'mixed_cavit_jeu'
        ]

        severity = np.ones(self.N) * 0.5
        transient = np.ones(self.N)

        for fault_type in fault_types:
            with self.subTest(fault_type=fault_type):
                signal = FaultModeler.generate_fault_signal(
                    fault_type=fault_type,
                    severity_curve=severity,
                    transient_modulation=transient,
                    omega=self.omega,
                    Omega=60.0,
                    load_factor=1.0,
                    temp_factor=1.0,
                    sommerfeld=0.5,
                    fs=self.fs,
                    N=self.N
                )

                # Check valid output
                self.assertEqual(len(signal), self.N)
                self.assertFalse(np.any(np.isnan(signal)))
                self.assertFalse(np.any(np.isinf(signal)))


class TestNoiseGenerator(unittest.TestCase):
    """Test noise model implementation."""

    def setUp(self):
        """Setup test fixtures."""
        self.N = 102400
        self.fs = 20480.0
        self.clean_signal = np.sin(2 * np.pi * 60 * np.arange(self.N) / self.fs)

    def test_noise_increases_signal_power(self):
        """Test that adding noise increases signal power."""
        config = DataConfig()

        # Clean signal RMS
        rms_clean = np.sqrt(np.mean(self.clean_signal**2))

        # Add noise
        noisy_signal, noise_info = NoiseGenerator.apply_noise_layers(
            self.clean_signal.copy(),
            config.noise,
            self.fs
        )

        # Noisy signal should have higher RMS
        rms_noisy = np.sqrt(np.mean(noisy_signal**2))
        self.assertGreater(rms_noisy, rms_clean)

    def test_no_noise_when_disabled(self):
        """Test that disabling noise leaves signal unchanged."""
        config = DataConfig()
        # Disable all noise sources
        config.noise.enable_measurement = False
        config.noise.enable_emi = False
        config.noise.enable_pink = False
        config.noise.enable_drift = False
        config.noise.enable_quantization = False
        config.noise.enable_sensor_drift = False
        config.noise.enable_impulse = False

        noisy_signal, noise_info = NoiseGenerator.apply_noise_layers(
            self.clean_signal.copy(),
            config.noise,
            self.fs
        )

        # Should be nearly identical (may have minor floating point differences)
        np.testing.assert_array_almost_equal(noisy_signal, self.clean_signal, decimal=10)

    def test_individual_noise_sources(self):
        """Test each noise source individually."""
        config = DataConfig()

        noise_sources = [
            'measurement', 'emi', 'pink', 'drift',
            'quantization', 'sensor_drift', 'impulse'
        ]

        for noise_type in noise_sources:
            with self.subTest(noise_type=noise_type):
                # Disable all
                config.noise.enable_measurement = False
                config.noise.enable_emi = False
                config.noise.enable_pink = False
                config.noise.enable_drift = False
                config.noise.enable_quantization = False
                config.noise.enable_sensor_drift = False
                config.noise.enable_impulse = False

                # Enable only this one
                setattr(config.noise, f'enable_{noise_type}', True)

                noisy_signal, noise_info = NoiseGenerator.apply_noise_layers(
                    self.clean_signal.copy(),
                    config.noise,
                    self.fs
                )

                # Should be different from clean signal
                self.assertFalse(np.array_equal(noisy_signal, self.clean_signal))


class TestSignalGenerator(unittest.TestCase):
    """Test complete signal generation pipeline."""

    def setUp(self):
        """Setup test fixtures."""
        setup_logging(console=False, file=False)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = DataConfig(
            num_signals_per_fault=5,
            output_dir=str(self.temp_dir / 'signals'),
            rng_seed=42
        )

    def tearDown(self):
        """Cleanup temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_dataset_generation(self):
        """Test full dataset generation."""
        set_seed(42)
        generator = SignalGenerator(self.config)
        dataset = generator.generate_dataset()

        # Check structure
        self.assertIn('signals', dataset)
        self.assertIn('metadata', dataset)
        self.assertIn('labels', dataset)
        self.assertIn('config', dataset)
        self.assertIn('statistics', dataset)

        # Check sizes
        signals = dataset['signals']
        labels = dataset['labels']
        metadata = dataset['metadata']

        # Should have 11 faults * 5 signals = 55 signals
        # (assuming all faults enabled)
        self.assertEqual(len(signals), len(labels))
        self.assertEqual(len(signals), len(metadata))
        self.assertGreater(len(signals), 0)

    def test_signal_shape(self):
        """Test generated signals have correct shape."""
        set_seed(42)
        generator = SignalGenerator(self.config)
        dataset = generator.generate_dataset()

        for signal in dataset['signals']:
            self.assertEqual(signal.shape, (self.config.signal.N,))

    def test_all_faults_present(self):
        """Test that all enabled faults are generated."""
        set_seed(42)
        generator = SignalGenerator(self.config)
        dataset = generator.generate_dataset()

        labels = dataset['labels']
        unique_labels = set(labels)

        # Check for expected faults (if enabled in config)
        if self.config.fault.enable_sain:
            self.assertIn('sain', unique_labels)
        if self.config.fault.enable_desalignement:
            self.assertIn('desalignement', unique_labels)

    def test_metadata_completeness(self):
        """Test that metadata contains all required fields."""
        set_seed(42)
        generator = SignalGenerator(self.config)
        dataset = generator.generate_dataset()

        metadata = dataset['metadata'][0]

        # Check required fields
        required_fields = [
            'fault_type', 'severity_level', 'severity_initial',
            'speed_rpm', 'load_percent', 'temperature_c',
            'sommerfeld_number', 'rms', 'peak', 'crest_factor'
        ]

        for field in required_fields:
            self.assertIn(field, metadata)

    def test_augmentation_ratio(self):
        """Test data augmentation generates correct number of signals."""
        config = DataConfig(
            num_signals_per_fault=10,
            rng_seed=42
        )
        config.augmentation.ratio = 0.5  # 50% augmented

        set_seed(42)
        generator = SignalGenerator(config)
        dataset = generator.generate_dataset()

        metadata = dataset['metadata']
        augmented_count = sum(1 for m in metadata if m.get('is_augmented', False))
        total_count = len(metadata)

        # Check augmentation ratio (should be close to 50%)
        actual_ratio = augmented_count / total_count
        self.assertAlmostEqual(actual_ratio, 0.5, delta=0.15)

    def test_severity_distribution(self):
        """Test severity levels are distributed across all levels."""
        config = DataConfig(num_signals_per_fault=20, rng_seed=42)

        set_seed(42)
        generator = SignalGenerator(config)
        dataset = generator.generate_dataset()

        # Extract severity levels
        severities = [m['severity_level'] for m in dataset['metadata']]
        unique_severities = set(severities)

        # Should have multiple severity levels
        self.assertGreaterEqual(len(unique_severities), 3)

    def test_signal_statistics(self):
        """Test signal statistics are reasonable."""
        set_seed(42)
        generator = SignalGenerator(self.config)
        dataset = generator.generate_dataset()

        for signal, metadata in zip(dataset['signals'], dataset['metadata']):
            # Check for NaN/Inf
            self.assertFalse(np.any(np.isnan(signal)))
            self.assertFalse(np.any(np.isinf(signal)))

            # Check RMS matches metadata
            computed_rms = np.sqrt(np.mean(signal**2))
            metadata_rms = metadata['rms']
            self.assertAlmostEqual(computed_rms, metadata_rms, places=5)

            # Check crest factor is positive
            self.assertGreater(metadata['crest_factor'], 0)


class TestPhysicsCalculations(unittest.TestCase):
    """Test physics-based calculations (Sommerfeld, etc.)."""

    def test_sommerfeld_calculation(self):
        """Test Sommerfeld number calculation."""
        # Parameters matching MATLAB
        mu = 0.05  # Pa·s (typical bearing oil)
        N = 60.0   # Hz (3600 RPM)
        P = 1e6    # Pa (load)
        R = 0.05   # m (radius)
        C = 0.0001 # m (clearance)

        # S = (μ * N / P) * (R / C)^2
        S_expected = (mu * N / P) * (R / C)**2

        # Should be reasonable value (typically 0.1 - 1.0 for hydrodynamic bearings)
        self.assertGreater(S_expected, 0.01)
        self.assertLess(S_expected, 10.0)

    def test_operating_conditions_variation(self):
        """Test that operating conditions vary within specified ranges."""
        config = DataConfig(num_signals_per_fault=50, rng_seed=42)

        set_seed(42)
        generator = SignalGenerator(config)
        dataset = generator.generate_dataset()

        speeds = [m['speed_rpm'] for m in dataset['metadata']]
        loads = [m['load_percent'] for m in dataset['metadata']]
        temps = [m['temperature_c'] for m in dataset['metadata']]

        # Check ranges
        self.assertGreater(max(speeds) - min(speeds), 10)  # Reasonable variation
        self.assertGreater(max(loads) - min(loads), 10)
        self.assertGreater(max(temps) - min(temps), 5)

        # Check within configured bounds
        self.assertGreaterEqual(min(loads), config.operating.load_percent_min)
        self.assertLessEqual(max(loads), config.operating.load_percent_max)
        self.assertGreaterEqual(min(temps), config.operating.temp_c_min)
        self.assertLessEqual(max(temps), config.operating.temp_c_max)


def run_tests(verbosity: int = 2):
    """
    Run all tests with specified verbosity.

    Args:
        verbosity: Test output verbosity (0=quiet, 1=normal, 2=verbose)

    Returns:
        TestResult object
    """
    # Setup logging
    setup_logging(console=True, file=False)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    # Run tests when executed directly
    result = run_tests(verbosity=2)

    # Exit with error code if tests failed
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
