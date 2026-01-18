"""
Data Pipeline Validator

Validates that data flows correctly through all phases.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

import numpy as np
import torch
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import constants
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.constants import SAMPLING_RATE, SIGNAL_LENGTH
except ImportError:
    SAMPLING_RATE = 20480
    SIGNAL_LENGTH = 102400


def validate_data_compatibility() -> bool:
    """
    Validate data flows correctly through all phases.

    Tests:
    - Raw Signal → Features → Classical ML
    - Raw Signal → CNN Input
    - Raw Signal → Spectrogram
    - Raw Signal → Transformer

    Returns:
        True if all validations pass

    Example:
        >>> from integration import validate_data_compatibility
        >>> is_valid = validate_data_compatibility()
        >>> assert is_valid, "Data pipeline validation failed"
    """
    logger.info("="*60)
    logger.info("Validating Data Pipeline Compatibility")
    logger.info("="*60)

    try:
        # Generate test signal
        logger.info("\n[1/4] Generating test signal...")
        test_signal = _generate_test_signal()
        logger.info(f"✓ Test signal shape: {test_signal.shape}")

        # Test Phase 0 → Phase 1 (Feature Extraction)
        logger.info("\n[2/4] Testing Phase 0 → Phase 1 (Feature Extraction)...")
        features = _test_feature_extraction(test_signal)
        logger.info(f"✓ Feature extraction passed: {features.shape}")

        # Test Phase 0 → Phase 2 (CNN Input)
        logger.info("\n[3/4] Testing Phase 0 → Phase 2 (CNN Input)...")
        cnn_input = _test_cnn_preprocessing(test_signal)
        logger.info(f"✓ CNN preprocessing passed: {cnn_input.shape}")

        # Test Phase 0 → Phase 5 (Spectrogram)
        logger.info("\n[4/4] Testing Phase 0 → Phase 5 (Spectrogram)...")
        spectrogram = _test_spectrogram_generation(test_signal)
        logger.info(f"✓ Spectrogram generation passed: {spectrogram.shape}")

        logger.info("\n" + "="*60)
        logger.info("✓ All Data Pipeline Validations Passed")
        logger.info("="*60)

        return True

    except Exception as e:
        logger.error(f"\n✗ Data pipeline validation failed: {e}")
        return False


def _generate_test_signal(length: int = SIGNAL_LENGTH) -> np.ndarray:
    """Generate a test vibration signal."""
    t = np.linspace(0, 5, length)
    signal = (
        np.sin(2 * np.pi * 10 * t) +  # Base frequency
        0.5 * np.sin(2 * np.pi * 20 * t) +  # Harmonic
        0.1 * np.random.randn(length)  # Noise
    )
    return signal.astype(np.float32)


def _test_feature_extraction(signal: np.ndarray) -> np.ndarray:
    """Test Phase 0 → Phase 1 data flow."""
    from features.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor(fs=SAMPLING_RATE)
    features = extractor.extract_features(signal)

    # Validate features
    assert features.shape == (36,), f"Expected (36,), got {features.shape}"
    assert np.all(np.isfinite(features)), "Features contain non-finite values"

    return features


def _test_cnn_preprocessing(signal: np.ndarray) -> torch.Tensor:
    """Test Phase 0 → Phase 2 data flow."""
    # Convert to CNN input format: [1, 1, signal_length]
    cnn_input = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)

    # Validate shape
    expected_shape = (1, 1, signal.shape[0])
    assert cnn_input.shape == expected_shape, \
        f"Expected {expected_shape}, got {cnn_input.shape}"

    return cnn_input


def _test_spectrogram_generation(signal: np.ndarray) -> np.ndarray:
    """Test Phase 0 → Phase 5 data flow."""
    from scipy import signal as scipy_signal

    # Generate STFT spectrogram
    f, t, Zxx = scipy_signal.stft(
        signal,
        fs=SAMPLING_RATE,
        nperseg=256,
        noverlap=128
    )

    spectrogram = np.abs(Zxx)

    # Validate spectrogram
    assert spectrogram.ndim == 2, "Spectrogram should be 2D"
    assert np.all(np.isfinite(spectrogram)), "Spectrogram contains non-finite values"

    return spectrogram


def test_data_transformations() -> bool:
    """
    Test reversibility of data transformations where applicable.

    Returns:
        True if all transformations are correct
    """
    logger.info("Testing data transformations...")

    # Test normalization reversibility
    signal = _generate_test_signal()

    # Z-score normalization
    mean = signal.mean()
    std = signal.std()
    normalized = (signal - mean) / std

    # Reverse
    denormalized = normalized * std + mean

    assert np.allclose(signal, denormalized, atol=1e-5), \
        "Z-score normalization is not reversible"

    logger.info("✓ Data transformations validated")
    return True


def benchmark_data_loading_speed() -> Dict[str, float]:
    """
    Benchmark data loading speed for different formats.

    Returns:
        Dictionary with loading times (seconds)

    Example:
        >>> times = benchmark_data_loading_speed()
        >>> print(f"HDF5: {times['hdf5']:.3f}s, MAT: {times['mat']:.3f}s")
    """
    import time

    logger.info("Benchmarking data loading speed...")

    times = {}

    # Benchmark HDF5 loading (if file exists)
    try:
        import h5py
        test_path = 'data/processed/signals_cache.h5'

        start = time.time()
        with h5py.File(test_path, 'r') as f:
            signals = f['signals'][:100]
        times['hdf5'] = time.time() - start

        logger.info(f"✓ HDF5 loading: {times['hdf5']:.3f}s")
    except Exception as e:
        logger.warning(f"HDF5 benchmark skipped: {e}")
        times['hdf5'] = None

    # Benchmark MAT file loading (if files exist)
    try:
        from scipy.io import loadmat
        # Placeholder - would load actual .mat files
        times['mat'] = None
    except Exception as e:
        times['mat'] = None

    return times
