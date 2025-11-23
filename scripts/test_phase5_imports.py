"""
Phase 5 Import Validation Test

Quick test to verify all Phase 5 modules can be imported correctly.
Run this after fixing bugs to ensure everything is properly connected.

Usage:
    python scripts/test_phase5_imports.py

Author: AI Assistant
Date: 2025-11-23
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all Phase 5 imports."""
    print("=" * 70)
    print("Phase 5 Import Validation Test")
    print("=" * 70)
    print()

    tests_passed = 0
    tests_failed = 0
    errors = []

    # Test 1: Data generators
    print("1. Testing data generators...")
    try:
        from data.spectrogram_generator import SpectrogramGenerator
        from data.wavelet_transform import WaveletTransform
        from data.wigner_ville import WignerVilleDistribution, generate_wvd
        print("   ✓ Data generators imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ FAILED: {e}")
        errors.append(f"Data generators: {e}")
        tests_failed += 1

    # Test 2: Dataset classes
    print("2. Testing dataset classes...")
    try:
        from data.tfr_dataset import (
            SpectrogramDataset,
            OnTheFlyTFRDataset,
            MultiTFRDataset,
            create_tfr_dataloaders
        )
        print("   ✓ Dataset classes imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ FAILED: {e}")
        errors.append(f"Dataset classes: {e}")
        tests_failed += 1

    # Test 3: Model architectures
    print("3. Testing model architectures...")
    try:
        from models.spectrogram_cnn import (
            resnet18_2d,
            resnet34_2d,
            resnet50_2d,
            efficientnet_b0,
            efficientnet_b1,
            efficientnet_b3,
            get_model,
            list_models
        )
        print("   ✓ Model architectures imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ FAILED: {e}")
        errors.append(f"Model architectures: {e}")
        tests_failed += 1

    # Test 4: Dual-stream CNN
    print("4. Testing dual-stream CNN...")
    try:
        from models.spectrogram_cnn.dual_stream_cnn import DualStreamCNN
        print("   ✓ Dual-stream CNN imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ FAILED: {e}")
        errors.append(f"Dual-stream CNN: {e}")
        tests_failed += 1

    # Test 5: Training components
    print("5. Testing training components...")
    try:
        from training.spectrogram_trainer import SpectrogramTrainer
        print("   ✓ Training components imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ FAILED: {e}")
        errors.append(f"Training components: {e}")
        tests_failed += 1

    # Test 6: Evaluation components
    print("6. Testing evaluation components...")
    try:
        from evaluation.spectrogram_evaluator import SpectrogramEvaluator
        from evaluation.time_vs_frequency_comparison import TimeVsFrequencyComparator
        print("   ✓ Evaluation components imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ FAILED: {e}")
        errors.append(f"Evaluation components: {e}")
        tests_failed += 1

    # Test 7: Visualization components
    print("7. Testing visualization components...")
    try:
        from visualization.spectrogram_plots import (
            plot_spectrogram,
            plot_spectrogram_comparison
        )
        from visualization.activation_maps_2d import (
            visualize_filters,
            visualize_feature_maps
        )
        print("   ✓ Visualization components imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ FAILED: {e}")
        errors.append(f"Visualization components: {e}")
        tests_failed += 1

    # Test 8: Scripts
    print("8. Testing script imports...")
    try:
        # These will import but not run
        import scripts.precompute_spectrograms
        import scripts.train_spectrogram_cnn
        print("   ✓ Scripts imported successfully")
        tests_passed += 1
    except (ImportError, AttributeError) as e:
        print(f"   ✗ FAILED: {e}")
        errors.append(f"Scripts: {e}")
        tests_failed += 1

    # Summary
    print()
    print("=" * 70)
    print(f"Test Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)

    if tests_failed > 0:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        print("\nPhase 5 has import errors. Please fix before using.")
        return False
    else:
        print("\n✅ ALL TESTS PASSED!")
        print("Phase 5 is ready to use.")
        return True


def test_basic_functionality():
    """Test basic functionality of key classes."""
    print("\n" + "=" * 70)
    print("Phase 5 Basic Functionality Test")
    print("=" * 70)
    print()

    try:
        import numpy as np
        from data.spectrogram_generator import SpectrogramGenerator
        from data.wavelet_transform import WaveletTransform
        from data.wigner_ville import WignerVilleDistribution

        # Generate a test signal
        print("1. Generating test signal...")
        fs = 20480
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration))
        signal = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
        print(f"   ✓ Test signal created: {len(signal)} samples")

        # Test STFT
        print("2. Testing STFT generation...")
        stft_gen = SpectrogramGenerator(fs=fs, nperseg=256, noverlap=128)
        spec, freqs, times = stft_gen.generate_stft_spectrogram(signal)
        print(f"   ✓ STFT spectrogram shape: {spec.shape}")

        # Test CWT
        print("3. Testing CWT generation...")
        cwt_gen = WaveletTransform(wavelet='morl', scales=128, fs=fs)
        scalogram, frequencies = cwt_gen.generate_cwt_scalogram(signal)
        print(f"   ✓ CWT scalogram shape: {scalogram.shape}")

        # Test WVD
        print("4. Testing WVD generation...")
        wvd_gen = WignerVilleDistribution(fs=fs)
        wvd, freqs_wvd, times_wvd = wvd_gen.generate_wvd(signal)
        print(f"   ✓ WVD shape: {wvd.shape}")

        print("\n✅ ALL FUNCTIONALITY TESTS PASSED!")
        print("Phase 5 core functions are working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ FUNCTIONALITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting Phase 5 validation...\n")

    # Run import tests
    import_success = test_imports()

    # Only run functionality tests if imports succeeded
    if import_success:
        try:
            func_success = test_basic_functionality()
        except Exception as e:
            print(f"\nCould not run functionality tests: {e}")
            func_success = False
    else:
        func_success = False

    # Exit with appropriate code
    if import_success and func_success:
        print("\n" + "=" * 70)
        print("🎉 Phase 5 is fully validated and ready to use!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("⚠️  Phase 5 validation failed. Please fix errors above.")
        print("=" * 70)
        sys.exit(1)
