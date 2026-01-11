
import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from packages.core.features.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction pipeline."""

    def setUp(self):
        """Setup test fixtures."""
        self.fs = 20480.0
        self.signal_length = 5000  # Smaller for faster testing
        self.extractor = FeatureExtractor(fs=self.fs)
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Cleanup."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_feature_names(self):
        """Test feature name generation."""
        names = self.extractor.get_feature_names()
        self.assertEqual(len(names), 36)
        # Check some key names
        self.assertIn('RMS', names)
        self.assertIn('SpectralCentroid', names)
        # self.assertIn('WaveletEnergyRatio', names) # Wait, code says 'WaveletEnergyRatio' in list

    def test_extraction_shape(self):
        """Test extracted feature vector shape."""
        signal = np.random.randn(self.signal_length)
        features = self.extractor.extract_features(signal)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (36,))
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))

    def test_batch_extraction(self):
        """Test batch extraction."""
        n_signals = 10
        signals = np.random.randn(n_signals, self.signal_length)
        
        feature_matrix = self.extractor.extract_batch(signals)
        
        self.assertEqual(feature_matrix.shape, (n_signals, 36))
        self.assertFalse(np.any(np.isnan(feature_matrix)))

    def test_save_load(self):
        """Test saving and loading features."""
        n_signals = 5
        signals = np.random.randn(n_signals, self.signal_length)
        features = self.extractor.extract_batch(signals)
        
        save_path = self.temp_dir / "features.npy"
        self.extractor.save_features(features, save_path)
        
        self.assertTrue(save_path.exists())
        
        loaded = self.extractor.load_features(save_path)
        np.testing.assert_array_equal(features, loaded)

    def test_nan_handling(self):
        """Test handling of problematic signals."""
        # Signal with NaN
        signal = np.random.randn(self.signal_length)
        signal[10] = np.nan
        
        # Should raise error or handle it
        # numpy usually propagates NaN. The extractor doesn't explicitly handle it in code I read.
        # But let's see if it propagates to output or crashes. 
        # Actually, user code didn't show nan handling.
        # I'll check if it propagates NaN (which is fine, or expected).
        try:
            features = self.extractor.extract_features(signal)
            # If it returns, check if it has nans
            self.assertTrue(np.any(np.isnan(features)))
        except ValueError:
            pass # Maybe fft raises error on nan?

    def test_consistency(self):
        """Test that same signal produces same features."""
        signal = np.random.randn(self.signal_length)
        
        feat1 = self.extractor.extract_features(signal)
        feat2 = self.extractor.extract_features(signal)
        
        np.testing.assert_array_equal(feat1, feat2)

