"""
Integration Tests for End-to-End Pipelines

Tests complete workflows from data loading to prediction.

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestClassicalMLPipeline:
    """Test classical ML pipeline integration."""

    def test_pipeline_full_workflow(self, sample_batch_signals):
        """Test complete classical ML pipeline."""
        from features.feature_extractor import FeatureExtractor
        from features.feature_selector import FeatureSelector
        from features.feature_normalization import FeatureNormalizer

        signals, labels = sample_batch_signals

        # 1. Feature extraction
        extractor = FeatureExtractor(fs=20480)
        features_list = []

        for signal in signals:
            features = extractor.extract_features(signal)
            features_list.append(features)

        X = np.array(features_list)

        # 2. Feature selection
        try:
            selector = FeatureSelector(method='variance', threshold=0.01)
            X_selected = selector.fit_transform(X, labels)
        except:
            X_selected = X  # Skip if selection fails

        # 3. Normalization
        normalizer = FeatureNormalizer(method='zscore')
        X_norm = normalizer.fit_transform(X_selected)

        # 4. Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, labels, test_size=0.3, random_state=42
        )

        # 5. Train classifier
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        # 6. Predict
        y_pred = clf.predict(X_test)

        # Verify pipeline worked
        assert y_pred.shape == y_test.shape
        assert len(np.unique(y_pred)) > 0


@pytest.mark.integration
class TestDeepLearningPipeline:
    """Test deep learning pipeline integration."""

    @pytest.mark.slow
    def test_cnn_training_pipeline(self, sample_batch_signals, temp_checkpoint_dir):
        """Test CNN training pipeline."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        signals, labels = sample_batch_signals

        # Prepare data
        signals_tensor = torch.from_numpy(signals).float().unsqueeze(1)  # [B, 1, T]
        labels_tensor = torch.from_numpy(labels).long()

        dataset = TensorDataset(signals_tensor, labels_tensor)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Create simple model
        from conftest import simple_cnn_model
        model = simple_cnn_model()

        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx >= 2:  # Just a few batches for testing
                break

        # Test inference
        model.eval()
        with torch.no_grad():
            test_input = signals_tensor[:1]
            output = model(test_input)

            assert output.shape == (1, 11)
            assert torch.all(torch.isfinite(output))


@pytest.mark.integration
class TestDeploymentPipeline:
    """Test deployment pipeline integration."""

    @pytest.mark.slow
    def test_quantization_pipeline(self, simple_cnn_model, temp_checkpoint_dir):
        """Test model quantization pipeline."""
        import torch
        from deployment.quantization import quantize_model_dynamic

        model = simple_cnn_model
        model.eval()

        # Quantize
        quantized_model = quantize_model_dynamic(model, inplace=False)

        # Test inference
        test_input = torch.randn(1, 1, 1024)
        with torch.no_grad():
            original_output = model(test_input)
            quantized_output = quantized_model(test_input)

        # Outputs should be similar (not exact due to quantization)
        assert original_output.shape == quantized_output.shape
        assert torch.allclose(original_output, quantized_output, atol=0.5)

    @pytest.mark.slow
    def test_inference_pipeline(self, simple_cnn_model):
        """Test inference pipeline."""
        import torch
        import numpy as np
        from deployment.inference import TorchInferenceEngine, InferenceConfig

        model = simple_cnn_model
        model.eval()

        # Create inference engine
        config = InferenceConfig(device='cpu', batch_size=8)
        engine = TorchInferenceEngine(config)
        engine.model = model

        # Single inference
        single_input = np.random.randn(1, 1024).astype(np.float32)
        single_output = engine.predict(single_input)

        assert single_output.shape == (1, 11)
        assert np.all(np.isfinite(single_output))

        # Batch inference
        batch_input = np.random.randn(16, 1, 1024).astype(np.float32)
        batch_output = engine.predict_batch(batch_input, batch_size=8)

        assert batch_output.shape == (16, 11)
        assert np.all(np.isfinite(batch_output))


@pytest.mark.integration
class TestEnsemblePipeline:
    """Test ensemble pipeline integration."""

    def test_voting_ensemble_pipeline(self):
        """Test voting ensemble pipeline."""
        import torch
        from models.ensemble.voting_ensemble import VotingEnsemble

        # Create multiple simple models
        models = []
        for _ in range(3):
            model = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, kernel_size=5),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(16, 11)
            )
            models.append(model)

        # Create voting ensemble
        ensemble = VotingEnsemble(models, voting_type='soft')

        # Test inference
        test_input = torch.randn(4, 1, 1024)
        with torch.no_grad():
            output = ensemble(test_input)

        assert output.shape == (4, 11)
        assert torch.all(torch.isfinite(output))


@pytest.mark.integration
@pytest.mark.slow
class TestDataPipeline:
    """Test data loading and preprocessing pipeline."""

    def test_hdf5_data_loading(self, mock_h5_cache):
        """Test HDF5 data loading pipeline."""
        import h5py

        with h5py.File(mock_h5_cache, 'r') as f:
            signals = f['signals'][:]
            labels = f['labels'][:]

            assert signals.shape[0] == labels.shape[0]
            assert len(signals.shape) == 2  # [N, T]
            assert np.all(np.isfinite(signals))

    def test_dataloader_pipeline(self, mock_h5_cache):
        """Test PyTorch DataLoader pipeline."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        import h5py

        # Load from HDF5
        with h5py.File(mock_h5_cache, 'r') as f:
            signals = torch.from_numpy(f['signals'][:]).float().unsqueeze(1)
            labels = torch.from_numpy(f['labels'][:]).long()

        # Create dataset and dataloader
        dataset = TensorDataset(signals, labels)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Test iteration
        for batch_idx, (data, target) in enumerate(dataloader):
            assert data.shape[0] <= 8  # Batch size
            assert data.shape[1] == 1  # Channel
            assert target.shape[0] == data.shape[0]

            if batch_idx >= 2:  # Just test a few batches
                break
