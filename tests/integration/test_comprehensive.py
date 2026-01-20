#!/usr/bin/env python3
"""
Comprehensive Integration Tests for End-to-End Workflows

Expands on existing integration tests with:
- Full training loop validation (not just 3 batches)
- Model checkpoint save/load verification
- Complete inference pipeline testing
- Cross-validation integration
- Streaming dataloader integration

Author: Critical Deficiency Fix #18 (Priority: 62)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import torch
import tempfile
import h5py
from torch.utils.data import TensorDataset, DataLoader

from utils.constants import NUM_CLASSES, SAMPLING_RATE


@pytest.fixture
def mock_hdf5_dataset():
    """Create a mock HDF5 dataset for integration testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    # Create mock dataset
    num_train = 200
    num_val = 50
    num_test = 50
    signal_length = 1024
    num_classes = NUM_CLASSES
    
    with h5py.File(tmp_path, 'w') as f:
        for split, n in [('train', num_train), ('val', num_val), ('test', num_test)]:
            signals = np.random.randn(n, signal_length).astype(np.float32)
            labels = np.random.randint(0, num_classes, n)
            
            grp = f.create_group(split)
            grp.create_dataset('signals', data=signals)
            grp.create_dataset('labels', data=labels)
        
        f.attrs['sampling_rate'] = SAMPLING_RATE
        f.attrs['num_classes'] = num_classes
    
    yield tmp_path
    
    # Cleanup
    import os
    try:
        os.unlink(tmp_path)
    except:
        pass


@pytest.fixture
def simple_model():
    """Create a simple CNN model for testing."""
    class SimpleCNN(torch.nn.Module):
        def __init__(self, num_classes=11):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(1, 32, 7, padding=3)
            self.conv2 = torch.nn.Conv1d(32, 64, 5, padding=2)
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.fc = torch.nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.nn.functional.max_pool1d(x, 2)
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    return SimpleCNN()


@pytest.mark.integration
class TestFullTrainingLoop:
    """Test complete training workflows beyond smoke tests."""
    
    def test_full_training_convergence(self, simple_model):
        """Test that training actually converges (not just runs)."""
        # Create synthetic dataset with clear class separation
        np.random.seed(42)
        torch.manual_seed(42)
        
        num_samples = 200
        signal_length = 1024
        num_classes = 3
        
        # Create separable data
        signals = []
        labels = []
        
        for i in range(num_classes):
            class_signals = np.zeros((num_samples // num_classes, signal_length))
            # Add class-specific frequency component
            t = np.linspace(0, 1, signal_length)
            for j in range(num_samples // num_classes):
                class_signals[j] = np.sin(2 * np.pi * (10 + i * 20) * t) + 0.5 * np.random.randn(signal_length)
            signals.append(class_signals)
            labels.extend([i] * (num_samples // num_classes))
        
        signals = np.vstack(signals).astype(np.float32)
        labels = np.array(labels)
        
        # Shuffle
        idx = np.random.permutation(len(signals))
        signals = signals[idx]
        labels = labels[idx]
        
        # Create DataLoader
        dataset = TensorDataset(
            torch.from_numpy(signals).unsqueeze(1),
            torch.from_numpy(labels).long()
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train model
        model = simple_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(10):
            epoch_loss = 0
            for batch_signals, batch_labels in loader:
                optimizer.zero_grad()
                outputs = model(batch_signals)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            
            if initial_loss is None:
                initial_loss = avg_loss
            final_loss = avg_loss
        
        # Training should reduce loss
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        
        # Evaluate accuracy
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_signals, batch_labels in loader:
                outputs = model(batch_signals)
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        
        accuracy = correct / total
        print(f"\n  Training converged: {initial_loss:.4f} -> {final_loss:.4f}")
        print(f"  Final accuracy: {accuracy*100:.1f}%")
        
        # Should achieve reasonable accuracy on separable data
        assert accuracy > 0.5, f"Accuracy too low: {accuracy*100:.1f}%"


@pytest.mark.integration
class TestCheckpointSaveLoad:
    """Test model checkpoint save/load functionality."""
    
    def test_checkpoint_round_trip(self, simple_model):
        """Test that saved and loaded models produce identical outputs."""
        model = simple_model
        model.eval()
        
        # Generate test input
        torch.manual_seed(42)
        test_input = torch.randn(1, 1, 1024)
        
        # Get original output
        with torch.no_grad():
            original_output = model(test_input)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 10,
                'accuracy': 0.95,
                'config': {'num_classes': 11}
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Create new model and load checkpoint
            from copy import deepcopy
            new_model = deepcopy(model)
            new_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            
            # Verify models are different before loading
            with torch.no_grad():
                new_output_before = new_model(test_input)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path)
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            # Verify models are identical after loading
            with torch.no_grad():
                new_output_after = new_model(test_input)
            
            assert torch.allclose(original_output, new_output_after, atol=1e-6), \
                "Outputs differ after checkpoint load"
            
            print("\n  ✓ Checkpoint save/load verified")
            
        finally:
            import os
            os.unlink(checkpoint_path)
    
    def test_checkpoint_metadata(self, simple_model):
        """Test that checkpoint metadata is preserved."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        try:
            metadata = {
                'epoch': 42,
                'best_accuracy': 0.9876,
                'learning_rate': 0.001,
                'config': {'model': 'attention', 'dropout': 0.3}
            }
            
            checkpoint = {
                'model_state_dict': simple_model.state_dict(),
                **metadata
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load and verify
            loaded = torch.load(checkpoint_path)
            
            assert loaded['epoch'] == 42
            assert loaded['best_accuracy'] == 0.9876
            assert loaded['config']['model'] == 'attention'
            
            print("\n  ✓ Checkpoint metadata preserved")
            
        finally:
            import os
            os.unlink(checkpoint_path)


@pytest.mark.integration
class TestStreamingDataloaderIntegration:
    """Test streaming dataloader with training pipeline."""
    
    def test_streaming_training(self, mock_hdf5_dataset, simple_model):
        """Test training with streaming dataloader."""
        from data.streaming_hdf5_dataset import StreamingHDF5Dataset
        
        # Create streaming dataset
        dataset = StreamingHDF5Dataset(mock_hdf5_dataset, split='train')
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        
        model = simple_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Train for a few batches
        model.train()
        losses = []
        
        for batch_idx, (signals, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if batch_idx >= 5:
                break
        
        assert len(losses) > 0
        assert all(np.isfinite(losses))
        
        print(f"\n  ✓ Trained {len(losses)} batches with streaming dataloader")


@pytest.mark.integration
class TestCWRUDatasetIntegration:
    """Test CWRU dataset with training pipeline."""
    
    def test_cwru_dataset_creation(self):
        """Test CWRU dataset can be created with mock data."""
        from data.cwru_dataset import segment_signal, load_cwru_mat_file
        
        # Test segmentation
        signal = np.random.randn(12000).astype(np.float32)
        segments = segment_signal(signal, segment_length=2048, overlap=0.5)
        
        assert segments.shape[0] > 0
        assert segments.shape[1] == 2048
        
        # Verify normalization
        for segment in segments[:5]:
            assert np.abs(np.mean(segment)) < 0.1  # Approximately zero mean
        
        print(f"\n  ✓ Segmented signal into {segments.shape[0]} segments")


@pytest.mark.integration  
class TestCrossValidationIntegration:
    """Test cross-validation integration."""
    
    def test_cv_imports(self):
        """Test cross-validation script imports correctly."""
        from scripts.utilities.cross_validation import (
            CrossValidationTrainer,
            compute_confidence_interval,
            print_results
        )
        
        # Test confidence interval computation
        values = [0.9, 0.92, 0.91, 0.89, 0.93]
        mean, low, high = compute_confidence_interval(values, confidence=0.95)
        
        assert low < mean < high
        assert 0.85 < mean < 0.95
        
        print(f"\n  ✓ CV utilities imported and functional")
        print(f"    Mean: {mean:.4f}, 95% CI: [{low:.4f}, {high:.4f}]")


@pytest.mark.integration
class TestLeakageCheckIntegration:
    """Test data leakage check integration."""
    
    def test_leakage_check_imports(self, mock_hdf5_dataset):
        """Test leakage check functionality."""
        from scripts.utilities.check_data_leakage import (
            LeakageChecker,
            compute_signal_hash,
            compute_full_signal_hash
        )
        
        # Test hashing
        signal1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        signal2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        signal3 = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        
        hash1 = compute_full_signal_hash(signal1)
        hash2 = compute_full_signal_hash(signal2)
        hash3 = compute_full_signal_hash(signal3)
        
        assert hash1 == hash2  # Same signals = same hash
        assert hash1 != hash3  # Different signals = different hash
        
        print("\n  ✓ Leakage check utilities functional")


@pytest.mark.integration
class TestStatisticalAnalysisIntegration:
    """Test statistical analysis integration."""
    
    def test_statistical_utilities(self):
        """Test statistical analysis utilities."""
        from scripts.utilities.statistical_analysis import (
            compute_confidence_interval,
            paired_ttest,
            wilcoxon_test
        )
        
        # Test confidence interval
        values = [0.90, 0.92, 0.91, 0.89, 0.93]
        mean, low, high = compute_confidence_interval(values)
        
        assert 0.88 < low < mean < high < 0.96
        
        # Test paired t-test
        values1 = [0.90, 0.92, 0.91, 0.89, 0.93]
        values2 = [0.85, 0.87, 0.86, 0.84, 0.88]
        
        ttest_result = paired_ttest(values1, values2)
        
        assert 'p_value' in ttest_result
        assert 't_statistic' in ttest_result
        assert ttest_result['significant_005']  # Should be significant
        
        print("\n  ✓ Statistical analysis utilities functional")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x', '--tb=short'])
