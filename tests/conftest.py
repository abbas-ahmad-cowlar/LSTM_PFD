"""
Pytest Configuration and Shared Fixtures

Provides common fixtures and configuration for all tests.

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import h5py


@pytest.fixture(scope="session")
def device():
    """Get available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_signal():
    """Generate a sample vibration signal."""
    np.random.seed(42)
    # Simulate bearing vibration: base frequency + harmonics + noise
    t = np.linspace(0, 1, 1024)
    signal = (
        np.sin(2 * np.pi * 10 * t) +  # Base frequency
        0.5 * np.sin(2 * np.pi * 20 * t) +  # Harmonic
        0.1 * np.random.randn(len(t))  # Noise
    )
    return signal.astype(np.float32)


@pytest.fixture
def sample_batch_signals():
    """Generate batch of sample signals."""
    np.random.seed(42)
    batch_size = 32
    signal_length = 1024

    signals = []
    labels = []

    for i in range(batch_size):
        t = np.linspace(0, 1, signal_length)
        # Different fault patterns
        label = i % 11

        if label == 0:  # Normal
            signal = np.sin(2 * np.pi * 10 * t) + 0.05 * np.random.randn(len(t))
        elif label == 1:  # Ball fault
            signal = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))
        else:  # Other faults
            signal = np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(len(t))

        signals.append(signal.astype(np.float32))
        labels.append(label)

    return np.array(signals), np.array(labels)


@pytest.fixture
def sample_features():
    """Generate sample feature vectors."""
    np.random.seed(42)
    num_samples = 100
    num_features = 15
    num_classes = 11

    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)

    return X, y


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_h5_cache(temp_data_dir, sample_batch_signals):
    """Create mock HDF5 cache file."""
    signals, labels = sample_batch_signals
    cache_path = Path(temp_data_dir) / "test_cache.h5"

    with h5py.File(cache_path, 'w') as f:
        f.create_dataset('signals', data=signals)
        f.create_dataset('labels', data=labels)

        # Add metadata
        f.attrs['num_samples'] = len(signals)
        f.attrs['signal_length'] = signals.shape[1]
        f.attrs['num_classes'] = 11

    return str(cache_path)


@pytest.fixture
def simple_cnn_model():
    """Create simple CNN model for testing."""
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=11):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            # x: [B, 1, T]
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = torch.adaptive_avg_pool1d(x, 1)
            x = x.squeeze(-1)
            x = self.fc(x)
            return x

    return SimpleCNN()


@pytest.fixture
def trained_model_checkpoint(temp_checkpoint_dir, simple_cnn_model):
    """Create a trained model checkpoint for testing."""
    checkpoint_path = Path(temp_checkpoint_dir) / "test_model.pth"

    checkpoint = {
        'model_state_dict': simple_cnn_model.state_dict(),
        'epoch': 10,
        'accuracy': 0.95,
        'metadata': {
            'model_type': 'SimpleCNN',
            'num_classes': 11
        }
    }

    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for pipelines"
    )
    config.addinivalue_line(
        "markers", "benchmark: Performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (>1 second)"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU"
    )
