# Testing Guide

> Patterns, conventions, and recipes for writing and running tests in the LSTM-PFD project.

## Overview

This guide covers practical patterns for extending the test suite. All examples are drawn from existing test code and verified against the codebase.

---

## 1. Writing New Tests

### Pattern A: Pytest Classes (Preferred)

Most tests use plain pytest classes with assertion statements. No `unittest.TestCase` inheritance needed.

```python
# tests/test_models.py style
import pytest
import torch
from utils.constants import NUM_CLASSES

class TestMyComponent:
    """Test MyComponent functionality."""

    def test_forward_pass(self):
        """Test forward pass output shape."""
        from packages.core.models import CNN1D

        model = CNN1D(num_classes=NUM_CLASSES, input_channels=1)
        x = torch.randn(4, 1, 5000)
        output = model(x)

        assert output.shape == (4, 11), f"Expected (4, 11), got {output.shape}"

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        from packages.core.models import CNN1D

        model = CNN1D(num_classes=NUM_CLASSES)
        x = torch.randn(2, 1, 5000)
        target = torch.tensor([0, 1])

        output = model(x)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
```

### Pattern B: unittest.TestCase (Classical Models)

Used for tests needing `setUp`/`tearDown` lifecycle. See `test_classical_models.py`.

```python
# tests/test_classical_models.py style
import unittest
import tempfile
from pathlib import Path

class TestMyClassicalModel(unittest.TestCase):
    """Test classical ML model."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        # Create test data...

    def tearDown(self):
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_train_predict(self):
        # ... train and predict ...
        self.assertEqual(len(preds), 20)
```

### Pattern C: Parametrized Tests

Used for testing multiple model architectures with the same assertions. See `test_all_models.py`.

```python
# tests/test_all_models.py style
import pytest
from packages.core.models.cnn.cnn_1d import CNN1D
from packages.core.models.resnet.resnet_1d import ResNet1D

@pytest.mark.parametrize("model_class, config", [
    (CNN1D, {}),
    (ResNet1D, {'base_filters': 16, 'n_blocks': [2, 2, 2, 2]}),
    # Add new models here...
])
def test_model_forward_backward(model_class, config):
    """Test instantiation, forward pass, and backward pass."""
    model = model_class(**config) if config else model_class()
    x = torch.randn(2, 1, 5000)

    y = model(x)
    assert y.shape[0] == 2
    assert y.shape[1] == 11

    loss = y.sum()
    loss.backward()
```

### Adding Markers

Use markers to categorize tests for selective execution:

```python
@pytest.mark.unit
class TestFeatureExtractor:
    """Runs with: pytest -m unit"""
    ...

@pytest.mark.integration
@pytest.mark.slow
class TestFullTrainingLoop:
    """Runs with: pytest -m integration; skipped with: pytest -m 'not slow'"""
    ...

@pytest.mark.gpu
def test_gpu_inference():
    """Skipped on CPU-only machines with: pytest -m 'not gpu'"""
    ...
```

---

## 2. Fixture Usage Guide

### Shared Fixtures from `conftest.py`

All fixtures below are automatically available to any test file under `tests/`.

#### `device` — PyTorch Device

```python
def test_model_on_device(self, device):
    """device is 'cuda' if available, else 'cpu'."""
    model = CNN1D().to(device)
    x = torch.randn(2, 1, 5000).to(device)
    output = model(x)
```

#### `sample_signal` — Single Vibration Signal

```python
def test_feature_extraction(self, sample_signal):
    """sample_signal: 1024-point float32 numpy array.
    Composed of base frequency (10 Hz) + harmonic (20 Hz) + noise."""
    extractor = FeatureExtractor(fs=20480.0)
    features = extractor.extract_features(sample_signal)
    assert features.shape == (36,)
```

#### `sample_batch_signals` — Batch of 32 Signals

```python
def test_batch_processing(self, sample_batch_signals):
    """Returns (signals, labels): signals=[32, 1024] float32, labels=[32] int."""
    signals, labels = sample_batch_signals
    assert signals.shape == (32, 1024)
    assert labels.shape == (32,)
```

#### `sample_features` — Feature Matrix

```python
def test_normalization(self, sample_features):
    """Returns (X, y): X=[100, 15] float32, y=[100] int."""
    X, y = sample_features
    normalizer = FeatureNormalizer(method='zscore')
    X_norm = normalizer.fit_transform(X)
```

#### `temp_checkpoint_dir` / `temp_data_dir` — Auto-Cleaned Temp Dirs

```python
def test_save_model(self, temp_checkpoint_dir):
    """Temporary directory, automatically removed after test."""
    path = Path(temp_checkpoint_dir) / "model.pth"
    torch.save(model.state_dict(), path)
    assert path.exists()
```

#### `mock_h5_cache` — Pre-Populated HDF5 File

```python
def test_hdf5_loading(self, mock_h5_cache):
    """HDF5 file with 'signals' and 'labels' datasets, plus metadata attrs."""
    import h5py
    with h5py.File(mock_h5_cache, 'r') as f:
        signals = f['signals'][:]
        labels = f['labels'][:]
```

#### `simple_cnn_model` — Lightweight Test Model

```python
def test_export(self, simple_cnn_model):
    """SimpleCNN from tests.models — picklable for ONNX export tests."""
    model = simple_cnn_model
    model.eval()
    x = torch.randn(1, 1, 1024)
    output = model(x)
    assert output.shape == (1, 11)
```

#### `trained_model_checkpoint` — Saved Checkpoint

```python
def test_load_checkpoint(self, trained_model_checkpoint):
    """Path to a .pth file with model_state_dict, epoch, accuracy, metadata."""
    checkpoint = torch.load(trained_model_checkpoint)
    assert 'model_state_dict' in checkpoint
    assert checkpoint['epoch'] == 10
```

---

## 3. Mocking and Patching Patterns

### Mock Models

Many test files define local `MockModel` classes for lightweight testing:

```python
# Pattern from test_deployment.py, test_evaluation_pipeline.py, test_xai.py
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 10, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### Environment Variable Mocking

Dashboard tests require environment variables set before imports:

```python
# Pattern from test_dashboard_sanity.py, test_phase1_verify.py
import os
os.environ['DATABASE_URL'] = "sqlite:///:memory:"
os.environ['SECRET_KEY'] = "test_secret_key_for_testing_purposes_1234567890"
os.environ['JWT_SECRET_KEY'] = "test_jwt_secret_key_for_testing_purposes_1234567890"
os.environ['JWT_ACCESS_TOKEN_EXPIRES'] = "3600"
os.environ['SKIP_CONFIG_VALIDATION'] = "True"
os.environ['ENV'] = "test"
os.environ['DEBUG'] = "True"

# Then import dashboard modules
import sys
sys.path.insert(0, str(DASHBOARD_DIR))
from packages.dashboard.app import app
```

### Import Guards

For optional dependencies, use try/except guards with `skipIf`:

```python
# Pattern from unit/test_api.py
try:
    from fastapi.testclient import TestClient
    from api.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPIEndpoints:
    ...
```

```python
# Pattern from unit/test_deployment.py — platform-specific skips
import platform
QUANTIZATION_AVAILABLE = platform.system() != 'Darwin'

@pytest.mark.skipif(not QUANTIZATION_AVAILABLE, reason="Quantization not available on macOS")
def test_dynamic_quantization(self, simple_cnn_model):
    ...
```

---

## 4. Integration Test Setup

Integration tests live in `tests/integration/` and test end-to-end workflows.

### Full Training Pipeline

`test_comprehensive.py` tests actual training convergence:

```python
@pytest.mark.integration
class TestFullTrainingLoop:
    def test_full_training_convergence(self, simple_model):
        # Creates synthetic data, trains for multiple epochs,
        # verifies loss decreases over time
        ...
```

### Pipeline Workflow

`test_pipelines.py` chains multiple components:

1. **Classical ML Pipeline:** Feature extraction → selection → normalization → sklearn classifier → predictions
2. **Deep Learning Pipeline:** Signal tensors → DataLoader → CNN training → inference
3. **Deployment Pipeline:** Model → quantization → inference engine comparison
4. **Ensemble Pipeline:** Multiple models → VotingEnsemble → joint inference
5. **Data Pipeline:** HDF5 loading → PyTorch DataLoader → batch iteration

### Creating Mock HDF5 Datasets

```python
# Pattern from test_comprehensive.py
@pytest.fixture
def mock_hdf5_dataset():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        path = f.name

    with h5py.File(path, 'w') as f:
        # Create train/val/test splits
        for split in ['train', 'val', 'test']:
            grp = f.create_group(split)
            n = 50 if split == 'train' else 20
            grp.create_dataset('signals', data=np.random.randn(n, 1024).astype(np.float32))
            grp.create_dataset('labels', data=np.random.randint(0, 11, n))

        f.attrs['signal_length'] = 1024
        f.attrs['num_classes'] = 11

    yield path
    os.unlink(path)
```

---

## 5. Stress and Load Testing

### Stress Tests (`stress_tests.py`)

Validates system stability under extreme conditions. Uses custom pytest options:

```bash
# Run with defaults (30s duration, 10000 samples)
pytest tests/stress_tests.py -v

# Custom configuration
pytest tests/stress_tests.py --stress-duration 120 --stress-samples 50000
```

**Test classes:**

| Class                      | Description                                                                  |
| -------------------------- | ---------------------------------------------------------------------------- |
| `TestLargeBatchProcessing` | Forward pass with large batches, extreme batch sizes, batch-size consistency |
| `TestMemoryLeakDetection`  | Inference memory stability (tracemalloc), training loop memory checks        |
| `TestGPUMemoryStress`      | GPU memory fragmentation, sustained GPU load (requires CUDA)                 |
| `TestConcurrentRequests`   | Concurrent inference with threading                                          |
| `TestNumericalStability`   | Extreme input values, gradient explosion detection                           |
| `TestModelRobustness`      | Different signal lengths, corrupted inputs, adversarial perturbations        |

### Load Tests (`load_tests.py`)

Simulates concurrent API users and monitors system resources. Can run against a live server or in mock mode.

```bash
# Run as pytest
pytest tests/load_tests.py -v

# Run standalone with custom parameters
python tests/load_tests.py
```

**Key components:**

| Component         | Description                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `ResourceMonitor` | Background thread capturing CPU%, memory%, GPU memory at configurable intervals  |
| `MockAPIScenario` | Simulates API requests with configurable latency and failure rate                |
| `HTTPAPIScenario` | Real HTTP requests against a running server                                      |
| `LoadTestSummary` | Collects p50/p95/p99 latencies, requests/second, error rate, peak resource usage |

---

## 6. Benchmarking

### Benchmark Suite (`benchmarks/benchmark_suite.py`)

Standalone script for performance benchmarking:

```bash
python tests/benchmarks/benchmark_suite.py --output results/benchmarks.json
python tests/benchmarks/benchmark_suite.py --model-path checkpoints/best_model.pth
python tests/benchmarks/benchmark_suite.py --api-url http://localhost:8000
```

**Available benchmarks:**

| Benchmark          | Method                           | What It Measures                                    |
| ------------------ | -------------------------------- | --------------------------------------------------- |
| Feature Extraction | `benchmark_feature_extraction()` | Throughput of `FeatureExtractor.extract_features()` |
| Model Inference    | `benchmark_model_inference()`    | Forward pass latency and throughput                 |
| Quantized Model    | `benchmark_quantized_model()`    | Quantized vs. original model comparison             |
| API Latency        | `benchmark_api_latency()`        | HTTP endpoint response times                        |
| Memory Usage       | `benchmark_memory_usage()`       | Model memory footprint                              |

---

## 7. CI/CD Integration

### Marker-Based Test Selection

Use markers to run different test subsets in CI:

```yaml
# Example CI configuration
steps:
  - name: Unit Tests (Fast)
    run: pytest tests/ -m "unit and not slow" -v --tb=short

  - name: Integration Tests
    run: pytest tests/ -m integration -v --tb=short

  - name: Full Suite (Excluding GPU)
    run: pytest tests/ -m "not gpu" -v --cov=packages/ --cov-report=xml
```

### Skipping Tests Appropriately

- **GPU tests:** Automatically skipped on CPU-only machines via `@pytest.mark.gpu`
- **Slow tests:** Skip with `pytest -m "not slow"`
- **Optional dependencies:** Tests skip gracefully with `pytest.importorskip("onnx")` or `@pytest.mark.skipif` guards
- **Platform-specific:** Quantization tests skip on macOS (`platform.system() != 'Darwin'`)

### Performance

> ⚠️ **Results pending.** Test suite metrics will be populated after experiments are run.

| Metric                     | Value       |
| -------------------------- | ----------- |
| Total test count           | `[PENDING]` |
| Unit test pass rate        | `[PENDING]` |
| Integration test pass rate | `[PENDING]` |
| Average suite runtime      | `[PENDING]` |
| Coverage percentage        | `[PENDING]` |
