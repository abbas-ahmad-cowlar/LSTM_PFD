# IDB 4.3: Testing Best Practices

> **Domain:** Infrastructure  
> **Extracted From:** `tests/` directory  
> **Date:** 2026-01-23

---

## 1. Test Organization Patterns

### 1.1 Directory Structure Convention

```
tests/
├── conftest.py          # Shared fixtures ONLY
├── unit/                # Isolated unit tests
│   └── test_<module>.py
├── integration/         # Cross-module workflow tests
│   └── test_<pipeline>.py
├── benchmarks/          # Performance benchmarks
│   └── benchmark_suite.py
└── models/              # Helper models for testing
    └── simple_cnn.py    # Reusable test fixtures
```

**Best Practice:** Separate concerns by test type, not by file being tested.

### 1.2 Test File Naming

```python
# Pattern: test_<subject>.py
test_models.py           # Tests for model architectures
test_feature_engineering.py  # Tests for feature pipeline
test_deployment.py       # Tests for deployment utilities

# Avoid: test_<class>.py per class (too granular)
```

### 1.3 Test Class Organization

```python
# Group related tests in classes with clear docstrings
class TestFeatureExtractor:
    """Test suite for FeatureExtractor."""

    def test_initialization(self):
        """Test default initialization."""

    def test_extract_time_domain_features(self, sample_signal):
        """Test time domain feature extraction."""
```

---

## 2. Fixture Conventions

### 2.1 Scope Selection Guide

| Scope      | Use Case                       | Example                                 |
| ---------- | ------------------------------ | --------------------------------------- |
| `session`  | Expensive, immutable resources | `device`, `database_connection`         |
| `module`   | Shared across test file        | `sample_batch_signals`, `trained_model` |
| `function` | Stateful, needs fresh state    | `temp_checkpoint_dir`                   |

### 2.2 Fixture With Cleanup Pattern

```python
@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir                    # Provide to test
    shutil.rmtree(temp_dir)           # Cleanup after test
```

### 2.3 Device-Agnostic Testing

```python
@pytest.fixture(scope="session")
def device():
    """Get available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Usage in tests
def test_model_forward(self, simple_model, device):
    model = simple_model.to(device)
    x = torch.randn(4, 1, 1024).to(device)
    output = model(x)
```

### 2.4 Constants Over Magic Numbers

```python
# ✅ Good: Import constants
from utils.constants import NUM_CLASSES, SAMPLING_RATE

@pytest.fixture
def sample_features():
    X = np.random.randn(100, 15).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, 100)  # Use constant
    return X, y

# ❌ Bad: Hardcoded magic numbers
y = np.random.randint(0, 11, 100)  # What is 11?
```

### 2.5 Composite Fixtures

```python
@pytest.fixture
def trained_model_checkpoint(temp_checkpoint_dir, simple_cnn_model):
    """Create a trained model checkpoint for testing."""
    checkpoint_path = Path(temp_checkpoint_dir) / "test_model.pth"

    checkpoint = {
        'model_state_dict': simple_cnn_model.state_dict(),
        'epoch': 10,
        'accuracy': 0.95,
        'metadata': {'model_type': 'SimpleCNN', 'num_classes': NUM_CLASSES}
    }

    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)
```

---

## 3. Mock Patterns

### 3.1 Synthetic Data Generation

```python
@pytest.fixture
def sample_signal():
    """Generate a sample vibration signal."""
    np.random.seed(42)  # Reproducibility
    t = np.linspace(0, 1, 1024)
    signal = (
        np.sin(2 * np.pi * 10 * t) +      # Base frequency
        0.5 * np.sin(2 * np.pi * 20 * t) + # Harmonic
        0.1 * np.random.randn(len(t))      # Noise
    )
    return signal.astype(np.float32)
```

### 3.2 Mock HDF5 Cache

```python
@pytest.fixture
def mock_h5_cache(temp_data_dir, sample_batch_signals):
    """Create mock HDF5 cache file."""
    signals, labels = sample_batch_signals
    cache_path = Path(temp_data_dir) / "test_cache.h5"

    with h5py.File(cache_path, 'w') as f:
        f.create_dataset('signals', data=signals)
        f.create_dataset('labels', data=labels)
        f.attrs['num_samples'] = len(signals)
        f.attrs['signal_length'] = signals.shape[1]
        f.attrs['num_classes'] = NUM_CLASSES

    return str(cache_path)
```

### 3.3 Reusable Test Models

```python
# tests/models/simple_cnn.py
class SimpleCNN(nn.Module):
    """Minimal CNN for testing (avoids import complexity)."""

    def __init__(self, num_classes=11):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
```

### 3.4 Optional Dependency Skip

```python
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPIEndpoints:
    """Test API endpoints."""
```

---

## 4. Assertion Patterns

### 4.1 Shape Assertions for Tensors

```python
def test_forward_pass(self):
    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1)
    x = torch.randn(4, 1, 5000)
    output = model(x)

    # Explicit expected shape
    assert output.shape == (4, 11), f"Expected (4, 11), got {output.shape}"
```

### 4.2 Numeric Assertions

```python
# Check finite values (no NaN/Inf)
assert torch.all(torch.isfinite(output))
assert np.all(np.isfinite(features))

# Approximate equality for floats
assert np.allclose(X_norm.mean(axis=0), 0, atol=0.1)
assert np.allclose(X_norm.std(axis=0), 1, atol=0.1)

# Range assertions
assert 0 <= accuracy <= 1.0
assert stats['compression_ratio'] >= 1.0
```

### 4.3 Gradient Flow Verification

```python
def test_gradient_flow(self):
    """Test gradients flow through all parameters."""
    model = CNN1D(num_classes=NUM_CLASSES)
    x = torch.randn(2, 1, 5000)
    target = torch.tensor([0, 1])

    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()

    # Verify all parameters received gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
```

### 4.4 Exception Testing

```python
def test_extract_features_empty_signal(self):
    """Test extraction with empty signal."""
    extractor = FeatureExtractor(fs=SAMPLING_RATE)

    with pytest.raises((ValueError, IndexError)):
        extractor.extract_features(np.array([]))

def test_transform_without_fit(self, sample_features):
    """Test transform without fit raises error."""
    X, _ = sample_features
    normalizer = FeatureNormalizer(method='zscore')

    with pytest.raises((RuntimeError, AttributeError, ValueError)):
        normalizer.transform(X)
```

### 4.5 Probability Sum Assertions

```python
def test_predict_proba(self):
    """Test probability outputs sum to 1."""
    proba = model.predict_proba(X_test)

    np.testing.assert_array_almost_equal(
        np.sum(proba, axis=1),
        np.ones(len(X_test))
    )
```

---

## 5. Test Naming Conventions

### 5.1 Method Naming Pattern

```python
# Pattern: test_<action>_<condition>
def test_forward_pass(self):                    # Basic functionality
def test_forward_pass_with_physics(self):       # Variant
def test_batch_size_consistency(self):          # Specific property
def test_extract_features_empty_signal(self):   # Edge case
def test_checkpoint_round_trip(self):           # Integration
```

### 5.2 Descriptive Docstrings

```python
def test_desalignement_harmonics(self):
    """Test desalignement produces 2X and 3X harmonics."""

def test_desequilibre_speed_dependence(self):
    """Test desequilibre amplitude scales with speed squared."""
```

---

## 6. Marker Usage

### 6.1 Standard Markers

```python
@pytest.mark.unit
class TestFeatureExtractor:
    """Unit tests - fast, isolated."""

@pytest.mark.integration
class TestDeepLearningPipeline:
    """Integration tests - cross-module workflows."""

@pytest.mark.slow
def test_full_training_convergence(self):
    """Tests that take >1 second."""

@pytest.mark.gpu
def test_gpu_memory_fragmentation(self):
    """Tests requiring GPU."""
```

### 6.2 Conditional Skipping

```python
# Skip on specific platform
@pytest.mark.skipif(platform.system() == 'Darwin', reason="Not supported on macOS")
def test_dynamic_quantization(self):

# Skip if dependency missing
@pytest.mark.slow
def test_onnx_export_basic(self, simple_cnn_model):
    pytest.importorskip("onnx")
```

---

## 7. Stress & Load Testing Patterns

### 7.1 Memory Leak Detection

```python
def test_inference_memory_stability(self, simple_model, device, stress_duration):
    """Check memory doesn't grow during repeated inference."""
    tracemalloc.start()

    for _ in range(100):
        x = torch.randn(32, 1, 1024).to(device)
        with torch.no_grad():
            _ = simple_model(x)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    growth_percent = ((peak - current) / current) * 100
    assert growth_percent < 10.0, f"Memory grew by {growth_percent:.1f}%"
```

### 7.2 Batch Size Consistency

```python
def test_batch_size_consistency(self, simple_model, device):
    """Verify model produces consistent outputs regardless of batch size."""
    model = simple_model.to(device).eval()
    x = torch.randn(16, 1, 1024).to(device)

    with torch.no_grad():
        full_batch = model(x)
        split_results = torch.cat([model(x[i:i+4]) for i in range(0, 16, 4)])

    assert torch.allclose(full_batch, split_results, atol=1e-5)
```

---

## 8. Integration Test Patterns

### 8.1 Pipeline Workflow Tests

```python
@pytest.mark.integration
def test_pipeline_full_workflow(self, sample_batch_signals):
    """Test complete classical ML pipeline."""
    signals, labels = sample_batch_signals

    # 1. Feature extraction
    extractor = FeatureExtractor(fs=SAMPLING_RATE)
    features = np.array([extractor.extract_features(s) for s in signals])

    # 2. Normalization
    normalizer = FeatureNormalizer(method='zscore')
    X_norm = normalizer.fit_transform(features)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, labels, test_size=0.3, random_state=42
    )

    # 4. Train & predict
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Verify pipeline worked
    assert y_pred.shape == y_test.shape
    assert len(np.unique(y_pred)) > 0
```

### 8.2 Checkpoint Round-Trip

```python
def test_checkpoint_round_trip(self, simple_model):
    """Test saved and loaded models produce identical outputs."""
    model = simple_model
    test_input = torch.randn(4, 1, 1024)

    with torch.no_grad():
        original_output = model(test_input)

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(model.state_dict(), f.name)

        loaded_model = type(model)()
        loaded_model.load_state_dict(torch.load(f.name))
        loaded_model.eval()

        with torch.no_grad():
            loaded_output = loaded_model(test_input)

    assert torch.allclose(original_output, loaded_output, atol=1e-6)
```

---

## Quick Reference Card

| Pattern             | Example                                   |
| ------------------- | ----------------------------------------- |
| **Fixture cleanup** | `yield value; cleanup()`                  |
| **Device agnostic** | `@pytest.fixture(scope="session")`        |
| **Shape assertion** | `assert output.shape == (batch, classes)` |
| **Finite check**    | `assert torch.all(torch.isfinite(x))`     |
| **Skip condition**  | `@pytest.mark.skipif(condition)`          |
| **Import skip**     | `pytest.importorskip("onnx")`             |
| **Exception test**  | `with pytest.raises(ValueError):`         |
| **Approx equal**    | `np.allclose(a, b, atol=0.1)`             |
| **Gradient check**  | `assert param.grad is not None`           |
| **Reproducibility** | `np.random.seed(42)`                      |

---

_Extracted from IDB 4.3 Testing Sub-Block Analysis_
