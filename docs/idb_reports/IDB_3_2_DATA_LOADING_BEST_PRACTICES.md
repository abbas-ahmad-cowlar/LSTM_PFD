# IDB 3.2: Data Loading Best Practices

**Source:** IDB 3.2 Analysis  
**Date:** 2026-01-23

---

## 1. Dataset Class Patterns

### 1.1 Factory Methods for Multiple Input Sources

**Location:** [dataset.py:142-290](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/dataset.py#L142-L290)

Use `@classmethod` factory methods to support multiple data sources:

```python
class BearingFaultDataset(Dataset):
    @classmethod
    def from_hdf5(cls, path: Path, split: str = 'train', transform=None):
        """Load from HDF5 with detailed docstring."""
        ...

    @classmethod
    def from_mat_file(cls, path: Path, transform=None):
        """Load from MATLAB .mat file."""
        ...

    @classmethod
    def from_generator_output(cls, output: Dict, transform=None):
        """Create from generator output dict."""
        ...
```

**Benefits:**

- Clear API for each data source
- Centralized path/split validation
- Consistent transform application

---

### 1.2 Consistent Label Mapping

**Location:** [dataset.py:67-77](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/dataset.py#L67-L77)

Always maintain bidirectional label mappings:

```python
# Create label mapping from data
if label_to_idx is None:
    unique_labels = sorted(set(labels))
    self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
else:
    self.label_to_idx = label_to_idx

# Reverse mapping for inference
self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
```

---

### 1.3 Document HDF5 File Structure

**Location:** [dataset.py:207-266](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/dataset.py#L207-L266)

Include complete HDF5 schema in docstrings:

```python
"""
HDF5 File Structure:
    f['train']/
        'signals'  - (N, signal_length) float32
        'labels'   - (N,) int64
    f['val']/ ...
    f['test']/ ...

File Attributes:
    f.attrs['num_classes']
    f.attrs['fault_types']
    f.attrs['signal_length']
"""
```

---

## 2. Transform Composition Patterns

### 2.1 Probabilistic Augmentations

**Location:** [cnn_transforms.py:157-194](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cnn_transforms.py#L157-L194)

Include probability parameter for optional augmentations:

```python
class RandomAmplitudeScale:
    def __init__(self, scale_range=(0.8, 1.2), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p  # Probability of application

    def __call__(self, signal):
        if np.random.rand() > self.p:
            return signal  # Return unchanged
        scale = np.random.uniform(*self.scale_range)
        return signal * scale
```

---

### 2.2 Pre-built Transform Pipelines

**Location:** [cnn_transforms.py:277-321](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cnn_transforms.py#L277-L321)

Provide factory functions for common pipelines:

```python
def get_train_transforms(augment: bool = True) -> Compose:
    """Standard training pipeline with toggleable augmentation."""
    if augment:
        return Compose([
            Normalize1D(),
            RandomAmplitudeScale(scale_range=(0.9, 1.1), p=0.5),
            AddGaussianNoise(noise_level=0.03, p=0.3),
            ToTensor1D()
        ])
    return Compose([Normalize1D(), ToTensor1D()])

def get_test_transforms() -> Compose:
    """Test pipeline: normalize only, no augmentation."""
    return Compose([Normalize1D(), ToTensor1D()])
```

---

### 2.3 Handle Both NumPy and Tensor Inputs

**Location:** [cnn_transforms.py:79-96](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cnn_transforms.py#L79-L96)

Support multiple input types in transforms:

```python
def __call__(self, signal: Union[np.ndarray, torch.Tensor]):
    if isinstance(signal, torch.Tensor):
        mean, std = signal.mean(), signal.std()
        return (signal - mean) / (std + self.eps)
    else:
        mean, std = signal.mean(), signal.std()
        return (signal - mean) / (std + self.eps)
```

---

## 3. Memory Management Patterns

### 3.1 Thread-Local File Handles

**Location:** [streaming_hdf5_dataset.py:90-94](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py#L90-L94)

Use `threading.local()` for multi-worker HDF5 access:

```python
class StreamingHDF5Dataset(Dataset):
    def __init__(self, path, ...):
        self._local = threading.local()
        # Read metadata once, don't keep file open
        with h5py.File(path, 'r') as f:
            self.num_samples = f[split]['signals'].shape[0]
            self.labels = f[split]['labels'][:]  # Small, keep in RAM

    def _get_file_handle(self):
        """Lazy-open per worker."""
        if not hasattr(self._local, 'file') or self._local.file is None:
            self._local.file = h5py.File(self.hdf5_path, 'r', swmr=True)
        return self._local.file
```

---

### 3.2 SWMR Mode for Concurrent Reads

**Location:** [streaming_hdf5_dataset.py:93](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py#L93)

Enable Single-Writer-Multiple-Reader mode:

```python
self._local.file = h5py.File(self.hdf5_path, 'r', swmr=True)
```

**Note:** Requires HDF5 file was created with `libver='latest'`.

---

### 3.3 Chunked Prefetching for Sequential Access

**Location:** [streaming_hdf5_dataset.py:181-207](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py#L181-L207)

Reduce I/O overhead by loading chunks:

```python
def __getitem__(self, idx):
    chunk_start = (idx // self.chunk_size) * self.chunk_size
    chunk_end = min(chunk_start + self.chunk_size, self.num_samples)

    if self._current_chunk_start != chunk_start:
        f = self._get_file_handle()
        self._current_chunk_data = f[self.split]['signals'][chunk_start:chunk_end]
        self._current_chunk_start = chunk_start

    return self._current_chunk_data[idx - chunk_start]
```

---

### 3.4 Memory-Mapped NPZ Loading

**Location:** [tfr_dataset.py:52](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/tfr_dataset.py#L52)

Use `mmap_mode='r'` for large NumPy files:

```python
data = np.load(spectrogram_file, mmap_mode='r')
self.spectrograms = data['spectrograms']  # No copy, direct disk access
```

---

## 4. Multi-Worker Best Practices

### 4.1 DataLoader Configuration Pattern

**Location:** [streaming_hdf5_dataset.py:258-266](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py#L258-L266)

Standard DataLoader setup with streaming datasets:

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=is_train,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,  # Avoid re-init overhead
    drop_last=is_train  # Consistent batch sizes for training
)
```

---

### 4.2 Worker Initialization for Reproducibility

**Location:** [dataloader.py:182-204](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/dataloader.py#L182-L204)

Seed workers uniquely:

```python
def worker_init_fn_seed(worker_id: int):
    """Set unique seed per worker for reproducibility."""
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
```

---

### 4.3 Proper File Handle Cleanup

**Location:** [streaming_hdf5_dataset.py:134-140](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py#L134-L140)

Close handles in `__del__` (defensive):

```python
def __del__(self):
    if hasattr(self._local, 'file') and self._local.file is not None:
        try:
            self._local.file.close()
        except Exception:
            pass  # File may already be closed
```

> [!NOTE]
> Prefer context managers when possible. Use `__del__` as fallback only.

---

## 5. Data Validation Conventions

### 5.1 Shape and Type Assertions

**Location:** [tfr_dataset.py:75-76](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/tfr_dataset.py#L75-L76)

Validate data at load time:

```python
assert len(self.spectrograms) == len(self.labels), \
    f"Mismatch: {len(self.spectrograms)} spectrograms, {len(self.labels)} labels"
```

---

### 5.2 Split Existence Validation

**Location:** [streaming_hdf5_dataset.py:74-76](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py#L74-L76)

Verify requested split exists:

```python
with h5py.File(self.hdf5_path, 'r') as f:
    if split not in f:
        available = list(f.keys())
        raise ValueError(f"Split '{split}' not found. Available: {available}")
```

---

### 5.3 File Existence with Helpful Errors

**Location:** [tfr_dataset.py:334-343](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/tfr_dataset.py#L334-L343)

Include remediation steps in errors:

```python
if not Path(file_path).exists():
    raise FileNotFoundError(
        f"{name.capitalize()} spectrogram file not found: {file_path}\n"
        f"Please run: python scripts/precompute_spectrograms.py"
    )
```

---

### 5.4 Stratified Splitting

**Location:** [cwru_dataset.py:375-403](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cwru_dataset.py#L375-L403)

Use stratified splits for classification:

```python
from sklearn.model_selection import train_test_split

train_idx, temp_idx = train_test_split(
    np.arange(len(signals)),
    test_size=0.3,
    random_state=seed,
    stratify=labels  # Maintain class balance
)
```

---

## Quick Reference

| Pattern                    | Key Code                   | Location                    |
| -------------------------- | -------------------------- | --------------------------- |
| Factory methods            | `@classmethod from_hdf5()` | `dataset.py`                |
| Thread-local handles       | `threading.local()`        | `streaming_hdf5_dataset.py` |
| SWMR mode                  | `swmr=True`                | `streaming_hdf5_dataset.py` |
| Probabilistic augmentation | `if np.random.rand() > p`  | `cnn_transforms.py`         |
| Memory-mapped loading      | `mmap_mode='r'`            | `tfr_dataset.py`            |
| Persistent workers         | `persistent_workers=True`  | `streaming_hdf5_dataset.py` |
| Stratified splits          | `stratify=labels`          | `cwru_dataset.py`           |

---

_End of IDB 3.2 Best Practices_
