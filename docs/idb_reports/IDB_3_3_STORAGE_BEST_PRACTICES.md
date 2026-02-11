# IDB 3.3 Storage Layer Best Practices

**IDB ID:** 3.3  
**Domain:** Data Engineering  
**Date:** 2026-01-23

---

## 1. Caching Patterns

### 1.1 Config-Based Cache Key Generation

**Source:** [cache_manager.py:251-272](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py#L251-L272)

```python
def compute_cache_key(self, config: Dict[str, Any]) -> str:
    """Compute hash key from configuration for auto-invalidation."""
    config_str = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()

# Usage
key = cache.compute_cache_key(data_config.to_dict())
cache_name = f'dataset_{key[:8]}'
```

**Why:** Automatically invalidates cache when any configuration parameter changes. The `sort_keys=True` ensures consistent hashing regardless of dict ordering.

---

### 1.2 Graceful Redis Fallback

**Source:** [cache_service.py:27-40](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/dashboard/services/cache_service.py#L27-L40)

```python
@staticmethod
def get(key: str) -> Optional[Any]:
    if redis_client is None:
        return None  # Graceful fallback

    try:
        value = redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.error(f"Cache get error for key '{key}': {e}")
        return None
```

**Why:** Application continues functioning when Redis is unavailable. Cache misses don't crash the system.

---

### 1.3 HDF5 Compression with Metadata

**Source:** [cache_manager.py:81-104](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py#L81-L104)

```python
with h5py.File(cache_path, 'w') as f:
    f.create_dataset(
        'signals', data=signals,
        compression='gzip', compression_opts=4
    )
    f.attrs['cache_name'] = cache_name
    f.attrs['num_signals'] = len(signals)
    f.attrs['cached_at'] = datetime.now().isoformat()
```

**Why:** Compression level 4 balances speed vs size. Metadata attributes enable cache introspection without loading data.

---

### 1.4 Stratified Split Caching

**Source:** [cache_manager.py:274-398](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py#L274-L398)

```python
def cache_dataset_with_splits(
    self, signals, labels,
    split_ratios=(0.7, 0.15, 0.15),
    stratify=True, random_seed=42
) -> Path:
    # Creates HDF5 structure:
    # f['train']['signals'], f['train']['labels']
    # f['val']['signals'], f['val']['labels']
    # f['test']['signals'], f['test']['labels']
```

**Why:** Pre-computed stratified splits ensure reproducibility and consistent class distribution across experiments.

---

## 2. File Format Handling Conventions

### 2.1 Flexible Field Detection

**Source:** [matlab_importer.py:152-176](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/matlab_importer.py#L152-L176)

```python
def _extract_signal(self, mat_data: Dict) -> np.ndarray:
    # Try common field names first
    for field_name in ['signal', 'x', 'data', 'y']:
        if field_name in mat_data:
            return np.asarray(mat_data[field_name]).flatten()

    # Fallback: find largest numeric array
    candidates = [(k, v) for k, v in mat_data.items()
                  if isinstance(v, np.ndarray) and v.size > 100]
    if candidates:
        largest_key, largest_array = max(candidates, key=lambda x: x[1].size)
        logger.warning(f"Using field '{largest_key}' as signal")
        return largest_array.flatten()
```

**Why:** Handles MATLAB exports with non-standard field names while providing explicit warnings for traceability.

---

### 2.2 Label Extraction with Filename Fallback

**Source:** [matlab_importer.py:207-233](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/matlab_importer.py#L207-L233)

```python
def _extract_label(self, mat_data, metadata, filename):
    # Try metadata fields
    for field in ['label', 'fault_type', 'class', 'fault']:
        if field in metadata:
            return str(metadata[field])

    # Parse from filename: "signal_<fault>_<num>.mat"
    parts = Path(filename).stem.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:-1])

    logger.warning(f"Could not extract label from {filename}")
    return 'unknown'
```

**Why:** Robust label extraction from multiple sources ensures compatibility with various data formats.

---

### 2.3 Thread-Local HDF5 Handles

**Source:** [streaming_hdf5_dataset.py:90-94](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py#L90-L94)

```python
def _get_file_handle(self):
    """Get thread-local file handle for safe multi-threaded access."""
    thread_id = threading.current_thread().ident
    if thread_id not in self._file_handles:
        self._file_handles[thread_id] = h5py.File(self.hdf5_path, 'r')
    return self._file_handles[thread_id]
```

**Why:** HDF5 files are not thread-safe; thread-local handles prevent corruption in DataLoader workers.

---

## 3. Validation Patterns

### 3.1 Multi-Level Signal Validation

**Source:** [data_validator.py:86-174](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/data_validator.py#L86-L174)

```python
def compare_signals(self, python_signal, matlab_signal, label):
    metrics = {}
    errors, warnings = [], []

    # Level 1: Statistical comparison
    stat_metrics = self._compare_statistics(python_signal, matlab_signal)

    # Level 2: Point-wise error analysis
    pointwise_metrics = self._compute_pointwise_error(...)

    # Level 3: Time-domain correlation
    corr_metrics = self._compute_correlation(...)

    # Level 4: Frequency-domain coherence
    coh_metrics = self._compute_coherence(...)

    return ValidationResult(len(errors) == 0, self.tolerance, metrics, errors, warnings)
```

**Why:** Multi-level validation catches issues that single-metric comparison would miss.

---

### 3.2 Structured Validation Results

**Source:** [data_validator.py:26-46](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/data_validator.py#L26-L46)

```python
@dataclass
class ValidationResult:
    passed: bool
    tolerance: float
    metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]

    def __repr__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"ValidationResult({status}, {len(self.errors)} errors)"
```

**Why:** Dataclass provides type safety, immutability, and meaningful string representation for debugging.

---

### 3.3 Configurable Thresholds

**Source:** [data_validator.py:68-84](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/data_validator.py#L68-L84)

```python
def __init__(
    self,
    tolerance: float = 0.01,           # 1% relative error
    correlation_threshold: float = 0.99,
    coherence_threshold: float = 0.95
):
    self.tolerance = tolerance
    self.correlation_threshold = correlation_threshold
    self.coherence_threshold = coherence_threshold
```

**Why:** Sensible defaults with configurability allows tuning for different validation contexts.

---

## 4. Error Recovery Patterns

### 4.1 Batch Loading with Continue-on-Error

**Source:** [matlab_importer.py:290-300](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/matlab_importer.py#L290-L300)

```python
def load_batch(self, mat_dir: Path, pattern='*.mat'):
    batch_data = []
    for mat_file in mat_files:
        try:
            data = self.load_mat_file(mat_file)
            batch_data.append(data)
        except Exception as e:
            logger.error(f"Failed to load {mat_file.name}: {e}")
            continue  # Don't fail entire batch

    logger.info(f"Loaded {len(batch_data)}/{len(mat_files)} files")
    return batch_data
```

**Why:** One corrupted file doesn't abort entire dataset loading; logs success rate.

---

### 4.2 Redis Connection Resilience

**Source:** [cache_service.py:14-21](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/dashboard/services/cache_service.py#L14-L21)

```python
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info(f"Connected to Redis at {REDIS_URL}")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}. Caching disabled.")
    redis_client = None
```

**Why:** Module-level connection failure sets client to `None`; all methods check before use.

---

### 4.3 Cache Info with Exception Handling

**Source:** [cache_manager.py:201-214](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py#L201-L214)

```python
def list_caches(self) -> List[Dict[str, Any]]:
    cache_list = []
    for cache_file in self.cache_dir.glob('*.h5'):
        try:
            with h5py.File(cache_file, 'r') as f:
                info = {'name': cache_file.stem, ...}
                cache_list.append(info)
        except Exception as e:
            logger.warning(f"Could not read cache info from {cache_file}: {e}")
    return sorted(cache_list, key=lambda x: x['name'])
```

**Why:** Corrupted cache files don't prevent listing valid caches.

---

### 4.4 FileNotFoundError with Helpful Messages

**Source:** [cache_manager.py:128-129](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py#L128-L129)

```python
if not cache_path.exists():
    raise FileNotFoundError(f"Cache not found: {cache_path}")
```

**Why:** Early validation with full paths in error messages speeds debugging.

---

## 5. Quick Reference Table

| Pattern                | Module                      | Key Benefit              |
| ---------------------- | --------------------------- | ------------------------ |
| Config hash caching    | `cache_manager.py`          | Auto-invalidation        |
| Graceful fallback      | `cache_service.py`          | No crash on Redis fail   |
| Thread-local handles   | `streaming_hdf5_dataset.py` | Thread-safe I/O          |
| Continue-on-error      | `matlab_importer.py`        | Robust batch loading     |
| Multi-level validation | `data_validator.py`         | Comprehensive checks     |
| Dataclass results      | `data_validator.py`         | Type-safe returns        |
| Stratified splits      | `cache_manager.py`          | Reproducible experiments |

---

## 6. Adoption Checklist

For teams adopting these patterns:

- [ ] Use config hashing for any cached data that depends on parameters
- [ ] Always check for `None` client before cache operations
- [ ] Use thread-local handles for HDF5 in multi-worker DataLoaders
- [ ] Log partial success rates when batch processing
- [ ] Use dataclasses for validation/result objects
- [ ] Catch exceptions at file level, not batch level
- [ ] Include full paths in FileNotFoundError messages

---

_Best practices extracted from IDB 3.3 Storage Layer Sub-Block_
