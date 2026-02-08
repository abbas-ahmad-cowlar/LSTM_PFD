# IDB 3.3 Storage Layer Analysis Report

**IDB ID:** 3.3  
**Domain:** Data Engineering  
**Analyst:** AI Agent  
**Date:** 2026-01-23

---

## Executive Summary

The Storage Layer sub-block provides caching, import, validation, and abstraction services for the LSTM_PFD project. The implementation is **mature and well-documented** with comprehensive HDF5 caching, MATLAB compatibility, and signal validation. However, several critical issues exist including **missing unit tests**, **cache invalidation gaps**, and an **unused storage package**.

| Metric                    | Value  |
| ------------------------- | ------ |
| Total Files Analyzed      | 7      |
| Total Lines of Code       | ~2,300 |
| P0 Critical Issues        | 2      |
| P1 High Priority Issues   | 4      |
| P2 Medium Priority Issues | 5      |

---

## 1. Current State Assessment

### 1.1 Caching Mechanisms

#### `data/cache_manager.py` (477 lines)

**Architecture:** HDF5-based dataset caching with gzip compression.

| Feature         | Implementation          |
| --------------- | ----------------------- |
| Storage Format  | HDF5 via `h5py`         |
| Compression     | gzip (level 4 default)  |
| Metadata        | JSON-serialized in HDF5 |
| Split Support   | train/val/test groups   |
| Hash-based Keys | SHA-256 config hashing  |

**Key Methods:**

```python
cache_dataset()          # Basic caching
cache_dataset_with_splits()  # Stratified train/val/test
load_cached_dataset()    # Load from cache
invalidate_cache()       # Delete cache file
compute_cache_key()      # SHA-256 config hash
```

> [!TIP]
> The `compute_cache_key()` method provides automatic cache invalidation when configuration changes - a good pattern for reproducibility.

#### `packages/dashboard/services/cache_service.py` (105 lines)

**Architecture:** Redis-based caching wrapper for dashboard services.

| Feature              | Implementation                         |
| -------------------- | -------------------------------------- |
| Backend              | Redis                                  |
| Fallback             | Graceful degradation when disconnected |
| TTL                  | Configurable via `CACHE_TTL_MEDIUM`    |
| Pattern Invalidation | `KEYS` + `DELETE` pattern              |

> [!WARNING]
> Uses `redis.keys(pattern)` for invalidation - this is O(n) and blocks Redis in production. Should use `SCAN` iterator instead.

#### `packages/dashboard/services/explanation_cache.py` (289 lines)

**Architecture:** Database-backed caching for XAI explanations.

| Feature    | Implementation                 |
| ---------- | ------------------------------ |
| Backend    | SQLAlchemy + database          |
| Cleanup    | Time-based (configurable days) |
| Statistics | Hit/miss tracking              |

---

### 1.2 MATLAB Import Compatibility

#### `data/matlab_importer.py` (496 lines)

**Purpose:** Load MATLAB `.mat` files for Python/MATLAB signal validation.

| Feature             | Implementation                          |
| ------------------- | --------------------------------------- |
| Format Support      | Single and batch `.mat` files           |
| Field Detection     | Auto-detect: `signal`, `x`, `data`, `y` |
| Fallback            | Uses largest numeric array              |
| Metadata Extraction | Struct parsing with `_fieldnames`       |
| Label Extraction    | From metadata or filename parsing       |

**Supported Fault Types (11 classes):**

```
sain, desalignement, desequilibre, jeu, lubrification,
cavitation, usure, oilwhirl, mixed_misalign_imbalance,
mixed_wear_lube, mixed_cavit_jeu
```

> [!NOTE]
> Robust field detection handles various MATLAB export formats, but no schema validation exists for expected structure.

---

### 1.3 Data Validation Coverage

#### `data/data_validator.py` (516 lines)

**Purpose:** Validate Python signals against MATLAB reference with 1% tolerance.

| Validation Type | Threshold         | Implementation              |
| --------------- | ----------------- | --------------------------- |
| Statistical     | 1% relative error | Mean, std, RMS, percentiles |
| Correlation     | 0.99              | Pearson coefficient         |
| Coherence       | 0.95              | Frequency-domain coherence  |
| Point-wise      | 1% max error      | Per-sample comparison       |

**Output:** `ValidationResult` dataclass with pass/fail, metrics, errors, warnings.

```python
@dataclass
class ValidationResult:
    passed: bool
    tolerance: float
    metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]
```

---

### 1.4 Storage Abstraction Layer

#### `packages/storage/` Directory

| Subdirectory | Status    | Purpose                         |
| ------------ | --------- | ------------------------------- |
| `datasets/`  | **EMPTY** | Intended for dataset storage    |
| `models/`    | **EMPTY** | Intended for model artifacts    |
| `results/`   | **EMPTY** | Intended for experiment results |
| `uploads/`   | **EMPTY** | Intended for user uploads       |

> [!CAUTION]
> The `packages/storage/` directory structure exists but is **completely unused**. No Python modules implement the abstraction layer.

#### `data/streaming_hdf5_dataset.py` (380 lines)

**Purpose:** Memory-efficient streaming dataset for large HDF5 files.

| Feature                | Implementation                          |
| ---------------------- | --------------------------------------- |
| On-demand Reading      | `__getitem__` reads single sample       |
| Thread Safety          | Thread-local file handles               |
| Chunked Prefetch       | `ChunkedStreamingDataset` variant       |
| DataLoader Integration | `create_streaming_dataloaders()` helper |

---

## 2. Critical Issues Identification

### P0 - Critical (Blocks Production)

| ID   | Issue                              | Location              | Impact                      |
| ---- | ---------------------------------- | --------------------- | --------------------------- |
| P0-1 | **No unit tests for cache/import** | `tests/`              | Cache corruption undetected |
| P0-2 | **Redis KEYS pattern blocking**    | `cache_service.py:74` | Production Redis blocking   |

### P1 - High Priority

| ID   | Issue                      | Location                     | Impact                         |
| ---- | -------------------------- | ---------------------------- | ------------------------------ |
| P1-1 | No cache versioning        | `cache_manager.py`           | Silent compatibility breaks    |
| P1-2 | Missing file locking       | `cache_manager.py`           | Concurrent write corruption    |
| P1-3 | Hardcoded fault type map   | `matlab_importer.py:416-428` | Not extensible                 |
| P1-4 | Unused storage abstraction | `packages/storage/`          | Dead code/misleading structure |

### P2 - Medium Priority

| ID   | Issue                          | Location                    | Impact                   |
| ---- | ------------------------------ | --------------------------- | ------------------------ |
| P2-1 | No cache size limits           | `cache_manager.py`          | Disk space exhaustion    |
| P2-2 | Magic number 256 nperseg       | `data_validator.py:298`     | Should use constant      |
| P2-3 | No retry on HDF5 read errors   | `streaming_hdf5_dataset.py` | Transient failures crash |
| P2-4 | Division by zero guard (1e-10) | `data_validator.py`         | Arbitrary small value    |
| P2-5 | Missing type hints on return   | Multiple files              | IDE/mypy warnings        |

---

## 3. Detailed Issue Analysis

### P0-1: No Unit Tests for Cache/Import

**Evidence:** `find_by_name` returned no project-specific tests for `cache`, `matlab`, or `validator`.

```
# Search results:
test_*cache* → Only venv/sympy test
test_*matlab* → Only venv/pywt tests
test_*validator* → Only test_config_validator.py (unrelated)
```

**Risk:** Cache corruption, MATLAB import failures, and validation bugs will reach production undetected.

**Recommendation:**

```python
# tests/test_cache_manager.py
def test_cache_roundtrip():
    """Cache and reload dataset, verify integrity."""

def test_cache_invalidation():
    """Verify invalidate_cache removes file."""

def test_cache_with_splits_stratification():
    """Verify stratified splits maintain class distribution."""
```

---

### P0-2: Redis KEYS Pattern Blocking

**Location:** [cache_service.py:68-81](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/dashboard/services/cache_service.py#L68-L81)

```python
def invalidate_pattern(pattern: str) -> int:
    keys = redis_client.keys(pattern)  # ⚠️ O(n) blocking operation
    if keys:
        return redis_client.delete(*keys)
```

**Risk:** In production with millions of keys, `KEYS *` blocks Redis for seconds.

**Fix:**

```python
def invalidate_pattern(pattern: str) -> int:
    count = 0
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
        if keys:
            count += redis_client.delete(*keys)
        if cursor == 0:
            break
    return count
```

---

### P1-1: No Cache Versioning

**Location:** [cache_manager.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py)

**Current state:** Caches store data without version metadata.

**Risk:** Code changes (e.g., normalization logic) silently use stale cached data.

**Recommendation:**

```python
# Add to cache attributes
f.attrs['cache_version'] = '1.0.0'
f.attrs['code_hash'] = compute_code_hash()  # Hash of signal_generator.py
```

---

### P1-2: Missing File Locking

**Location:** [cache_manager.py:81](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py#L81)

```python
with h5py.File(cache_path, 'w') as f:  # No locking
```

**Risk:** Parallel training jobs writing to same cache create corrupted HDF5 files.

**Recommendation:**

```python
import fcntl  # Unix or portalocker for Windows

with open(cache_path.with_suffix('.lock'), 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with h5py.File(cache_path, 'w') as f:
        # ... write operations
```

---

### P1-3: Hardcoded Fault Type Map

**Location:** [matlab_importer.py:416-428](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/matlab_importer.py#L416-L428)

```python
FAULT_TYPE_MAP = {
    'sain': 0,
    'desalignement': 1,
    # ... 11 hardcoded entries
}
```

**Risk:** Adding new fault types requires code changes in multiple places.

**Recommendation:** Move to `utils/constants.py` or configuration file.

---

## 4. Good Practices to Adopt

| Practice                 | Location                               | Description                                 |
| ------------------------ | -------------------------------------- | ------------------------------------------- |
| Config hash caching      | `compute_cache_key()`                  | Auto-invalidate on config changes           |
| Graceful Redis fallback  | `CacheService`                         | Returns `None` when disconnected            |
| Comprehensive validation | `SignalValidator`                      | Multi-level (stats, correlation, coherence) |
| Dataclass containers     | `MatlabSignalData`, `ValidationResult` | Type-safe data transfer                     |
| Thread-local handles     | `StreamingHDF5Dataset`                 | Safe multi-threaded access                  |
| Docstring examples       | All files                              | Runnable usage examples                     |
| Stratified splits        | `cache_dataset_with_splits()`          | Maintains class distribution                |

---

## 5. Rewrite Retrospective

### Should `packages/storage/` Be Deleted or Implemented?

**Current State:** Empty directory structure with no code.

**Options:**

1. **Delete** - Remove misleading empty structure
2. **Implement** - Create storage abstraction layer:
   - `StorageBackend` ABC with `save()`, `load()`, `delete()`
   - `LocalStorage`, `S3Storage`, `GCSStorage` implementations
   - Unified API for models, datasets, results

**Recommendation:** Option 2 - The project would benefit from a unified storage interface, especially for cloud deployment.

---

### Should Caching Be Unified?

**Current State:** Three separate caching systems:

- `CacheManager` (HDF5 for signals)
- `CacheService` (Redis for dashboard)
- `ExplanationCache` (DB for XAI)

**Assessment:** This separation is **appropriate** - each cache serves different use cases:

- HDF5 for large array data
- Redis for ephemeral session/request data
- DB for persistent explanation artifacts

---

## 6. Verification Plan

### Automated Tests (To Be Created)

```bash
# Unit tests
pytest tests/test_cache_manager.py -v
pytest tests/test_matlab_importer.py -v
pytest tests/test_data_validator.py -v

# Integration tests
pytest tests/integration/test_storage_layer.py -v
```

### Manual Verification

1. **Cache integrity:** Generate dataset, cache, reload, compare arrays
2. **MATLAB compatibility:** Load reference `.mat` files from each fault type
3. **Validation accuracy:** Run `validate_against_matlab()` with known-good signals

---

## 7. Technical Debt Inventory

| Priority | Issue                        | Effort  | Impact |
| -------- | ---------------------------- | ------- | ------ |
| P0       | Add unit tests               | 2 days  | High   |
| P0       | Fix Redis SCAN               | 1 hour  | High   |
| P1       | Add cache versioning         | 4 hours | Medium |
| P1       | Add file locking             | 4 hours | Medium |
| P1       | Centralize fault type map    | 2 hours | Low    |
| P1       | Delete/implement storage pkg | 1 day   | Medium |
| P2       | Add cache size limits        | 4 hours | Medium |
| P2       | Extract magic numbers        | 1 hour  | Low    |
| P2       | Add HDF5 read retries        | 2 hours | Low    |

**Total Estimated Effort:** 4-5 developer days

---

## 8. File Reference

| File                                                                                                               | Lines | Purpose                     |
| ------------------------------------------------------------------------------------------------------------------ | ----- | --------------------------- |
| [cache_manager.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py)                                | 477   | HDF5 dataset caching        |
| [matlab_importer.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/matlab_importer.py)                            | 496   | MATLAB .mat file loading    |
| [data_validator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/data_validator.py)                              | 516   | Python vs MATLAB validation |
| [cache_service.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/dashboard/services/cache_service.py)         | 105   | Redis caching wrapper       |
| [explanation_cache.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/dashboard/services/explanation_cache.py) | 289   | XAI explanation caching     |
| [streaming_hdf5_dataset.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py)              | 380   | Memory-efficient streaming  |

---

_Report generated by AI Agent for IDB 3.3 Storage Layer Analysis_
