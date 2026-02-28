# Domain 3: Data Engineering — Consolidated Analysis

## 1. Domain Overview

- **Purpose:** Data Engineering provides the complete data pipeline for the LSTM_PFD project, including synthetic signal generation with physics-based fault modeling, dataset loading with multiple memory strategies, and caching/storage services for reproducibility and performance.
- **Sub-blocks:**
  - **IDB 3.1 — Signal Generation:** Physics-based synthetic vibration signal generation with 11 fault classes
  - **IDB 3.2 — Data Loading:** Dataset classes, transforms, and DataLoader integration (12+ dataset classes, 15+ transforms)
  - **IDB 3.3 — Storage Layer:** Caching (HDF5, Redis, DB), MATLAB import, and validation services
- **Overall Independence Score:** (9 + 8 + 8) / 3 = **8.3/10** — High independence, clean interfaces
- **Key Interfaces:**
  - `SignalGenerator.generate_dataset()` → HDF5/MAT output
  - `BearingFaultDataset.from_hdf5()` / `from_mat_file()` → PyTorch Dataset
  - `CacheManager.cache_dataset_with_splits()` → Stratified HDF5 caching
  - `SignalValidator.compare_signals()` → Python vs MATLAB validation

---

## 2. Current State Summary

### What's Implemented

| Sub-block                 | Key Components                                                                                      | Lines of Code |
| ------------------------- | --------------------------------------------------------------------------------------------------- | ------------- |
| **3.1 Signal Generation** | `SignalGenerator`, `FaultModeler`, `NoiseGenerator`, `SpectrogramGenerator`, 8 augmentation classes | ~1,866        |
| **3.2 Data Loading**      | 12+ dataset classes, 15+ transform classes, streaming/chunked HDF5, CWRU benchmark support          | ~2,500        |
| **3.3 Storage Layer**     | `CacheManager`, `CacheService`, `ExplanationCache`, `MatlabImporter`, `SignalValidator`             | ~2,300        |

**Signal Generation (IDB 3.1):**

- 11 fault classes matching MATLAB `generator.m` port
- Sommerfeld number calculation with Arrhenius viscosity model
- 8-layer noise model (measurement, EMI, pink, drift, quantization, sensor drift, aliasing, impulse)
- 8 augmentation techniques (Mixup, TimeWarping, MagnitudeWarping, Jittering, Scaling, TimeShift, WindowSlicing, Compose)
- STFT and Mel spectrogram generation with 4 normalization strategies

**Data Loading (IDB 3.2):**

- Multiple memory strategies: in-memory, cached, streaming, memory-mapped
- Format support: HDF5, MAT, NPZ, Pickle
- Thread-local HDF5 handles with SWMR mode
- Factory methods (`from_hdf5()`, `from_mat_file()`, `from_generator_output()`)
- Pre-built transform pipelines with probabilistic augmentations

**Storage Layer (IDB 3.3):**

- HDF5 caching with gzip compression and SHA-256 config hashing
- Redis caching for dashboard with graceful fallback
- Database-backed XAI explanation caching
- MATLAB import with flexible field detection
- Multi-level signal validation (statistical, correlation, coherence, point-wise)

### What's Working Well

1. **Physics-based modeling** — Sommerfeld scaling, temperature-dependent viscosity, and fault harmonics are physically correct (IDB 3.1)
2. **Layered noise architecture** — Independent toggleable noise sources enable ablation studies (IDB 3.1)
3. **Thread-local HDF5 handles** — Safe multi-worker DataLoader access (IDB 3.2, 3.3)
4. **Factory methods** — Clean API for multiple data sources (IDB 3.2)
5. **Config-hash caching** — Automatic cache invalidation on config changes (IDB 3.3)
6. **Graceful Redis fallback** — Dashboard continues when cache unavailable (IDB 3.3)
7. **Multi-level validation** — Catches issues single-metric comparison would miss (IDB 3.3)
8. **Comprehensive metadata** — `SignalMetadata` dataclass with 28 fields for reproducibility (IDB 3.1)
9. **Stratified splits** — Maintains class distribution across train/val/test (IDB 3.1, 3.2, 3.3)
10. **SWMR mode for concurrent reads** — HDF5 Single-Writer-Multiple-Reader support (IDB 3.2)

### What's Problematic

1. **Monolithic files** — `signal_generator.py` is 37KB/935 lines with 4 classes (IDB 3.1)
2. **Documentation mismatch** — Docstring says "7-layer" noise but implements 8 layers (IDB 3.1)
3. **HDF5 file handle leaks** — `OnTheFlyTFRDataset` doesn't close handles when `cache_in_memory=False` (IDB 3.2)
4. **Bare except clauses** — Swallow all exceptions including `KeyboardInterrupt` (IDB 3.2)
5. **Thread-safety issues** — LRU cache not thread-safe across workers (IDB 3.2)
6. **Duplicate code** — Two identical `Compose` classes in `transforms.py` and `cnn_transforms.py` (IDB 3.2)
7. **Per-sample HDF5 open** — `CachedRawSignalDataset` opens file on every `__getitem__` (IDB 3.2)
8. **No unit tests** — Zero tests for cache_manager, matlab_importer, data_validator (IDB 3.3)
9. **Redis KEYS blocking** — O(n) operation blocks production Redis (IDB 3.3)
10. **Unused storage package** — `packages/storage/` is completely empty (IDB 3.3)

---

## 3. Critical Issues Inventory

### P0 Issues (Critical — Production Blockers)

| IDB | Issue                                         | Impact                                            | Effort | Dependencies |
| --- | --------------------------------------------- | ------------------------------------------------- | ------ | ------------ |
| 3.2 | HDF5 file handle leak in `OnTheFlyTFRDataset` | Exhausts file descriptors with DataLoader workers | 2h     | None         |
| 3.2 | Bare `except:` clauses swallowing errors      | Debugging impossible, silent data corruption      | 1h     | None         |
| 3.2 | LRU cache not thread-safe                     | Race conditions in multi-worker DataLoaders       | 4h     | None         |
| 3.3 | No unit tests for cache/import/validation     | Cache corruption reaches production undetected    | 2 days | None         |
| 3.3 | Redis `KEYS` pattern blocking                 | Production Redis blocked during invalidation      | 1h     | None         |

### P1 Issues (High Priority)

| IDB | Issue                                          | Impact                                       | Effort   | Dependencies |
| --- | ---------------------------------------------- | -------------------------------------------- | -------- | ------------ |
| 3.1 | Monolithic 37KB signal_generator.py            | Cognitive load, hard to test components      | 2-3 days | None         |
| 3.1 | Noise layer count mismatch (7 vs 8)            | Documentation confusion for researchers      | 5 min    | None         |
| 3.1 | Undocumented magic numbers                     | Researcher trust, reproducibility concerns   | 1 day    | None         |
| 3.2 | Duplicate `Compose` class                      | Code duplication, inconsistent behavior risk | 30 min   | None         |
| 3.2 | `CachedRawSignalDataset` opens HDF5 per-sample | Extremely slow I/O                           | 4h       | None         |
| 3.2 | Inconsistent label handling across datasets    | Integration bugs                             | 4h       | None         |
| 3.2 | Hardcoded segmentation params in CWRU          | Not configurable                             | 1h       | None         |
| 3.3 | No cache versioning                            | Silent compatibility breaks on code changes  | 4h       | None         |
| 3.3 | Missing file locking                           | Concurrent write corruption                  | 4h       | None         |
| 3.3 | Hardcoded fault type map                       | Not extensible                               | 2h       | None         |
| 3.3 | Unused `packages/storage/` abstraction         | Dead code, misleading structure              | 1 day    | None         |

### P2 Issues (Medium Priority)

| IDB | Issue                                      | Impact                                     | Effort | Dependencies |
| --- | ------------------------------------------ | ------------------------------------------ | ------ | ------------ |
| 3.1 | Non-deterministic seeding without flag     | Different thread order = different results | 2h     | None         |
| 3.1 | Missing signal validation                  | No NaN/Inf checks post-generation          | 4h     | None         |
| 3.1 | Mixed random modules (random vs np.random) | Breaks reproducibility                     | 1h     | None         |
| 3.1 | Spectrogram shape estimation off-by-one    | Minor boundary issues                      | 30 min | None         |
| 3.2 | Incomplete type annotations                | IDE/mypy warnings                          | 4h     | None         |
| 3.2 | Inconsistent transform type handling       | NumPy vs Tensor mismatch                   | 2h     | None         |
| 3.2 | Test functions in production code          | Code organization smell                    | 2h     | None         |
| 3.2 | Missing data validation                    | No NaN/Inf checks on load                  | 4h     | None         |
| 3.3 | No cache size limits                       | Disk space exhaustion                      | 4h     | None         |
| 3.3 | Magic number 256 nperseg                   | Should use constant                        | 1h     | None         |
| 3.3 | No HDF5 read retries                       | Transient failures crash                   | 2h     | None         |
| 3.3 | Division by zero guard (1e-10) arbitrary   | Should document rationale                  | 30 min | None         |

---

## 4. Technical Debt Registry

### Quick Wins (< 1 hour)

| IDB | Task                                                | Benefit                |
| --- | --------------------------------------------------- | ---------------------- |
| 3.1 | Fix noise layer count docstring (7 → 8)             | Documentation accuracy |
| 3.1 | Unify random module usage to `np.random`            | Reproducibility        |
| 3.2 | Delete duplicate `Compose` from `cnn_transforms.py` | Code deduplication     |
| 3.2 | Replace bare `except:` with specific exceptions     | Debuggability          |
| 3.3 | Replace Redis `KEYS` with `SCAN` iterator           | Production safety      |
| 3.3 | Extract magic number 256 to constant                | Maintainability        |

### Medium Tasks (1-4 hours)

| IDB | Task                                                          | Benefit              |
| --- | ------------------------------------------------------------- | -------------------- |
| 3.1 | Add post-generation signal validation                         | Data quality         |
| 3.1 | Document physics constants with references                    | Researcher trust     |
| 3.2 | Add context managers for HDF5 file handles                    | Resource safety      |
| 3.2 | Add `threading.Lock` to LRU cache                             | Thread safety        |
| 3.2 | Refactor `CachedRawSignalDataset` to use worker-local handles | Performance          |
| 3.2 | Standardize label handling across datasets                    | Consistency          |
| 3.2 | Move constants to `utils/constants.py`                        | Maintainability      |
| 3.3 | Add cache versioning with code hash                           | Compatibility safety |
| 3.3 | Add file locking for concurrent writes                        | Data integrity       |
| 3.3 | Centralize fault type map to config                           | Extensibility        |

### Large Refactors (1+ days)

| IDB | Task                                                     | Benefit                      |
| --- | -------------------------------------------------------- | ---------------------------- |
| 3.1 | Split `signal_generator.py` into 4-5 focused modules     | Maintainability, testability |
| 3.1 | Externalize physics parameters to YAML                   | Researcher configurability   |
| 3.1 | Consider Builder pattern for signal configuration        | API simplicity               |
| 3.2 | Create `BaseSignalDataset` ABC with unified interface    | Consistency                  |
| 3.2 | Move all test functions to `tests/` directory            | Code organization            |
| 3.3 | Implement or delete `packages/storage/` abstraction      | Architecture clarity         |
| 3.3 | Add comprehensive unit tests for cache/import/validation | Quality assurance            |

---

## 5. "If We Could Rewrite" Insights

### Cross-Sub-Block Themes

1. **Monolithic files** — Both `signal_generator.py` (935 lines) and data loading files contain too many classes
2. **Missing tests** — No unit tests across the domain for critical components
3. **Magic numbers** — Physics constants, segmentation params, and thresholds scattered without documentation
4. **Inconsistent randomness** — Mixed use of `random` and `np.random` modules breaks reproducibility
5. **Resource management** — HDF5 file handles need consistent context manager usage
6. **Lack of unified interfaces** — No common base class for datasets or transforms

### Fundamental Architectural Changes

1. **Unified Dataset Interface** — Create `BaseSignalDataset` ABC with:
   - `get_signal_shape()`, `get_num_classes()`, `get_class_names()`
   - `from_hdf5()`, `from_mat()` factory methods
   - `is_streaming` property for memory mode detection

2. **Physics Parameter Registry** — Externalize all physics constants to YAML:

   ```yaml
   sommerfeld:
     base: 0.25
     range: [0.05, 0.5]
   fault_signatures:
     desalignement:
       harmonics: { 2X: 0.35, 3X: 0.20 }
   ```

3. **Modular Signal Generator** — Extract to separate modules:
   - `fault_modeler.py`, `noise_generator.py`, `signal_metadata.py`
   - `io/hdf5_writer.py`, `io/mat_writer.py`

4. **Transform Hierarchy** — Single `BaseTransform` ABC with:
   - `SignalTransforms`, `AugmentationTransforms`, `TensorTransforms` subclasses

5. **Storage Abstraction Layer** — Implement `packages/storage/`:
   - `StorageBackend` ABC with `save()`, `load()`, `delete()`
   - `LocalStorage`, `S3Storage`, `GCSStorage` implementations

### Patterns to Preserve

| Pattern                          | Location                                  | Why It Works                        |
| -------------------------------- | ----------------------------------------- | ----------------------------------- |
| Layered noise with tracking dict | `NoiseGenerator.apply_noise_layers()`     | Ablation studies, reproducibility   |
| Config-hash caching              | `CacheManager.compute_cache_key()`        | Auto-invalidation on config changes |
| Thread-local HDF5 handles        | `StreamingHDF5Dataset._get_file_handle()` | Multi-worker safety                 |
| Factory methods for data sources | `BearingFaultDataset.from_*()`            | Clean API separation                |
| Graceful Redis fallback          | `CacheService.get()`                      | No crash on cache failure           |
| Multi-level validation           | `SignalValidator.compare_signals()`       | Comprehensive error detection       |
| Stratified splits with seed      | `cache_dataset_with_splits()`             | Reproducible experiments            |
| Comprehensive metadata dataclass | `SignalMetadata`                          | Generation provenance               |
| SWMR mode for concurrent reads   | `h5py.File(..., swmr=True)`               | DataLoader worker safety            |
| Chunked prefetching              | `ChunkedStreamingDataset`                 | I/O performance                     |

### Patterns to Eliminate

| Anti-Pattern                | Location                                 | Replacement                      |
| --------------------------- | ---------------------------------------- | -------------------------------- |
| Bare `except:` clauses      | Multiple files in IDB 3.2?3.3            | Specific exception types         |
| Per-sample HDF5 open        | `CachedRawSignalDataset.__getitem__`     | Worker-local handles             |
| Mixed random modules        | `signal_augmentation.py`                 | Unified `np.random`              |
| Duplicate class definitions | `Compose` in 2 files                     | Single source in `transforms.py` |
| Test code in production     | All dataset files                        | Move to `tests/`                 |
| Redis `KEYS` pattern        | `cache_service.py`                       | `SCAN` iterator                  |
| Hardcoded constants         | `signal_generator.py`, `cwru_dataset.py` | Config files or `constants.py`   |
| Empty package structure     | `packages/storage/`                      | Implement or delete              |

---

## 6. Best Practices Observed

### Code Conventions

- **Naming:** Snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Imports:** Grouped by stdlib → third-party → local, with type imports separated
- **Docstrings:** Google-style with Args/Returns/Raises sections; HDF5 structure documented inline
- **Type Hints:** Present on public methods; some gaps in internal functions and return types

### Design Patterns Worth Preserving

1. **Factory Methods** — `BearingFaultDataset.from_hdf5()`, `from_mat_file()`, `from_generator_output()`
2. **Dataclass Containers** — `SignalMetadata`, `ValidationResult`, `MatlabSignalData`
3. **Config Hash Invalidation** — SHA-256 of config dict for cache keying
4. **Layered Noise Architecture** — Independent toggleable noise sources with tracking
5. **Severity-Curve Modulation** — Time-varying severity multiplied with fault signatures
6. **Thread-Local State** — `threading.local()` for worker-safe file handles
7. **Graceful Degradation** — Cache returns `None` on failure, doesn't crash
8. **Continue-on-Error Batch Loading** — Log failures, process valid files

### Testing Patterns

> [!WARNING]
> **Critical Gap:** No unit tests exist for `cache_manager.py`, `matlab_importer.py`, or `data_validator.py`.

**Recommended Test Structure:**

```
tests/
├── test_cache_manager.py      # Roundtrip, invalidation, stratification
├── test_matlab_importer.py    # Field detection, label extraction
├── test_data_validator.py     # Multi-level validation thresholds
├── test_signal_generator.py   # Physics correctness, reproducibility
└── test_datasets.py           # All dataset classes consolidated
```

**Current State:** Test functions embedded in production code (`cnn_dataset.py:346-406`, `streaming_hdf5_dataset.py:271-374`, `cwru_dataset.py:470-539`) — should be migrated.

### Interface Contracts

| Interface                                  | Contract                                                                     |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| `SignalGenerator.generate_dataset()`       | Returns `Dict` with `'signals'`, `'labels'`, `'metadata'`; saves to HDF5/MAT |
| `BearingFaultDataset.__getitem__(idx)`     | Returns `Tuple[torch.Tensor, int]` (signal, label)                           |
| `CacheManager.cache_dataset_with_splits()` | Creates HDF5 with `train/val/test` groups                                    |
| `SignalValidator.compare_signals()`        | Returns `ValidationResult` with `passed: bool`                               |
| `MatlabImporter.load_mat_file()`           | Returns `MatlabSignalData` dataclass                                         |

---

## 7. Cross-Domain Dependencies

### Inbound Dependencies

| From Domain            | Consumes                             | Purpose                                   |
| ---------------------- | ------------------------------------ | ----------------------------------------- |
| **Infrastructure (4)** | `DataConfig`, `DataGenerationConfig` | Configuration for signal generation       |
| **Infrastructure (4)** | `utils.constants`                    | Fault types, physics constants            |
| **Infrastructure (4)** | Database (via `CacheService`)        | Redis connection, XAI explanation storage |

### Outbound Dependencies

| To Domain           | Provides                                          | Purpose                |
| ------------------- | ------------------------------------------------- | ---------------------- |
| **Core ML (1)**     | `BearingFaultDataset`, `StreamingHDF5Dataset`     | Training data          |
| **Core ML (1)**     | `get_train_transforms()`, `get_test_transforms()` | Preprocessing          |
| **Dashboard (2)**   | `CacheService.get/set()`                          | Request caching        |
| **Dashboard (2)**   | `ExplanationCache`                                | XAI result persistence |
| **Research (5)**    | `SignalGenerator`, `CWRUDataset`                  | Experiment data        |
| **Integration (6)** | All datasets via `model_factory`                  | Unified pipeline       |

### Integration Risks

| Risk                        | Affected Domains     | Mitigation                                         |
| --------------------------- | -------------------- | -------------------------------------------------- |
| HDF5 file format changes    | Core ML, Research    | Add cache versioning (P1-1 in 3.3)                 |
| Label mapping inconsistency | Core ML              | Standardize to single `label_to_idx` (P1-3 in 3.2) |
| Cache corruption            | All consumers        | Add unit tests, file locking (P0-1, P1-2 in 3.3)   |
| Redis unavailability        | Dashboard            | Already handled with graceful fallback ✅          |
| MATLAB format changes       | Research, Validation | May require schema validation in importer          |

---

## 8. Domain-Specific Recommendations

### Top 3 Quick Wins

1. **Fix bare `except:` clauses** (IDB 3.2) — Replace with specific exception types in `streaming_hdf5_dataset.py:139-140` and `tfr_dataset.py:206-207`. **Impact:** Debuggability. **Effort:** 1 hour.

2. **Delete duplicate `Compose` class** (IDB 3.2) — Remove from `cnn_transforms.py:241-274`, import from `transforms.py`. **Impact:** Code deduplication. **Effort:** 30 minutes.

3. **Replace Redis `KEYS` with `SCAN`** (IDB 3.3) — Fix `cache_service.py:68-81` to use iterator pattern. **Impact:** Production safety. **Effort:** 1 hour.

### Top 3 Strategic Improvements

1. **Add comprehensive unit tests for storage layer** (IDB 3.3) — Create `tests/test_cache_manager.py`, `tests/test_matlab_importer.py`, `tests/test_data_validator.py`. **Impact:** Quality assurance for critical path. **Effort:** 2 days.

2. **Split `signal_generator.py` into focused modules** (IDB 3.1) — Extract `FaultModeler`, `NoiseGenerator`, `SignalMetadata`, and I/O writers. **Impact:** Maintainability, testability. **Effort:** 2-3 days.

3. **Create unified dataset interface** (IDB 3.2) — Implement `BaseSignalDataset` ABC with standard methods. **Impact:** Consistency across all dataset classes. **Effort:** 1-2 days.

### Team Coordination Requirements

| Change                          | Requires Coordination With                       |
| ------------------------------- | ------------------------------------------------ |
| HDF5 format/structure changes   | Core ML (trainers), Research (scripts)           |
| Label mapping standardization   | Core ML (model factory), Dashboard (predictions) |
| Cache versioning implementation | All consumers of cached datasets                 |
| Storage abstraction layer       | Dashboard (checkpoints), Research (results)      |
| Fault type map centralization   | Signal Generator, MATLAB Importer, Datasets      |

---

## Appendix A: File Inventory

| File                                                                                                               | Lines | Sub-block | Purpose                                       |
| ------------------------------------------------------------------------------------------------------------------ | ----- | --------- | --------------------------------------------- |
| [signal_generator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/signal_generator.py)                          | 935   | 3.1       | Main physics engine + orchestration           |
| [signal_augmentation.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/signal_augmentation.py)                    | 500   | 3.1       | 8 augmentation techniques                     |
| [spectrogram_generator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/spectrogram_generator.py)                | 431   | 3.1       | Time-frequency analysis                       |
| [dataset.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/dataset.py)                                            | ~400  | 3.2       | BearingFaultDataset, AugmentedBearingDataset  |
| [cnn_dataset.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cnn_dataset.py)                                    | ~400  | 3.2       | RawSignalDataset, CachedRawSignalDataset      |
| [streaming_hdf5_dataset.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/streaming_hdf5_dataset.py)              | 380   | 3.2       | StreamingHDF5Dataset, ChunkedStreamingDataset |
| [cwru_dataset.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cwru_dataset.py)                                  | ~500  | 3.2       | CWRUDataset (benchmark)                       |
| [tfr_dataset.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/tfr_dataset.py)                                    | ~350  | 3.2       | SpectrogramDataset, OnTheFlyTFRDataset        |
| [transforms.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/transforms.py)                                      | ~200  | 3.2       | Core transforms (Normalize, Filter, etc.)     |
| [cnn_transforms.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cnn_transforms.py)                              | ~320  | 3.2       | CNN-specific transforms + augmentations       |
| [cache_manager.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cache_manager.py)                                | 477   | 3.3       | HDF5 dataset caching                          |
| [matlab_importer.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/matlab_importer.py)                            | 496   | 3.3       | MATLAB .mat file loading                      |
| [data_validator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/data_validator.py)                              | 516   | 3.3       | Python vs MATLAB validation                   |
| [cache_service.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/dashboard/services/cache_service.py)         | 105   | 3.3       | Redis caching wrapper                         |
| [explanation_cache.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/dashboard/services/explanation_cache.py) | 289   | 3.3       | XAI explanation caching                       |

---

## Appendix B: Signal Generation Investigation

### 37KB File Analysis (IDB 3.1)

The `signal_generator.py` file is 37KB (935 lines) because it contains:

1. **4 Classes in one file:**
   - `SignalGenerator` — Main orchestrator (300+ lines)
   - `FaultModeler` — 11 fault type implementations (200+ lines)
   - `NoiseGenerator` — 8-layer noise model (150+ lines)
   - `SignalMetadata` — 28-field dataclass (50+ lines)

2. **Dataset I/O logic** — HDF5 and MAT export (100+ lines)

3. **Severity/transient configuration** — Operating conditions logic (100+ lines)

**Verdict:** File is **overly complex** but not malicious. The size is due to comprehensive physics modeling. Recommended refactoring into 4-5 focused modules would improve maintainability without sacrificing functionality.

---

_Report generated by Domain 3 Consolidation Agent — 2026-01-24_
