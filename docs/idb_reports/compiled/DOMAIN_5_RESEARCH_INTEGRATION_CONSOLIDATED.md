# Domain #5: Research & Science + Integration Layer — Consolidated Analysis

**Consolidation Date:** 2026-01-24  
**Input Files:** IDB 5.1, IDB 5.2, IDB 6.0 (Analysis + Best Practices)  
**Total Lines Analyzed:** ~10,000+ across 43+ files

---

## 1. Domain Overview

### Purpose

This combined domain encompasses two smaller but critical areas:

1. **Research & Science (5.x)**: Experiment scripts for reproducible research and publication-quality visualizations
2. **Integration Layer (6.0)**: Cross-cutting bridge between Core ML Engine and Dashboard Platform

### Sub-blocks

| IDB | Sub-Block         | Files     | LOC    | Primary Function                                                     |
| --- | ----------------- | --------- | ------ | -------------------------------------------------------------------- |
| 5.1 | Research Scripts  | 9 scripts | ~5,034 | Systematic experimentation (ablation, benchmarks, OOD testing)       |
| 5.2 | Visualization     | 13 files  | ~4,300 | Publication-quality plots (attention maps, t-SNE/UMAP, spectrograms) |
| 6.0 | Integration Layer | 20 files  | ~4,500 | Pipeline orchestration, adapters, utilities                          |

### Overall Independence Score

| Sub-Block            | Independence | Justification                                     |
| -------------------- | ------------ | ------------------------------------------------- |
| 5.1 Research Scripts | 7/10         | Uses Core ML heavily but self-contained execution |
| 5.2 Visualization    | 8/10         | Mostly standalone, minimal external dependencies  |
| 6.0 Integration      | 3/10         | By design, highly coupled to all domains          |

**Average:** (7 + 8 + 3) / 3 = **6.0/10**

### Key Interfaces

| Interface                          | Direction        | Consumer(s)                 |
| ---------------------------------- | ---------------- | --------------------------- |
| `Phase0Adapter.generate()`         | Dashboard → Core | Celery Tasks                |
| `Phase1Adapter.train()`            | Dashboard → Core | Training UI                 |
| `DeepLearningAdapter.train()`      | Dashboard → Core | DL Training                 |
| `ModelRegistry.register_model()`   | Core → Storage   | All training pipelines      |
| `utils/constants.py`               | Shared           | All domains (50+ constants) |
| `utils/reproducibility.set_seed()` | Shared           | Research scripts, training  |

---

## 2. Current State Summary

### What's Implemented

#### Research Scripts (IDB 5.1)

- **9 comprehensive scripts** covering ablation studies, physics-aware pretraining, PINN comparisons, transformer benchmarks, OOD testing, and XAI metrics
- **100% CLI coverage**: All scripts have `argparse` interfaces
- **100% seed control**: All scripts support reproducibility
- **Multi-format output**: JSON, CSV, PNG, HTML, LaTeX tables

#### Visualization (IDB 5.2)

- **13 visualization files** (~4,300 LOC) covering:
  - Transformer attention heatmaps (`attention_viz.py`)
  - CNN analysis (Grad-CAM, saliency maps, filter visualization)
  - Dimensionality reduction (t-SNE/UMAP clusters)
  - Spectrogram comparisons (STFT, CWT, WVD)
  - Interactive XAI dashboard (Streamlit)

#### Integration Layer (IDB 6.0)

- **Unified Pipeline** with 10-phase ML workflow definition
- **3 Dashboard Adapters**: Phase0, Phase1, DeepLearning
- **Model Registry**: SQLite-backed tracking
- **11 utility modules** (constants, device management, checkpointing, reproducibility)
- **629 LOC of centralized constants** with no magic numbers

### What's Working Well

| Pattern                         | Location                                      | Benefit                                    |
| ------------------------------- | --------------------------------------------- | ------------------------------------------ |
| Dataclass configurations        | `ablation_study.py`                           | Type-safe, serializable experiment configs |
| Multi-seed experiments          | `ablation_study.py`, `contrastive_physics.py` | Statistical validity for publications      |
| McNemar's significance test     | `pinn_ablation.py`                            | Proper model comparison                    |
| `--quick` development mode      | All research scripts                          | Fast iteration during development          |
| Optional dependency pattern     | `HAS_TORCH`, `HAS_PLOTTING`                   | Graceful degradation                       |
| Context manager device handling | `utils/device_manager.py`                     | Automatic GPU cleanup                      |
| Timestamped outputs             | Research scripts                              | No accidental overwrites                   |
| Adapter pattern with callbacks  | `integrations/*.py`                           | Clean Dashboard ↔ Core ML bridge           |
| Top-k checkpoint retention      | `checkpoint_manager.py`                       | Disk-efficient training                    |

### What's Problematic

| Issue                                | Scope                                                  | Impact                                     |
| ------------------------------------ | ------------------------------------------------------ | ------------------------------------------ |
| **DPI inconsistency** (150 vs 300)   | 7 of 13 visualization files                            | Figures rejected by journals               |
| **No unified style config**          | All visualization files                                | 13 files with independent styling          |
| **Duplicate CLASS_COLORS**           | `feature_visualization.py`, `latent_space_analysis.py` | Visual inconsistency, maintenance burden   |
| **`jet` colormap used**              | `activation_maps_2d.py`                                | Not colorblind-accessible                  |
| **`sys.path.insert()` anti-pattern** | All adapters                                           | Fragile imports, debugging difficulty      |
| **7 placeholder phases**             | `unified_pipeline.py`                                  | Pipeline non-functional E2E                |
| **Duplicate logging modules**        | `utils/logger.py` vs `utils/logging.py`                | Confusion about which to use               |
| **Code duplication in scripts**      | ~200 lines across 6 scripts                            | Training loops, data loading reimplemented |

---

## 3. Critical Issues Inventory

### P0 Issues (Critical - Production Blockers)

| IDB | Issue                                                                    | Impact                                                  | Effort | Dependencies      |
| --- | ------------------------------------------------------------------------ | ------------------------------------------------------- | ------ | ----------------- |
| 5.1 | **OOD Testing requires actual data** (`ood_testing.py:446-486`)          | Demo mode works, real data path has no validation       | 2h     | Data Domain       |
| 5.1 | **PINN model import fallback hides errors** (`pinn_ablation.py:109-113`) | Falls back to ResNet18 silently if HybridPINN missing   | 1h     | None              |
| 5.2 | **DPI inconsistency (150 vs 300)** across 7 files                        | Publications rejected; figures look different           | 2h     | None              |
| 5.2 | **No unified style config** - 13 files with independent styling          | Maintenance nightmare, visual inconsistency             | 4h     | None              |
| 5.2 | **Duplicate CLASS_COLORS** in 2 files                                    | Different color values causing visual drift             | 1h     | None              |
| 6.0 | **Unified Pipeline has 7 placeholder phases**                            | Pipeline cannot execute full E2E workflow               | 8-16h  | All domains       |
| 6.0 | **`sys.path.insert()` anti-pattern** in all adapters                     | Fragile imports, pollutes sys.path, namespace conflicts | 4h     | Package structure |

### P1 Issues (High Priority)

| IDB | Issue                                                          | Impact                                              | Effort | Dependencies  |
| --- | -------------------------------------------------------------- | --------------------------------------------------- | ------ | ------------- |
| 5.1 | Code duplication: Training loops (~200 lines across 6 scripts) | Bug propagation, maintenance burden                 | 4h     | None          |
| 5.1 | Inconsistent seed handling (local vs shared utility)           | Non-reproducible results                            | 1h     | None          |
| 5.1 | Hardcoded output directories                                   | No consistent root configuration                    | 1h     | Config module |
| 5.1 | Missing statistical tests in `transformer_benchmark.py`        | No significance testing between models              | 2h     | None          |
| 5.2 | **`jet` colormap used** (`activation_maps_2d.py:310,322`)      | Not colorblind-accessible (~8% males affected)      | 30m    | None          |
| 5.2 | **Font settings not global** (12 of 13 files)                  | Inconsistent label sizes                            | 2h     | Style config  |
| 5.2 | **`plt.close()` inconsistent**                                 | Memory leaks in batch processing                    | 1h     | None          |
| 5.2 | Bare except clauses (`feature_visualization.py:337-348`)       | Swallows errors silently                            | 30m    | None          |
| 6.0 | Hardcoded fallback for SAMPLING_RATE                           | Silent fallback could mask import issues            | 30m    | None          |
| 6.0 | Model Registry uses SQLite without connection pooling          | Performance issues under concurrent dashboard usage | 2h     | None          |
| 6.0 | Deep Learning Adapter duplicates early stopping logic          | Doesn't use existing `utils/early_stopping.py`      | 1h     | None          |
| 6.0 | Duplicate logging modules (`logger.py` vs `logging.py`)        | Confusion about which to use                        | 1h     | None          |

### P2 Issues (Medium Priority)

| IDB | Issue                                             | Impact                                      | Effort | Dependencies |
| --- | ------------------------------------------------- | ------------------------------------------- | ------ | ------------ |
| 5.1 | Bare `except:` clause in `ood_testing.py:247-248` | Swallows exceptions                         | 30m    | None         |
| 5.1 | `xai_metrics.py` has no CLI                       | Only usable as library import               | 1h     | None         |
| 5.1 | Incomplete type hints                             | Some functions lack return type annotations | 2h     | None         |
| 5.1 | `ablation_study.py` too large (1,175 lines)       | Should be split into modules                | 4h     | None         |
| 5.2 | Missing figure size standardization               | Inconsistent plot dimensions                | 1h     | Style config |
| 5.2 | No PDF/SVG vector output option                   | Limited for publications                    | 2h     | None         |
| 5.2 | Missing progress bars for t-SNE/UMAP              | Poor UX for large datasets                  | 1h     | None         |
| 6.0 | No type hints on adapter return types             | Reduced IDE support                         | 2h     | None         |
| 6.0 | Magic number `num_workers=2` in data loading      | Should use `get_optimal_num_workers()`      | 30m    | None         |
| 6.0 | `validate_cross_phase_compatibility()` is a stub  | Returns True without validation             | 2h     | None         |
| 6.0 | No integration test for `model_registry.py`       | SQLite operations untested in CI            | 2h     | None         |

---

## 4. Technical Debt Registry

### Quick Wins (< 1 hour)

| IDB | Task                                                        | Benefit                              |
| --- | ----------------------------------------------------------- | ------------------------------------ |
| 5.1 | Fix bare `except` in `ood_testing.py:247`                   | Proper error visibility              |
| 5.1 | Consolidate to `utils.reproducibility.set_seed` everywhere  | Consistent reproducibility           |
| 5.1 | Add explicit error when HybridPINN import fails             | Fail-fast instead of silent fallback |
| 5.2 | Replace `jet` with `viridis` in `activation_maps_2d.py`     | Colorblind accessibility             |
| 5.2 | Add `plt.close(fig)` after all save operations              | Prevent memory leaks                 |
| 5.2 | Fix bare except in `feature_visualization.py`               | Proper error visibility              |
| 6.0 | Add WARNING log to SAMPLING_RATE fallback                   | Visibility of import issues          |
| 6.0 | Use existing `EarlyStopping` class in `DeepLearningAdapter` | Reduces code duplication             |
| 6.0 | Use `get_optimal_num_workers()` in data loading             | Optimal performance                  |

### Medium Tasks (1-4 hours)

| IDB | Task                                                                   | Benefit                     |
| --- | ---------------------------------------------------------------------- | --------------------------- |
| 5.1 | Add CLI to `xai_metrics.py`                                            | Usable as standalone script |
| 5.1 | Standardize output directories via environment variable or config      | Consistent output locations |
| 5.2 | Create `visualization/style_config.py` with unified DPI, colors, fonts | Single source of truth      |
| 5.2 | Extract single `CLASS_COLORS` and `CLASS_NAMES` definition             | No duplicate definitions    |
| 5.2 | Add optional `ax` parameter to all plotting functions                  | Composable figures          |
| 5.2 | Add vector output option (PDF/SVG)                                     | Publication-ready           |
| 6.0 | Consolidate `logger.py` and `logging.py` into single module            | Clarity                     |
| 6.0 | Add connection pooling to `ModelRegistry`                              | Performance under load      |
| 6.0 | Add integration tests for `ModelRegistry` and adapters                 | Regression prevention       |
| 6.0 | Add type hints to all adapter methods                                  | IDE support, documentation  |

### Large Refactors (1+ days)

| IDB | Task                                                                  | Benefit                                |
| --- | --------------------------------------------------------------------- | -------------------------------------- |
| 5.1 | Extract common `ExperimentRunner` base class                          | Unified training loop, less bugs       |
| 5.1 | Create shared data loading module                                     | Eliminate HDF5 loading duplication     |
| 5.1 | Split `ablation_study.py` into modules (configs, models, runner, CLI) | Maintainability                        |
| 5.1 | Add experiment tracking (MLflow/W&B) integration                      | Reproducibility, experiment management |
| 5.1 | Create publication utilities module (LaTeX tables, styling, stats)    | Publication-ready outputs              |
| 5.2 | Implement colorblind mode toggle across all visualizations            | Accessibility (8% of males)            |
| 5.2 | Add progress bars for dimensionality reduction                        | UX improvement                         |
| 6.0 | Replace `sys.path.insert()` with proper Python packaging              | Clean imports, no global pollution     |
| 6.0 | Implement or remove unified_pipeline placeholder phases               | Functional E2E pipeline                |
| 6.0 | Create abstract base class for adapters                               | Extensibility, interface contracts     |
| 6.0 | Split `utils/` by domain (core, io, viz, training)                    | Better organization at scale           |

---

## 5. "If We Could Rewrite" Insights

### Cross-Sub-Block Themes

1. **Centralized Configuration Needed**
   - Research scripts need unified `ExperimentRunner` with consistent training loops
   - Visualization needs single `style_config.py` for DPI, fonts, colors
   - Integration layer needs proper Python packaging (no `sys.path` hacks)

2. **Duplication is Rampant**
   - ~200 lines of training loop code duplicated across 6 research scripts
   - `CLASS_COLORS` defined twice with different values
   - Early stopping implemented inline instead of using existing utility
   - Two logging modules (`logger.py` vs `logging.py`)

3. **Publication-Readiness Gaps**
   - Half of visualization files use 150 DPI (insufficient for print)
   - Missing statistical significance tests in some benchmarks
   - LaTeX table generation only in 1 of 9 research scripts
   - No colorblind-safe mode

4. **Incomplete Bridge Implementation**
   - 7 of 10 unified pipeline phases are placeholders
   - No abstract adapter interface
   - Progress callbacks are ad-hoc (no shared protocol)

### Fundamental Architectural Changes

1. **Extract Common Experiment Infrastructure**

   ```
   scripts/research/
   ├── _common/
   │   ├── experiment.py      # Base ExperimentRunner class
   │   ├── data_loading.py    # Unified HDF5 loading
   │   ├── evaluation.py      # Shared metrics computation
   │   └── visualization.py   # Common plotting utilities
   ├── ablation_study.py      # Calls _common.experiment
   └── ...
   ```

2. **Create Visualization Style System**

   ```python
   # visualization/style_config.py
   @dataclass
   class PublicationStyle:
       dpi: int = 300
       font_size: int = 12
       colormap_sequential: str = 'viridis'
       colormap_categorical: List[str] = CLASS_COLORS
   ```

3. **Implement Abstract Adapter Interface**

   ```python
   class BaseAdapter(ABC):
       @abstractmethod
       def train(self, config: Dict, progress_callback: Optional[Callable]) -> Dict: ...

       @abstractmethod
       def validate_config(self, config: Dict) -> bool: ...
   ```

4. **Reorganize Integration Layer**
   ```
   integration/
   ├── adapters/           # Thin adapters only
   ├── registry/           # Model tracking
   ├── validators/         # Validation logic
   └── orchestration/      # Pipeline orchestration
   ```

### Patterns to Preserve

| Pattern                              | Location                   | Why Keep                  |
| ------------------------------------ | -------------------------- | ------------------------- |
| Dataclass configs with `.to_dict()`  | `ablation_study.py`        | Type-safe, serializable   |
| Multi-seed experiments with mean±std | Research scripts           | Statistical validity      |
| McNemar's significance test          | `pinn_ablation.py`         | Proper model comparison   |
| `--quick` flag for development       | All CLI scripts            | Fast iteration            |
| `HAS_TORCH`, `HAS_PLOTTING` pattern  | Optional imports           | Graceful degradation      |
| `DeviceManager` context manager      | `utils/device_manager.py`  | Automatic cleanup         |
| Top-k checkpoint retention           | `checkpoint_manager.py`    | Disk efficiency           |
| Adapter with callbacks               | `integrations/*.py`        | Clean cross-domain bridge |
| Timestamped filenames                | Research scripts output    | No overwrites             |
| `bbox_inches='tight'` on all saves   | Visualization              | Clean figure exports      |
| UMAP availability check              | `feature_visualization.py` | Optional dependency       |
| Demo functions pattern               | Visualization files        | Documentation + testing   |

### Patterns to Eliminate

| Anti-Pattern              | Location                                  | Replacement                      |
| ------------------------- | ----------------------------------------- | -------------------------------- |
| `sys.path.insert()`       | All adapters                              | Proper Python packaging          |
| Inline training loops     | 6 research scripts                        | Shared `ExperimentRunner`        |
| Duplicate `CLASS_COLORS`  | 2 visualization files                     | Single `style_config.py`         |
| `jet` colormap            | `activation_maps_2d.py`                   | `viridis` (perceptually uniform) |
| Silent fallbacks          | `pinn_ablation.py`, `unified_pipeline.py` | Explicit errors or warnings      |
| Bare `except:` clauses    | Multiple files                            | Specific exception handling      |
| Missing `plt.close()`     | Visualization files                       | Always close after save          |
| 150 DPI for saved figures | 7 visualization files                     | 300 DPI standard                 |
| Two logging modules       | `utils/`                                  | Single consolidated module       |
| Hardcoded `num_workers=2` | `deep_learning_adapter.py`                | Use utility function             |

---

## 6. Best Practices Observed

### Code Conventions

- **Naming:**
  - Scripts: `snake_case.py` describing experiment type
  - Classes: `PascalCase` (e.g., `ExperimentConfig`, `ModelRegistry`)
  - Functions: `verb_noun()` pattern (e.g., `run_ablation()`, `save_results()`)
  - Constants: `UPPER_SNAKE_CASE`

- **Imports:**

  ```python
  # Standard library
  import argparse
  from datetime import datetime

  # Third-party (with optional checks)
  import numpy as np
  try:
      import torch
      HAS_TORCH = True
  except ImportError:
      HAS_TORCH = False

  # Project imports
  from utils.reproducibility import set_seed
  ```

- **Docstrings:** Google-style with Args, Returns, Examples sections

- **Type Hints:** Present on most function signatures, missing on some return types

### Design Patterns Worth Preserving

1. **Adapter Pattern** (`Phase0Adapter`, `Phase1Adapter`, `DeepLearningAdapter`)
   - Static methods for stateless Celery invocation
   - `success` boolean in all return dicts
   - Optional `progress_callback` parameter

2. **Dataclass Configuration** (`AblationConfig`, `AblationResult`)
   - Type-safe experiment parameters
   - `.to_dict()` for JSON serialization
   - `field(default_factory=list)` for mutable defaults

3. **Context Manager Resources** (`DeviceManager`, `Timer`)
   - `__enter__`/`__exit__` for automatic cleanup
   - GPU memory cleared on exit

4. **Registry Pattern** (`ModelRegistry`)
   - SQLite-backed with JSON metadata
   - Auto-creates directories
   - Returns generated IDs for chaining

5. **Optional Dependency Pattern**
   ```python
   try:
       import umap
       HAS_UMAP = True
   except ImportError:
       HAS_UMAP = False
   ```

### Testing Patterns

| Test Type            | Coverage                                   | Location                                             |
| -------------------- | ------------------------------------------ | ---------------------------------------------------- |
| Integration tests    | Training loops, checkpoints, streaming, CV | `tests/integration/test_comprehensive.py` (410+ LOC) |
| Pipeline integration | Phase execution                            | `tests/integration/test_pipelines.py` (~200 LOC)     |
| Missing              | Model Registry CRUD                        | N/A                                                  |
| Missing              | Dashboard Adapter error handling           | N/A                                                  |
| Missing              | Cross-phase data format compatibility      | N/A                                                  |

### Interface Contracts

**Adapter Train Interface:**

```python
def train(config: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Args:
        config: Flat dict with model_type, cache_path, num_epochs, etc.
        progress_callback: Optional fn(epoch: int, metrics: dict) -> None

    Returns:
        {"success": True, "test_accuracy": float, ...} on success
        {"success": False, "error": str} on failure
    """
```

**Progress Callback Protocol:**

```python
ProgressCallback = Callable[[int, Dict[str, Any]], None]
# Called as: callback(epoch, {"train_loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.95})
```

**Standard Config Keys:**

- Required: `model_type`, `cache_path`
- Training: `num_epochs`, `batch_size`, `learning_rate`
- Optional: `device`, `early_stopping_patience`, `optimizer`, `scheduler`
- Model-specific: Nested in `hyperparameters` dict

---

## 7. Cross-Domain Dependencies

### Inbound Dependencies

| From Domain                  | Component                          | What's Used                                 |
| ---------------------------- | ---------------------------------- | ------------------------------------------- |
| **Domain 1: Core ML**        | `packages/core/models/*`           | CNN1D, ResNet, Transformer, HybridPINN      |
| **Domain 1: Core ML**        | `packages/core/training/*`         | Training loops (though often reimplemented) |
| **Domain 3: Data**           | `data_generation/signal_generator` | Signal synthesis                            |
| **Domain 3: Data**           | HDF5 caches                        | `signals_cache.h5`                          |
| **Domain 4: Infrastructure** | `config/data_config`               | Dataset configuration                       |
| **Domain 4: Infrastructure** | `utils/*`                          | Constants, device management, logging       |

### Outbound Dependencies

| Consumer           | What's Consumed                                         |
| ------------------ | ------------------------------------------------------- |
| Dashboard Tasks    | `Phase0Adapter`, `Phase1Adapter`, `DeepLearningAdapter` |
| Dashboard UI       | Model Registry for model listing                        |
| Research Pipelines | All visualization utilities                             |
| All Domains        | `utils/constants.py` (50+ constants)                    |
| All Domains        | `utils/reproducibility.py` (seed setting)               |
| All Domains        | `utils/device_manager.py` (GPU handling)                |
| All Domains        | `utils/checkpoint_manager.py` (model saving)            |

### Integration Risks

> [!CAUTION]
> **HIGH COUPLING RISK:** The Integration Layer directly imports from:
>
> - `data_generation.signal_generator`
> - `config.data_config`
> - `pipelines.classical_ml_pipeline`
> - `packages.core.models`
> - `features.feature_extractor`
>
> Changes to any of these modules may break the integration layer. Consider implementing interface contracts.

| Risk                            | Probability | Impact | Mitigation                      |
| ------------------------------- | ----------- | ------ | ------------------------------- |
| Core ML model interface changes | Medium      | High   | Define abstract model interface |
| `sys.path` order conflicts      | High        | Medium | Proper packaging                |
| HDF5 format changes             | Low         | High   | Version field in cache          |
| Constants drift                 | Medium      | Medium | Add validation on load          |
| Visualization style drift       | High        | Low    | Centralize in style_config.py   |

---

## 8. Domain-Specific Recommendations

### Top 3 Quick Wins

1. **Create `visualization/style_config.py`** (2h)
   - Centralize DPI=300, fonts, `CLASS_COLORS`, `CLASS_NAMES`
   - Single `apply_publication_style()` function
   - Import at top of all 13 visualization files

2. **Replace `jet` colormap and add `plt.close()`** (1h)
   - Change `jet` → `viridis` in `activation_maps_2d.py`
   - Add `plt.close(fig)` after all `savefig()` calls
   - Immediate accessibility and memory improvement

3. **Consolidate seed handling to shared utility** (30m)
   - Replace local `_set_seed()` and `set_seeds()` with `from utils.reproducibility import set_seed`
   - Affects: `ablation_study.py`, `pinn_ablation.py`, `contrastive_physics.py`

### Top 3 Strategic Improvements

1. **Extract Common Experiment Infrastructure** (5 days)
   - Create `scripts/research/_common/` with `ExperimentRunner`, shared data loading, evaluation
   - Reduce ~200 lines of duplication per script
   - Enable MLflow/W&B integration point

2. **Replace `sys.path.insert()` with Proper Packaging** (3 days)
   - Add `pyproject.toml` with entry points
   - Convert to relative imports with proper `__init__.py` chains
   - Fixes all adapter import fragility

3. **Implement or Remove Unified Pipeline** (3-5 days)
   - Option A: Complete phase 2-9 implementations
   - Option B: Document as roadmap and remove placeholders
   - Currently misleading - looks functional but isn't

### Team Coordination Requirements

| Change                             | Domains Affected | Coordination Needed                     |
| ---------------------------------- | ---------------- | --------------------------------------- |
| Style config creation              | 5.2              | Agree on color palette with design team |
| Experiment infrastructure refactor | 5.1              | Review by research team                 |
| Package restructure                | All              | Full team alignment, CI/CD updates      |
| Unified pipeline completion        | 1, 3, 4          | Requires input from all domain owners   |
| Adapter base class                 | 2, 6             | Dashboard team review                   |

---

## 9. Integration Layer Special Section

### Unified ML Pipeline Status

| Phase | Name                     | Status         | Notes                               |
| ----- | ------------------------ | -------------- | ----------------------------------- |
| 0     | Signal Generation        | ⚠️ Partial     | Has logic, needs data validation    |
| 1     | Classical ML             | ⚠️ Partial     | Delegates to adapter                |
| 2-9   | Deep Learning (8 phases) | ❌ Placeholder | Returns `{'status': 'placeholder'}` |

### Utility Module Coverage

| Module                   | LOC  | Functions                          | Coverage         |
| ------------------------ | ---- | ---------------------------------- | ---------------- |
| `constants.py`           | 629  | 50+ constants, 5 utilities         | ✅ Comprehensive |
| `device_manager.py`      | 419  | 15+ functions, 1 context manager   | ✅ Comprehensive |
| `file_io.py`             | 520  | 15+ functions (pickle, JSON, YAML) | ✅ Comprehensive |
| `checkpoint_manager.py`  | 434  | Full class with top-k              | ✅ Comprehensive |
| `early_stopping.py`      | 375  | 2 classes (base + warmup)          | ✅ Comprehensive |
| `reproducibility.py`     | 140  | 4 functions                        | ✅ Comprehensive |
| `visualization_utils.py` | 350+ | 8+ plotting utilities              | ✅ Comprehensive |
| `timer.py`               | 350+ | Timer, Profiler, decorators        | ✅ Comprehensive |
| `logger.py`              | 50+  | `setup_logger()`                   | ⚠️ Duplicate     |
| `logging.py`             | 150+ | `get_logger()`, `setup_logging()`  | ⚠️ Duplicate     |

### Cross-Phase Data Formats

| Transition           | Format                          | Validation              |
| -------------------- | ------------------------------- | ----------------------- |
| Phase 0 → HDF5       | HDF5 with train/val/test splits | `DataPipelineValidator` |
| HDF5 → Phase 1       | NumPy arrays via h5py           | ✅ Works                |
| HDF5 → Deep Learning | PyTorch Tensors                 | ✅ Works                |
| Training → Registry  | SQLite + JSON metadata          | Partial validation      |

---

## 10. Consolidated Technical Debt Summary

| Category                | P0    | P1     | P2     | Total  | Est. Effort |
| ----------------------- | ----- | ------ | ------ | ------ | ----------- |
| Research Scripts (5.1)  | 2     | 4      | 4      | 10     | 16-24h      |
| Visualization (5.2)     | 4     | 4      | 3      | 11     | 12-18h      |
| Integration Layer (6.0) | 2     | 4      | 5      | 11     | 18-36h      |
| **Domain Total**        | **8** | **12** | **12** | **32** | **46-78h**  |

### Priority Matrix

```
         High Impact
              │
    P0-1,P0-2 │ P0-3,P0-4,P0-5
    (sys.path │ (style config,
     unified) │  DPI, colors)
              │
Low ──────────┼────────────── High
 Effort       │                Effort
              │
    P1 Quick  │ Strategic
    Fixes     │ Refactors
              │
         Low Impact
```

---

_Consolidated analysis generated from IDB 5.1, 5.2, and 6.0 reports_
