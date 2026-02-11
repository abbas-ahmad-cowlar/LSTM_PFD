# Domain 1: Core ML Engine — Consolidated Analysis

**Consolidation Date:** 2026-01-24  
**Source IDBs:** 1.1 Models, 1.2 Training, 1.3 Evaluation, 1.4 Features, 1.5 XAI  
**Total Files Analyzed:** 10 reports (~110 source Python files)

---

## 1. Domain Overview

### Purpose

The Core ML Engine is the central machine learning infrastructure for the LSTM_PFD bearing fault diagnosis system. It provides neural network architectures, training pipelines, evaluation frameworks, feature engineering utilities, and explainability (XAI) methods.

### Sub-Blocks

| IDB | Sub-Block  | Scope                                     | Independence Score |
| --- | ---------- | ----------------------------------------- | ------------------ |
| 1.1 | Models     | `packages/core/models/` (61 files)        | 9/10               |
| 1.2 | Training   | `packages/core/training/` (23 files)      | 7/10               |
| 1.3 | Evaluation | `packages/core/evaluation/` (16 files)    | 8/10               |
| 1.4 | Features   | `packages/core/features/` (12 files)      | 9/10               |
| 1.5 | XAI        | `packages/core/explainability/` (8 files) | 8/10               |

### Overall Independence Score: **8.2/10** _(Average)_

### Key Interfaces

| API                                            | Location                                 | Consumers              |
| ---------------------------------------------- | ---------------------------------------- | ---------------------- |
| `create_model(name, **kwargs)`                 | `models/model_factory.py`                | Training, Dashboard    |
| `BaseModel.forward(x)`                         | `models/base_model.py`                   | All trainers           |
| `Trainer.fit(epochs)`                          | `training/trainer.py`                    | Dashboard, Scripts     |
| `CNNEvaluator.evaluate(loader)`                | `evaluation/cnn_evaluator.py`            | Dashboard, Scripts     |
| `FeatureExtractor.extract_features(signal)`    | `features/feature_extractor.py`          | Classical ML pipelines |
| `IntegratedGradientsExplainer.explain(signal)` | `explainability/integrated_gradients.py` | Dashboard XAI tab      |

---

## 2. Current State Summary

### What's Implemented

**Models (IDB 1.1)**

- **55+ model architectures** across 13 subdirectories (CNN, ResNet, Transformer, EfficientNet, hybrid, PINN, ensemble, spectrogram)
- `BaseModel` abstract class with `forward()`, `get_config()`, `save_checkpoint()`, `freeze_backbone()`
- `model_factory` with registry pattern for model creation
- Preset configurations (`*_small`, `*_base`, `*_large`)

**Training (IDB 1.2)**

- **8 trainer classes** (Trainer, CNNTrainer, PINNTrainer, SpectrogramTrainer, etc.)
- **Two callback systems** (callbacks.py, cnn_callbacks.py) with EarlyStopping, ModelCheckpoint, TensorBoard, MLflow
- **11 loss functions** including FocalLoss, LabelSmoothing, SupConLoss, physics losses
- **11 schedulers** (cosine, warmup, Noam, OneCycle, etc.)
- Advanced techniques: mixed precision, progressive resizing, knowledge distillation, CutMix/Mixup

**Evaluation (IDB 1.3)**

- **5 specialized evaluators** (ModelEvaluator, CNNEvaluator, EnsembleEvaluator, PINNEvaluator, SpectrogramEvaluator)
- Metrics: accuracy, precision, recall, F1 (per-class and macro), ROC/AUC, ensemble diversity
- Visualization: confusion matrix, attention heatmaps, physics interpretability
- Architecture comparison with FLOPs/params/Pareto analysis

**Features (IDB 1.4)**

- **36 base features** across 5 domains (time, frequency, envelope, wavelet, bispectrum)
- **16 advanced features** optional (CWT, WPT, nonlinear dynamics, spectrogram)
- MRMR feature selection, variance threshold
- Normalization: z-score, min-max, robust (IQR-based)
- Comprehensive validation utilities (NaN/Inf detection, distribution checks)

**XAI (IDB 1.5)**

- **7 XAI methods**: Integrated Gradients, GradientSHAP, DeepSHAP, KernelSHAP, LIME, MC Dropout, CAVs/TCAV, Partial Dependence, Anchors
- **12 visualization functions** for attributions, uncertainty, TCAV results
- Scientific references for all methods
- Convergence validation for Integrated Gradients

### What's Working Well

| Strength                      | Sub-Blocks | Evidence                                                           |
| ----------------------------- | ---------- | ------------------------------------------------------------------ |
| **Abstract base patterns**    | 1.1, 1.3   | `BaseModel`, `ModelEvaluator` provide consistent interfaces        |
| **Factory pattern**           | 1.1        | `model_factory` with case-insensitive registry and aliases         |
| **Preset configurations**     | 1.1, 1.2   | `*_small`, `*_base`, `*_large` presets for quick setup             |
| **Scientific documentation**  | All        | Docstrings include formulas, references, units                     |
| **Numerical stability**       | 1.4, 1.5   | Division guards (`if x > 0 else 0.0`), epsilon for logs            |
| **Sklearn compatibility**     | 1.4        | `FeatureNormalizer` implements `BaseEstimator`, `TransformerMixin` |
| **Graceful library fallback** | 1.5        | SHAP library optional with native PyTorch fallback                 |
| **Domain-specific metrics**   | 1.3        | PINN physics-aware evaluation, ensemble diversity metrics          |

### What's Problematic

| Category                | Issue                                                                  | Sub-Blocks Affected |
| ----------------------- | ---------------------------------------------------------------------- | ------------------- |
| **Code Duplication**    | Trainer hierarchy fragmentation (5/8 trainers don't inherit from base) | 1.2                 |
| **Code Duplication**    | Two separate callback systems with overlapping functionality           | 1.2                 |
| **Code Duplication**    | `FocalLoss`, `LabelSmoothingCrossEntropy` implemented twice            | 1.2                 |
| **Code Duplication**    | 4/5 evaluators don't inherit from `ModelEvaluator`                     | 1.3                 |
| **Code Duplication**    | `PositionalEncoding` implemented 3x in transformer models              | 1.1                 |
| **Missing Features**    | `explanation_cache.py` does not exist despite expected                 | 1.5                 |
| **Hardcoded Values**    | `sys.path.append('/home/user/LSTM_PFD')` in 8+ files                   | 1.1, 1.2, 1.3       |
| **Hardcoded Values**    | Magic numbers: `102400`, `20480`, `512` across all sub-blocks          | All                 |
| **Empty Exports**       | `__init__.py` empty in Training, Evaluation packages                   | 1.2, 1.3            |
| **Incomplete Registry** | Only ~15 of ~55 models registered in `model_factory`                   | 1.1                 |
| **Scientific Issues**   | KernelSHAP uses plain LinearRegression, not SHAP kernel weighting      | 1.5                 |

---

## 3. Critical Issues Inventory

### P0 Issues (Critical - Production Blockers)

| IDB | Issue                                                                               | Impact                                                         | Effort | Dependencies            |
| --- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------- | ------ | ----------------------- |
| 1.1 | **Hardcoded `sys.path` manipulation** (`/home/user/LSTM_PFD`)                       | Breaks on any other machine                                    | 2h     | None                    |
| 1.1 | **Duplicate HybridPINN implementations** (`hybrid_pinn.py` + `pinn/hybrid_pinn.py`) | Import confusion, maintenance burden                           | 4h     | Update all imports      |
| 1.1 | **Duplicate ResNet1D implementations** (`resnet_1d.py` + `resnet/resnet_1d.py`)     | Same class defined twice                                       | 4h     | Update all imports      |
| 1.2 | **Trainer hierarchy fragmentation** — 5/8 trainers don't inherit from base          | Bug fixes require 5x effort                                    | 2d     | Regression testing      |
| 1.2 | **Duplicate callback systems** (`callbacks.py` + `cnn_callbacks.py`)                | Inconsistent interfaces, maintenance nightmare                 | 1d     | Update all trainers     |
| 1.2 | **Duplicate loss functions** (`FocalLoss`, `LabelSmoothing` in two files)           | Confusion, potential drift between implementations             | 4h     | None                    |
| 1.2 | **Empty `__init__.py`** in training package                                         | No public API, import confusion                                | 30m    | None                    |
| 1.3 | **Empty `__init__.py`** in evaluation package                                       | No module exports, requires full paths                         | 30m    | None                    |
| 1.3 | **Inconsistent evaluator inheritance** — 3/5 evaluators don't inherit from base     | 500+ lines of duplicated code                                  | 2d     | Regression testing      |
| 1.5 | **Missing `explanation_cache.py`**                                                  | No caching for expensive XAI computations                      | 1d     | Design caching strategy |
| 1.5 | **KernelSHAP incorrect weighting** — uses plain `LinearRegression()`                | Violates SHAP's theoretical guarantees, incorrect attributions | 4h     | None                    |

### P1 Issues (High Priority)

| IDB | Issue                                                                               | Impact                                 | Effort | Dependencies         |
| --- | ----------------------------------------------------------------------------------- | -------------------------------------- | ------ | -------------------- |
| 1.1 | **Hardcoded magic numbers** (`102400`, `20480`, `5000`) in 45+ occurrences          | Violates DRY, hard to maintain         | 4h     | `utils/constants.py` |
| 1.1 | **Classical models don't inherit BaseModel**                                        | Inconsistent interface                 | 4h     | ClassicalML wrapper  |
| 1.1 | **TSMixer, PatchTST don't inherit BaseModel**                                       | Missing standard interface methods     | 2h     | None                 |
| 1.1 | **No ONNX export method in BaseModel**                                              | Missing standardized export capability | 1d     | None                 |
| 1.1 | **model_factory registry incomplete** — only ~15 of ~55 models                      | Many models not discoverable           | 4h     | None                 |
| 1.2 | **Mixed precision implemented 3x** (trainer.py, cnn_trainer.py, mixed_precision.py) | Inconsistent behavior                  | 4h     | Unify mixins         |
| 1.2 | **LR scheduler step() called incorrectly** for ReduceLROnPlateau                    | Silent failure of LR reduction         | 1h     | None                 |
| 1.2 | **No reproducibility enforcement** — trainers don't set seeds                       | Non-reproducible experiments           | 2h     | None                 |
| 1.2 | **Checkpoint format inconsistency** — each trainer saves different structures       | Evaluation can't reliably load         | 4h     | Standardize schema   |
| 1.3 | **Memory issue in ErrorAnalyzer** — stores full signals for misclassified samples   | OOM on large test sets                 | 2h     | Store indices only   |
| 1.3 | **No standard output schema** across evaluators                                     | Hard to compare results                | 4h     | Define dataclass     |
| 1.4 | **Hardcoded `fs=20480`** in multiple files                                          | Violates DRY                           | 2h     | Use constants        |
| 1.4 | **Feature count mismatch** — docs say 52, implementation has 36                     | Confusion                              | 1h     | Update docs          |
| 1.5 | **MC Dropout mode not restored on exception**                                       | Model left in training mode            | 30m    | Add try/finally      |
| 1.5 | **CAV gradient hook memory leak risk**                                              | Memory growth on large N               | 1h     | Use context manager  |
| 1.5 | **GradientSHAP baseline handling** — uses mean instead of per-sample                | Attribution accuracy reduced           | 2h     | None                 |
| 1.5 | **LIME segment boundary edge case** — last segment larger than others               | Importance bias                        | 1h     | None                 |

### P2 Issues (Medium Priority)

| IDB | Issue                                                   | Impact                        | Effort | Dependencies           |
| --- | ------------------------------------------------------- | ----------------------------- | ------ | ---------------------- |
| 1.1 | Duplicate `ConvBlock` implementations                   | DRY violation                 | 1h     | Consolidate            |
| 1.1 | Missing docstrings in helper modules                    | Reduced maintainability       | 2h     | None                   |
| 1.1 | Test code embedded in module files (`if __name__`)      | Should be separate            | 3h     | None                   |
| 1.1 | NAS search space has no actual NAS algorithm            | Incomplete feature            | 3-5d   | Research               |
| 1.2 | Test code in production files                           | Pattern violation             | 2h     | None                   |
| 1.2 | HPO only for classical ML, not deep learning            | Incomplete                    | 1d     | Extend factory         |
| 1.2 | PINNTrainer has `train()` not `fit()`                   | Inconsistent API              | 1h     | None                   |
| 1.3 | Incomplete `RobustnessTester` — missing temporal drift  | Partial feature               | 1d     | None                   |
| 1.3 | Missing ROC plotting in `roc_analyzer.py`               | Incomplete analysis           | 2h     | None                   |
| 1.3 | Simplified FLOPs estimation                             | Inaccurate complexity metrics | 2h     | Use thop/fvcore        |
| 1.4 | No parallel batch extraction                            | Slow processing               | 4h     | Add joblib             |
| 1.4 | SHAP claimed in features block but not implemented      | Misleading docs               | 1h     | Fix docs               |
| 1.4 | Sample entropy O(N²) very slow                          | Performance                   | 4h     | Downsample             |
| 1.5 | `sys.path` manipulation in all XAI files                | Import hygiene                | 2h     | Use relative imports   |
| 1.5 | Hardcoded parameters (n_segments=20, num_samples=1000)  | Not configurable              | 2h     | Add parameters         |
| 1.5 | Visualization uses `Reds` colormap for +/- attributions | Misleading                    | 1h     | Use diverging colormap |

---

## 4. Technical Debt Registry

### Quick Wins (< 1 hour)

| IDB | Task                                                              | Benefit              |
| --- | ----------------------------------------------------------------- | -------------------- |
| 1.1 | Replace hardcoded `sys.path` paths in 8 files                     | Portability          |
| 1.1 | Add `__all__` to missing `__init__.py` files (`hybrid/`, `pinn/`) | Clean imports        |
| 1.1 | Standardize `Dict` vs `dict` type hints                           | Consistency          |
| 1.2 | Populate `__init__.py` with public API exports                    | Discoverability      |
| 1.2 | Fix LR scheduler step for ReduceLROnPlateau                       | Correct behavior     |
| 1.3 | Populate `__init__.py` with standard exports                      | Clean imports        |
| 1.4 | Fix documentation — 36 vs 52 feature count                        | Accuracy             |
| 1.5 | Add try/finally to MC Dropout mode restoration                    | Model state safety   |
| 1.5 | Fix LIME segment boundary edge case                               | Attribution accuracy |

### Medium Tasks (1-4 hours)

| IDB | Task                                                         | Benefit                |
| --- | ------------------------------------------------------------ | ---------------------- |
| 1.1 | Make TSMixer, PatchTST inherit BaseModel                     | Interface consistency  |
| 1.1 | Consolidate duplicate PositionalEncoding (3 implementations) | DRY                    |
| 1.1 | Move test code to `tests/` directory (10+ files)             | Clean separation       |
| 1.1 | Replace magic numbers with constants (45+ occurrences)       | Maintainability        |
| 1.1 | Add missing models to model_factory registry                 | Discoverability        |
| 1.2 | Consolidate duplicate loss functions                         | Single source of truth |
| 1.3 | Fix memory issue in ErrorAnalyzer — store indices only       | Large-scale analysis   |
| 1.3 | Add ROC plotting function                                    | Complete analysis      |
| 1.4 | Replace hardcoded `fs=20480` with constant imports           | Maintainability        |
| 1.4 | Add input validation in `extract_features()`                 | Robustness             |
| 1.5 | Fix KernelSHAP weighting — add SHAP kernel                   | Scientific correctness |
| 1.5 | Add input validation across all XAI methods                  | Error prevention       |

### Large Refactors (1+ days)

| IDB | Task                                                          | Est. | Benefit                |
| --- | ------------------------------------------------------------- | ---- | ---------------------- |
| 1.1 | Remove duplicate model files, keep subdirectory versions only | 1d   | Single source of truth |
| 1.1 | Add ONNX export to BaseModel + ONNX tests for all models      | 2d   | Deployment readiness   |
| 1.1 | Implement decorator-based auto-registration for factory       | 1d   | Automatic discovery    |
| 1.1 | Create comprehensive config system (pydantic)                 | 2d   | Type-safe configs      |
| 1.1 | Refactor to backbone/head separation pattern                  | 3d   | Composability          |
| 1.2 | Create unified BaseTrainer with mixin architecture            | 2d   | Eliminate duplication  |
| 1.2 | Merge callback systems into single implementation             | 1d   | Maintenance simplicity |
| 1.2 | Standardize checkpoint format with version field              | 1d   | Forward compatibility  |
| 1.2 | Implement distributed training (DDP)                          | 2d   | Scalability            |
| 1.3 | Refactor evaluators to inherit from common base               | 2d   | 500+ lines removed     |
| 1.3 | Create `EvaluationResult` dataclass for consistent output     | 1d   | Cross-evaluator compat |
| 1.3 | Add lazy metric computation with caching                      | 2d   | Performance            |
| 1.4 | Add parallel batch extraction with joblib                     | 4h   | Performance            |
| 1.4 | Implement feature versioning                                  | 1d   | Tracking               |
| 1.4 | Add streaming support with windowed extraction                | 2d   | Real-time use          |
| 1.5 | Implement full `explanation_cache.py` with disk persistence   | 1d   | Performance            |
| 1.5 | Create unified `ExplainerProtocol` interface                  | 1d   | Consistency            |
| 1.5 | Add proper unit tests (not just `__main__` blocks)            | 2d   | Quality                |

---

## 5. "If We Could Rewrite" Insights

### Cross-Sub-Block Themes

1. **Inheritance Fragmentation**: Both Training (5/8 trainers) and Evaluation (3/5 evaluators) have specialized classes that don't inherit from the base class, duplicating hundreds of lines of code.

2. **Duplicate Implementations**: Pattern appears across all sub-blocks:
   - Models: `HybridPINN`, `ResNet1D` defined twice
   - Training: `FocalLoss`, `LabelSmoothing`, callbacks defined twice
   - Evaluation: Base evaluation loop repeated 5x
   - XAI: Similar patterns in explainer interfaces

3. **Empty/Missing `__init__.py`**: Training and Evaluation packages lack proper exports, making imports awkward and API unclear.

4. **Magic Numbers**: Hardcoded constants (`102400`, `20480`, `5000`) scattered across all sub-blocks instead of using centralized `utils/constants.py`.

5. **`sys.path` Manipulation**: Anti-pattern of appending absolute paths appears in Models, Training, and XAI.

### Fundamental Architectural Changes

1. **Unified Trainer with Plugin Architecture**

   ```python
   class UnifiedTrainer:
       def __init__(self, model, config, plugins=None, callbacks=None):
           self.plugins = PluginManager([
               MixedPrecisionPlugin(),
               GradientClippingPlugin(),
               PhysicsLossPlugin(),  # Optional for PINN
           ])
   ```

2. **Single BaseEvaluator with Abstract Hooks**

   ```python
   class BaseEvaluator:
       def evaluate(self, dataloader) -> EvaluationResult:
           self.model.eval()
           with torch.no_grad():
               for batch in dataloader:
                   self._process_batch(batch)  # Abstract
           return self._compute_metrics()      # Abstract
   ```

3. **Decorator-Based Model Registration**

   ```python
   @register_model("attention_cnn")
   class AttentionCNN1D(BaseModel):
       ...
   ```

4. **Unified Explainer Protocol**

   ```python
   class ExplainerProtocol(Protocol):
       def explain(self, input: Tensor, target_class: int) -> ExplanationResult: ...
       def visualize(self, explanation: ExplanationResult) -> Figure: ...
   ```

5. **Structured Checkpoint Format with Versioning**
   ```python
   checkpoint = {
       'version': '2.0',
       'model': {'state_dict': ..., 'architecture': ..., 'config': ...},
       'training': {'epoch': ..., 'best_metric': ..., 'history': ...},
       'random_state': {'torch': ..., 'numpy': ..., 'python': ...},
   }
   ```

### Patterns to Preserve

| Pattern                                     | Sub-Block | Location                      | Why Keep                |
| ------------------------------------------- | --------- | ----------------------------- | ----------------------- |
| Factory registry with aliases               | 1.1       | `model_factory.py`            | Flexible model creation |
| Preset configurations (`*_small`, `*_base`) | 1.1       | Various model files           | Quick setup             |
| `get_config()` returning dict               | 1.1       | `BaseModel`                   | Checkpoint metadata     |
| He/Kaiming weight initialization            | 1.1       | `_initialize_weights()`       | Proper init for ReLU    |
| Division guards (`if x > 0 else 0.0`)       | 1.4       | Feature extractors            | Numerical stability     |
| Epsilon for log operations                  | 1.4       | `compute_spectral_entropy()`  | Prevents log(0)         |
| Sklearn compatibility (`fit`, `transform`)  | 1.4       | `FeatureNormalizer`           | Pipeline integration    |
| Graceful library fallback                   | 1.5       | `SHAPExplainer`               | Optional dependencies   |
| Convergence validation                      | 1.5       | `compute_convergence_delta()` | Attribution quality     |
| Scientific references in docstrings         | 1.5       | All XAI methods               | Academic credibility    |

### Patterns to Eliminate

| Anti-Pattern                        | Sub-Blocks    | Replacement                       |
| ----------------------------------- | ------------- | --------------------------------- |
| `sys.path.append('/home/user/...')` | 1.1, 1.2, 1.5 | Proper package structure          |
| Hardcoded magic numbers             | All           | `from utils.constants import ...` |
| Test code in `if __name__` blocks   | 1.1, 1.2, 1.3 | `tests/` directory with pytest    |
| Duplicate implementations           | 1.1, 1.2, 1.3 | Single source + imports           |
| Empty `__init__.py`                 | 1.2, 1.3      | Explicit `__all__` exports        |
| Class per file without inheritance  | 1.2, 1.3      | Proper class hierarchies          |

---

## 6. Best Practices Observed

### Code Conventions

- **Naming:**
  - Model classes: `PascalCase` (e.g., `CNN1D`, `VisionTransformer1D`)
  - Factory functions: `snake_case` with `create_` prefix (e.g., `create_cnn1d`)
  - Features: `PascalCase` matching scientific literature (e.g., `SpectralCentroid`, `WaveletEnergy_D1`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `NUM_CLASSES`, `SIGNAL_LENGTH`)

- **Imports:** Standard library → Third-party → Project absolute → Local relative

- **Docstrings:** Google style with Args/Returns/Example sections, formulas for derived features

- **Type Hints:** Function signatures fully typed, return types specified

### Design Patterns Worth Preserving

1. **Abstract Base Class Pattern** (IDB 1.1)

   ```python
   class BaseModel(nn.Module, ABC):
       @abstractmethod
       def forward(self, x: torch.Tensor) -> torch.Tensor: ...
       @abstractmethod
       def get_config(self) -> Dict[str, Any]: ...
   ```

2. **Factory Pattern with Registry** (IDB 1.1)

   ```python
   MODEL_REGISTRY = {'cnn1d': create_cnn1d, 'resnet18': create_resnet18_1d}
   def create_model(name: str, **kwargs) -> BaseModel:
       return MODEL_REGISTRY[name.lower()](**kwargs)
   ```

3. **Preset Configurations** (IDB 1.1)

   ```python
   def cnn_transformer_small(num_classes=NUM_CLASSES, **kwargs):
       return create_cnn_transformer_hybrid(d_model=256, num_heads=4, **kwargs)
   ```

4. **Division Guard Pattern** (IDB 1.4)

   ```python
   return peak / rms if rms > 0 else 0.0
   ```

5. **Sklearn Compatibility** (IDB 1.4)

   ```python
   class FeatureNormalizer(BaseEstimator, TransformerMixin):
       def fit(self, X, y=None): return self
       def transform(self, X): return self.scaler.transform(X)
   ```

6. **Graceful Library Fallback** (IDB 1.5)
   ```python
   try:
       import shap
       self.shap_available = True
   except ImportError:
       warnings.warn("SHAP not installed. Using native implementation.")
       self.shap_available = False
   ```

### Testing Patterns

- **In-module testing:** Currently uses `if __name__ == '__main__':` blocks for quick verification
- **Recommended migration:** Move to `tests/` with pytest fixtures
- **Coverage requirements:**
  - Instantiation with default args
  - Forward pass shape verification
  - Gradient flow (`loss.backward()`)
  - Config retrieval (`get_config()`)
  - Device transfer (`model.to('cuda')`)
  - Checkpoint save/load consistency

### Interface Contracts

**Model Interface:**
| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `forward(x)` | `[B, C, T]` tensor | `[B, num_classes]` logits | NOT softmax |
| `get_config()` | None | `Dict` with `model_type`, `num_classes`, `num_parameters` | Required |
| `count_parameters()` | None | `Dict` with total/trainable/non-trainable | Inherited |

**Trainer Interface:**
| Method | Signature | Notes |
|--------|-----------|-------|
| `fit(num_epochs)` | `-> Dict[str, List[float]]` | Returns history |
| `train_epoch()` | `-> Dict[str, float]` | Returns `train_loss`, `train_acc` |
| `validate_epoch()` | `-> Dict[str, float]` | Returns `val_loss`, `val_acc`, `lr` |

**Evaluator Interface:**
| Method | Signature | Notes |
|--------|-----------|-------|
| `evaluate(dataloader)` | `-> Dict` | Returns `accuracy`, `confusion_matrix`, `per_class_metrics` |
| `predict(inputs)` | `-> np.ndarray` | Returns predicted labels |
| `save_results(path)` | `-> None` | Saves JSON with numpy→list conversion |

---

## 7. Cross-Domain Dependencies

### Inbound Dependencies (What Core ML Needs)

- **From Data Domain (IDB 3.x):**
  - `DataLoaders`, `Datasets` for training and evaluation
  - Signal preprocessing utilities
  - Data augmentation pipelines

- **From Infrastructure (IDB 4.x):**
  - `TrainingConfig`, `EvaluationConfig` from configuration system
  - Database for checkpoint storage (Dashboard)
  - Logging infrastructure

### Outbound Dependencies (What Depends on Core ML)

- **Dashboard (IDB 2.x) consumes:**
  - `model_factory.list_available_models()` for UI dropdowns
  - `Trainer.history` for training progress plots
  - `Trainer.best_val_acc`, `Trainer.current_epoch` for status display
  - `CNNEvaluator.evaluate()` for test results
  - `IntegratedGradientsExplainer.explain()` for XAI visualizations

- **Research Scripts (IDB 5.x) consume:**
  - All Core ML modules directly
  - Experiment configs and hyperparameter search

### Integration Risks

| Risk                                       | Location                   | Impact                         | Mitigation                     |
| ------------------------------------------ | -------------------------- | ------------------------------ | ------------------------------ |
| Checkpoint format changes                  | Training ↔ Evaluation      | Saved models unloadable        | Version field + migration      |
| History key changes                        | Training ↔ Dashboard       | Plots break                    | Document required keys         |
| `forward()` signature changes              | Models ↔ Training          | Training loop crashes          | Abstract interface enforcement |
| Different percentage scales (0-1 vs 0-100) | Evaluation ↔ Dashboard     | Incorrect displays             | Standardize on 0-100           |
| PINN dual-input models                     | Models ↔ Standard trainers | Standard trainers can't handle | Document exceptions            |

---

## 8. Domain-Specific Recommendations

### Top 3 Quick Wins

1. **Populate `__init__.py` in Training and Evaluation packages** (1 hour total)
   - Export public APIs explicitly
   - Add `__all__` lists
   - Immediate benefit: Clean imports, API clarity

2. **Fix LR scheduler step for ReduceLROnPlateau** (30 minutes)

   ```python
   if isinstance(scheduler, ReduceLROnPlateau):
       scheduler.step(val_metrics['loss'])
   else:
       scheduler.step()
   ```

   - Fixes silent learning rate reduction failure

3. **Remove hardcoded `sys.path` manipulations** (2 hours)
   - Replace with proper relative imports
   - Makes codebase portable across machines

### Top 3 Strategic Improvements

1. **Unify Trainer Hierarchy with Plugin Architecture** (2-3 days)
   - Create `BaseTrainer` with template method pattern
   - Implement mixins: `MixedPrecisionMixin`, `PhysicsLossMixin`, `GradientAccumulationMixin`
   - All specialized trainers inherit from `BaseTrainer`
   - **Impact:** Eliminates 1000+ lines of duplicate code, centralizes bug fixes

2. **Merge Callback and Loss Systems** (1-2 days)
   - Single canonical location for callbacks (`training/callbacks/`)
   - Single canonical location for losses (`training/losses/`)
   - Deprecate duplicate implementations
   - **Impact:** Single source of truth, easier maintenance

3. **Implement Explanation Cache** (1-2 days)
   - Create `ExplanationCache` with three-tier hierarchy (results, intermediate, model)
   - Cache expensive computations (IG, SHAP, LIME)
   - Invalidate on model weight changes
   - **Impact:** 10x speedup for repeated XAI queries in dashboard

### Team Coordination Requirements

| Change                   | Teams to Coordinate             | Action Required                     |
| ------------------------ | ------------------------------- | ----------------------------------- |
| Checkpoint format change | Training, Evaluation, Dashboard | Versioned migration, update loaders |
| History key changes      | Training, Dashboard             | Update plotting code                |
| New model architecture   | Training, Dashboard             | Add to registry, update UI          |
| Evaluator output schema  | Evaluation, Dashboard           | Update display components           |
| XAI interface changes    | XAI, Dashboard                  | Update XAI tab callbacks            |

---

## Appendix: Source Reports

| Report                            | Location                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------ |
| IDB 1.1 Models Analysis           | [IDB_1_1_MODELS_ANALYSIS.md](./IDB_1_1_MODELS_ANALYSIS.md)                     |
| IDB 1.1 Models Best Practices     | [IDB_1_1_MODELS_BEST_PRACTICES.md](./IDB_1_1_MODELS_BEST_PRACTICES.md)         |
| IDB 1.2 Training Analysis         | [IDB_1_2_TRAINING_ANALYSIS.md](./IDB_1_2_TRAINING_ANALYSIS.md)                 |
| IDB 1.2 Training Best Practices   | [IDB_1_2_TRAINING_BEST_PRACTICES.md](./IDB_1_2_TRAINING_BEST_PRACTICES.md)     |
| IDB 1.3 Evaluation Analysis       | [IDB_1_3_EVALUATION_ANALYSIS.md](./IDB_1_3_EVALUATION_ANALYSIS.md)             |
| IDB 1.3 Evaluation Best Practices | [IDB_1_3_EVALUATION_BEST_PRACTICES.md](./IDB_1_3_EVALUATION_BEST_PRACTICES.md) |
| IDB 1.4 Features Analysis         | [IDB_1_4_FEATURES_ANALYSIS.md](./IDB_1_4_FEATURES_ANALYSIS.md)                 |
| IDB 1.4 Features Best Practices   | [IDB_1_4_FEATURES_BEST_PRACTICES.md](./IDB_1_4_FEATURES_BEST_PRACTICES.md)     |
| IDB 1.5 XAI Analysis              | [IDB_1_5_XAI_ANALYSIS.md](./IDB_1_5_XAI_ANALYSIS.md)                           |
| IDB 1.5 XAI Best Practices        | [IDB_1_5_XAI_BEST_PRACTICES.md](./IDB_1_5_XAI_BEST_PRACTICES.md)               |

---

_Consolidated by Domain 1 Consolidation Agent — 2026-01-24_
