# IDB 1.1 Models Sub-Block — Comprehensive Audit Report (2026-03)

> **Scope**: `packages/core/models/` — All neural network architectures, classical ML wrappers, physics models, factory, and registry.
>
> **Supersedes**: `IDB_1_1_MODELS_ANALYSIS.md` and `IDB_1_1_MODELS_BEST_PRACTICES.md`

---

## Executive Summary

The Models sub-block has **~55 model classes across 65+ files in 12 subdirectories**. Since the last analysis, several P0 issues have been partially resolved (duplicate `HybridPINN`/`ResNet1D` files converted to re-export shims). However, **8 files still contain `sys.path` hacks**, **7+ model classes still don't inherit `BaseModel`**, and the factory registry is incomplete. Embedded test code remains in **27 files**.

| Severity | Count | Status |
|----------|-------|--------|
| **P0 – Critical** | 4 issues | 1 partially fixed, 3 open |
| **P1 – High** | 6 issues | All open |
| **P2 – Medium** | 7 issues | All open |

---

## P0 — Critical Issues (Blocks portability / correctness)

### P0-1: `sys.path` Hacks (8 files)

Hardcoded `sys.path.insert()` breaks portability across environments and CI/CD.

| File | Line | Pattern |
|------|------|---------|
| `transformer/vision_transformer_1d.py` | L24 | `sys.path.insert(0, os.path.dirname(...))` |
| `hybrid/cnn_transformer.py` | L22 | `sys.path.insert(0, os.path.dirname(...))` |
| `fusion/early_fusion.py` | L29 | `sys.path.insert(0, str(Path(...).parent...))` |
| `fusion/late_fusion.py` | L24 | `sys.path.insert(0, str(Path(...).parent...))` |
| `ensemble/voting_ensemble.py` | L23 | `sys.path.insert(0, str(Path(...).parent...))` |
| `ensemble/stacking_ensemble.py` | L21 | `sys.path.insert(0, str(Path(...).parent...))` |
| `ensemble/boosting_ensemble.py` | L21 | `sys.path.insert(0, str(Path(...).parent...))` |
| `ensemble/mixture_of_experts.py` | L27 | `sys.path.insert(0, str(Path(...).parent...))` |

**Fix**: Replace `sys.path` hack + absolute import with relative imports (e.g., `from ..base_model import BaseModel`).

### P0-2: Duplicate `HybridPINN` (PARTIALLY FIXED)

Top-level `hybrid_pinn.py` is now a re-export shim that delegates to `pinn/hybrid_pinn.py`. However, it uses absolute import `from packages.core.models.pinn.hybrid_pinn import ...` which couples to project layout. Should use relative import.

### P0-3: Duplicate `ResNet1D` (PARTIALLY FIXED)

Top-level `resnet_1d.py` is now a re-export shim. Same absolute-import concern as P0-2.

### P0-4: Duplicate `CNN1D` (NEW — NOT FIXED)

Two completely different `CNN1D` classes exist:

| Location | Architecture | Lines |
|----------|-------------|-------|
| `models/cnn_1d.py` (top-level) | 6-conv, kernel 3-7, 256 channels | 232 lines |
| `models/cnn/cnn_1d.py` (subdirectory) | 5-conv blocks, kernel 4-64, 512 channels | 313 lines |

The top-level `CNN1D` is imported by `__init__.py` and `model_factory.py`. The `cnn/cnn_1d.py` version (which uses `ConvBlock1D` from `conv_blocks.py`) is the intended canonical version but is only used by `test_all_models.py` and `pinn/hybrid_pinn.py`. **One must be deprecated**.

---

## P1 — High Impact Issues

### P1-1: Models Not Inheriting `BaseModel`

The following model classes inherit from `nn.Module` directly, breaking the unified interface contract (no `get_config()`, `count_parameters()`, `save_checkpoint()`, etc.):

| Class | File | Should Inherit |
|-------|------|----------------|
| `PatchTST` | `transformer/patchtst.py` | `BaseModel` |
| `TSMixer` | `transformer/tsmixer.py` | `BaseModel` |
| `AttentionCNN1D` | `cnn/attention_cnn.py` | `BaseModel` |
| `LightweightAttentionCNN` | `cnn/attention_cnn.py` | `BaseModel` |
| `MultiScaleCNN1D` | `cnn/multi_scale_cnn.py` | `BaseModel` |
| `DilatedMultiScaleCNN` | `cnn/multi_scale_cnn.py` | `BaseModel` |
| `SignalEncoder` | `contrastive/signal_encoder.py` | `BaseModel` |
| `ContrastiveClassifier` | `contrastive/classifier.py` | `BaseModel` |

### P1-2: Hardcoded Magic Numbers

- `PatchTST`: `num_classes=11`, `input_length=102400` instead of `NUM_CLASSES`, `SIGNAL_LENGTH`
- `TSMixer`: Same hardcoded defaults
- Various test blocks assert `== (4, 11)` instead of `== (4, NUM_CLASSES)`

### P1-3: Incomplete Model Factory Registry

Models present in codebase but **missing from `MODEL_REGISTRY`** in `model_factory.py`:

| Missing Model | File |
|---------------|------|
| `AttentionCNN1D` | `cnn/attention_cnn.py` |
| `LightweightAttentionCNN` | `cnn/attention_cnn.py` |
| `MultiScaleCNN1D` | `cnn/multi_scale_cnn.py` |
| `DilatedMultiScaleCNN` | `cnn/multi_scale_cnn.py` |
| `MultiScaleCNN` (hybrid) | `hybrid/multiscale_cnn.py` |
| `CNNLSTM` | `hybrid/cnn_lstm.py` |
| `CNNTCN` | `hybrid/cnn_tcn.py` |
| `SEResNet1D` | `resnet/se_resnet.py` |
| `WideResNet1D` | `resnet/wide_resnet.py` |
| `ResNet2DSpectrogram` variants | `spectrogram_cnn/resnet2d_spectrogram.py` |
| `EfficientNet2DSpectrogram` variants | `spectrogram_cnn/efficientnet2d_spectrogram.py` |

### P1-4: Existing Tests Have Commented-Out Models

In `test_all_models.py`:
- `VisionTransformer1D` is commented out (likely fails due to `sys.path` issue)
- `PhysicsConstrainedCNN` is commented out
- Several models have incorrect config (e.g., `ResNet1D` is instantiated with `base_filters` and `n_blocks` which aren't valid kwargs)

### P1-5: `HybridPINN.forward()` Signature Deviation

`HybridPINN.forward(signal, metadata)` takes a dict `metadata` as second arg, breaking the standard `forward(x)` contract. This means the factory's `create_model → model(x)` workflow won't work correctly with `HybridPINN` — it will use default metadata with a warning.

### P1-6: Re-export Shims Use Absolute Imports

`hybrid_pinn.py` and `resnet_1d.py` top-level shims use `from packages.core.models.pinn.hybrid_pinn import ...` instead of relative imports. This couples to the top-level package layout and breaks if the project is installed or invoked differently.

---

## P2 — Medium Impact Issues

### P2-1: 3× Duplicate `PositionalEncoding`

| File | Class Name | Variant |
|------|-----------|---------|
| `transformer/signal_transformer.py` | `PositionalEncoding` | Learnable + sinusoidal |
| `transformer/patchtst.py` | `PositionalEncoding` | Sinusoidal only |
| `transformer/vision_transformer_1d.py` | `PositionalEncoding1D` | Learnable only |

**Fix**: Extract to a shared module (e.g., `transformer/positional_encoding.py`) and import everywhere.

### P2-2: Embedded Test Code in 27 Files

Every model file contains `if __name__ == '__main__': test_xxx()` blocks. These should be migrated to the `tests/` directory using pytest fixtures.

### P2-3: `legacy_ensemble.py` Still Present

386 lines of legacy ensemble code (`VotingEnsemble`, `StackedEnsemble`, `EnsembleModel`) coexist with newer implementations in `ensemble/voting_ensemble.py`, `ensemble/stacking_ensemble.py`, etc. The `__init__.py` imports both with `V2` suffixes for new versions. This creates confusion.

### P2-4: `ConvBlock` Duplication

- `models/cnn_1d.py` defines its own `ConvBlock` (lightweight)
- `models/cnn/conv_blocks.py` defines `ConvBlock1D` (full-featured with pooling, activation choices)

### P2-5: Inconsistent Import Styles

Mix of relative imports (`from ..base_model import BaseModel`) and absolute imports (`from packages.core.models.base_model import BaseModel`). The codebase should standardize on one approach.

### P2-6: Missing `__init__.py` Exports

Many subdirectory models are not exported through their `__init__.py` or the top-level `__init__.py`, making them invisible to users who import from `packages.core.models`.

### P2-7: Classical ML Models Outside BaseModel Contract

Classical models (`RandomForestClassifier`, `SVMClassifier`, `GradientBoostingClassifier`, etc.) have their own interface (`train/predict/save/load`) that doesn't align with `BaseModel`. While this is somewhat acceptable (sklearn vs PyTorch), the `model_selector.py` in `classical/` duplicates functionality that could be handled by the model factory.

---

## What's Working Well ✅

1. **`BaseModel` abstract class**: Well-designed with `forward()`, `get_config()`, `count_parameters()`, `save_checkpoint()`, `load_checkpoint()`, and utility methods.
2. **Model Factory**: `model_factory.py` has a comprehensive registry with ~50 entries and supports `create_model()`, `create_model_from_config()`, `load_pretrained()`, `register_model()`.
3. **Canonical model implementations**: `ResNet1D`, `SignalTransformer`, `VisionTransformer1D`, `CNNTransformerHybrid`, `HybridPINN`, `EfficientNet1D` all properly inherit `BaseModel` and implement `get_config()`.
4. **Weight initialization**: Most models use He/Kaiming or Xavier initialization consistently.
5. **2D input handling**: Most models handle both `[B, T]` and `[B, 1, T]` input gracefully.
6. **Preset configurations**: ViT, CNN-Transformer, and EfficientNet have factory functions for common sizes (tiny/small/base/large/b0-b7).
7. **Formal test suite**: `tests/test_models.py`, `test_all_models.py`, `test_classical_models.py` exist with good coverage for core models.
8. **Physics integration**: `BearingDynamics` provides differentiable physics computations usable in both numpy and torch.

---

## Recommended Fix Priority Order

### Phase 1 — P0 Fixes (Immediate, blocks CI/CD)

1. **Remove all 8 `sys.path` hacks** — replace with relative imports
2. **Convert shim absolute imports to relative** — `hybrid_pinn.py`, `resnet_1d.py`
3. **Resolve `CNN1D` duplication** — deprecate `models/cnn_1d.py` in favor of `cnn/cnn_1d.py`, update `model_factory.py` and `__init__.py` to point to canonical version

### Phase 2 — P1 Fixes (High impact, interface consistency)

4. **Make all models inherit `BaseModel`** — `PatchTST`, `TSMixer`, `AttentionCNN1D`, etc.
5. **Replace hardcoded `11` / `102400`** with `NUM_CLASSES` / `SIGNAL_LENGTH` constants
6. **Complete model factory registry** — register all missing models
7. **Fix test suite** — uncomment `VisionTransformer1D`, fix `ResNet1D` config, add tests for new models

### Phase 3 — P2 Cleanup (Quality / maintainability)

8. **Extract shared `PositionalEncoding`** to common module
9. **Remove embedded test code** from all 27 model files
10. **Deprecate `legacy_ensemble.py`** — migrate consumers to `ensemble/` versions
11. **Standardize import style** — all relative imports within `packages/core/models/`
12. **Update `__init__.py` exports** for all subdirectories

---

## File Inventory (65 Python files)

| Subdirectory | Files | Models | BaseModel? | In Registry? |
|-------------|-------|--------|------------|-------------|
| Top-level | 5 | CNN1D, (shims) | CNN1D: ✅ | ✅ |
| `cnn/` | 4 | CNN1D, AttentionCNN1D, LightweightAttentionCNN, MultiScaleCNN1D, DilatedMultiScaleCNN | CNN1D: ✅, rest: ❌ | CNN1D: ❌ (shadow), rest: ❌ |
| `transformer/` | 4 | SignalTransformer, VisionTransformer1D, PatchTST, TSMixer | ST: ✅, ViT: ✅, rest: ❌ | ✅ |
| `resnet/` | 4 | ResNet1D, SEResNet1D, WideResNet1D | ✅ | ResNet1D: ✅, rest: ❌ |
| `ensemble/` | 5 | VotingEnsembleV2, StackingEnsembleV2, BoostingEnsemble, MixtureOfExperts, ModelSelector | ✅ | ✅ (via `__init__.py`) |
| `hybrid/` | 4 | CNNTransformerHybrid, CNNLSTM, CNNTCN, MultiScaleCNN | ✅ | CNNTransformerHybrid: ✅, rest: ❌ |
| `pinn/` | 4 | HybridPINN, PhysicsConstrainedCNN, MultitaskPINN, KnowledgeGraphPINN | ✅ | ✅ |
| `fusion/` | 2 | EarlyFusion, LateFusion | ✅ | ✅ |
| `efficientnet/` | 2 | EfficientNet1D | ✅ | ✅ |
| `spectrogram_cnn/` | 3 | ResNet2DSpectrogram, EfficientNet2DSpectrogram, DualStreamCNN | ✅ | Partial |
| `physics/` | 3 | BearingDynamics, FaultSignatures, OperatingConditions | N/A (not nn.Module) | N/A |
| `contrastive/` | 4 | SignalEncoder, ContrastiveClassifier | ❌ | ✅ |
| `classical/` | 6 | RF, SVM, GBM, NN, StackedEnsemble, ModelSelector | N/A (sklearn) | N/A |

---

*Report generated: 2026-03-15 by IDB 1.1 Agent*
