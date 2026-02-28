# Chunk 4: Data Engineering (Domain 3)

**Verdict: 🟡 Moderately Over-Engineered — Core is Solid, Periphery is Bloated**

---

## The Numbers

| Category          | Files                                           | Total Size | Assessment                           |
| ----------------- | ----------------------------------------------- | ---------- | ------------------------------------ |
| Signal Generation | 1 .py (37 KB)                                   | 37 KB      | 🟡 Monolith — should split           |
| Augmentation      | 3 .py                                           | ~39 KB     | 🔴 Duplicate implementations         |
| Datasets          | 5 .py                                           | ~74 KB     | 🟡 Proliferation                     |
| Transforms        | 3 .py                                           | ~35 KB     | 🟡 Partial overlap                   |
| Spectrograms      | 2 .py                                           | ~26 KB     | 🟢 Fine                              |
| Data Loading      | 2 .py                                           | ~24 KB     | 🟢 Fine                              |
| Supporting        | 5 .py (cache, import, validator, HDF5, wavelet) | ~82 KB     | 🟢 Mostly justified                  |
| **Documentation** | **6 .md**                                       | **~83 KB** | 🟡 More docs than some modules' code |

---

## Problem 1: 🔴 Duplicate Augmentation — NumPy vs PyTorch

Two files implement **identical augmentation methods** in different frameworks:

| Method         | `augmentation.py` (NumPy)          | `signal_augmentation.py` (PyTorch) |
| -------------- | ---------------------------------- | ---------------------------------- |
| Mixup          | `SignalAugmenter.mixup()`          | `Mixup.__call__()`                 |
| Time Warp      | `SignalAugmenter.time_warp()`      | `TimeWarping.__call__()`           |
| Magnitude Warp | `SignalAugmenter.magnitude_warp()` | `MagnitudeWarping.__call__()`      |
| Jittering      | `SignalAugmenter.jittering()`      | `Jittering.__call__()`             |
| Scaling        | `SignalAugmenter.scaling()`        | `Scaling.__call__()`               |
| Cutout         | `SignalAugmenter.cutout()`         | `Cutout.__call__()`                |
| Permutation    | `SignalAugmenter.permutation()`    | `Permutation.__call__()`           |

Plus `spectrogram_augmentation.py` (12 KB) adds 2D-specific augmentations (SpecAugment, frequency/time masking).

**What should exist:** One augmentation module with a unified API that handles both numpy and torch tensors internally (PyTorch can do both). The spectrogram augmentation is genuinely different and should stay.

**Verdict:**

- **MERGE** `augmentation.py` + `signal_augmentation.py` → single `augmentation.py`
- **KEEP** `spectrogram_augmentation.py` (genuinely different domain)

---

## Problem 2: 🟡 5 Dataset Classes — Unclear Boundaries

| File                        | Size  | What It Does                                      | Used?               |
| --------------------------- | ----- | ------------------------------------------------- | ------------------- |
| `dataset.py`                | 19 KB | Base `BearingDataset` — loads HDF5, `.mat`, numpy | ✅ Core             |
| `cnn_dataset.py`            | 13 KB | CNN-specific dataset with transforms              | ✅ Used             |
| `streaming_hdf5_dataset.py` | 13 KB | Memory-efficient HDF5 streaming                   | ✅ Used             |
| `tfr_dataset.py`            | 14 KB | Time-frequency representation dataset             | 🟡 Niche            |
| `cwru_dataset.py`           | 17 KB | **CWRU Bearing Dataset**                          | 🔴 **Out of scope** |

> [!WARNING]
> **`cwru_dataset.py` loads the Case Western Reserve University bearing dataset — which is for rolling element bearings.** Your project explicitly states "Out of Scope: Rolling element bearings (ball/roller)." This file contradicts the project's defined boundaries.

**Verdict:**

- **REMOVE** `cwru_dataset.py` — out of project scope
- **KEEP** the other 4 — they serve different access patterns (full-load vs streaming vs CNN-specific transforms)

---

## Problem 3: 🟡 `signal_generator.py` — 935-Line Monolith

At 37 KB, this is the largest Python file in the core codebase. It contains:

- `SignalMetadata` dataclass (50 lines)
- `FaultModeler` class (~200 lines) — physics-based fault signal equations
- `NoiseGenerator` class (~100 lines) — 7-layer noise model
- `SignalGenerator` class (~540 lines) — orchestrator with generation, augmentation, and HDF5 saving

**Is it too big?** The code is well-structured internally (clear class responsibilities), but at 935 lines it's hard to navigate. The `FaultModeler` and `NoiseGenerator` could be separate files.

**Verdict:** **CONSIDER splitting** into `fault_modeler.py`, `noise_generator.py`, and `signal_generator.py`. Low priority — this is working code with clear internal structure.

---

## Problem 4: 🔴 Dead Code — `contrast_learning_tfr.py`

11.6 KB file implementing contrastive learning data transforms. **Zero imports** anywhere else in the project. This is research scaffolding that was never integrated.

**Verdict:** **REMOVE.**

---

## Problem 5: 🟡 Transform Overlap

| File                | Framework | What It Does                                             |
| ------------------- | --------- | -------------------------------------------------------- |
| `transforms.py`     | NumPy     | Signal transforms (normalize, detrend, bandpass, etc.)   |
| `cnn_transforms.py` | PyTorch   | CNN-specific transforms (compose pipeline, spec augment) |

Less duplicated than augmentation — `transforms.py` is signal-processing focused while `cnn_transforms.py` is PyTorch `Dataset` transform pipeline. Overlap is minor.

**Verdict:** **KEEP both.** They serve different pipeline stages.

---

## Problem 6: 🟡 Documentation Ratio

6 markdown files totaling **83 KB of documentation** for 21 Python files:

| Doc                           | Size  | Covers             |
| ----------------------------- | ----- | ------------------ |
| `DATA_LOADING_README.md`      | 16 KB | Dataset classes    |
| `STORAGE_README.md`           | 14 KB | HDF5 caching       |
| `SIGNAL_GENERATION_README.md` | 13 KB | Signal generator   |
| `DATASET_GUIDE.md`            | 13 KB | Dataset usage      |
| `HDF5_GUIDE.md`               | 12 KB | HDF5 format        |
| `PHYSICS_MODEL_GUIDE.md`      | 12 KB | Physics model math |

**Not necessarily bad** — this is domain-specific physics code that benefits from documentation. But 83 KB of docs for ~300 KB of code is a high ratio. Consider merging `DATA_LOADING_README.md` + `DATASET_GUIDE.md` into one, and `STORAGE_README.md` + `HDF5_GUIDE.md` into one.

---

## Summary Scorecard

| Action                                   | Impact                               |
| ---------------------------------------- | ------------------------------------ |
| Merge 2 augmentation files → 1           | -1 file, cleaner API                 |
| Remove `cwru_dataset.py`                 | -1 file, -17 KB of out-of-scope code |
| Remove `contrast_learning_tfr.py`        | -1 file, -11 KB of dead code         |
| Consider splitting `signal_generator.py` | Better navigability (low priority)   |
| Merge overlapping docs                   | -2 doc files                         |

> [!IMPORTANT]
> **This domain is the healthiest so far.** The core signal generation and data loading code is well-designed and justified. The issues are at the edges: duplicate augmentation, one out-of-scope dataset, and one dead file.

---

_Next: Chunk 5 — Infrastructure (Domain 4) — Database, Deployment (Helm/K8s), Testing, Configuration_
