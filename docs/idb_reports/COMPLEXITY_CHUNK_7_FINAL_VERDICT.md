# Chunk 7: Cross-Cutting Concerns + Final Project Verdict

---

## Part A: Cross-Cutting Concerns

### 🔴 `integration/` — Dead Orchestration Layer

| File                         | Size   | Imports Found |
| ---------------------------- | ------ | ------------- |
| `unified_pipeline.py`        | 8.9 KB | **Zero**      |
| `model_registry.py`          | 8.5 KB | **Zero**      |
| `configuration_validator.py` | 8.7 KB | **Zero**      |
| `data_pipeline_validator.py` | 6 KB   | **Zero**      |

**~33 KB of integration code that nothing imports.** This was designed as a cross-domain orchestration layer but was never wired in. The dashboard connects to core ML directly, bypassing these abstractions entirely.

**Verdict:** **REMOVE the entire `integration/` directory** (keep the README if useful as a design doc).

---

### 🟡 `utils/` — Dual Logging + Constants Dumping Ground

| File                     | Size      | Issue                                                                                |
| ------------------------ | --------- | ------------------------------------------------------------------------------------ |
| `constants.py`           | **23 KB** | 🔴 Dumping ground — fault types, signal params, UI strings, error messages all mixed |
| `logger.py`              | 1.7 KB    | 🟡 Used by dashboard (via `from utils.logger`)                                       |
| `logging.py`             | 4.6 KB    | 🟡 Used by core + scripts (via `from utils.logging`)                                 |
| `visualization_utils.py` | 10 KB     | 🟡 Overlaps with `visualization/` package                                            |

**Two logging modules** co-existing: `logger.py` (simple) and `logging.py` (configured). Both are actively used by different parts of the project. This causes confusion.

**`constants.py` at 23 KB** is the largest utils file. It mixes domain constants (fault types, sampling rates) with UI constants (color palettes, dashboard strings) and error messages. Should be split by domain.

**Verdict:**

- **MERGE** `logger.py` → `logging.py` (one logging entry point)
- **SPLIT** `constants.py` into domain-specific constant files
- **MOVE** `visualization_utils.py` → `visualization/utils.py`

---

### 🔴 `milestones/` — 203 Frozen Files

4 milestone directories containing **complete frozen snapshots** of past project states:

| Directory      | Files | Purpose                  |
| -------------- | ----- | ------------------------ |
| `milestone-1/` | 82    | Early CNN implementation |
| `milestone-2/` | 41    | Multi-model expansion    |
| `milestone-3/` | 65    | Dashboard integration    |
| `milestone-4/` | 15    | Enterprise hardening     |

These are full copies of `data/`, `training/`, `models/`, `utils/`, `scripts/` from each milestone. They import from `utils.logging`, reference old paths, and **cannot run** without the original project state.

**Verdict:** **REMOVE entirely.** Git history preserves every milestone. If you need checkpoints, use git tags.

---

### 🟡 GitHub Actions — 6 CI/CD Workflows

| Workflow       | Size | Purpose                |
| -------------- | ---- | ---------------------- |
| `ci.yml`       | 6 KB | Test + lint pipeline   |
| `ci-cd.yml`    | 5 KB | Duplicate CI + deploy? |
| `deploy.yml`   | 2 KB | K8s deployment         |
| `release.yml`  | 4 KB | Automated release      |
| `docs.yml`     | 2 KB | MkDocs deploy          |
| `security.yml` | 2 KB | Security scanning      |

`ci.yml` and `ci-cd.yml` overlap. `deploy.yml` targets a K8s cluster that may not exist.

**Verdict:** **MERGE** `ci.yml` + `ci-cd.yml`. **REVIEW** whether `deploy.yml` targets a real cluster.

---

### 🟢 Other Cross-Cutting — Fine

| Directory                        | Files           | Verdict                        |
| -------------------------------- | --------------- | ------------------------------ |
| `reproducibility/`               | 4               | 🟢 KEEP — zenodo config, seeds |
| `deliverables/HANDOVER_PACKAGE/` | 3 subdirs       | 🟢 KEEP — handover artifacts   |
| `.pre-commit-config.yaml`        | 1               | 🟢 KEEP                        |
| `docs/paper/`                    | 2 (LaTeX + bib) | 🟢 KEEP                        |

---

## Part B: Final Project Verdict

### Project Health by Domain

| Domain                 | Severity | Top Issue                                                      |
| ---------------------- | -------- | -------------------------------------------------------------- |
| **Core ML Engine**     | 🔴       | 40+ models, only 15 registered; 113 KB dead transformers       |
| **Dashboard Platform** | 🔴       | Enterprise SaaS for a research tool; 300 KB of unused features |
| **Data Engineering**   | 🟡       | Duplicate augmentation; out-of-scope CWRU dataset              |
| **Infrastructure**     | 🟡       | Full K8s platform + load testing before first user             |
| **Research & Scripts** | 🔴       | Worst written-vs-used ratio; 38 KB dead contrastive code       |
| **Cross-Cutting**      | 🔴       | 33 KB dead integration layer; 203 frozen milestone files       |

---

### Prioritized Cleanup Roadmap

#### 🚨 Phase 1: Dead Code Removal (Low Risk, High Impact)

| Target                                                                                            | Files Removed | KB Saved    |
| ------------------------------------------------------------------------------------------------- | ------------- | ----------- |
| `models/transformers/advanced/`                                                                   | 7             | 113         |
| `scripts/research/contrastive_physics.py`                                                         | 1             | 38          |
| `integration/` (entire dir)                                                                       | 5             | 33          |
| Dead dashboard files (`auth_utils_improved.py`, phase adapters)                                   | 3             | 26          |
| Duplicate root-level models (`cnn_1d.py`, `resnet_1d.py`, `hybrid_pinn.py`, `legacy_ensemble.py`) | 4             | 56          |
| `visualization/xai_dashboard.py` + `counterfactual_explanations.py`                               | 2             | 33          |
| `data/cwru_dataset.py` + `contrast_learning_tfr.py`                                               | 2             | 29          |
| Third ONNX export copy                                                                            | 1             | 10          |
| **Phase 1 Total**                                                                                 | **~25 files** | **~338 KB** |

#### ⚡ Phase 2: Consolidation (Medium Risk)

| Target                                             | Impact               |
| -------------------------------------------------- | -------------------- |
| Merge 4 trainers → 1 configurable trainer          | -3 files             |
| Merge 4 evaluators → 1                             | -3 files             |
| Merge 2 augmentation files → 1                     | -1 file              |
| Merge CNN visualization files                      | -1 file              |
| Merge `logger.py` + `logging.py`                   | -1 file              |
| Consolidate `experiments/` → `scripts/research/`   | -1 directory         |
| Remove root-level scripts with package equivalents | -6 files             |
| Merge duplicate CI workflows                       | -1 file              |
| **Phase 2 Total**                                  | **~16 files, 1 dir** |

#### 🏗️ Phase 3: Structural Decisions (Needs Discussion)

| Decision                                  | Question                                            |
| ----------------------------------------- | --------------------------------------------------- |
| Remove enterprise dashboard features      | Are Slack/Teams/Webhooks/Email/API Keys/NAS needed? |
| Remove `milestones/`                      | Can git tags replace frozen snapshots?              |
| Remove `deploy/` K8s infra                | Is there a K8s cluster?                             |
| Remove load/stress tests                  | Will there be >5 concurrent users?                  |
| Prune model zoo to registered-only models | Which unregistered models produced useful results?  |

---

### By The Numbers

| Metric                            | Current   | After Phase 1 | After All Phases |
| --------------------------------- | --------- | ------------- | ---------------- |
| Dead code files                   | ~25       | 0             | 0                |
| Dead code KB                      | ~338 KB   | 0             | 0                |
| Duplicate implementations         | ~16       | ~16           | 0                |
| Enterprise features (if unneeded) | ~20 files | ~20           | 0                |
| Milestone snapshots               | 203 files | 203           | 0                |

> [!IMPORTANT]
> **Phase 1 is entirely safe** — removing files with zero imports cannot break anything. Start there. Phase 2 requires careful refactoring. Phase 3 requires your input on project scope.
