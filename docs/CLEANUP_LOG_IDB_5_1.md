# Cleanup Log — IDB 5.1: Research Scripts

**Domain:** Research & Science
**Date:** 2026-02-08
**Scope:** `scripts/research/` (9 scripts), `docs/research/` (8 supporting files, protected)

---

## Phase 1: Archive & Extract

### 1A. Archive (`scripts/research/`)

No `.md` files were found in `scripts/research/`. Phase 1 archiving is trivially complete.

### 1B. In-Place `[PENDING]` Replacements (`docs/research/`)

The following 6 files had unverified claimed results replaced with `[PENDING — run experiment to fill]`. Files were modified **in-place** per IDB standards — no content was removed, only specific numeric values were replaced.

| File                            | Values Replaced                                                                                                                                                                                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ablation-studies.md`           | 15 accuracy values across 3 tables (architecture ablations: 97.8%, -1.2%, -0.8%, -0.3%, -1.5%; physics loss ablations: 97.8%, -0.6%, -0.4%, -0.3%, -1.4%; ensemble ablations: 98.4%, -0.4%, -0.3%, -0.6%, -0.2%)                                                             |
| `ensemble-strategies.md`        | Header accuracy claim (98-99%); 4 model accuracies in Mermaid diagram (96.8%, 96.5%, 97.8%, 96.2%); 2 ensemble accuracies in Mermaid (98.1%, 98.4%); 5×3 values in results table (accuracy, latency, model size); 3×2 values in diversity table (correlation, error overlap) |
| `index.md`                      | Ensemble accuracy claim "98-99%" (line 55); citation note "98-99% accuracy" (line 103)                                                                                                                                                                                       |
| `interactive-visualizations.md` | Confusion matrix accuracy "98.4%" (line 12); ROC AUC "0.988" (line 23)                                                                                                                                                                                                       |
| `pinn-theory.md`                | Results table: baseline accuracy "96.4%", PINN accuracy "97.8%", physics consistency "94.2%"                                                                                                                                                                                 |
| `project-timeline.md`           | 7 accuracy ranges in Mermaid flowchart node labels; 7 bar chart values; 6 accuracy values in milestones table                                                                                                                                                                |

**No changes needed:**

- `reproducibility.md` — contains no numeric performance claims
- `xai-methods.md` — contains no numeric performance claims

---

## Phase 2: New Documentation Created

| File                                   | Description                                                                                                                                   |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/research/README.md`           | Script catalog (9 scripts), CLI arguments, output locations, quick start guide, reproducibility notes                                         |
| `scripts/research/EXPERIMENT_GUIDE.md` | Per-script deep-dive: purpose, methodology, inputs, full CLI reference, expected outputs, interpretation guidance, PENDING performance tables |
| `docs/CLEANUP_LOG_IDB_5_1.md`          | This file                                                                                                                                     |

---

## Decisions Made

1. **No archiving needed** — `scripts/research/` contained only `.py` files, no documentation to archive.
2. **Protected files respected** — all `docs/research/` files remain in place; only numeric claims were replaced.
3. **Aspirational content removed** — the "excellent" qualifier was removed from the ROC curves description in `interactive-visualizations.md`.
4. **Mermaid diagram handling** — accuracy labels in Mermaid node labels were replaced with `PENDING` (short form) to avoid breaking diagram syntax; bar chart data was zeroed out with a `[PENDING]` note in the title.
5. **xai_metrics.py documented** — despite having no argparse CLI, this script was included in the catalog for completeness since it is part of the research scripts suite.
