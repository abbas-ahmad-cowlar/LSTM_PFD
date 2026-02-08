# Cleanup Log — IDB 1.4: Features Sub-Block

**Date:** 2026-02-08
**Agent:** IDB 1.4 (Features — Core ML Engine)
**Scope:** `packages/core/features/` (12 Python files)

---

## Files Archived

**None.** No `.md`, `.rst`, or `.txt` documentation files existed in `packages/core/features/` prior to this overhaul. There was nothing to archive.

## Files Created

| File                                        | Description                                                                                                                                                                                                          |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `packages/core/features/README.md`          | Module overview with Mermaid architecture diagram, quick start, component table, dependencies, and configuration                                                                                                     |
| `packages/core/features/FEATURE_CATALOG.md` | Definitive catalog of all 52 features (36 base + 16 advanced) with verified names, formulas, source files, and line numbers                                                                                          |
| `packages/core/features/API.md`             | Full API reference for all classes (`FeatureExtractor`, `FeatureSelector`, `VarianceThresholdSelector`, `FeatureNormalizer`, `RobustNormalizer`) and standalone functions (validation, importance analysis, helpers) |
| `docs/CLEANUP_LOG_IDB_1_4.md`               | This file                                                                                                                                                                                                            |

## Information Extracted

- **Feature count verification:** The "52-feature" claim was verified as accurate — 36 base features extracted by `FeatureExtractor` plus 16 optional advanced features in `advanced_features.py`
- **Canonical feature order:** Documented from `FeatureExtractor._get_feature_names()` (lines 239–283 of `feature_extractor.py`)
- **Dual naming convention:** `FeatureExtractor.extract_time_domain_features()` provides both capitalized (`'RMS'`) and lowercase (`'rms'`) keys for backward compatibility

## Decisions Made

| Decision                                                | Rationale                                                                                 |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| No files archived                                       | No pre-existing documentation existed in the features directory                           |
| Split catalog into Base (36) and Advanced (16) sections | Advanced features are optional and ~10× slower; separate sections clarify the distinction |
| Included helper function tables in API.md               | Makes standalone functions discoverable without reading source code                       |
| Used `[PENDING]` for all performance metrics            | Per Rule 1: no unverified performance claims                                              |
