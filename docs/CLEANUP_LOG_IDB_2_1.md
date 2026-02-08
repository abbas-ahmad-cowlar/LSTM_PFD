# Cleanup Log — IDB 2.1: Frontend/UI — Dashboard Platform

**Date:** 2026-02-09
**Scope:** `packages/dashboard/layouts/`, `packages/dashboard/components/`, `packages/dashboard/assets/`, and root-level `.md` files in `packages/dashboard/`

---

## Phase 1: Archive & Extract

### Files Archived

| Original Location                                     | Archive Location                                          | Category    | Rationale                                                                                                                                                                     |
| ----------------------------------------------------- | --------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `packages/dashboard/README.md`                        | `docs/archive/DASHBOARD_README.md`                        | WRONG/STALE | 784 lines of phase-based marketing language, unverified performance targets ("page load <2 seconds"), and aspirational feature claims. Directory layout was partially useful. |
| `packages/dashboard/GUI_QUICKSTART.md`                | `docs/archive/DASHBOARD_GUI_QUICKSTART.md`                | PARTIAL     | 655-line tutorial with useful step-by-step workflow, but claims "96% accuracy" and references stale phase numbering.                                                          |
| `packages/dashboard/FEATURE_2_COMPARISON.md`          | `docs/archive/DASHBOARD_FEATURE_2_COMPARISON.md`          | PARTIAL     | Useful architecture info (McNemar/Friedman tests, ComparisonService API). References "Phase 11C" and contains aspirational TODOs.                                             |
| `packages/dashboard/FEATURE_3_EMAIL_NOTIFICATIONS.md` | `docs/archive/DASHBOARD_FEATURE_3_EMAIL_NOTIFICATIONS.md` | PARTIAL     | Detailed email config and template info, but content spans IDB 2.2 (services) and IDB 2.4 (tasks) scope.                                                                      |
| `packages/dashboard/FEATURE_5_TAGS_SEARCH.md`         | `docs/archive/DASHBOARD_FEATURE_5_TAGS_SEARCH.md`         | PARTIAL     | Backend API endpoints and query syntax. Spans IDB 2.2 scope (TagService, SearchService).                                                                                      |

### Information Extracted

- **Page catalog structure** from `README.md` — used as starting point for layouts catalog (all routes re-verified against code)
- **Directory layout** from `README.md` — validated and incorporated into UI Guide
- **Comparison architecture** from `FEATURE_2_COMPARISON.md` — route `/compare?ids=` and `ComparisonService` confirmed in code
- **Navigation groups** from `README.md` — cross-referenced with `NAV_ITEMS` dict in `sidebar.py`

### Information NOT Carried Forward

- Performance targets (page load, filter response times) — unverified
- Accuracy claims ("96%", "96.2%", "98-99%") — unverified
- Phase numbering (11A, 11B, 11C, 11D) — stale organizational scheme
- "Production Ready" badge claims — unverified deployment status

---

## Phase 2: Files Created

| File                                      | Description                                                                                                                                                            |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `packages/dashboard/layouts/README.md`    | Page catalog with all 21+ routes verified from `callbacks/__init__.py`, organized by navigation group, with Mermaid architecture diagram                               |
| `packages/dashboard/components/README.md` | Component catalog covering 6 modules with verified function signatures, shell architecture diagram, and full API reference                                             |
| `packages/dashboard/UI_GUIDE.md`          | Practical development guide: architecture overview, adding pages (4-step verified workflow), creating components, asset management, theming, and running the dashboard |

---

## Decisions Made

1. **Archived `README.md` entirely** rather than updating in-place — too much unverified content and marketing language; a clean rewrite across three focused docs was cleaner.
2. **Did not create a new `packages/dashboard/README.md`** — the three new files (`layouts/README.md`, `components/README.md`, `UI_GUIDE.md`) serve the same purpose with better organization.
3. **Feature-specific docs (FEATURE_2, 3, 5) archived** — their backend content belongs in IDB 2.2 (services) scope; the UI-relevant portions were incorporated into the page and component catalogs.
4. **No unverified performance claims** in any output file — all content verified against source code.
