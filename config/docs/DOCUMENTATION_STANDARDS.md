# Documentation Standards

> Templates, conventions, and quality standards for all LSTM-PFD documentation.

## General Rules

### Rule 1: No False Claims

**Never** state accuracy, performance, F1 scores, precision, recall, benchmark results, or any quantitative results unless you have personally verified them by running the code.

For any performance metrics, use this exact placeholder format:

```
Accuracy: [PENDING — run experiment to fill]
F1 Score: [PENDING — run experiment to fill]
Inference Time: [PENDING — run experiment to fill]
```

### Rule 2: Documentation Must Match Code

Every statement must be verifiable by reading the source code. If uncertain, mark it as `[NEEDS VERIFICATION]`.

### Rule 3: No Aspirational Content

Document what **exists**, not what _should_ exist or what was _planned_. For partial implementations:

```
Feature X: Partially implemented. The class exists in `module.py` but [specific gap].
```

### Rule 4: Archive, Don't Delete

Move old `.md` files to `docs/archive/`. Never permanently delete documentation.

---

## Standard README Template

Every module gets a `README.md` following this structure:

```markdown
# {Module Name}

> One-line description.

## Overview

2-3 paragraphs explaining purpose and role in the system.

## Architecture

(Mermaid diagram if applicable)

## Quick Start

(Minimal working code example)

## Key Components

| Component | Description | File |
| --------- | ----------- | ---- |
| ...       | ...         | ...  |

## API Summary

Brief overview with link to full API.md if needed.

## Dependencies

- **Requires:** (what this module needs)
- **Provides:** (what this module exports/exposes)

## Configuration

(If applicable)

## Testing

(How to run tests for this module)

## Related Documentation

(Links to related IDB docs)
```

---

## Standard API Reference Template

For modules with significant APIs, create an `API.md`:

```markdown
# {Module} API Reference

## Classes

### `ClassName`

> Description

**Constructor:**

`ClassName(param1: Type, param2: Type = default)`

| Parameter | Type | Default  | Description  |
| --------- | ---- | -------- | ------------ |
| param1    | Type | required | What it does |

**Methods:**

#### `method_name(args) -> ReturnType`

Description.

**Example:**

(Working code example)
```

---

## Performance Placeholder Convention

Whenever results would normally appear:

```markdown
## Performance

> ⚠️ **Results pending.** Performance metrics below will be populated
> after experiments are run on the current codebase.

| Metric         | Value       |
| -------------- | ----------- |
| Accuracy       | `[PENDING]` |
| F1 Score       | `[PENDING]` |
| Precision      | `[PENDING]` |
| Recall         | `[PENDING]` |
| Inference Time | `[PENDING]` |
```

---

## Naming Conventions

| Item            | Convention                              | Example                          |
| --------------- | --------------------------------------- | -------------------------------- |
| Module README   | `README.md` in module root              | `packages/core/models/README.md` |
| API reference   | `API.md` in module root                 | `packages/core/models/API.md`    |
| Guide documents | `UPPER_SNAKE_CASE.md`                   | `TRAINING_GUIDE.md`              |
| Cleanup logs    | `CLEANUP_LOG_IDB_{X}_{Y}.md` in `docs/` | `docs/CLEANUP_LOG_IDB_1_1.md`    |
| Archive files   | Flat in `docs/archive/`                 | `docs/archive/OLD_README.md`     |

## Diagram Requirements

- Use **Mermaid** for all architecture and flow diagrams
- Keep diagrams simple — one concept per diagram
- Use `graph TB` (top-to-bottom) for hierarchies, `flowchart LR` (left-to-right) for data flows
- Label all nodes clearly

## Required Sections by Doc Type

| Doc Type      | Required Sections                                               |
| ------------- | --------------------------------------------------------------- |
| Module README | Overview, Key Components, Dependencies, Related Docs            |
| API Reference | Classes, Methods with signatures, Examples                      |
| Guide         | Overview, Steps/Instructions, Examples, Troubleshooting         |
| Cleanup Log   | Files Archived, Files Created, Information Extracted, Decisions |

## Cleanup Log Template

```markdown
# Cleanup Log — IDB {X.Y}: {Team Name}

## Summary

Brief description of what was done.

## Files Archived

| Original Location | Archive Location | Category | Reason |
| ----------------- | ---------------- | -------- | ------ |

## Files Created

| File | Description |
| ---- | ----------- |

## Information Extracted

Summary of key info salvaged from archived files.

## Decisions Made

- Decision 1: why
- Decision 2: why
```

---

## Protected Paths

These paths must **never** be archived, deleted, or rewritten:

- `docs/idb_reports/` — Active IDB analysis reports
- `docs/idb_reports/compiled/` — Compiled reports
- `docs/paper/` — Active research paper (LaTeX)
- `docs/research/` — Only in-place `[PENDING]` replacement allowed
- `docs/idb_reports/docs-cleanup/` — Overhaul prompt files
