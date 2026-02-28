# Documentation Overhaul — IDB 4.2: Deployment — Infrastructure

> Copy this entire document and paste it as the prompt for the IDB 4.2 AI agent.

---

## Mission Overview

### The Problem

The project currently has **97+ markdown files**, many of which are:

- **Outdated** — References to old phases, removed features, and stale plans
- **Redundant** — Multiple docs saying the same thing
- **Inaccurate** — Claiming results/performance numbers that were never verified
- **Cluttered** — Legacy planning docs mixed with active documentation

### The Goal

A **clean, professional, up-to-date** documentation set where:

- Every document reflects the **actual current state** of the codebase
- No false accuracy claims, performance numbers, or unverified results
- New developers can onboard quickly
- Teams can understand each other's work
- Documentation is easy to maintain going forward

### The Approach: Two Phases

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: ARCHIVE & EXTRACT                   │
│                                                                      │
│  For EVERY .md file in your scope:                                  │
│  1. Read it thoroughly                                              │
│  2. Extract any still-relevant information                          │
│  3. Move the file to docs/archive/ (don't delete)                   │
│  4. Build an extraction summary                                     │
│                                                                      │
│  PROTECTED: docs/idb_reports/ — DO NOT TOUCH                        │
├──────────────────────────────────────────────────────────────────────┤
│                        PHASE 2: CREATE NEW DOCS                      │
│                                                                      │
│  Using extracted info + actual codebase inspection:                  │
│  1. Create fresh documentation from scratch                         │
│  2. Base everything on what the code ACTUALLY does                   │
│  3. Leave performance/accuracy as placeholders                      │
│  4. Follow the standard templates                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

# AI Agent Role: Deployment Sub-Block — Documentation Overhaul

# IDB ID: 4.2

# Domain: Infrastructure

## YOUR ROLE

You own all documentation for the Deployment sub-block.

## YOUR SCOPE

**Primary Directories and Files:**

- `packages/deployment/` (API + optimization)
- `deploy/` (deployment configs, 16 files)
- `Dockerfile`, `docker-compose.yml`

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in `deploy/` and `packages/deployment/`
2. Check if Docker/K8s configs match documentation
3. Verify ONNX/quantization documentation
4. Extract valid deployment configurations
5. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `deploy/README.md`

1. **Deployment Options Overview:**
   | Method | Files | Status |
   |--------|-------|--------|
   | Docker | Dockerfile, docker-compose.yml | [Verify] |
   | Kubernetes | deploy/\*.yaml | [Verify] |
2. **Quick Start:** Docker local deployment
3. **Environment Variables:** From .env.example
4. **Health Checks:** How to verify deployment

### Task 2.2: Create `deploy/DEPLOYMENT_GUIDE.md`

Comprehensive guide:

1. Local development with Docker
2. Production deployment
3. Environment configuration
4. Monitoring and health checks
5. Troubleshooting common issues

### Task 2.3: Create `packages/deployment/README.md`

1. Model optimization pipeline (ONNX, quantization)
2. Inference API documentation
3. Optimization workflows

Do NOT claim deployment performance or inference speeds.
Use `[PENDING BENCHMARKS]`.

## OUTPUT FILES

1. `deploy/README.md`
2. `deploy/DEPLOYMENT_GUIDE.md`
3. `packages/deployment/README.md`
4. `docs/CLEANUP_LOG_IDB_4_2.md`

---

═══════════════════════════════════════════════════════════════════
                    GENERIC INSTRUCTIONS FOR ALL TEAMS
              (Appended to every IDB team prompt)
═══════════════════════════════════════════════════════════════════

## CRITICAL RULES — READ THESE FIRST

### Rule 1: NO FALSE CLAIMS
⚠️ ABSOLUTELY DO NOT claim any accuracy, performance, F1 scores, precision,
recall, benchmark results, or any quantitative results ANYWHERE in your
documentation UNLESS you have personally verified them by running the code.

For any performance metrics, use this exact placeholder format:
  - Accuracy: `[PENDING — run experiment to fill]`
  - F1 Score: `[PENDING — run experiment to fill]`
  - Benchmark: `[PENDING — run experiment to fill]`

This applies to ALL documentation you create. The previous documentation
contained unverified claims — do NOT carry them forward.

### Rule 2: PROTECTED FILES
DO NOT delete, move, archive, or rewrite anything in:
  - `docs/idb_reports/` — These are active analysis reports
  - `docs/idb_reports/compiled/` — These are compiled reports
  - `docs/paper/` — Active research paper (LaTeX). DO NOT TOUCH.
  - `docs/research/` — Supporting research docs. DO NOT archive.
    Only update claimed results/values with `[PENDING]` placeholders
    IN-PLACE. Do not move, rename, or rewrite these files.

These folders and their contents are off-limits for archival.
The ONLY allowed action on `docs/research/` files is replacing
claimed results, accuracies, or benchmark numbers with `[PENDING]`.

### Rule 3: ARCHIVE, DON'T DELETE
Move old .md files to `docs/archive/` — do NOT permanently delete them.
If docs/archive/ already has files, that's fine — add more there.
Create a flat structure inside archive (no deep nesting).

### Rule 4: DOCUMENTATION MUST MATCH CODE
Every statement in your documentation must be verifiable by reading the
actual source code. If you're not sure about something, check the code.
If you can't verify it, mark it as `[NEEDS VERIFICATION]`.

### Rule 5: NO ASPIRATIONAL CONTENT
Document what EXISTS, not what SHOULD exist or what was PLANNED.
If a feature is partially implemented, say so:
  "Feature X: Partially implemented. The class exists but [specific gap]."
Do NOT say: "Feature X provides comprehensive [thing]" if it doesn't.

---

## PHASE 1: ARCHIVE & EXTRACT (Do this FIRST)

### Step 1: Find All .md Files in Your Scope
- Use file search to find every .md file in your assigned directories
- Also check for .rst, .txt documentation files

### Step 2: Read and Categorize Each File
For each documentation file found, categorize it:

| Category | Action | Criteria |
|----------|--------|----------|
| **STALE** | Archive | References removed features, old phases, or outdated APIs |
| **WRONG** | Archive | Contains incorrect information or unverified claims |
| **REDUNDANT** | Archive | Same info exists elsewhere or in IDB reports |
| **PARTIAL** | Extract & Archive | Has some useful info mixed with outdated content |
| **CURRENT** | Keep & Update | Accurately reflects current code state |

### Step 3: Extract Relevant Information
Before archiving a file, extract anything still relevant:
- Accurate technical descriptions of how components work
- Valid configuration examples
- Correct API signatures (verify against code)
- Useful architectural explanations
- Valid dependency information

Save extractions to a temporary working document for use in Phase 2.

### Step 4: Move Files to Archive
Move categorized files to `docs/archive/`

### Step 5: Create Archive Summary
Create `docs/archive/ARCHIVE_INDEX.md` (or append to it if it exists):

```markdown
# Archive Index

## Files Archived by IDB {your_id}

| Original Location | Archive Location | Category | Key Info Extracted | Date |
|-------------------|------------------|----------|-------------------|------|
| `path/to/old.md` | `docs/archive/OLD.md` | STALE | None | 2026-02-08 |
| `path/to/other.md` | `docs/archive/OTHER.md` | PARTIAL | API signatures | 2026-02-08 |
````

---

## PHASE 2: CREATE NEW DOCUMENTATION (Do this AFTER Phase 1)

### Documentation Quality Standards

1. **Structure:** Use consistent headers (H1 for title, H2 for sections, H3 for subsections)
2. **Tone:** Professional, concise, factual — no marketing language
3. **Examples:** Every API/class must have at least one working code example
4. **Diagrams:** Use Mermaid for architecture/flow diagrams
5. **Tables:** Use tables for catalogs, parameters, and comparisons
6. **Links:** Link to related docs in other IDB scopes using relative paths

### Standard README Template

Every module gets a README.md following this structure:

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

### Standard API Reference Template

For modules with significant APIs, create an API.md:

```markdown
# {Module} API Reference

## Classes

### `ClassName`

> Description

**Constructor:**
ClassName(param1: Type, param2: Type = default)

| Parameter | Type | Default  | Description  |
| --------- | ---- | -------- | ------------ |
| param1    | Type | required | What it does |

**Methods:**

#### `method_name(args) -> ReturnType`

Description.

**Example:**
(Working code example)
```

### Performance Placeholder Convention

Whenever you would normally state a result, use:

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

### Cleanup Log

Every team must produce a `CLEANUP_LOG_IDB_{id}.md` in docs/ with:

- Files archived (with reasons)
- Files created (with descriptions)
- Information extracted (summary)
- Decisions made (why certain things were kept/removed)

═══════════════════════════════════════════════════════════════════
END OF GENERIC INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

---

# Expected Final Documentation Structure

After ALL teams complete their work, the documentation should look like this:

```

LSTM*PFD/
│
├── README.md ← IDB 0.0 (Global)
├── CONTRIBUTING.md ← IDB 0.0
├── CHANGELOG.md ← IDB 0.0
│
├── docs/
│ ├── index.md ← IDB 0.0 (nav hub)
│ ├── ARCHITECTURE.md ← IDB 0.0
│ ├── GETTING_STARTED.md ← IDB 0.0
│ ├── DOCUMENTATION_STANDARDS.md ← IDB 0.0
│ ├── INDEPENDENT_DEVELOPMENT_BLOCKS.md ← KEEP (reference)
│ │
│ ├── archive/ ← All archived files
│ │ ├── ARCHIVE_MASTER_INDEX.md ← IDB 0.0
│ │ ├── ARCHIVE_INDEX.md ← All teams append
│ │ └── (all archived .md files)
│ │
│ ├── idb_reports/ ← PROTECTED
│ │ ├── (38 analysis/best practices files)
│ │ └── compiled/
│ │ └── (12 compiled reports)
│ │
│ └── CLEANUP_LOG_IDB*\*.md ← One per team
│
├── packages/
│ ├── core/
│ │ ├── models/
│ │ │ ├── README.md ← IDB 1.1
│ │ │ └── API.md ← IDB 1.1
│ │ ├── training/
│ │ │ ├── README.md ← IDB 1.2
│ │ │ ├── TRAINING_GUIDE.md ← IDB 1.2
│ │ │ └── API.md ← IDB 1.2
│ │ ├── evaluation/
│ │ │ ├── README.md ← IDB 1.3
│ │ │ ├── METRICS_GUIDE.md ← IDB 1.3
│ │ │ └── API.md ← IDB 1.3
│ │ ├── features/
│ │ │ ├── README.md ← IDB 1.4
│ │ │ ├── FEATURE_CATALOG.md ← IDB 1.4
│ │ │ └── API.md ← IDB 1.4
│ │ └── explainability/
│ │ ├── README.md ← IDB 1.5
│ │ ├── XAI_GUIDE.md ← IDB 1.5
│ │ └── API.md ← IDB 1.5
│ │
│ ├── dashboard/
│ │ ├── layouts/README.md ← IDB 2.1
│ │ ├── components/README.md ← IDB 2.1
│ │ ├── UI_GUIDE.md ← IDB 2.1
│ │ ├── services/
│ │ │ ├── README.md ← IDB 2.2
│ │ │ ├── SERVICE_CATALOG.md ← IDB 2.2
│ │ │ └── API.md ← IDB 2.2
│ │ ├── callbacks/
│ │ │ ├── README.md ← IDB 2.3
│ │ │ └── CALLBACK_GUIDE.md ← IDB 2.3
│ │ ├── tasks/
│ │ │ ├── README.md ← IDB 2.4
│ │ │ └── TASK_GUIDE.md ← IDB 2.4
│ │ ├── database/
│ │ │ ├── README.md ← IDB 4.1
│ │ │ └── SCHEMA_GUIDE.md ← IDB 4.1
│ │ └── models/README.md ← IDB 4.1
│ │
│ └── deployment/README.md ← IDB 4.2
│
├── data/
│ ├── SIGNAL_GENERATION_README.md ← IDB 3.1
│ ├── PHYSICS_MODEL_GUIDE.md ← IDB 3.1
│ ├── DATA_LOADING_README.md ← IDB 3.2
│ ├── DATASET_GUIDE.md ← IDB 3.2
│ ├── STORAGE_README.md ← IDB 3.3
│ └── HDF5_GUIDE.md ← IDB 3.3
│
├── config/
│ ├── README.md ← IDB 4.4
│ └── CONFIGURATION_GUIDE.md ← IDB 4.4
│
├── tests/
│ ├── README.md ← IDB 4.3
│ └── TESTING_GUIDE.md ← IDB 4.3
│
├── deploy/
│ ├── README.md ← IDB 4.2
│ └── DEPLOYMENT_GUIDE.md ← IDB 4.2
│
├── integration/
│ ├── README.md ← IDB 6.0
│ └── INTEGRATION_GUIDE.md ← IDB 6.0
│
├── utils/README.md ← IDB 6.0
│
├── visualization/
│ ├── README.md ← IDB 5.2
│ └── VISUALIZATION_GUIDE.md ← IDB 5.2
│
└── scripts/research/
├── README.md ← IDB 5.1
└── EXPERIMENT_GUIDE.md ← IDB 5.1

```

---

## Execution Order Recommendation

```

PARALLEL WAVE 1 (All independent teams):
├── IDB 1.1 Models
├── IDB 1.3 Evaluation
├── IDB 1.4 Features
├── IDB 1.5 XAI
├── IDB 3.1 Signal Generation
├── IDB 3.2 Data Loading
├── IDB 3.3 Storage Layer
├── IDB 4.2 Deployment
├── IDB 4.3 Testing
├── IDB 4.4 Configuration
├── IDB 5.1 Research Scripts
├── IDB 5.2 Visualization
└── IDB 6.0 Integration

PARALLEL WAVE 2 (Depend on Wave 1 outputs):
├── IDB 1.2 Training (needs Models docs for context)
├── IDB 2.2 Backend Services (needs Database docs)
├── IDB 2.4 Async Tasks (needs Services skeleton)
└── IDB 2.1 Frontend/UI

PARALLEL WAVE 3 (Highest coupling):
└── IDB 2.3 Callbacks (needs both UI and Services docs)

FINAL:
└── IDB 0.0 Global Documentation (runs last, ties everything together)

```
