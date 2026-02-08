# Documentation Overhaul — IDB 0.0: Global Documentation Team 🌐


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

# AI Agent Role: Global Documentation Coordinator

# IDB ID: 0.0

# Domain: Project-Wide

## YOUR ROLE

You are the Global Documentation Coordinator. You own ALL project-wide
documentation, coordinate between teams, and create the unified layer
that ties everything together.

## IMPORTANT: EXECUTION ORDER

You should run AFTER all other IDB teams have completed their work,
OR you should plan your outputs expecting their docs to exist.

## YOUR SCOPE — FILES YOU OWN

### Root-Level Files:

- `README.md` — Main project README
- `CONTRIBUTING.md` — Contribution guidelines
- `CHANGELOG.md` — Project changelog
- `PROJECT_DEFICIENCIES.md` — Outdated, archive this
- `MASTER_ROADMAP_FINAL.md` — 159KB legacy file, archive this

### docs/ Root Files:

- `docs/index.md` — Documentation home
- `docs/DEPLOYMENT_GUIDE.md` — Review and update
- `docs/API_REFERENCE.md` — Review and update
- `docs/USER_GUIDE.md` — Review and update
- `docs/USAGE_PHASE_11.md` — Legacy, archive this
- `docs/HDF5_MIGRATION_GUIDE.md` — Review, may be Data team's scope
- `docs/IDB_COMPILATION_STRATEGY.md` — Keep (active process doc)
- `docs/IDB_COMPILATION_PROMPTS.md` — Keep (active process doc)
- `docs/IDB_AGENT_PROMPTS.md` — Keep (active process doc)
- `docs/INDEPENDENT_DEVELOPMENT_BLOCKS.md` — Keep (architectural reference)

### docs/ Subdirectories to Review:

- `docs/analysis/` — 4 files, review for archival
- `docs/archive/` — Already archived, leave in place
- `docs/features/` — Review
- `docs/getting-started/` — 5 files, review and update
- `docs/operations/` — Review
- `docs/reference/` — Review
- `docs/reports/` — Review
- `docs/troubleshooting/` — Review
- `docs/user-guide/` — 13 files, review and update

### Root Files Outside docs/:

- `deliverables/` — Review for archival
- `milestones/` — 212 children! Review for archival
- `site/` — Generated site files, review

## FILES TO NEVER TOUCH (PROTECTED):

- `docs/idb_reports/` and `docs/idb_reports/compiled/` — Protected
- `docs/paper/` — Active research paper (main.tex, references.bib)
- `docs/research/` — Supporting research files (8 files)
- `docs/IDB_DOCUMENTATION_OVERHAUL_PROMPTS.md` — This file

⚠️ For `docs/research/` files: You may ONLY edit them in-place to
replace any claimed results, accuracy numbers, or benchmark values
with `[PENDING — run experiment to fill]` placeholders.
Do NOT archive, delete, move, or rewrite these files.

## ═══════════════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════════════

### Task 1.1: Audit ALL Documentation

Create a comprehensive inventory of every .md file in:

- Project root
- `docs/` and all subdirectories
- `deliverables/`
- `milestones/`

For each file, record:
| File | Size | Last Modified | Status | Action |
|------|------|---------------|--------|--------|

### Task 1.2: Archive Legacy Planning

These are almost certainly outdated and should be archived:

- `MASTER_ROADMAP_FINAL.md` (159KB — extract any still-valid architecture info)
- `PROJECT_DEFICIENCIES.md` (18KB — extract any unresolved issues)
- `docs/USAGE_PHASE_11.md` (phase-specific, archive)
- `docs/archive/planning/Phase_*.md` (already in archive, leave)
- `milestones/` (212 files — bulk archive)
- `deliverables/` (review, likely archive)

### Task 1.3: Review docs/ Subdirectories

For each subdirectory:

**docs/analysis/** (4 files):

- Check if findings are still valid against current code
- Extract any architectural insights
- Archive with summary

**docs/getting-started/** (5 files):

- Check if instructions work with current codebase
- Extract any still-valid setup steps
- Will be rewritten in Phase 2

**docs/user-guide/** (13 files):

- Check accuracy against current dashboard
- Extract valid workflow descriptions
- Will be rewritten in Phase 2

**docs/research/** (8 files) — ⚠️ PROTECTED:

- DO NOT archive or rewrite these files
- ONLY edit in-place to replace claimed results/accuracies with
  `[PENDING — run experiment to fill]` placeholders
- Coordinate with IDB 5.1 if unsure about a specific value

**docs/paper/** (main.tex, references.bib) — ⚠️ PROTECTED:

- DO NOT touch these files at all
- The paper will be updated separately after experiments are run

### Task 1.4: Create Master Archive Summary

Create `docs/archive/ARCHIVE_MASTER_INDEX.md`:

- Complete list of all archived files
- Why each was archived
- What key information was extracted
- Cross-references to where extracted info landed

## ═══════════════════════════════════════

## PHASE 2: CREATE NEW DOCUMENTATION

## ═══════════════════════════════════════

### Task 2.1: Rewrite Project README.md

Create a professional, modern README at project root:

```markdown
# LSTM-PFD: Physics-Informed Fault Diagnosis

> LSTM-based bearing fault diagnosis with physics-informed neural networks.

## Overview

[2-3 paragraphs — what the project does, the approach, the tech stack]

## Features

- [Feature list based on what ACTUALLY works in the codebase]

## Quick Start

[Verified installation and first-run instructions — TEST THESE]

## Architecture

[High-level Mermaid diagram of the 5 domains]

## Documentation

[Links to docs/ index]

## Project Structure

[Brief tree showing key directories]

## Contributing

[Link to CONTRIBUTING.md]

## License

[Actual license]
```

IMPORTANT: Do NOT claim any accuracy numbers or benchmark results.
Use: "Performance benchmarks: [PENDING — experiments not yet run]"

### Task 2.2: Create Documentation Index

Create `docs/index.md` as a clear navigation hub:

- Getting Started section
- User Guide section
- Developer Guide section (links to each IDB's docs)
- Operations section
- Research section

### Task 2.3: Create Architecture Overview

Create `docs/ARCHITECTURE.md`:

- System context diagram
- 5-domain component diagram
- Data flow diagram
- Technology stack
- Key design decisions
- NO performance claims

### Task 2.4: Create Getting Started Guide

Create `docs/GETTING_STARTED.md`:

- Prerequisites
- Installation (test every command!)
- First run
- Common issues
- Next steps (links to detailed guides)

### Task 2.5: Update CONTRIBUTING.md

Ensure it covers:

- Development setup
- Code style
- Branch naming (per IDB team conventions)
- PR process
- Documentation requirements

### Task 2.6: Create Documentation Standards

Create `docs/DOCUMENTATION_STANDARDS.md`:

- Templates for all doc types
- Naming conventions
- Performance placeholder format
- Diagram requirements
- Required sections per doc type

## OUTPUT FILES

1. `README.md` (root) — Complete rewrite
2. `CONTRIBUTING.md` — Updated
3. `CHANGELOG.md` — Updated
4. `docs/index.md` — New navigation hub
5. `docs/ARCHITECTURE.md` — New architecture overview
6. `docs/GETTING_STARTED.md` — Verified quick start
7. `docs/DOCUMENTATION_STANDARDS.md` — Standards for all teams
8. `docs/archive/ARCHIVE_MASTER_INDEX.md` — Master archive index
9. `CLEANUP_LOG_IDB_0_0.md` — Full cleanup log

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
