# IDB Documentation Overhaul — Team Agent Prompts

**Purpose:** Complete documentation cleanup, archival, and rewrite for the LSTM_PFD project.  
**Strategy:** 2-phase approach — Archive first, then create fresh docs.  
**Last Updated:** February 8, 2026

---

## Table of Contents

1. [Mission Overview](#mission-overview)
2. [Prompt Strategy (1 or 2 per team?)](#prompt-strategy)
3. [GENERIC INSTRUCTIONS — Append to Every Team Prompt](#generic-instructions)
4. [IDB 0.0 — Global Documentation Team 🌐](#idb-00--global-documentation-team-)
5. [IDB 1.1 — Models](#idb-11--models)
6. [IDB 1.2 — Training](#idb-12--training)
7. [IDB 1.3 — Evaluation](#idb-13--evaluation)
8. [IDB 1.4 — Features](#idb-14--features)
9. [IDB 1.5 — Explainability (XAI)](#idb-15--explainability-xai)
10. [IDB 2.1 — Frontend/UI](#idb-21--frontendui)
11. [IDB 2.2 — Backend Services](#idb-22--backend-services)
12. [IDB 2.3 — Callbacks](#idb-23--callbacks)
13. [IDB 2.4 — Async Tasks](#idb-24--async-tasks)
14. [IDB 3.1 — Signal Generation](#idb-31--signal-generation)
15. [IDB 3.2 — Data Loading](#idb-32--data-loading)
16. [IDB 3.3 — Storage Layer](#idb-33--storage-layer)
17. [IDB 4.1 — Database](#idb-41--database)
18. [IDB 4.2 — Deployment](#idb-42--deployment)
19. [IDB 4.3 — Testing](#idb-43--testing)
20. [IDB 4.4 — Configuration](#idb-44--configuration)
21. [IDB 5.1 — Research Scripts](#idb-51--research-scripts)
22. [IDB 5.2 — Visualization](#idb-52--visualization)
23. [IDB 6.0 — Integration Layer](#idb-60--integration-layer)
24. [Expected Final Structure](#expected-final-structure)

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

## Prompt Strategy

### Recommendation: **1 Prompt Per Team (with 2 phases inside)**

I recommend **a single prompt** per team that contains both Phase 1 and Phase 2, because:

1. **Context Continuity** — The extraction from Phase 1 directly informs Phase 2
2. **No Info Loss** — Agent has both old doc knowledge and codebase context in the same session
3. **Efficiency** — One pass through the codebase is better than two

The prompt structure for each team is:

- **Section A:** Role, scope, and boundaries
- **Section B:** Phase 1 — Archive & Extract
- **Section C:** Phase 2 — Create New Docs
- **Section D:** Generic Instructions (appended from below)

If your agent/model has context length issues, split at the phase boundary.

---

## GENERIC INSTRUCTIONS

> **COPY-PASTE THE BLOCK BELOW AND APPEND IT TO EVERY TEAM PROMPT.**

````
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

```

---
---

# TEAM-SPECIFIC PROMPTS

Each prompt below is self-contained. Copy the prompt, append the Generic Instructions block above, and give it to the team agent.

---

# IDB 0.0 — Global Documentation Team 🌐

> **NEW TEAM** — Coordinates project-wide docs. Runs AFTER all other teams.

```

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

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

# Domain 1: Core ML Engine 🧠

---

## IDB 1.1 — Models

```

# AI Agent Role: Models Sub-Block — Documentation Overhaul

# IDB ID: 1.1

# Domain: Core ML Engine

## YOUR ROLE

You own all documentation for the Models sub-block. Your job is to
archive outdated docs, extract useful info, and create fresh
documentation based on the actual current state of the code.

## YOUR SCOPE

**Primary Directory:** `packages/core/models/`

- 61 files across 13 subdirectories + 7 top-level files
- Sub-directories: `classical/`, `cnn/`, `transformer/`, `pinn/`,
  `ensemble/`, `fusion/`

**You are NOT allowed to touch:**

- `packages/core/training/` (IDB 1.2 owns this)
- `packages/core/evaluation/` (IDB 1.3 owns this)
- `docs/idb_reports/` (protected)
- Any other packages

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Task 1.1: Find All Documentation in Scope

Search `packages/core/models/` for all .md, .rst, .txt files.
Also check for inline documentation quality in .py files.

### Task 1.2: Audit Existing Docs

For each doc found:

- Does it accurately describe the current model implementations?
- Are the claimed architectures still present in the code?
- Are there API signatures that no longer match?
- Any performance claims? (These must be removed or marked PENDING)

### Task 1.3: Extract Useful Information

Before archiving, extract:

- Accurate descriptions of model architectures
- Valid BaseModel interface documentation
- Correct model_factory usage patterns
- Any valid diagrams showing model hierarchy
- Configuration parameter documentation

### Task 1.4: Archive Old Files

Move outdated docs to `docs/archive/`, update ARCHIVE_INDEX.md

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `packages/core/models/README.md`

Document based on ACTUAL CODE inspection:

1. **Overview:** What model architectures actually exist (verify each!)
2. **Model Catalog Table:**
   | Model Class | Type | File | Status |
   |-------------|------|------|--------|
   (List ONLY models that exist in the codebase)
3. **Inheritance Diagram:** Mermaid class diagram from actual code
4. **Factory Usage:** How create_model() actually works
5. **Quick Start:** Verified working example
6. **Adding New Models:** Based on actual BaseModel requirements

### Task 2.2: Create `packages/core/models/API.md`

For each model class that ACTUALLY EXISTS:

- Class name, inheritance, file location
- Constructor parameters (verify against **init** signature)
- forward() signature and return type
- Any additional public methods
- NO performance claims — just architecture description

### Task 2.3: Create Sub-Directory READMEs (if complex enough)

- `pinn/README.md` — Physics-informed models need extra explanation
- `ensemble/README.md` — Ensemble strategy documentation
- Only if the sub-directory has 3+ files

## OUTPUT FILES

1. `packages/core/models/README.md` — Module overview
2. `packages/core/models/API.md` — API reference
3. Sub-directory READMEs (optional)
4. `docs/CLEANUP_LOG_IDB_1_1.md` — Cleanup log
5. Updates to `docs/archive/ARCHIVE_INDEX.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 1.2 — Training

```

# AI Agent Role: Training Sub-Block — Documentation Overhaul

# IDB ID: 1.2

# Domain: Core ML Engine

## YOUR ROLE

You own all documentation for the Training sub-block.

## YOUR SCOPE

**Primary Directory:** `packages/core/training/`

- 23 files
- Key: trainers, callbacks, loss functions, optimizers, schedulers

**Boundaries:** Only training logic. Models = IDB 1.1, Evaluation = IDB 1.3.

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Task 1.1: Find All .md Files

Search `packages/core/training/` for documentation files.

### Task 1.2: Audit & Extract

- Check if trainer class descriptions match actual code
- Verify callback system documentation against implementation
- Extract correct loss function catalog
- Remove any claims about training accuracy/convergence speed
- Extract checkpoint format documentation if accurate

### Task 1.3: Archive Old Files

Move to `docs/archive/`, update index.

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `packages/core/training/README.md`

Based on actual code:

1. **Overview:** Training infrastructure description
2. **Trainer Catalog:**
   | Trainer | Specialization | File |
   |---------|---------------|------|
   (Only trainers that exist)
3. **Callback System:** How callbacks work (from actual code)
4. **Loss Functions:** Catalog with formulas (verify against code)
5. **Quick Start:** Working training example
6. **Checkpoint Format:** What's actually saved (verify code)

### Task 2.2: Create `packages/core/training/TRAINING_GUIDE.md`

Practical guide:

1. Basic training workflow
2. Choosing the right trainer
3. Configuring callbacks
4. Loss function selection
5. Hyperparameter optimization options
6. Reproducibility setup (seed handling from actual code)

IMPORTANT: Do NOT claim convergence speeds, training times,
or any performance benchmarks. Use `[PENDING]` placeholders.

### Task 2.3: Create `packages/core/training/API.md`

Full API reference for all public classes.

## OUTPUT FILES

1. `packages/core/training/README.md`
2. `packages/core/training/TRAINING_GUIDE.md`
3. `packages/core/training/API.md`
4. `docs/CLEANUP_LOG_IDB_1_2.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 1.3 — Evaluation

```

# AI Agent Role: Evaluation Sub-Block — Documentation Overhaul

# IDB ID: 1.3

# Domain: Core ML Engine

## YOUR ROLE

You own all documentation for the Evaluation sub-block.

## YOUR SCOPE

**Primary Directory:** `packages/core/evaluation/` (16 files)
Key: evaluators, metrics computation, confusion analysis, ROC curves.

**Boundaries:** Only evaluation logic. Do not document models or training.

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in `packages/core/evaluation/`
2. Check if metric descriptions match actual implementations
3. Remove any reported benchmark results (these are unverified)
4. Extract accurate evaluator class documentation
5. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `packages/core/evaluation/README.md`

1. Evaluator types (verify each exists in code)
2. Metrics computed (list only those implemented)
3. Output format documentation
4. Quick start example

### Task 2.2: Create `packages/core/evaluation/METRICS_GUIDE.md`

For each metric ACTUALLY IMPLEMENTED:

1. Metric name and formula
2. When to use it
3. Interpretation guidance
4. Code example

CRITICAL: Do NOT include any claimed benchmark results.
Every results table must use `[PENDING]` placeholders.

### Task 2.3: Create `packages/core/evaluation/API.md`

API reference for all evaluator classes.

## OUTPUT FILES

1. `packages/core/evaluation/README.md`
2. `packages/core/evaluation/METRICS_GUIDE.md`
3. `packages/core/evaluation/API.md`
4. `docs/CLEANUP_LOG_IDB_1_3.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 1.4 — Features

```

# AI Agent Role: Features Sub-Block — Documentation Overhaul

# IDB ID: 1.4

# Domain: Core ML Engine

## YOUR ROLE

You own all documentation for the Features sub-block.

## YOUR SCOPE

**Primary Directory:** `packages/core/features/` (12 files)
Key: feature extraction, selection, importance analysis.

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in `packages/core/features/`
2. Verify the "52-feature" claim — count actual features in code
3. Check if feature formulas/descriptions are correct
4. Extract accurate pipeline documentation
5. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `packages/core/features/README.md`

1. Feature extraction pipeline overview
2. Feature categories (time-domain, frequency-domain, wavelet)
3. Pipeline flow diagram (from actual code)
4. Quick start example

### Task 2.2: Create `packages/core/features/FEATURE_CATALOG.md`

For EACH feature actually implemented (verify in code!):
| # | Feature Name | Category | Formula | File | Line |
|---|-------------|----------|---------|------|------|

This is the definitive catalog. Every entry must be verified.

### Task 2.3: Create `packages/core/features/API.md`

API reference for extractors, selectors, and importance analyzers.

## OUTPUT FILES

1. `packages/core/features/README.md`
2. `packages/core/features/FEATURE_CATALOG.md`
3. `packages/core/features/API.md`
4. `docs/CLEANUP_LOG_IDB_1_4.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 1.5 — Explainability (XAI)

```

# AI Agent Role: XAI Sub-Block — Documentation Overhaul

# IDB ID: 1.5

# Domain: Core ML Engine

## YOUR ROLE

You own all documentation for the Explainability sub-block.

## YOUR SCOPE

**Primary Directory:** `packages/core/explainability/` (8 files)
Key: SHAP, LIME, Integrated Gradients, MC Dropout uncertainty.

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in scope
2. Verify XAI method claims against actual implementations
3. Check if caching documentation is accurate
4. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `packages/core/explainability/README.md`

1. XAI methods overview (only those implemented)
2. Method selection guide (which method for which scenario)
3. Quick start example
4. Caching mechanism documentation (verify against code)

### Task 2.2: Create `packages/core/explainability/XAI_GUIDE.md`

For each XAI method ACTUALLY IMPLEMENTED:

1. Method name and theoretical background
2. Implementation status (complete/partial)
3. Usage example (verified working)
4. Output format
5. Limitations

CRITICAL: Do NOT claim any XAI quality metrics or interpretation
accuracy. These need experimental validation.

### Task 2.3: Create `packages/core/explainability/API.md`

API reference for all explainer classes.

## OUTPUT FILES

1. `packages/core/explainability/README.md`
2. `packages/core/explainability/XAI_GUIDE.md`
3. `packages/core/explainability/API.md`
4. `docs/CLEANUP_LOG_IDB_1_5.md`

[APPEND GENERIC INSTRUCTIONS HERE]

---

# Domain 2: Dashboard Platform 📊

---

## IDB 2.1 — Frontend/UI

```
# AI Agent Role: Frontend/UI Sub-Block — Documentation Overhaul
# IDB ID: 2.1
# Domain: Dashboard Platform

## YOUR ROLE
You own all documentation for the Frontend/UI sub-block.

## YOUR SCOPE
**Primary Directories:**
- `packages/dashboard/layouts/` (24 files)
- `packages/dashboard/components/` (6 files)
- `packages/dashboard/assets/`
- `packages/dashboard/README.md` (if it exists)

**Boundaries:**
- Services = IDB 2.2
- Callbacks = IDB 2.3
- Tasks = IDB 2.4
- Do NOT document backend logic

## ═══════════════════════════════
## PHASE 1: ARCHIVE & EXTRACT
## ═══════════════════════════════

### Tasks:
1. Find all .md files in layout, component, and asset directories
2. Check `packages/dashboard/README.md` — is it accurate?
3. Verify page/route descriptions match actual layouts
4. Extract valid component documentation
5. Remove any screenshots or references to UI that has changed
6. Archive old files

## ═══════════════════════════════
## PHASE 2: CREATE NEW DOCS
## ═══════════════════════════════

### Task 2.1: Create `packages/dashboard/layouts/README.md`
Based on actual code inspection:
1. **Page Catalog:**
   | Page | File | Route | Complexity |
   |------|------|-------|------------|
   (Verify each page exists and its route)
2. **Layout Architecture:** How pages are structured
3. **Routing:** Actual URL routing from code
4. **Component ID Conventions:** For callback integration

### Task 2.2: Create `packages/dashboard/components/README.md`
1. Component catalog (verify each exists)
2. Usage examples from actual code
3. Props/parameters for each component
4. Styling conventions

### Task 2.3: Create `packages/dashboard/UI_GUIDE.md`
Practical guide:
1. Dashboard architecture overview
2. Adding a new page (verified workflow)
3. Creating reusable components
4. Asset management (CSS, JS, images)
5. Theming and styling

IMPORTANT: Do NOT include screenshots of the UI. The current
state hasn't been validated. Just document the code structure.

## OUTPUT FILES
1. `packages/dashboard/layouts/README.md`
2. `packages/dashboard/components/README.md`
3. `packages/dashboard/UI_GUIDE.md`
4. `docs/CLEANUP_LOG_IDB_2_1.md`
```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 2.2 — Backend Services

```
# AI Agent Role: Backend Services Sub-Block — Documentation Overhaul
# IDB ID: 2.2
# Domain: Dashboard Platform

## YOUR ROLE
You own all documentation for the Backend Services sub-block.

## YOUR SCOPE
**Primary Directory:** `packages/dashboard/services/` (24+ services)
**Independence Score: 5/10** — This is the central orchestration layer.

**Boundaries:**
- UI/Layouts = IDB 2.1
- Callbacks = IDB 2.3
- Database models = IDB 4.1

## ═══════════════════════════════
## PHASE 1: ARCHIVE & EXTRACT
## ═══════════════════════════════

### Tasks:
1. Find all .md files in `packages/dashboard/services/`
2. Verify service descriptions match actual implementations
3. Check for documented services that no longer exist
4. Extract accurate dependency mapping
5. Remove any performance claims about services
6. Archive old files

## ═══════════════════════════════
## PHASE 2: CREATE NEW DOCS
## ═══════════════════════════════

### Task 2.1: Create `packages/dashboard/services/README.md`
1. **Service Architecture:** How services are organized
2. **Service Catalog:**
   | Service | File | Responsibility | Dependencies |
   |---------|------|----------------|--------------|
   (Verify each service exists and its actual responsibilities)
3. **Dependency Diagram:** Mermaid graph of service dependencies
4. **Usage Pattern:** How callbacks/tasks consume services

### Task 2.2: Create `packages/dashboard/services/SERVICE_CATALOG.md`
For each service:
1. Class name and file
2. Public methods (verify signatures against code)
3. Dependencies (what it imports/uses)
4. Error handling patterns
5. Usage example

### Task 2.3: Create `packages/dashboard/services/API.md`
Full API reference for all service classes.

## OUTPUT FILES
1. `packages/dashboard/services/README.md`
2. `packages/dashboard/services/SERVICE_CATALOG.md`
3. `packages/dashboard/services/API.md`
4. `docs/CLEANUP_LOG_IDB_2_2.md`
```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 2.3 — Callbacks

```
# AI Agent Role: Callbacks Sub-Block — Documentation Overhaul
# IDB ID: 2.3
# Domain: Dashboard Platform

## YOUR ROLE
You own all documentation for the Callbacks sub-block.

## YOUR SCOPE
**Primary Directory:** `packages/dashboard/callbacks/` (29 modules)
**Independence Score: 4/10** — HIGHEST COUPLING in the system.

This is the "glue" between UI (IDB 2.1) and Services (IDB 2.2).

## ═══════════════════════════════
## PHASE 1: ARCHIVE & EXTRACT
## ═══════════════════════════════

### Tasks:
1. Find all .md files in `packages/dashboard/callbacks/`
2. Check if callback-to-layout mapping is documented and accurate
3. Verify Input/Output documentation against actual decorators
4. Extract any useful patterns
5. Archive old files

## ═══════════════════════════════
## PHASE 2: CREATE NEW DOCS
## ═══════════════════════════════

### Task 2.1: Create `packages/dashboard/callbacks/README.md`
1. **Callback Architecture:** How Dash callbacks work in this project
2. **Callback-Layout Mapping:**
   | Callback Module | Layout(s) Served | Service(s) Used |
   |----------------|------------------|-----------------|
   (Verify each mapping against actual code)
3. **Common Patterns:** Identified from actual code
4. **Known Pitfalls:** Circular dependencies, common bugs

### Task 2.2: Create `packages/dashboard/callbacks/CALLBACK_GUIDE.md`
1. How to create a new callback
2. Input/Output naming conventions (from actual code)
3. State management patterns
4. Error handling patterns
5. Performance considerations
6. Testing callbacks

## OUTPUT FILES
1. `packages/dashboard/callbacks/README.md`
2. `packages/dashboard/callbacks/CALLBACK_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_2_3.md`
```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 2.4 — Async Tasks

```
# AI Agent Role: Async Tasks Sub-Block — Documentation Overhaul
# IDB ID: 2.4
# Domain: Dashboard Platform

## YOUR ROLE
You own all documentation for the Async Tasks sub-block.

## YOUR SCOPE
**Primary Directory:** `packages/dashboard/tasks/` (11 modules)
Task types: training, HPO, NAS, data generation, XAI, deployment, testing.

## ═══════════════════════════════
## PHASE 1: ARCHIVE & EXTRACT
## ═══════════════════════════════

### Tasks:
1. Find all .md files in `packages/dashboard/tasks/`
2. Verify task descriptions against actual Celery task definitions
3. Check if task signatures are documented correctly
4. Extract any useful configuration info
5. Archive old files

## ═══════════════════════════════
## PHASE 2: CREATE NEW DOCS
## ═══════════════════════════════

### Task 2.1: Create `packages/dashboard/tasks/README.md`
1. **Task Architecture:** Celery setup from actual code
2. **Task Catalog:**
   | Task Name | Celery Signature | Module | Purpose |
   |-----------|-----------------|--------|---------|
   (Verify each task against actual @celery.task decorators)
3. **Queue Configuration:** From actual celery config
4. **Task State Management:** How progress is reported

### Task 2.2: Create `packages/dashboard/tasks/TASK_GUIDE.md`
1. Creating new Celery tasks (based on existing patterns)
2. Task state management patterns
3. Progress reporting conventions
4. Error handling and retry patterns
5. Task chaining examples
6. Monitoring and debugging

IMPORTANT: Do NOT claim any task execution times or throughput.
Use `[PENDING]` placeholders.

## OUTPUT FILES
1. `packages/dashboard/tasks/README.md`
2. `packages/dashboard/tasks/TASK_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_2_4.md`
```

[APPEND GENERIC INSTRUCTIONS HERE]

---

# Domain 3: Data Engineering 💾

---

## IDB 3.1 — Signal Generation

```
# AI Agent Role: Signal Generation Sub-Block — Documentation Overhaul
# IDB ID: 3.1
# Domain: Data Engineering

## YOUR ROLE
You own all documentation for the Signal Generation sub-block.

## YOUR SCOPE
**Primary Files:**
- `data/signal_generator.py` (37KB — largest data file)
- `data/signal_augmentation.py`
- `data/spectrogram_generator.py`

**Boundaries:**
- Data Loading = IDB 3.2
- Storage = IDB 3.3

## ═══════════════════════════════
## PHASE 1: ARCHIVE & EXTRACT
## ═══════════════════════════════

### Tasks:
1. Find all .md files related to signal generation in `data/` directory
2. Check if physics model documentation matches actual code
3. Verify fault type descriptions against implementation
4. Extract valid parameter documentation
5. Remove any claims about signal fidelity or validation results
6. Archive old files

### SPECIAL ATTENTION:
The project has a known scope issue regarding journal bearing
physics vs ball bearing (CWRU) dataset. Document what the code
ACTUALLY implements without making claims about physical accuracy.

## ═══════════════════════════════
## PHASE 2: CREATE NEW DOCS
## ═══════════════════════════════

### Task 2.1: Create `data/SIGNAL_GENERATION_README.md`
1. **Overview:** What the signal generator does (from code)
2. **Physics Model:** The mathematical model ACTUALLY used
   - Document parameters, equations, and assumptions from code
   - Do NOT claim physical accuracy — use `[PENDING VALIDATION]`
3. **Fault Types Implemented:**
   | Fault Type | Class/Function | Parameters |
   |-----------|---------------|------------|
   (Only those that exist in code)
4. **Signal Parameters:**
   | Parameter | Type | Default | Range | Meaning |
   |-----------|------|---------|-------|---------|
5. **Quick Start:** Working generation example
6. **Output Formats:** What files are produced

### Task 2.2: Create `data/PHYSICS_MODEL_GUIDE.md`
Document the actual physics:
1. Sommerfeld-scaled bearing model (from code)
2. Parameter ranges and their physical meaning
3. Fault injection mechanism
4. Noise model
5. Known limitations and assumptions

Mark all validation status as `[PENDING]`.

## OUTPUT FILES
1. `data/SIGNAL_GENERATION_README.md`
2. `data/PHYSICS_MODEL_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_3_1.md`
```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 3.2 — Data Loading

```
# AI Agent Role: Data Loading Sub-Block — Documentation Overhaul
# IDB ID: 3.2
# Domain: Data Engineering

## YOUR ROLE
You own all documentation for the Data Loading sub-block.

## YOUR SCOPE
**Primary Files:**
- `data/dataset.py`, `data/dataloader.py`
- `data/cnn_dataset.py`, `data/streaming_hdf5_dataset.py`
- `data/cwru_dataset.py`, `data/tfr_dataset.py`
- `data/transforms.py`, `data/cnn_transforms.py`

**Boundaries:**
- Signal Generation = IDB 3.1
- Storage = IDB 3.3

## ═══════════════════════════════
## PHASE 1: ARCHIVE & EXTRACT
## ═══════════════════════════════

### Tasks:
1. Find all .md files related to data loading in `data/`
2. Verify dataset class descriptions match actual code
3. Check transform documentation accuracy
4. Extract valid data format documentation
5. Archive old files

## ═══════════════════════════════
## PHASE 2: CREATE NEW DOCS
## ═══════════════════════════════

### Task 2.1: Create `data/DATA_LOADING_README.md`
1. **Dataset Types:**
   | Dataset Class | File | Use Case | Interface |
   |--------------|------|----------|-----------|
   (Verify each exists)
2. **DataLoader Configuration:** From actual code
3. **Transform Pipeline:** Document all transforms
4. **CWRU Benchmark:** How to use the CWRU dataset
5. **Memory Management:** Streaming/chunked loading options
6. **Quick Start:** Working example

### Task 2.2: Create `data/DATASET_GUIDE.md`
1. Choosing the right dataset class
2. Custom dataset creation (following existing patterns)
3. Data augmentation options (from actual transforms)
4. Streaming for large files
5. Data format specifications

Do NOT claim data loading speeds. Use `[PENDING BENCHMARKS]`.

## OUTPUT FILES
1. `data/DATA_LOADING_README.md`
2. `data/DATASET_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_3_2.md`
```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 3.3 — Storage Layer

```
# AI Agent Role: Storage Layer Sub-Block — Documentation Overhaul
# IDB ID: 3.3
# Domain: Data Engineering

## YOUR ROLE
You own all documentation for the Storage Layer sub-block.

## YOUR SCOPE
**Primary Files:**
- `data/cache_manager.py`
- `data/matlab_importer.py`
- `data/data_validator.py`
- `packages/storage/` (if exists)

## ═══════════════════════════════
## PHASE 1: ARCHIVE & EXTRACT
## ═══════════════════════════════

### Tasks:
1. Find all .md files in scope (including `docs/HDF5_MIGRATION_GUIDE.md`
   if Global Team delegates it to you)
2. Check if cache documentation matches actual implementation
3. Verify HDF5 schema documentation
4. Extract valid format specifications
5. Archive old files

## ═══════════════════════════════
## PHASE 2: CREATE NEW DOCS
## ═══════════════════════════════

### Task 2.1: Create `data/STORAGE_README.md`
1. **Storage Architecture:** How data is stored/cached
2. **File Formats Supported:**
   | Format | Reader | Writer | Notes |
   |--------|--------|--------|-------|
3. **Cache Manager:** How caching works (from code)
4. **Data Validation:** Validation pipeline docs
5. **MATLAB Import:** How to import MAT files

### Task 2.2: Create `data/HDF5_GUIDE.md`
1. HDF5 schema used in the project
2. Reading/writing HDF5 data
3. Dataset structure within HDF5 files
4. Migration from other formats

Do NOT claim cache hit rates. Use `[PENDING BENCHMARKS]`.

## OUTPUT FILES
1. `data/STORAGE_README.md`
2. `data/HDF5_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_3_3.md`
[APPEND GENERIC INSTRUCTIONS HERE]

---

# Domain 4: Infrastructure 🛠️

---

## IDB 4.1 — Database

```

# AI Agent Role: Database Sub-Block — Documentation Overhaul

# IDB ID: 4.1

# Domain: Infrastructure

## YOUR ROLE

You own all documentation for the Database sub-block.

## YOUR SCOPE

**Primary Directories:**

- `packages/dashboard/database/` (14 files)
- `packages/dashboard/models/` (25 ORM models)

**Boundaries:**

- Services (IDB 2.2) consume database models
- You own the schema, they own the business logic

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in database/ and models/ directories
2. Check if schema documentation matches actual SQLAlchemy models
3. Verify migration documentation
4. Extract valid ER diagrams if any exist
5. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `packages/dashboard/database/README.md`

1. **Database Architecture:** Connection management from code
2. **Migration Workflow:** How to run/create migrations
3. **Connection Pooling:** Configuration from actual code
4. **Seed Data:** Available seed scripts

### Task 2.2: Create `packages/dashboard/database/SCHEMA_GUIDE.md`

1. **ER Diagram:** Mermaid diagram of ALL tables
2. **Table Catalog:**
   | Table | Model Class | File | Key Columns |
   |-------|------------|------|-------------|
   (Verify each model against actual code)
3. **Relationships:** Foreign keys and associations
4. **Indexing Strategy:** Documented from actual models

### Task 2.3: Create `packages/dashboard/models/README.md`

Catalog of all ORM models with their files and purposes.

## OUTPUT FILES

1. `packages/dashboard/database/README.md`
2. `packages/dashboard/database/SCHEMA_GUIDE.md`
3. `packages/dashboard/models/README.md`
4. `docs/CLEANUP_LOG_IDB_4_1.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 4.2 — Deployment

```

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

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 4.3 — Testing

```

# AI Agent Role: Testing Sub-Block — Documentation Overhaul

# IDB ID: 4.3

# Domain: Infrastructure

## YOUR ROLE

You own all documentation for the Testing sub-block.

## YOUR SCOPE

**Primary Directory:** `tests/` (27+ files)
Key: unit tests, integration tests, stress tests, conftest fixtures.

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in `tests/`
2. Check if test documentation matches actual test files
3. Verify fixture documentation
4. Extract valid test patterns
5. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `tests/README.md`

1. **Test Structure:**
   | Category | Directory/File | Count | Description |
   |----------|---------------|-------|-------------|
   (Count actual test files)
2. **Running Tests:**
   ```bash
   # Exact pytest commands that work
   pytest tests/
   pytest tests/models/ -v
   pytest --cov=packages/ --cov-report=html
   ```
3. **Coverage:** Current setup (do NOT claim coverage percentage,
   use `[RUN pytest --cov TO CHECK]`)
4. **Fixtures:** Key fixtures from conftest.py

### Task 2.2: Create `tests/TESTING_GUIDE.md`

1. Writing new tests (patterns from existing code)
2. Fixture usage guide
3. Mocking/patching patterns
4. Integration test setup
5. Stress/load testing
6. CI/CD integration details

## OUTPUT FILES

1. `tests/README.md`
2. `tests/TESTING_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_4_3.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 4.4 — Configuration

```

# AI Agent Role: Configuration Sub-Block — Documentation Overhaul

# IDB ID: 4.4

# Domain: Infrastructure

## YOUR ROLE

You own all documentation for the Configuration sub-block.

## YOUR SCOPE

**Primary Directory:** `config/` (6 config files)
**Independence Score: 10/10** — Pure data, no logic dependencies.

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in `config/`
2. Verify config parameter documentation against actual code
3. Check if documented defaults match actual defaults
4. Extract valid configuration examples
5. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `config/README.md`

1. **Configuration System Overview:** How config loading works
2. **Config File Catalog:**
   | Config File | Purpose | Key Parameters |
   |------------|---------|----------------|
   (Verify each file exists and its purpose)
3. **Environment Variables:** .env mapping
4. **Quick Start:** Minimal config for first run

### Task 2.2: Create `config/CONFIGURATION_GUIDE.md`

For each config file, create a complete parameter reference:
| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|

Verify EVERY default and type against the actual config code.
Do NOT copy from old docs — read the source code directly.

## OUTPUT FILES

1. `config/README.md`
2. `config/CONFIGURATION_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_4_4.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

# Domain 5: Research & Science 🔬

---

## IDB 5.1 — Research Scripts

```

# AI Agent Role: Research Scripts Sub-Block — Documentation Overhaul

# IDB ID: 5.1

# Domain: Research & Science

## YOUR ROLE

You own all documentation for the Research Scripts sub-block.

## YOUR SCOPE

**Primary Directory:** `scripts/research/` (8 scripts)

Key scripts:

- `ablation_study.py`
- `contrastive_physics.py`
- `hyperparameter_sensitivity.py`
- `pinn_comparison.py`
- `ood_testing.py`

## PROTECTED FILES (in your awareness but NOT your scope to archive):

- `docs/research/` — 8 supporting research files. DO NOT archive.
- `docs/paper/` — Active LaTeX paper. DO NOT touch at all.

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in `scripts/research/` ONLY
2. Check if script descriptions match actual implementations
3. Extract valid usage instructions
4. Archive old .md files in `scripts/research/` only

### SPECIAL TASK — Update docs/research/ IN-PLACE:

The 8 files in `docs/research/` are PROTECTED and must NOT be
archived. However, they contain claimed results that need updating.
For each file in `docs/research/`:

1. Read the file
2. Find any claimed accuracy, F1 scores, benchmark comparisons,
   performance numbers, or stated results
3. Replace those specific values with `[PENDING — run experiment to fill]`
4. Do NOT change anything else in these files
5. Log what values you replaced in your cleanup log

### CRITICAL:

`docs/paper/` is completely off-limits. Do not read, modify, or reference it.

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `scripts/research/README.md`

1. **Script Catalog:**
   | Script | Purpose | CLI Args | Output |
   |--------|---------|----------|--------|
   (Verify each script's argparse args)
2. **Running Experiments:**
   ```bash
   # Exact commands to run each script
   python scripts/research/ablation_study.py --help
   ```
3. **Output Locations:** Where results are saved
4. **Reproducibility:** Seed handling, deterministic settings

### Task 2.2: Create `scripts/research/EXPERIMENT_GUIDE.md`

For each research script:

1. Purpose and methodology
2. Required inputs
3. Configuration options (from argparse)
4. Expected outputs (formats, not values)
5. How to interpret results

ALL results sections use `[PENDING — run experiment to fill]`.

## OUTPUT FILES

1. `scripts/research/README.md`
2. `scripts/research/EXPERIMENT_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_5_1.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

## IDB 5.2 — Visualization

```

# AI Agent Role: Visualization Sub-Block — Documentation Overhaul

# IDB ID: 5.2

# Domain: Research & Science

## YOUR ROLE

You own all documentation for the Visualization sub-block.

## YOUR SCOPE

**Primary Directory:** `visualization/` (13 files)
Key: attention maps, latent space (t-SNE/UMAP), feature importance,
physics embeddings, publication-quality figures.

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in `visualization/`
2. Check if visualization descriptions match actual code
3. Remove any example figures that may not be reproducible
4. Extract valid styling/configuration documentation
5. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `visualization/README.md`

1. **Visualization Types:**
   | Type | File | Input | Output Format |
   |------|------|-------|---------------|
   (Verify each visualization exists)
2. **Styling:** Publication quality settings from actual code
3. **Quick Start:** Generate a basic visualization
4. **Output Formats:** PNG, PDF, SVG support

### Task 2.2: Create `visualization/VISUALIZATION_GUIDE.md`

1. Creating each visualization type
2. Customization options (from actual code parameters)
3. Publication-quality settings
4. Integration with research scripts
5. Matplotlib/Plotly conventions used

Do NOT include example images — they may be outdated.
Just document how to generate them.

## OUTPUT FILES

1. `visualization/README.md`
2. `visualization/VISUALIZATION_GUIDE.md`
3. `docs/CLEANUP_LOG_IDB_5_2.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---

# Domain 6: Integration Layer 🔗

---

## IDB 6.0 — Integration Layer

```

# AI Agent Role: Integration Layer — Documentation Overhaul

# IDB ID: 6.0

# Domain: Integration (Cross-Cutting)

## YOUR ROLE

You own all documentation for the Integration Layer and shared utilities.

## YOUR SCOPE

**Primary Directories:**

- `integration/` — Unified pipeline, model registry, validators
- `packages/dashboard/integrations/` — Celery ↔ Core ML bridge
- `utils/` — Shared utilities (11 files)

## ═══════════════════════════════

## PHASE 1: ARCHIVE & EXTRACT

## ═══════════════════════════════

### Tasks:

1. Find all .md files in `integration/`, `utils/`
2. Verify integration documentation matches actual code
3. Check if pipeline documentation is accurate
4. Extract valid interface documentation
5. Archive old files

## ═══════════════════════════════

## PHASE 2: CREATE NEW DOCS

## ═══════════════════════════════

### Task 2.1: Create `integration/README.md`

1. **Integration Architecture:**
   - How Core ML ↔ Dashboard communication works
   - Model registry usage
   - Unified pipeline flow
2. **Component Catalog:**
   | Component | File | Purpose | Connects |
   |-----------|------|---------|----------|
3. **Data Flow Diagram:** Mermaid diagram showing how data flows
   between domains through integration layer

### Task 2.2: Create `integration/INTEGRATION_GUIDE.md`

1. Adding new integration points
2. Cross-domain data flow patterns
3. Error handling across boundaries
4. Validator usage

### Task 2.3: Create `utils/README.md`

Document ALL utility modules:
| Utility | File | Purpose | Used By |
|---------|------|---------|---------|
(Verify each utility file and its imports)

## OUTPUT FILES

1. `integration/README.md`
2. `integration/INTEGRATION_GUIDE.md`
3. `utils/README.md`
4. `docs/CLEANUP_LOG_IDB_6_0.md`

```

[APPEND GENERIC INSTRUCTIONS HERE]

---
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

---

**Document prepared for team distribution.**
_Each team should receive their specific prompt section + the Generic Instructions block._
_IDB 0.0 (Global Team) should run LAST after all other teams complete._
```
