# IDB Compilation Prompts — All 10 Agents

**Strategy:** 3-Phase Hierarchical Bottom-Up Compilation  
**Based on:** `IDB_COMPILATION_STRATEGY.md`  
**Last Updated:** January 24, 2026

---

# PHASE 1: Domain Consolidation (5 Parallel Agents)

## Agent 1: Core ML Engine Compiler

```
# ROLE: Domain 1 Consolidation Agent
# DOMAIN: Core ML Engine 🧠

## INPUT FILES (10 documents)
Analysis Reports:
- docs/idb_reports/IDB_1_1_MODELS_ANALYSIS.md
- docs/idb_reports/IDB_1_2_TRAINING_ANALYSIS.md
- docs/idb_reports/IDB_1_3_EVALUATION_ANALYSIS.md
- docs/idb_reports/IDB_1_4_FEATURES_ANALYSIS.md
- docs/idb_reports/IDB_1_5_XAI_ANALYSIS.md

Best Practices Reports:
- docs/idb_reports/IDB_1_1_MODELS_BEST_PRACTICES.md
- docs/idb_reports/IDB_1_2_TRAINING_BEST_PRACTICES.md
- docs/idb_reports/IDB_1_3_EVALUATION_BEST_PRACTICES.md
- docs/idb_reports/IDB_1_4_FEATURES_BEST_PRACTICES.md
- docs/idb_reports/IDB_1_5_XAI_BEST_PRACTICES.md

## YOUR MISSION
Consolidate all Core ML Engine findings into ONE comprehensive domain report.

## CRITICAL INSTRUCTIONS
1. READ all 10 input files completely
2. PRESERVE all technical details - do not summarize away important findings
3. MERGE duplicate findings across sub-blocks
4. IDENTIFY cross-sub-block patterns and themes
5. MAINTAIN traceability (which IDB each finding came from)

## OUTPUT STRUCTURE
Follow this exact structure:

# Domain 1: Core ML Engine — Consolidated Analysis

## 1. Domain Overview
- Purpose: [What this domain does]
- Sub-blocks: Models (1.1), Training (1.2), Evaluation (1.3), Features (1.4), XAI (1.5)
- Overall Independence Score: [Average: (9+7+8+9+8)/5 = 8.2]
- Key Interfaces: [List critical APIs]

## 2. Current State Summary
### What's Implemented
[Synthesize from all 5 sub-blocks]

### What's Working Well
[Strengths across the domain]

### What's Problematic
[Common issues across sub-blocks]

## 3. Critical Issues Inventory

### P0 Issues (Critical - Production Blockers)
| IDB | Issue | Impact | Effort | Dependencies |
|-----|-------|--------|--------|--------------|
| [e.g., 1.2] | [Training memory leak] | [Crashes on large datasets] | [8h] | [None] |

### P1 Issues (High Priority)
[Same table format]

### P2 Issues (Medium Priority)
[Same table format]

## 4. Technical Debt Registry

### Quick Wins (< 1 hour)
| IDB | Task | Benefit |
|-----|------|---------|

### Medium Tasks (1-4 hours)
[Table]

### Large Refactors (1+ days)
[Table]

## 5. "If We Could Rewrite" Insights

### Cross-Sub-Block Themes
[Patterns that appear in multiple sub-blocks]

### Fundamental Architectural Changes
1. [e.g., Unify trainer classes into plugin architecture]
2. [...]

### Patterns to Preserve
[What's working well and should NOT be changed]

### Patterns to Eliminate
[Anti-patterns found across sub-blocks]

## 6. Best Practices Observed

### Code Conventions
- **Naming:** [Pattern]
- **Imports:** [Pattern]
- **Docstrings:** [Format used]
- **Type Hints:** [Usage pattern]

### Design Patterns Worth Preserving
1. [e.g., BaseModel abstraction]
2. [model_factory pattern]
3. [...]

### Testing Patterns
[How tests are structured in this domain]

### Interface Contracts
[Critical APIs documented]

## 7. Cross-Domain Dependencies

### Inbound Dependencies (What Core ML needs)
- From Data Domain: [DataLoaders, Datasets]
- From Infrastructure: [Configuration, Database for checkpoints]

### Outbound Dependencies (What depends on Core ML)
- Dashboard consumes: [Model factory, Training API, Evaluation API]
- Research scripts consume: [All Core ML modules]

### Integration Risks
[Where cross-domain coupling could cause issues]

## 8. Domain-Specific Recommendations

### Top 3 Quick Wins
1. [Specific actionable task]
2. [...]
3. [...]

### Top 3 Strategic Improvements
1. [Larger architectural change]
2. [...]
3. [...]

### Team Coordination Requirements
[When changes need coordination with other domains]

## OUTPUT FILE
docs/idb_reports/compiled/DOMAIN_1_CORE_ML_CONSOLIDATED.md
```

---

## Agent 2: Dashboard Platform Compiler

```
# ROLE: Domain 2 Consolidation Agent
# DOMAIN: Dashboard Platform 📊

## INPUT FILES (8 documents)
Analysis:
- IDB_2_1_UI_ANALYSIS.md
- IDB_2_2_SERVICES_ANALYSIS.md
- IDB_2_3_CALLBACKS_ANALYSIS.md
- IDB_2_4_TASKS_ANALYSIS.md

Best Practices:
- IDB_2_1_UI_BEST_PRACTICES.md
- IDB_2_2_SERVICES_BEST_PRACTICES.md
- IDB_2_3_CALLBACKS_BEST_PRACTICES.md
- IDB_2_4_TASKS_BEST_PRACTICES.md

## SPECIAL ATTENTION
- IDB 2.3 (Callbacks) has independence score 4/10 - HIGHEST COUPLING!
- UI sub-block has 42KB settings.py file - flag this prominently

## OUTPUT
docs/idb_reports/compiled/DOMAIN_2_DASHBOARD_CONSOLIDATED.md

[Use same structure as Agent 1]
```

---

## Agent 3: Data Engineering Compiler

```
# ROLE: Domain 3 Consolidation Agent
# DOMAIN: Data Engineering 💾

## INPUT FILES (6 documents)
Analysis:
- IDB_3_1_SIGNAL_GEN_ANALYSIS.md
- IDB_3_2_DATA_LOADING_ANALYSIS.md
- IDB_3_3_STORAGE_ANALYSIS.md

Best Practices:
- IDB_3_1_SIGNAL_GEN_BEST_PRACTICES.md
- IDB_3_2_DATA_LOADING_BEST_PRACTICES.md
- IDB_3_3_STORAGE_BEST_PRACTICES.md

## SPECIAL ATTENTION
- Signal generation has 37KB file - investigate if overly complex
- High independence scores (8-9/10) - should be clean

## OUTPUT
docs/idb_reports/compiled/DOMAIN_3_DATA_CONSOLIDATED.md

[Use same structure as Agent 1]
```

---

## Agent 4: Infrastructure Compiler

```
# ROLE: Domain 4 Consolidation Agent
# DOMAIN: Infrastructure 🛠️

## INPUT FILES (8 documents)
Analysis:
- IDB_4_1_DATABASE_ANALYSIS.md
- IDB_4_2_DEPLOYMENT_ANALYSIS.md
- IDB_4_3_TESTING_ANALYSIS.md
- IDB_4_4_CONFIG_ANALYSIS.md

Best Practices:
- IDB_4_1_DATABASE_BEST_PRACTICES.md
- IDB_4_2_DEPLOYMENT_BEST_PRACTICES.md
- IDB_4_3_TESTING_BEST_PRACTICES.md
- IDB_4_4_CONFIG_BEST_PRACTICES.md

## SPECIAL ATTENTION
- Config has 10/10 independence - pure data, should be exemplary
- Database is critical - schema changes affect everyone

## OUTPUT
docs/idb_reports/compiled/DOMAIN_4_INFRASTRUCTURE_CONSOLIDATED.md

[Use same structure as Agent 1]
```

---

## Agent 5: Research & Integration Compiler

```
# ROLE: Domain 5 + Integration Layer Consolidation Agent
# DOMAINS: Research & Science 🔬 + Integration Layer

## INPUT FILES (6 documents)
Analysis:
- IDB_5_1_RESEARCH_ANALYSIS.md
- IDB_5_2_VISUALIZATION_ANALYSIS.md
- IDB_6_0_INTEGRATION_ANALYSIS.md

Best Practices:
- IDB_5_1_RESEARCH_BEST_PRACTICES.md
- IDB_5_2_VISUALIZATION_BEST_PRACTICES.md
- IDB_6_0_INTEGRATION_BEST_PRACTICES.md

## SPECIAL NOTE
You're combining TWO domains because they're smaller and Integration bridges everything.

## OUTPUT
docs/idb_reports/compiled/DOMAIN_5_RESEARCH_INTEGRATION_CONSOLIDATED.md

[Use same structure but add section for Integration Layer]
```

---

# PHASE 2: Thematic Extraction (4 Sequential Agents)

## Agent 6: Critical Issues Extractor

````
# ROLE: Critical Issues Matrix Builder
# MISSION: Extract ALL critical issues across ALL domains into priority matrix

## INPUT FILES (5 documents)
- DOMAIN_1_CORE_ML_CONSOLIDATED.md
- DOMAIN_2_DASHBOARD_CONSOLIDATED.md
- DOMAIN_3_DATA_CONSOLIDATED.md
- DOMAIN_4_INFRASTRUCTURE_CONSOLIDATED.md
- DOMAIN_5_RESEARCH_INTEGRATION_CONSOLIDATED.md

## YOUR TASK
1. READ section 3 (Critical Issues Inventory) from all 5 domain reports
2. COMBINE all P0/P1/P2 issues into unified tables
3. IDENTIFY which issues block other issues
4. CREATE dependency graph
5. RECOMMEND remediation order

## OUTPUT STRUCTURE

# Critical Issues Matrix — Project-Wide

## Executive Summary
- Total Issues: P0=[X], P1=[Y], P2=[Z]
- By Domain: [Table showing count per domain]
- Total Estimated Effort: [X person-days]
- Critical Path: [Which issues must be fixed first]

## Priority Matrix (Combined View)
| Priority | Domain | IDB | Issue | Impact | Effort | Blocks | Status |
|----------|--------|-----|-------|--------|--------|--------|--------|
| P0 | Core ML | 1.2 | Training memory leak | Production crash | 8h | - | ⬜ |
| P0 | Dashboard | 2.3 | Callback circular dependency | UI breaks | 4h | - | ⬜ |
[... all issues ...]

## P0 Issues Deep Dive
### Issue 1: [Title]
- **Domain:** [X]
- **IDB:** [X.X]
- **Description:** [Detailed]
- **Impact:** [What breaks]
- **Root Cause:** [Why it happens]
- **Remediation Steps:**
  1. [Step]
  2. [Step]
- **Testing:** [How to verify fix]
- **Estimated Effort:** [Xh]

[Repeat for all P0 issues]

## P1 Issues by Category

### Category: Code Quality
[All P1 code quality issues]

### Category: Performance
[All P1 performance issues]

### Category: Testing Gaps
[...]

## P2 Issues by Category
[Same categorization]

## Issue Dependency Graph
```mermaid
graph TD
    I1[Issue 1: Database migration] --> I5[Issue 5: Service layer]
    I2[Issue 2: Model export] --> I7[Issue 7: Deployment]
````

## Clustering Analysis

### Security Issues

[All security-related issues across domains]

### Performance Issues

[All performance issues]

### Maintainability Issues

[...]

## Recommended Remediation Order

Based on dependencies and impact:

### Week 1 (Must Fix First)

1. [Issue X] - Blocks [Y, Z]
2. [...]

### Week 2

[...]

### Week 3-4

[...]

## OUTPUT FILE

docs/idb_reports/compiled/CRITICAL_ISSUES_MATRIX.md

```

---

## Agent 7: Architectural Insights Extractor

```

# ROLE: Architectural Recommendations Synthesizer

# MISSION: Extract "If We Could Rewrite" insights into strategic guidance

## INPUT FILES (5 documents)

[Same 5 domain consolidated reports]

## YOUR TASK

1. READ section 5 ("If We Could Rewrite") from all domains
2. IDENTIFY cross-cutting architectural themes
3. SYNTHESIZE strategic recommendations
4. PROVIDE migration paths from current to ideal state

## OUTPUT STRUCTURE

# Architectural Recommendations — Strategic Insights

## Executive Summary

### Top 5 Architectural Changes Recommended

1. [e.g., Reduce Dashboard coupling through event-driven architecture]
2. [...]

### Patterns to Preserve

[What's working well across all domains]

### Patterns to Eliminate

[Anti-patterns found]

## Cross-Cutting Architectural Themes

### Theme 1: Excessive Coupling in Dashboard Layer

**Evidence:**

- Domain 2 (Dashboard) has lowest independence: 5.0/10 avg
- IDB 2.3 (Callbacks) specifically at 4/10
- Services tightly couple to Core ML and Database

**Root Causes:**

- [Analysis]

**Recommended Solution:**

- [Specific architectural pattern]

**Migration Path:**

- [Step-by-step evolution]

### Theme 2: [Next theme]

[...]

## Domain-Specific Deep Dives

### Core ML Engine

**Current Architecture Assessment:**

- [Strengths]
- [Weaknesses]

**Recommended Changes:**

1. [Specific change]
2. [...]

**Migration Path:**

- Phase 1: [...]
- Phase 2: [...]

### Dashboard Platform

[Same structure]

### Data Engineering

[...]

### Infrastructure

[...]

### Research & Integration

[...]

## Interface Redesign Recommendations

### Interfaces to Refactor

| Interface | Current State | Recommended State | Breaking? | Migration |
| --------- | ------------- | ----------------- | --------- | --------- |

### Backward Compatibility Considerations

[How to avoid breaking existing code]

## Technology Stack Recommendations

### Libraries to Add

| Library | Purpose | Impact |
| ------- | ------- | ------ |

### Libraries to Remove/Replace

[...]

### Framework Upgrades

[...]

## Greenfield Guidance: "If Starting From Scratch..."

### Architecture Patterns We Would Use

1. [e.g., Clean Architecture with clear boundaries]
2. [Event-driven for Dashboard ↔ Core ML communication]
3. [...]

### Technology Choices

- [What we'd pick today vs. what's used]

### File Organization

- [Ideal structure]

### What We Got Right (Don't Change)

- [Patterns already good]

## OUTPUT FILE

docs/idb_reports/compiled/ARCHITECTURAL_RECOMMENDATIONS.md

```

---

## Agent 8: Best Practices Curator

```

# ROLE: Master Style Guide Compiler

# MISSION: Create unified style guide from all best practices

## INPUT FILES (5 documents)

[Same 5 domain consolidated reports]

## YOUR TASK

1. READ section 6 (Best Practices Observed) from all domains
2. IDENTIFY common patterns vs. inconsistencies
3. CREATE unified conventions
4. DOCUMENT anti-patterns to avoid

## OUTPUT STRUCTURE

# Master Style Guide — LSTM_PFD Project

## Purpose

Unified code standards across all domains for consistency and onboarding.

## 1. Python Code Conventions

### Naming Conventions

**Classes:**

- Pattern: PascalCase (e.g., `SignalGenerator`, `CNNTrainer`)
- Observed in: [All domains]
- Exceptions: [If any]

**Functions:**

- Pattern: snake_case (e.g., `create_model`, `extract_features`)
- Observed in: [All domains]

**Variables:**

- Pattern: snake_case
- Constants: UPPER_SNAKE_CASE
- Private: \_leading_underscore

**Files:**

- Pattern: snake_case.py
- Observed in: [All domains]

### Import Organization

Standard order observed:

```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
import numpy as np
import torch

# 3. Local application
from packages.core.models import BaseModel
from config import settings
```

### Type Hints

**Usage Pattern:**

- Function signatures: [Percentage observed: X%]
- Variable annotations: [Percentage: Y%]
- Format: [typing module vs. PEP 604 style]

**Example:**

```python
def create_model(name: str, config: dict) -> BaseModel:
    ...
```

### Docstrings

**Format:** [Google / NumPy / reStructuredText - which is most common?]
**Example:**

```python
def feature_extractor(signal: np.ndarray) -> np.ndarray:
    """Extract features from signal.

    Args:
        signal: Input signal array

    Returns:
        Feature vector
    """
```

## 2. Architecture Patterns

### Model Patterns

1. **BaseModel Abstraction** (Domain 1)
   - All models inherit from BaseModel
   - Required methods: forward(), reset()
   - Registration via model_factory

2. **Factory Pattern** (Domain 1)
   - Used for: Models, Losses, Optimizers

### Service Patterns

1. **Stateless Services** (Domain 2)
   - Methods are pure functions or have clear side effects
   - No global state

### Callback Patterns

1. **Dash Callbacks** (Domain 2)
   - [Pattern observed]

## 3. Testing Standards

### Test Organization

- File naming: `test_<module>.py`
- Directory: `tests/` mirrors `packages/` structure

### Fixture Patterns

- Scope: [function vs module vs session - when to use which]
- Location: conftest.py for shared fixtures

### Mock Patterns

- When to mock: [Guidelines]
- Preferred library: [unittest.mock vs pytest-mock]

### Coverage Requirements

| Domain    | Minimum Coverage |
| --------- | ---------------- |
| Core ML   | 85%              |
| Dashboard | 70%              |
| Data      | 80%              |

## 4. Interface Contract Standards

### API Design Principles

1. [e.g., Accept dictionaries, return dataclasses]
2. [Raise specific exceptions]
3. [...]

### Configuration Patterns

[How configs should be structured]

### Error Handling

```python
# Standard exception pattern
class ModelNotFoundError(ValueError):
    """Raised when model name not in registry."""
    pass
```

## 5. Documentation Standards

### Code Comments

- When to comment: [Complex logic, non-obvious decisions]
- When NOT to comment: [Self-explanatory code]

### Module Docstrings

Required in every module:

```python
"""Module purpose.

This module provides...
"""
```

### README Requirements

Each package needs:

- Purpose
- Usage examples
- API reference link

## 6. Domain-Specific Conventions

### Core ML

- Model configs use dataclasses
- All models must be ONNX-exportable

### Dashboard

- Callback IDs follow pattern: `{page}-{component}-{action}`

### Data Engineering

- Dataset classes must inherit from torch.utils.data.Dataset

## 7. Anti-Patterns to Avoid

### 1. God Classes

**Problem:** [e.g., settings.py at 42KB]
**Solution:** [Split into focused modules]

### 2. Circular Dependencies

**Problem:** [Found in Dashboard callbacks]
**Solution:** [Use dependency injection]

### 3. [More anti-patterns from analysis]

[...]

## 8. Onboarding Checklist

Before contributing to this codebase:

- [ ] Read this style guide
- [ ] Read EXECUTIVE_DASHBOARD.md
- [ ] Read relevant domain consolidated report
- [ ] Review INTERFACE_CONTRACTS_CATALOG.md
- [ ] Run existing tests to verify setup

## OUTPUT FILE

docs/idb_reports/compiled/MASTER_STYLE_GUIDE.md

```

---

## Agent 9: Integration Mapper

```

# ROLE: Interface Contracts Cataloger

# MISSION: Document all cross-domain interfaces

## INPUT FILES (5 documents)

[Same 5 domain consolidated reports]

## YOUR TASK

1. READ section 7 (Cross-Domain Dependencies) from all domains
2. MAP every interface where domains interact
3. DOCUMENT contract details
4. ASSESS stability and risk

## OUTPUT STRUCTURE

# Interface Contracts Catalog — Cross-Domain APIs

## Purpose

Comprehensive documentation of boundaries where domains interact.

## 1. Interface Inventory

| From Domain | To Domain | Interface Name  | Location                              | Stability | Risk |
| ----------- | --------- | --------------- | ------------------------------------- | --------- | ---- |
| Core ML     | Dashboard | Model Factory   | packages/core/models/model_factory.py | STABLE    | LOW  |
| Core ML     | Dashboard | Training API    | packages/core/training/trainer.py     | STABLE    | LOW  |
| Dashboard   | Core ML   | Task Submission | packages/dashboard/tasks/             | FRAGILE   | HIGH |

[... all interfaces ...]

## 2. Core ML ↔ Dashboard Interfaces

### Interface 1: Model Factory

**API Signature:**

```python
def create_model(name: str, config: dict) -> BaseModel:
    """Create model instance by name."""
```

**Location:** `packages/core/models/model_factory.py`

**Contract Details:**

- **Input:**
  - `name` (str): Registered model name
  - `config` (dict): Model configuration
- **Output:** PyTorch nn.Module instance
- **Exceptions:**
  - `ModelNotFoundError`: Name not registered
  - `InvalidConfigError`: Config validation fails

**Consumers:**

- `packages/dashboard/services/experiment_service.py`
- `packages/dashboard/tasks/training_tasks.py`

**Stability:** ⬜ STABLE — Do not break!

**Testing:**

```python
# How to test this interface
from packages.core.models import create_model
model = create_model("cnn", {"input_dim": 52})
assert isinstance(model, BaseModel)
```

**Change Protocol:**
If this interface must change:

1. Discuss with Dashboard team
2. Create deprecation path
3. Update consumers in same PR

### Interface 2: [Next interface]

[Same detailed documentation]

## 3. Data ↔ Core ML Interfaces

[Document all Data->CoreML interfaces]

## 4. Services ↔ Callbacks Interfaces

⚠️ **HIGH COUPLING AREA**

[Document this carefully - it's the 4/10 independence zone]

## 5. Services ↔ Celery Tasks Interfaces

[...]

## 6. Database ↔ Services Interfaces

### ORM Models Used

| Service           | Models Used             | Operations |
| ----------------- | ----------------------- | ---------- |
| ExperimentService | Experiment, Run, Metric | CRUD       |

[...]

## 7. Integration Risk Assessment

### High-Risk Interfaces (Many Dependencies)

1. **ExperimentService** → Used by 15+ callbacks
   - Risk: Changes break many UI components
   - Mitigation: [Versioning strategy]

2. [...]

### Stable Interfaces (Safe to Depend On)

- Model Factory: Used widely, well-tested
- DataLoader: PyTorch standard

### Deprecated Interfaces

[Any planned for removal]

## 8. Versioning Strategy

For future interface changes:

1. Semantic versioning for major APIs
2. Deprecation warnings before removal
3. 2-version support period

## OUTPUT FILE

docs/idb_reports/compiled/INTERFACE_CONTRACTS_CATALOG.md

```

---

# PHASE 3: Executive Synthesis (1 Final Agent)

## Agent 10: Executive Synthesizer

```

# ROLE: Executive Synthesizer

# MISSION: Create executive summary and action plan

## INPUT FILES (9 documents)

Phase 1 outputs:

- DOMAIN_1_CORE_ML_CONSOLIDATED.md
- DOMAIN_2_DASHBOARD_CONSOLIDATED.md
- DOMAIN_3_DATA_CONSOLIDATED.md
- DOMAIN_4_INFRASTRUCTURE_CONSOLIDATED.md
- DOMAIN_5_RESEARCH_INTEGRATION_CONSOLIDATED.md

Phase 2 outputs:

- CRITICAL_ISSUES_MATRIX.md
- ARCHITECTURAL_RECOMMENDATIONS.md
- MASTER_STYLE_GUIDE.md
- INTERFACE_CONTRACTS_CATALOG.md

## YOUR TASK

Create TWO executive documents for leadership/stakeholders

## OUTPUT 1: EXECUTIVE_DASHBOARD.md

See IDB_COMPILATION_STRATEGY.md for exact structure.

Key sections:

- Project Health Scorecard with grades
- Domain Health Matrix
- Critical Issues Summary
- Strategic Recommendations
- Next Steps

## OUTPUT 2: ACTION_PLAN_AND_ROADMAP.md

See IDB_COMPILATION_STRATEGY.md for exact structure.

Key sections:

- Phased Remediation Plan (0-3)
- Dependency-Ordered Task List
- Team Allocation
- Progress Tracking metrics

## OUTPUT FILES

- docs/idb_reports/compiled/EXECUTIVE_DASHBOARD.md
- docs/idb_reports/compiled/ACTION_PLAN_AND_ROADMAP.md

```

---

## Agent 11: Scope & Limitations Extractor

```

# ROLE: Project Scope Analyst

# MISSION: Extract implicit scope, detect mismatches, define project limitations

## CONTEXT

This project claims to be a "bearing fault diagnosis system" but:

- "PFD" in LSTM_PFD is never defined
- The physics model uses Sommerfeld equation (journal bearings)
- CWRU benchmark is ball bearing data (different physics)
- The README lists inconsistent fault types vs. actual implementation

## INPUT FILES

Primary Sources:

- data/signal_generator.py (37KB - physics model)
- utils/constants.py (fault types, physics parameters)
- config/data_config.py (operating conditions)
- data/cwru_dataset.py (CWRU loader)
- packages/core/models/physics/ (bearing_dynamics.py, operating_conditions.py)
- README.md (claims made)
- docs/research/pinn-theory.md (physics theory)

IDB Reports (for cross-reference):

- All 5 domain consolidated reports

## YOUR TASKS

### Task 1: Define What "PFD" Means

Search the entire codebase and documentation. What does PFD stand for?
If undefined, propose a definition based on what the code actually does.

### Task 2: Identify Implicit Scope (What the code ACTUALLY does)

#### Bearing Type Analysis

- What physics equations are used? (Sommerfeld, Reynolds, etc.)
- What bearing geometry is assumed?
- Is this for: Journal bearings? Rolling element? Both?

#### Operating Envelope

- What speed range is supported? (from config)
- What load range? (from config)
- What temperature range? (from config)
- What sampling rate? (from constants)

#### Fault Types

- What faults are actually implemented in signal_generator.py?
- Do these match what README claims?
- Are these journal bearing faults or ball bearing faults?

### Task 3: Detect Mismatches

#### Synthetic Data vs. Benchmarks

| Aspect        | Synthetic Data | CWRU Dataset     | Match? |
| ------------- | -------------- | ---------------- | ------ |
| Bearing Type  | [?]            | Ball bearing     | [?]    |
| Fault Types   | [list]         | Inner/Outer/Ball | [?]    |
| Physics Model | Sommerfeld     | N/A              | [?]    |
| Sampling Rate | 20480 Hz       | 12000 Hz         | ❌     |

#### README vs. Implementation

| Claim in README    | Actually Implemented      | Match? |
| ------------------ | ------------------------- | ------ |
| "11 Fault Types"   | [count from constants.py] | [?]    |
| Fault names listed | Fault names in code       | [?]    |

### Task 4: Assess Research Viability

#### Can This Be Published?

- What scope claims can the paper realistically make?
- What datasets can be used for validation?
- What are the necessary caveats/limitations sections?

Draft limitation statement for a paper:

```
This work focuses on [X] bearings and has been validated on [Y].
Generalization to [Z] would require [additional work].
```

### Task 5: Assess Commercial Viability

#### What Markets Is This Suitable For?

| Application                     | Fit   | Reason   |
| ------------------------------- | ----- | -------- |
| Turbomachinery (oil-lubricated) | ✅/❌ | [reason] |
| CNC Spindles                    | ✅/❌ | [reason] |
| Pumps/Compressors               | ✅/❌ | [reason] |
| Electric Motors (ball bearings) | ✅/❌ | [reason] |
| Automotive                      | ✅/❌ | [reason] |

### Task 6: Recommendations

#### Immediate Documentation Fixes

1. Define PFD in README
2. Correct fault type descriptions
3. Add explicit scope section

#### Code Fixes for Consistency

1. [List mismatches that need code changes]

#### For Expanding Scope (Future Work)

What would it take to support:

- Ball bearings?
- Different sampling rates?
- Different operating conditions?

## OUTPUT STRUCTURE

# Project Scope Analysis — LSTM_PFD

## 1. What is PFD?

- Current definition: [None found / Found at X]
- Proposed definition: [Based on code analysis]

## 2. Discovered Implicit Scope

### Bearing Type

- **Implemented:** [Journal / Ball / Both]
- **Evidence:** [files and code snippets]
- **Physics equations used:** [list]

### Operating Envelope

| Parameter          | Min | Max | Configurable? |
| ------------------ | --- | --- | ------------- |
| Speed (RPM)        |     |     | Yes/No        |
| Load (%)           |     |     | Yes/No        |
| Temperature (°C)   |     |     | Yes/No        |
| Sampling Rate (Hz) |     |     | Yes/No        |

### Fault Types (Actual Implementation)

| ID  | Internal Name | Physics Basis       | Category |
| --- | ------------- | ------------------- | -------- |
| 0   | sain          | Healthy             | Baseline |
| 1   | desalignement | Misalignment forces | Journal  |
| ... | ...           | ...                 | ...      |

### Signal Characteristics

- Length: [X samples = Y seconds]
- Frequency range: [0 - Nyquist]

## 3. Mismatch Analysis

### Synthetic Data ↔ CWRU Benchmark

[Table with detailed comparison]

**Severity:** 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW

**Impact:** [What this means for claims]

### README ↔ Implementation

[Table with detailed comparison]

**Recommendation:** [Fix README / Fix Code / Both]

## 4. Explicit Limitations to Document

### Must Add to README

```markdown
## Scope & Limitations

This system is designed for:

- **Bearing Type:** [X]
- **Operating Conditions:** [Y]
- **Fault Types:** [Z]

Not suitable for:

- [List]
```

### Must Add to Research Paper

```latex
\section{Limitations}
The proposed method has been developed and validated for
[specific scope]. Extension to [other domains] would require
[specific modifications].
```

## 5. Research Paper Scope Statement (Draft)

### Valid Claims

- "State-of-the-art for [X] bearings"
- "Validated on synthetic data with physics-informed constraints"

### Invalid Claims (Without Additional Work)

- "Generalizes to all bearing types"
- "Validated on CWRU" (if physics mismatch not addressed)

### Recommended Title Adjustment

Current: "Advanced Bearing Fault Diagnosis System"
Suggested: "[More specific title based on actual scope]"

## 6. Commercial Applicability

### Strong Fit ✅

| Industry | Application | Why |
| -------- | ----------- | --- |

### Poor Fit ❌

| Industry | Application | Why Not |
| -------- | ----------- | ------- |

### Could Fit With Modifications

| Industry | Required Changes | Effort |
| -------- | ---------------- | ------ |

## 7. Recommendations Summary

### Priority 1: Documentation (< 1 day)

1. [Action]
2. [Action]

### Priority 2: Code Fixes (1-3 days)

1. [Action]
2. [Action]

### Priority 3: Scope Expansion (Future Work)

1. [Action with effort estimate]
2. [Action with effort estimate]

## OUTPUT FILE

docs/idb_reports/compiled/PROJECT_SCOPE_ANALYSIS.md

```

---

# Execution Instructions

## Step 1: Run Phase 1 (Parallel)
Can run ALL 5 agents simultaneously:
- Agent 1 (Core ML)
- Agent 2 (Dashboard)
- Agent 3 (Data)
- Agent 4 (Infrastructure)
- Agent 5 (Research+Integration)

## Step 2: Run Phase 2 (Sequential)
Must run IN ORDER (each needs Phase 1 complete):
- Agent 6 (Critical Issues)
- Agent 7 (Architectural Insights)
- Agent 8 (Best Practices)
- Agent 9 (Integration Mapper)
- **Agent 11 (Scope & Limitations)** ← NEW

## Step 3: Run Phase 3
- Agent 10 (Executive Synthesis) — needs ALL previous outputs including Scope Analysis

---

**Total Prompts:** 11
**Total Output Files:** 12 (includes PROJECT_SCOPE_ANALYSIS.md)
**Execution Time:** ~2-3 hours with parallel Phase 1

```
