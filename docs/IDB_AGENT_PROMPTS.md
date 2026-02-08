# IDB Agent Prompts — AI Development Team Instructions

**Purpose:** This document contains structured prompts for AI agents assigned to analyze each Independent Development Block (IDB). Each agent will perform a comprehensive review and produce deliverables for quality improvement and consistency.

**Last Updated:** January 22, 2026

---

## Overview

Each AI agent will receive:

1. **Prompt 1: Block Analysis & Critical Evaluation** — Deep dive into the current implementation
2. **Prompt 2: Best Practices Extraction** — Document patterns for future developers

The prompts are designed so each agent:

- Understands their **exact scope** (files, directories, responsibilities)
- Knows their **boundaries** (what NOT to touch)
- Produces **consistent deliverables** (same format across all teams)
- Focuses on **professional quality improvement**

---

## Deliverables Format (Standard for All Agents)

Each agent will produce two documents:

### Deliverable 1: `IDB_{BLOCK_ID}_ANALYSIS.md`

```markdown
# IDB Analysis Report: {Block Name}

## 1. Executive Summary

## 2. Current State Assessment

## 3. Critical Issues (P0/P1/P2)

## 4. If I Could Rewrite This Block (Retrospective Analysis)

## 5. Recommended Fixes (Quick Wins vs Major Refactors)

## 6. Integration Risk Assessment

## 7. Technical Debt Inventory
```

### Deliverable 2: `IDB_{BLOCK_ID}_BEST_PRACTICES.md`

```markdown
# Best Practices: {Block Name}

## 1. Patterns Worth Preserving

## 2. Code Style & Conventions Observed

## 3. Interface Contract Documentation

## 4. Testing Patterns

## 5. Recommendations for Future Developers

## 6. Cross-Team Coordination Notes
```

---

# Domain 1: Core ML Engine 🧠

---

## IDB 1.1 — Models Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Models Sub-Block Analyst
# IDB ID: 1.1
# Domain: Core ML Engine

## YOUR SCOPE
You are assigned to analyze the **Models Sub-Block** of the LSTM_PFD project. Your scope is STRICTLY LIMITED to:

**Primary Directory:** `packages/core/models/`
**Files to Analyze:** All 61 files (13 subdirectories + 7 top-level files)
**Sub-directories in scope:**
- `models/classical/` — SVM, Random Forest, Gradient Boosting wrappers
- `models/cnn/` — Advanced CNN variants
- `models/transformer/` — ViT-1D, PatchTST, TSMixer
- `models/pinn/` — Physics-Informed Neural Networks
- `models/ensemble/` — Voting, Stacking, Boosting, MoE
- `models/fusion/` — Early and Late Fusion architectures

**You are NOT allowed to analyze or modify:**
- Training logic (`packages/core/training/`)
- Evaluation logic (`packages/core/evaluation/`)
- Dashboard code
- Any other packages

## KEY INTERFACES YOU MUST UNDERSTAND
1. `BaseModel` abstract class — all models inherit from this
2. `model_factory.py` — `create_model()`, `create_model_from_config()`, `register_model()`
3. Standard PyTorch module interface (`forward()`, `state_dict()`)

## ANALYSIS TASKS

### Task 1: Current State Assessment
- List all model architectures implemented
- Document the inheritance hierarchy
- Identify which models follow the BaseModel pattern and which deviate
- Check for consistency in forward() signatures
- Verify ONNX exportability compliance

### Task 2: Critical Issues Identification
Rate each issue as:
- **P0 (Critical)**: Blocks production, causes crashes, data loss
- **P1 (High)**: Major functionality broken, significant tech debt
- **P2 (Medium)**: Code smells, inefficiencies, maintainability issues

Look specifically for:
- Hardcoded magic numbers (batch sizes, dimensions, etc.)
- Inconsistent model interfaces
- Missing or incorrect type hints
- Copy-paste code (DRY violations)
- Dead code or unused models
- Missing docstrings or documentation
- Incorrect inheritance patterns
- Memory leaks in forward passes
- Device handling issues (CPU/GPU)

### Task 3: "If I Could Rewrite This" Retrospective
Assume you're starting fresh. Answer:
1. What fundamental architectural decisions would you change?
2. Are there models that shouldn't exist (redundant/obsolete)?
3. Is the BaseModel abstraction correct, or is it over/under-engineered?
4. Is the model_factory pattern the right choice?
5. How would you restructure the 13 subdirectories?
6. What external dependencies would you add/remove?
7. What design patterns are missing that should be there?

### Task 4: Technical Debt Inventory
Create a prioritized list of all technical debt items with estimated effort:
- **Quick Win (< 1 hour)**: Typos, simple refactors
- **Medium (1-4 hours)**: Interface fixes, documentation
- **Large (1+ days)**: Architecture changes, rewrites

## OUTPUT REQUIREMENTS
Produce: `IDB_1_1_MODELS_ANALYSIS.md`
Save to: `docs/idb_reports/` directory

Format your findings professionally with:
- Tables for comparisons
- Code snippets for evidence
- Mermaid diagrams for architecture visualization
- Severity ratings for all issues
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Models Sub-Block — Best Practices Curator
# IDB ID: 1.1
# Domain: Core ML Engine

## CONTEXT
You have just completed the analysis of the Models Sub-Block. Now extract all best practices, patterns, and conventions that should be preserved and shared with other teams.

## YOUR MISSION
Document everything a NEW developer would need to know before touching this codebase. This document will be:
1. Shared with other IDB teams for consistency verification
2. Used as onboarding material for future developers
3. Compiled into a master style guide

## EXTRACTION TASKS

### Task 1: Patterns Worth Preserving
Identify coding patterns that are GOOD and should be replicated:
- How are models registered?
- How is the factory pattern implemented?
- How are configurations handled?
- How are models made serializable?
- What naming conventions are followed?

### Task 2: Code Style & Conventions
Document observed conventions:
- Naming: classes, functions, variables, files
- Import ordering
- Docstring format (Google/NumPy/reStructuredText?)
- Type hint usage
- Comment styles
- Module organization within files

### Task 3: Interface Contracts
Document the contracts that MUST BE PRESERVED:
- BaseModel interface requirements
- model_factory registration protocol
- Input/output tensor shapes and types
- Device handling conventions
- State dict structure

### Task 4: Testing Patterns
Document how models should be tested:
- Unit test structure
- Mock patterns
- Fixture usage
- What coverage is expected?

### Task 5: Future Developer Recommendations
Based on your analysis, what should future developers:
- ALWAYS do when adding a new model?
- NEVER do?
- Be CAREFUL about?

### Task 6: Cross-Team Coordination Notes
Document interfaces that touch other teams:
- How Training sub-block consumes models
- How Evaluation sub-block uses models
- How Dashboard displays model information
- What changes would require cross-team coordination?

## OUTPUT REQUIREMENTS
Produce: `IDB_1_1_MODELS_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 1.2 — Training Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Training Sub-Block Analyst
# IDB ID: 1.2
# Domain: Core ML Engine

## YOUR SCOPE
You are assigned to analyze the **Training Sub-Block** of the LSTM_PFD project.

**Primary Directory:** `packages/core/training/`
**Files to Analyze:** All 23 files in this directory
**Key Components:**
- `trainer.py` — Standard training interface
- `cnn_trainer.py`, `pinn_trainer.py`, `spectrogram_trainer.py` — Specialized trainers
- `cnn_callbacks.py` — Callback system (EarlyStopping, ModelCheckpoint, etc.)
- `losses.py`, `cnn_losses.py`, `physics_loss_functions.py` — Loss functions
- Optimizer and scheduler configurations
- Advanced techniques: Mixed Precision, Progressive Resizing, Knowledge Distillation

**You are NOT allowed to analyze or modify:**
- Model architectures (`packages/core/models/`)
- Data loading (`data/`)
- Dashboard code
- Any other packages

## DEPENDENCIES TO UNDERSTAND
**Inbound:**
- `packages/core/models/` — Models being trained
- `data/` — DataLoaders providing batches
- `config/training_config.py` — Training configurations

**Outbound:**
- Checkpoints → `checkpoints/`
- Metrics → `packages/core/evaluation/`

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Map all trainer classes and their inheritance
- Document the callback system architecture
- List all loss functions and their use cases
- Identify optimizer and scheduler options
- Check for advanced training techniques implementation status

### Task 2: Critical Issues Identification
Rate each issue P0/P1/P2. Look specifically for:
- Training loop bugs (gradient accumulation, loss computation)
- Callback ordering issues
- Memory leaks during training
- Checkpoint save/load reliability
- Mixed precision implementation correctness
- Distributed training support
- Reproducibility issues (seed handling)
- Learning rate scheduler bugs
- Early stopping edge cases

### Task 3: "If I Could Rewrite This" Retrospective
1. Is the trainer hierarchy correct? Too many specialized trainers?
2. Should there be one unified trainer with plugin architecture?
3. Is the callback system well-designed?
4. How would you handle distributed training differently?
5. Are loss functions in the right place?
6. Is the checkpoint format future-proof?

### Task 4: Integration Risk Assessment
Identify risks for teams that depend on Training:
- What happens if training API changes?
- Which dashboard features would break?
- Are there hidden assumptions about model interfaces?

## OUTPUT REQUIREMENTS
Produce: `IDB_1_2_TRAINING_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Training Sub-Block — Best Practices Curator
# IDB ID: 1.2
# Domain: Core ML Engine

## YOUR MISSION
Document all training-related best practices for consistency across teams.

## EXTRACTION TASKS

### Task 1: Training Loop Patterns
- How is the training loop structured?
- What is the epoch/batch logging convention?
- How are metrics accumulated?

### Task 2: Callback System Conventions
- How are callbacks registered?
- What callback hooks are available?
- How should new callbacks be implemented?

### Task 3: Loss Function Patterns
- How are loss functions organized?
- How to add a new loss function?
- Multi-task loss weighting conventions

### Task 4: Checkpoint Conventions
- What is saved in a checkpoint?
- Filename conventions
- Best checkpoint selection logic

### Task 5: Reproducibility Requirements
- Seed setting requirements
- Deterministic mode expectations
- What should be logged for reproducibility?

### Task 6: Cross-Team Coordination Notes
- How Evaluation team should load trained models
- How Dashboard surfaces training progress
- Configuration passing conventions

## OUTPUT REQUIREMENTS
Produce: `IDB_1_2_TRAINING_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 1.3 — Evaluation Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Evaluation Sub-Block Analyst
# IDB ID: 1.3
# Domain: Core ML Engine

## YOUR SCOPE
**Primary Directory:** `packages/core/evaluation/`
**Files to Analyze:** All 16 files
**Key Components:**
- `evaluator.py` — Base evaluation interface
- `cnn_evaluator.py`, `ensemble_evaluator.py`, `pinn_evaluator.py` — Specialized evaluators
- `attention_visualization.py`, `physics_interpretability.py` — Visualization tools

**Boundaries:** Only analyze evaluation logic, not training or models.

## DEPENDENCIES
**Inbound:** Trained models, Test data loaders
**Outbound:** Results → `results/`, Visualizations → `visualization/`

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document all metric types computed
- Map evaluator inheritance hierarchy
- Check output format consistency
- Verify visualization tools work standalone

### Task 2: Critical Issues Identification
Look for:
- Metric computation bugs
- Memory issues with large test sets
- Inconsistent output formats
- Missing metrics for specific model types
- Visualization quality issues

### Task 3: "If I Could Rewrite This" Retrospective
- Is there too much duplication across evaluators?
- Should metrics be computed lazily or eagerly?
- Is the results storage format extensible?

## OUTPUT REQUIREMENTS
Produce: `IDB_1_3_EVALUATION_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Evaluation Sub-Block — Best Practices Curator
# IDB ID: 1.3

## EXTRACTION TASKS
1. Metric computation patterns
2. Results serialization conventions
3. Visualization styling requirements
4. Cross-team result format contracts
5. Testing patterns for evaluators

## OUTPUT REQUIREMENTS
Produce: `IDB_1_3_EVALUATION_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 1.4 — Features Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Features Sub-Block Analyst
# IDB ID: 1.4
# Domain: Core ML Engine

## YOUR SCOPE
**Primary Directory:** `packages/core/features/`
**Files to Analyze:** All 12 files
**Key Components:**
- `feature_extractor.py` — Main extraction pipeline
- `feature_selector.py` — Feature selection algorithms
- `feature_importance.py` — SHAP/Permutation importance
- Domain-specific: `time_domain.py`, `frequency_domain.py`, `wavelet_features.py`

## ANALYSIS TASKS

### Task 1: Current State Assessment
- List all feature types extracted (target: 52 features)
- Document the extraction pipeline flow
- Check normalization/validation logic
- Verify feature importance tools work

### Task 2: Critical Issues Identification
Look for:
- Numerical stability issues
- Feature computation correctness
- Missing feature documentation
- Hardcoded parameters
- Inefficient computations

### Task 3: "If I Could Rewrite This" Retrospective
- Is the feature set scientifically sound?
- Are there features that should be removed/added?
- Is the pipeline efficient for streaming data?

## OUTPUT REQUIREMENTS
Produce: `IDB_1_4_FEATURES_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Features Sub-Block — Best Practices Curator
# IDB ID: 1.4

## EXTRACTION TASKS
1. Feature naming conventions
2. Vectorization patterns
3. Numerical stability practices
4. Feature validation patterns
5. Documentation requirements for features

## OUTPUT REQUIREMENTS
Produce: `IDB_1_4_FEATURES_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 1.5 — Explainability (XAI) Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Explainability Sub-Block Analyst
# IDB ID: 1.5
# Domain: Core ML Engine

## YOUR SCOPE
**Primary Directory:** `packages/core/explainability/`
**Files to Analyze:** All 8 files
**Key Components:**
- `SHAPExplainer` — SHAP-based explanations
- `LIMEExplainer` — LIME-based explanations
- `IntegratedGradientsExplainer` — Gradient-based attribution
- `UncertaintyQuantifier` — MC Dropout uncertainty

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document all XAI methods implemented
- Check compliance with XAI best practices
- Verify caching mechanism (`explanation_cache.py`)
- Test visualization outputs

### Task 2: Critical Issues Identification
Look for:
- Incorrect attribution implementations
- Performance issues with large models
- Cache invalidation bugs
- Misleading visualizations

### Task 3: "If I Could Rewrite This" Retrospective
- Are all XAI methods scientifically sound?
- Is the caching strategy optimal?
- Should there be a unified explainer interface?

## OUTPUT REQUIREMENTS
Produce: `IDB_1_5_XAI_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Explainability Sub-Block — Best Practices Curator
# IDB ID: 1.5

## EXTRACTION TASKS
1. Explainer interface patterns
2. Caching conventions
3. Visualization styling for XAI
4. Performance optimization patterns
5. Scientific correctness validation

## OUTPUT REQUIREMENTS
Produce: `IDB_1_5_XAI_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

# Domain 2: Dashboard Platform 📊

---

## IDB 2.1 — Frontend/UI Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Frontend/UI Sub-Block Analyst
# IDB ID: 2.1
# Domain: Dashboard Platform

## YOUR SCOPE
**Primary Directories:**
- `packages/dashboard/layouts/` (24 files)
- `packages/dashboard/components/` (6 files)
- `packages/dashboard/assets/`

**Key Pages:**
- `home.py` — Home Dashboard (Low complexity)
- `experiments.py`, `experiment_wizard.py`, `experiment_results.py` — Experiments (High)
- `data_explorer.py` — Data Explorer (Medium)
- `signal_viewer.py` — Signal Viewer (Medium)
- `xai_dashboard.py` — XAI Dashboard (High)
- `settings.py` — Settings (Very High - 42KB!)

**You are NOT allowed to analyze:**
- Backend services (`packages/dashboard/services/`)
- Callbacks (`packages/dashboard/callbacks/`)
- Database models

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Map all layouts and their components
- Document component reuse patterns
- Check CSS/styling consistency
- Identify the largest/most complex files

### Task 2: Critical Issues Identification
Look for:
- Overly large files (especially `settings.py` at 42KB)
- Inconsistent styling/theming
- Accessibility issues
- Mobile responsiveness issues
- Hardcoded strings (i18n issues)
- Dead code in layouts
- Component duplication

### Task 3: "If I Could Rewrite This" Retrospective
- Should `settings.py` be split?
- Are components properly reusable?
- Is the layout structure logical?
- Should routing be restructured?

## OUTPUT REQUIREMENTS
Produce: `IDB_2_1_UI_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Frontend/UI Sub-Block — Best Practices Curator
# IDB ID: 2.1

## EXTRACTION TASKS
1. Layout organization patterns
2. Component design conventions
3. CSS/styling patterns
4. Asset management conventions
5. Accessibility requirements
6. Responsive design patterns

## OUTPUT REQUIREMENTS
Produce: `IDB_2_1_UI_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 2.2 — Backend Services Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Backend Services Sub-Block Analyst
# IDB ID: 2.2
# Domain: Dashboard Platform

## YOUR SCOPE
**Primary Directory:** `packages/dashboard/services/` (24 services + notification providers)

**Key Services:**
- `DataService` — Dataset metadata and signal extraction
- `ExperimentService` — Experiment lifecycle
- `AuthenticationService` — User auth and session management
- `NotificationService` — Multi-channel notifications
- `XAIService` — Explainability orchestration
- `HPOService`, `NASService` — Hyperparameter/Architecture search

**Independence Score: 5/10** — This is a high-coupling central orchestration layer!

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Map all services and their responsibilities
- Document service dependencies (which services call which)
- Check for proper error handling
- Verify transaction management

### Task 2: Critical Issues Identification
Look for:
- God-class services (doing too much)
- Circular dependencies between services
- Missing exception handling
- Database session management issues
- Authentication/authorization holes
- Rate limiting gaps
- Caching inconsistencies

### Task 3: "If I Could Rewrite This" Retrospective
- Are services properly scoped?
- Should some services be merged/split?
- Is dependency injection used properly?
- Is there a proper service layer abstraction?

## OUTPUT REQUIREMENTS
Produce: `IDB_2_2_SERVICES_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Backend Services Sub-Block — Best Practices Curator
# IDB ID: 2.2

## EXTRACTION TASKS
1. Service method conventions
2. Error handling patterns
3. Transaction management
4. Caching strategies
5. Logging conventions
6. Dependency injection patterns

## OUTPUT REQUIREMENTS
Produce: `IDB_2_2_SERVICES_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 2.3 — Callbacks Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Callbacks Sub-Block Analyst
# IDB ID: 2.3
# Domain: Dashboard Platform

## YOUR SCOPE
**Primary Directory:** `packages/dashboard/callbacks/` (29 callback modules)

**Independence Score: 4/10** — HIGHEST COUPLING in the entire system!
This is the "glue" layer between UI and services.

## CRITICAL WARNING
This sub-block has the highest coupling. Your analysis must identify:
- Which callbacks are tightly coupled to which layouts
- Which callbacks are tightly coupled to which services
- Circular dependency risks

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Map callbacks to their layouts (1:1? 1:many? many:1?)
- Document Input/Output patterns used
- Check for callback chaining patterns
- Identify any clientside callbacks

### Task 2: Critical Issues Identification
Look for:
- Circular callback dependencies (Dash's #1 bug)
- Callbacks that do too much (should be in services)
- Missing error boundaries
- State management issues
- Performance bottlenecks (slow callbacks)
- Memory leaks in callbacks

### Task 3: "If I Could Rewrite This" Retrospective
- Should callbacks be thinner (just orchestration)?
- Is the callback file organization logical?
- Are patterns like clientside callbacks underutilized?

## OUTPUT REQUIREMENTS
Produce: `IDB_2_3_CALLBACKS_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Callbacks Sub-Block — Best Practices Curator
# IDB ID: 2.3

## EXTRACTION TASKS
1. Callback organization patterns
2. Input/Output conventions
3. State management patterns
4. Error handling in callbacks
5. Performance optimization patterns
6. Testing callback patterns

## OUTPUT REQUIREMENTS
Produce: `IDB_2_3_CALLBACKS_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 2.4 — Async Tasks Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Async Tasks Sub-Block Analyst
# IDB ID: 2.4
# Domain: Dashboard Platform

## YOUR SCOPE
**Primary Directory:** `packages/dashboard/tasks/` (11 task modules)

**Task Types:**
- Training tasks
- HPO tasks
- NAS tasks
- Data generation tasks
- XAI tasks
- Deployment tasks
- Testing tasks

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Map all Celery tasks and their signatures
- Document task result patterns
- Check task state update mechanisms
- Verify error handling and retry logic

### Task 2: Critical Issues Identification
Look for:
- Task timeout issues
- Result backend inconsistencies
- Missing task monitoring
- Retry logic bugs
- Task chaining issues
- Resource cleanup failures

### Task 3: "If I Could Rewrite This" Retrospective
- Is task granularity appropriate?
- Should some tasks be split/merged?
- Is the result backend choice optimal?

## OUTPUT REQUIREMENTS
Produce: `IDB_2_4_TASKS_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Async Tasks Sub-Block — Best Practices Curator
# IDB ID: 2.4

## EXTRACTION TASKS
1. Task definition patterns
2. Result handling conventions
3. Retry and error handling patterns
4. Monitoring conventions
5. Task chaining patterns

## OUTPUT REQUIREMENTS
Produce: `IDB_2_4_TASKS_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

# Domain 3: Data Engineering 💾

---

## IDB 3.1 — Signal Generation Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Signal Generation Sub-Block Analyst
# IDB ID: 3.1
# Domain: Data Engineering

## YOUR SCOPE
**Primary Files:**
- `data/signal_generator.py` (37KB — LARGEST data file!)
- `data/signal_augmentation.py`
- `data/spectrogram_generator.py`

**Key Components:**
- `SignalGenerator` — Main signal generation class
- `SpectrogramGenerator` — Time-frequency representation
- Physics parameters: Sommerfeld number, fault types, severity

**Independence Score: 9/10** — Highly isolated, pure physics simulation!

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document all signal types generated
- Verify physics correctness (Sommerfeld scaling)
- Check fault injection mechanisms
- Verify noise generation

### Task 2: Critical Issues Identification
Look for:
- Physics formula errors
- Numerical precision issues
- Large file size (37KB!) — is it doing too much?
- Missing signal validation
- Non-deterministic generation

### Task 3: "If I Could Rewrite This" Retrospective
- Is the class responsible for too much?
- Should physics parameters be externalized?
- Is the API intuitive for researchers?

## OUTPUT REQUIREMENTS
Produce: `IDB_3_1_SIGNAL_GEN_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Signal Generation Sub-Block — Best Practices Curator
# IDB ID: 3.1

## EXTRACTION TASKS
1. Physics parameter conventions
2. Signal generation patterns
3. Validation patterns
4. Documentation requirements for physics
5. Reproducibility patterns

## OUTPUT REQUIREMENTS
Produce: `IDB_3_1_SIGNAL_GEN_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 3.2 — Data Loading Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Data Loading Sub-Block Analyst
# IDB ID: 3.2
# Domain: Data Engineering

## YOUR SCOPE
**Primary Files:**
- `data/dataset.py`, `data/dataloader.py`
- `data/cnn_dataset.py`
- `data/streaming_hdf5_dataset.py`
- `data/cwru_dataset.py`
- `data/tfr_dataset.py`
- `data/transforms.py`, `data/cnn_transforms.py`

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Map all dataset classes
- Document supported file formats
- Check streaming implementation
- Verify transform pipelines

### Task 2: Critical Issues Identification
Look for:
- Memory leaks in data loading
- Inefficient I/O patterns
- Missing data validation
- Transform inconsistencies
- Multi-worker issues

### Task 3: "If I Could Rewrite This" Retrospective
- Is the dataset hierarchy correct?
- Should there be a unified dataset interface?
- Are transforms composable?

## OUTPUT REQUIREMENTS
Produce: `IDB_3_2_DATA_LOADING_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Data Loading Sub-Block — Best Practices Curator
# IDB ID: 3.2

## EXTRACTION TASKS
1. Dataset class patterns
2. Transform composition patterns
3. Memory management patterns
4. Multi-worker best practices
5. Data validation conventions

## OUTPUT REQUIREMENTS
Produce: `IDB_3_2_DATA_LOADING_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 3.3 — Storage Layer Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Storage Layer Sub-Block Analyst
# IDB ID: 3.3
# Domain: Data Engineering

## YOUR SCOPE
**Primary Files:**
- `data/cache_manager.py`
- `data/matlab_importer.py`
- `data/data_validator.py`
- `packages/storage/`

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document caching mechanisms
- Check MATLAB import compatibility
- Verify data validation coverage
- Map storage abstraction layer

### Task 2: Critical Issues Identification
Look for:
- Cache invalidation bugs
- File corruption risks
- Import failures for edge cases
- Validation gaps

## OUTPUT REQUIREMENTS
Produce: `IDB_3_3_STORAGE_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Storage Layer Sub-Block — Best Practices Curator
# IDB ID: 3.3

## EXTRACTION TASKS
1. Caching patterns
2. File format handling conventions
3. Validation patterns
4. Error recovery patterns

## OUTPUT REQUIREMENTS
Produce: `IDB_3_3_STORAGE_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

# Domain 4: Infrastructure 🛠️

---

## IDB 4.1 — Database Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Database Sub-Block Analyst
# IDB ID: 4.1
# Domain: Infrastructure

## YOUR SCOPE
**Primary Directories:**
- `packages/dashboard/database/` (14 files)
- `packages/dashboard/models/` (25 files)

**Key Components:**
- `connection.py` — Database session management
- SQLAlchemy ORM models
- Migrations: `database/migrations/`

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Map all database models
- Document relationships and constraints
- Check migration history
- Verify connection pooling

### Task 2: Critical Issues Identification
Look for:
- N+1 query problems
- Missing indexes
- Relationship issues
- Migration conflicts
- Session management bugs

### Task 3: "If I Could Rewrite This" Retrospective
- Is the schema normalized correctly?
- Are relationships properly defined?
- Is there schema evolution support?

## OUTPUT REQUIREMENTS
Produce: `IDB_4_1_DATABASE_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Database Sub-Block — Best Practices Curator
# IDB ID: 4.1

## EXTRACTION TASKS
1. Model definition conventions
2. Relationship patterns
3. Migration patterns
4. Query optimization patterns
5. Session management conventions

## OUTPUT REQUIREMENTS
Produce: `IDB_4_1_DATABASE_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 4.2 — Deployment Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Deployment Sub-Block Analyst
# IDB ID: 4.2
# Domain: Infrastructure

## YOUR SCOPE
**Primary Files:**
- `packages/deployment/` (API + optimization)
- `deploy/`
- `Dockerfile`
- `docker-compose.yml`

**Key Components:**
- Dockerfile for container builds
- `docker-compose.yml` for local orchestration
- `packages/deployment/optimization/` — ONNX export, quantization
- `packages/deployment/api/` — Inference API

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document container architecture
- Check ONNX export coverage
- Verify quantization support
- Map deployment scripts

### Task 2: Critical Issues Identification
Look for:
- Security vulnerabilities in Dockerfile
- Missing health checks
- Resource limit issues
- ONNX export failures
- Quantization accuracy issues

## OUTPUT REQUIREMENTS
Produce: `IDB_4_2_DEPLOYMENT_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Deployment Sub-Block — Best Practices Curator
# IDB ID: 4.2

## EXTRACTION TASKS
1. Dockerfile patterns
2. Container orchestration conventions
3. Model optimization patterns
4. Health check conventions
5. CI/CD integration patterns

## OUTPUT REQUIREMENTS
Produce: `IDB_4_2_DEPLOYMENT_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 4.3 — Testing Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Testing Sub-Block Analyst
# IDB ID: 4.3
# Domain: Infrastructure

## YOUR SCOPE
**Primary Directory:** `tests/` (27 files: 14 test files + 6 subdirectories)

**Key Components:**
- pytest conventions
- `conftest.py` — Shared fixtures
- Specialized: `stress_tests.py`, `load_tests.py`

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Map test coverage per sub-block
- Document fixture patterns
- Check test isolation
- Identify slow tests

### Task 2: Critical Issues Identification
Look for:
- Flaky tests
- Missing coverage areas
- Test isolation failures
- Fixture leaks
- Slow test suites

### Task 3: "If I Could Rewrite This" Retrospective
- Is test organization logical?
- Are fixtures properly scoped?
- Is there over-mocking?

## OUTPUT REQUIREMENTS
Produce: `IDB_4_3_TESTING_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Testing Sub-Block — Best Practices Curator
# IDB ID: 4.3

## EXTRACTION TASKS
1. Test organization patterns
2. Fixture conventions
3. Mock patterns
4. Assertion patterns
5. Test naming conventions

## OUTPUT REQUIREMENTS
Produce: `IDB_4_3_TESTING_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 4.4 — Configuration Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Configuration Sub-Block Analyst
# IDB ID: 4.4
# Domain: Infrastructure

## YOUR SCOPE
**Primary Directory:** `config/` (6 config files)

**Key Files:**
- `base_config.py` — Core configuration loader
- `data_config.py`, `model_config.py`, `training_config.py`, `experiment_config.py`

**Independence Score: 10/10** — Pure data, no logic dependencies!

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document all configuration schemas
- Check validation logic
- Verify environment variable handling
- Map configuration inheritance

### Task 2: Critical Issues Identification
Look for:
- Missing validation
- Insecure defaults
- Undocumented options
- Configuration sprawl

## OUTPUT REQUIREMENTS
Produce: `IDB_4_4_CONFIG_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Configuration Sub-Block — Best Practices Curator
# IDB ID: 4.4

## EXTRACTION TASKS
1. Configuration structure patterns
2. Validation patterns
3. Environment variable conventions
4. Default value conventions
5. Documentation requirements

## OUTPUT REQUIREMENTS
Produce: `IDB_4_4_CONFIG_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

# Domain 5: Research & Science 🔬

---

## IDB 5.1 — Research Scripts Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Research Scripts Sub-Block Analyst
# IDB ID: 5.1
# Domain: Research & Science

## YOUR SCOPE
**Primary Directory:** `scripts/research/` (8 scripts)

**Key Scripts:**
- `ablation_study.py` — Systematic ablation experiments
- `contrastive_physics.py` — Self-supervised pretraining
- `hyperparameter_sensitivity.py` — HPO analysis
- `pinn_comparison.py` — PINN vs baseline comparison
- `ood_testing.py` — Out-of-distribution evaluation

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document script purposes and CLI interfaces
- Check reproducibility (seeds, determinism)
- Verify output formats
- Map dependencies on core packages

### Task 2: Critical Issues Identification
Look for:
- Non-reproducible experiments
- Missing CLI arguments
- Incomplete logging
- Hardcoded paths
- Missing error handling

### Task 3: "If I Could Rewrite This" Retrospective
- Are scripts modular enough?
- Should there be a unified experiment runner?
- Are outputs publication-ready?

## OUTPUT REQUIREMENTS
Produce: `IDB_5_1_RESEARCH_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Research Scripts Sub-Block — Best Practices Curator
# IDB ID: 5.1

## EXTRACTION TASKS
1. Experiment script patterns
2. CLI argument conventions
3. Logging conventions
4. Reproducibility requirements
5. Output format conventions

## OUTPUT REQUIREMENTS
Produce: `IDB_5_1_RESEARCH_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

## IDB 5.2 — Visualization Sub-Block

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Visualization Sub-Block Analyst
# IDB ID: 5.2
# Domain: Research & Science

## YOUR SCOPE
**Primary Directory:** `visualization/` (13 files)

**Purpose:** Publication-quality visualizations: attention maps, latent space (t-SNE/UMAP), feature importance, physics embeddings.

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document all visualization types
- Check styling consistency
- Verify DPI/font requirements for publications
- Map input/output formats

### Task 2: Critical Issues Identification
Look for:
- Inconsistent styling
- Non-publication-quality outputs
- Memory issues with large visualizations
- Missing colorblind-friendly options

### Task 3: "If I Could Rewrite This" Retrospective
- Should there be a unified style configuration?
- Are visualization functions composable?
- Is there a consistent color palette?

## OUTPUT REQUIREMENTS
Produce: `IDB_5_2_VISUALIZATION_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Visualization Sub-Block — Best Practices Curator
# IDB ID: 5.2

## EXTRACTION TASKS
1. Visualization styling conventions
2. Figure size/DPI conventions
3. Color palette conventions
4. Font conventions
5. Accessibility considerations
6. Output format conventions

## OUTPUT REQUIREMENTS
Produce: `IDB_5_2_VISUALIZATION_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

# Integration Layer (Cross-Cutting Concerns)

---

## IDB 6.0 — Integration Layer

### PROMPT 1: Block Analysis & Critical Evaluation

```
# AI Agent Role: Integration Layer Analyst
# IDB ID: 6.0
# Domain: Cross-Cutting Concerns

## YOUR SCOPE
**Primary Directories:**
- `integration/` — Unified pipeline, model registry, validators
- `packages/dashboard/integrations/` — Celery ↔ Core ML bridge
- `utils/` — Shared utilities

**WARNING:** This layer has HIGH COUPLING — it bridges Core ML ↔ Dashboard!

## ANALYSIS TASKS

### Task 1: Current State Assessment
- Document all integration points
- Map data flow across domains
- Check for proper abstraction layers
- Verify utility function coverage

### Task 2: Critical Issues Identification
Look for:
- Tight coupling that could be loosened
- Missing abstraction layers
- Utility duplication
- Integration test gaps

### Task 3: "If I Could Rewrite This" Retrospective
- Are integration points minimal and well-defined?
- Should some utilities be split by domain?
- Is the bridge layer properly abstracted?

## OUTPUT REQUIREMENTS
Produce: `IDB_6_0_INTEGRATION_ANALYSIS.md`
Save to: `docs/idb_reports/` directory
```

### PROMPT 2: Best Practices Extraction

```
# AI Agent Role: Integration Layer — Best Practices Curator
# IDB ID: 6.0

## EXTRACTION TASKS
1. Integration patterns
2. Utility function conventions
3. Cross-domain communication patterns
4. Error propagation patterns
5. Logging conventions for cross-domain flows

## OUTPUT REQUIREMENTS
Produce: `IDB_6_0_INTEGRATION_BEST_PRACTICES.md`
Save to: `docs/idb_reports/` directory
```

---

# Final Compilation Prompt

After all IDB teams complete their work, use this prompt to compile findings:

```
# AI Agent Role: Master Compiler
# Task: Compile all IDB reports into unified documents

## YOUR MISSION
You have access to all IDB analysis and best practices reports:
- 18 Analysis reports (`IDB_*_ANALYSIS.md`)
- 18 Best Practices reports (`IDB_*_BEST_PRACTICES.md`)

## DELIVERABLES

### 1. `MASTER_ANALYSIS_SUMMARY.md`
- Executive summary of all critical issues across all IDBs
- Priority matrix (P0/P1/P2 issues by IDB)
- Cross-cutting themes and patterns
- Recommended action plan

### 2. `MASTER_STYLE_GUIDE.md`
- Unified code style conventions
- Naming conventions across all domains
- Interface contract documentation
- Testing requirements
- Documentation standards

### 3. `MASTER_IMPROVEMENT_BACKLOG.md`
- All technical debt items prioritized
- Estimated effort for each
- Dependencies between items
- Recommended implementation order

### 4. `CROSS_TEAM_ALIGNMENT_REPORT.md`
- Identify inconsistencies between teams
- Highlight areas where teams diverged
- Recommendations for alignment
- Integration risk assessment

## OUTPUT LOCATION
Save all files to: `docs/idb_reports/compiled/`
```

---

# Summary

| Domain             | Sub-Blocks | Total Prompts  |
| ------------------ | ---------- | -------------- |
| Core ML Engine     | 5          | 10             |
| Dashboard Platform | 4          | 8              |
| Data Engineering   | 3          | 6              |
| Infrastructure     | 4          | 8              |
| Research & Science | 2          | 4              |
| Integration Layer  | 1          | 2              |
| Final Compilation  | 1          | 1              |
| **TOTAL**          | **20**     | **39 prompts** |

---

**Document created for IDB Analysis Initiative.**
_Ready for distribution to AI Agent teams._
