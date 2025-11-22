# 📋 PLOTLY DASH APPLICATION: COMPREHENSIVE PLANNING DOCUMENT

**Context:** After reviewing all Phases 0-10, this plan outlines how to build an enterprise-grade Plotly Dash application that integrates with the complete ML pipeline for bearing fault diagnosis.

---

## EXECUTIVE SUMMARY

**What We're Building:** A professional web-based dashboard that allows users to interact with the entire ML pipeline (data generation, training, evaluation, deployment) without writing code.

**Why 4 Phases:** 
- Phase 11A: Foundation (can't skip this)
- Phase 11B: Core value delivery (training/monitoring)
- Phase 11C: Advanced features (XAI, analytics)
- Phase 11D: Production hardening (scale, security)

**Critical Success Factors:**
1. Seamless integration with existing Phases 0-10 code
2. No duplication of logic (dashboard wraps existing code, doesn't reimplement)
3. Multi-user support from day 1 (not retrofitted later)
4. Real-time feedback during long operations (training takes 15 min)

---

# PHASE 11A: FOUNDATION & DATA EXPLORATION

**Duration:** 2 weeks  
**Objective:** Build architectural foundation and basic data visualization capabilities

---

## 11A.1 PRE-DEVELOPMENT DECISIONS

### Technology Stack Choices

**Decision 1: Database Selection**
- **Choice:** PostgreSQL (not SQLite)
- **Rationale:** 
  - Multi-user concurrent access (SQLite locks)
  - Stores: experiment metadata, user sessions, dataset info, model registry
  - Does NOT store: actual signals (too large), model weights (use file storage)
- **Schema Design Principles:**
  - Each experiment = 1 row (links to model files, config, results)
  - Normalized design (no JSON blobs)
  - Indexed on: timestamp, user_id, model_type, accuracy

**Decision 2: Caching Strategy**
- **Choice:** Redis for session caching
- **What to Cache:**
  - User selections (dropdown values, filters)
  - Expensive computations (t-SNE projections, correlation matrices)
  - Recently viewed signals
- **What NOT to Cache:**
  - Training progress (use WebSockets instead)
  - Real-time metrics (defeats purpose)

**Decision 3: Task Queue**
- **Choice:** Celery with Redis broker
- **Why:** Training takes 15 minutes → cannot block web request
- **Architecture:**
  ```
  User clicks "Train" → Dash creates Celery task → Returns task_id
  → Frontend polls task status every 2 seconds → Shows progress bar
  → Task completes → Updates database → Dashboard shows results
  ```

**Decision 4: File Storage**
- **Choice:** MinIO (S3-compatible) or local filesystem with clear structure
- **Directory Structure:**
  ```
  storage/
  ├── datasets/
  │   └── {dataset_id}/
  │       ├── signals.h5           # HDF5 cache from Phase 0
  │       └── metadata.json
  ├── models/
  │   └── {experiment_id}/
  │       ├── model.pth            # PyTorch model
  │       ├── model.onnx           # ONNX export
  │       └── config.yaml          # Exact config used
  ├── results/
  │   └── {experiment_id}/
  │       ├── confusion_matrix.png
  │       ├── roc_curves.png
  │       └── metrics.json
  └── uploads/
      └── {user_id}/
          └── {timestamp}_signal.npy
  ```

---

## 11A.2 ARCHITECTURAL PRINCIPLES

### Principle 1: Separation of Concerns

**Three-Layer Architecture:**

1. **Presentation Layer (Dash)**
   - Responsibility: UI rendering, user input collection
   - Does NOT: Contain business logic, direct database access
   - Pattern: Layouts define structure, callbacks handle events

2. **Service Layer (Python classes)**
   - Responsibility: Business logic, orchestration
   - Examples: `DataService`, `TrainingService`, `EvaluationService`
   - Pattern: Each service wraps Phases 0-10 functionality
   - Key Point: Services are **testable independently** of Dash

3. **Data Layer (PostgreSQL + Files)**
   - Responsibility: Persistence only
   - Pattern: SQLAlchemy ORM models
   - Key Point: Database schema is **version controlled** (Alembic migrations)

**Critical Insight:** If you can't test a function without starting the Dash server, it's in the wrong layer.

---

### Principle 2: Integration Strategy with Phases 0-10

**DO:**
- ✅ Import existing Phase 0-10 modules directly
- ✅ Wrap them in service classes with error handling
- ✅ Add progress callbacks where Phase code supports them
- ✅ Cache expensive operations (feature extraction)

**DON'T:**
- ❌ Copy-paste Phase code into Dash app
- ❌ Reimplement algorithms (use what exists)
- ❌ Modify Phase 0-10 code to fit Dash (wrap, don't change)
- ❌ Tight coupling (service layer acts as abstraction)

**Example Integration Pattern:**
```
Phase 0: data_generator_v4.m (MATLAB) → Already ported to signal_generator.py

Dash Integration:
  layouts/data_explorer.py (UI)
    ↓ calls
  services/data_service.py (wrapper)
    ↓ calls
  integrations/phase0_adapter.py (adapter pattern)
    ↓ calls
  data/signal_generator.py (existing Phase 0 code - UNCHANGED)
```

**Why Adapter Pattern:** If Phase 0 code changes (bug fix, enhancement), only `phase0_adapter.py` needs update, not entire Dash app.

---

### Principle 3: Callback Design Pattern

**Dash Callbacks Are Event Handlers:**

**Structure:**
```
Every callback follows this pattern:

1. INPUT VALIDATION
   - Check if inputs are valid (not None, correct type)
   - Return empty state if invalid
   
2. BUSINESS LOGIC (delegated to service layer)
   - Call service function
   - Handle exceptions gracefully
   
3. OUTPUT FORMATTING
   - Convert service output to Dash components
   - Apply consistent styling
   
4. ERROR HANDLING
   - Try-except around ALL callbacks
   - Return user-friendly error message
   - Log detailed error to backend
```

**Anti-Pattern to Avoid:**
```
BAD: 200-line callback doing everything
  ├─ Load data from database
  ├─ Compute statistics
  ├─ Generate 5 different plots
  ├─ Format tables
  └─ Return 10 outputs

GOOD: Callback delegates to service
  ├─ Call service.get_dashboard_data()
  ├─ Service returns pre-computed results
  ├─ Callback just formats for display
  └─ Each service function is 20-30 lines, testable
```

---

### Principle 4: State Management

**Challenge:** Dash is stateless (each callback independent)

**Solutions:**

1. **Session Storage (dcc.Store):**
   - Use for: User preferences, filter selections, temp data
   - Lifespan: Browser session (cleared on tab close)
   - Example: Current dataset selection, zoom level on plots

2. **Database:**
   - Use for: Persistent data, experiments, results
   - Lifespan: Permanent
   - Example: Trained models, experiment history

3. **Redis Cache:**
   - Use for: Expensive computations, temporary results
   - Lifespan: 5-60 minutes (configurable TTL)
   - Example: t-SNE projection (takes 30 seconds, cache for 10 min)

**Critical Rule:** Never use global variables for state (breaks multi-user support).

---

## 11A.3 FILE STRUCTURE AND RESPONSIBILITIES

### Directory Structure (58 files total for Phase 11A)

```
dash_app/
├── app.py                          # Application entry point
├── config.py                       # Configuration (database, Redis URLs, secrets)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Multi-service orchestration (Dash, PostgreSQL, Redis, Celery)
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── README.md                       # Setup and usage instructions
│
├── assets/                         # Static assets (served by Flask)
│   ├── custom.css                  # Application-wide styling
│   ├── logo.svg                    # Company/project logo
│   ├── favicon.ico                 # Browser tab icon
│   └── custom.js                   # Optional: Custom JavaScript (rarely needed)
│
├── components/                     # Reusable UI components (7 files)
│   ├── __init__.py
│   ├── header.py                   # Top navigation bar (logo, user menu, notifications)
│   ├── sidebar.py                  # Left navigation menu (links to all pages)
│   ├── footer.py                   # Footer (version, links, copyright)
│   ├── cards.py                    # Stat cards, info boxes (reusable templates)
│   ├── tables.py                   # Styled AG-Grid tables (common configurations)
│   ├── charts.py                   # Plotly figure templates (consistent styling)
│   └── modals.py                   # Modal dialogs (confirmations, forms)
│
├── layouts/                        # Page layouts (Phase 11A: 5 pages)
│   ├── __init__.py
│   ├── home.py                     # Landing page (dashboard overview, quick stats)
│   ├── data_explorer.py            # Dataset exploration (filter, visualize, statistics)
│   ├── signal_viewer.py            # Individual signal inspection (time/freq/spectrogram)
│   ├── dataset_manager.py          # Dataset CRUD (create, upload, delete datasets)
│   └── system_health.py            # System monitoring (disk space, GPU, database status)
│
├── callbacks/                      # Dash callbacks (Phase 11A: 4 files)
│   ├── __init__.py                 # Registers all callbacks with app
│   ├── data_explorer_callbacks.py  # Handles: filter changes, plot updates, table refresh
│   ├── signal_viewer_callbacks.py  # Handles: signal selection, plot type toggle, export
│   ├── dataset_callbacks.py        # Handles: dataset create/delete, validation
│   └── common_callbacks.py         # Shared callbacks (navigation, notifications)
│
├── services/                       # Business logic layer (Phase 11A: 6 files)
│   ├── __init__.py
│   ├── data_service.py             # Dataset operations (list, load, validate, stats)
│   ├── signal_service.py           # Signal processing (load, preprocess, extract features)
│   ├── cache_service.py            # Redis caching wrapper (get, set, invalidate)
│   ├── file_service.py             # File I/O (upload, download, cleanup)
│   ├── validation_service.py       # Input validation (signal format, config values)
│   └── notification_service.py     # User notifications (success, error, warning messages)
│
├── integrations/                   # Phase 0-10 integration adapters (Phase 11A: 2 files)
│   ├── __init__.py
│   ├── phase0_adapter.py           # Wraps Phase 0 data generation
│   └── pipeline_adapter.py         # Wraps UnifiedMLPipeline from Phase 10
│
├── models/                         # Database models (SQLAlchemy) (Phase 11A: 5 files)
│   ├── __init__.py
│   ├── base.py                     # Base model class (common fields: id, created_at, updated_at)
│   ├── dataset.py                  # Dataset metadata (name, num_signals, fault_types, created_by)
│   ├── signal.py                   # Signal records (dataset_id, fault_class, file_path)
│   ├── system_log.py               # System events (user_action, timestamp, status)
│   └── user.py                     # User accounts (username, email, role) [Phase 11D adds auth]
│
├── database/                       # Database management (4 files)
│   ├── __init__.py
│   ├── connection.py               # SQLAlchemy engine, session factory
│   ├── seed_data.py                # Initial data seeding (fault classes, default datasets)
│   └── migrations/                 # Alembic migration files (auto-generated)
│       ├── versions/
│       └── alembic.ini
│
├── utils/                          # Utility functions (Phase 11A: 7 files)
│   ├── __init__.py
│   ├── plotting.py                 # Plotly figure helpers (consistent themes, layouts)
│   ├── formatting.py               # Data formatting (numbers, dates, percentages)
│   ├── validation.py               # Input validators (signal shape, config ranges)
│   ├── constants.py                # Global constants (FAULT_CLASSES, COLORS, PLOT_CONFIGS)
│   ├── logger.py                   # Logging configuration (file + console handlers)
│   └── exceptions.py               # Custom exception classes (DataValidationError, etc.)
│
├── tasks/                          # Celery tasks (Phase 11B adds training tasks)
│   ├── __init__.py
│   └── data_tasks.py               # Background data processing (large file uploads, preprocessing)
│
└── tests/                          # Testing suite (Phase 11A: 6 files)
    ├── __init__.py
    ├── conftest.py                 # Pytest fixtures (mock database, test client)
    ├── test_services/
    │   ├── test_data_service.py
    │   └── test_signal_service.py
    ├── test_integrations/
    │   └── test_phase0_adapter.py
    └── test_callbacks/
        └── test_data_explorer_callbacks.py
```

---

## 11A.4 DETAILED PAGE SPECIFICATIONS

### Page 1: Home Dashboard

**Purpose:** Landing page providing system overview and navigation hub

**Layout Zones:**
1. **Header Bar** (top, 60px height)
   - Logo (left)
   - Page title (center)
   - User dropdown menu (right): Profile, Settings, Logout

2. **Quick Stats Row** (4 cards, equal width)
   - Card 1: Total Signals (icon: 📊, color: blue)
   - Card 2: Fault Classes (icon: 🏷️, color: green)
   - Card 3: Best Model Accuracy (icon: 🎯, color: purple)
   - Card 4: Experiments Run (icon: 🧪, color: orange)

3. **Main Content (2 columns)**
   - **Left Column (33% width):** Quick Actions
     - Large buttons (vertical stack):
       - "Explore Datasets" → /data-explorer
       - "View Signals" → /signal-viewer
       - "Train Model" → /training (Phase 11B)
       - "Predict Fault" → /inference (Phase 11B)
   
   - **Right Column (67% width):** Recent Experiments
     - Scrollable list (5 most recent)
     - Each item shows: timestamp, model type, accuracy, status badge
     - Click item → navigate to detailed results

4. **Bottom Row (2 charts)**
   - Left: System Health Gauge (0-100 score)
   - Right: Dataset Class Distribution (pie chart)

**Data Sources:**
- Database: experiments table (recent 5)
- Service: `data_service.get_dataset_stats()` (cached 5 min)
- Real-time: system_health check (every 30 sec via interval callback)

**Interactions:**
- All buttons are links (no callbacks needed)
- Hover on stat cards → tooltip with trend (↑ +12% this week)
- Click experiment item → open modal with full details

---

### Page 2: Data Explorer

**Purpose:** Interactive exploration of generated datasets (Phase 0 output)

**Layout Zones:**

1. **Filter Panel (left sidebar, 250px width)**
   - **Dataset Selector:** Dropdown (lists all datasets from database)
   - **Fault Class Filter:** Checklist (11 checkboxes, default: all selected)
   - **Severity Filter:** Checklist (incipient, mild, moderate, severe)
   - **Sample Size Slider:** Range slider (50-200 signals per fault)
   - **Apply Filters Button:** Triggers data reload
   - **Reset Button:** Clear all filters

2. **Main Visualization Area (3 tabs)**

   **Tab 1: Overview Statistics**
   - Top: Summary table (rows = fault classes, cols = count, mean amplitude, dominant frequency)
   - Bottom Left: Class distribution bar chart (x-axis: fault, y-axis: count)
   - Bottom Right: Severity distribution stacked bar chart

   **Tab 2: Feature Distributions**
   - Dropdown: Select feature (36 features from Phase 1)
   - Visualization: Box plots (one box per fault class)
   - Purpose: See feature separability (wide spread = good discriminator)
   - Interaction: Click box → histogram overlay appears

   **Tab 3: Dimensionality Reduction**
   - Dropdown: Select method (t-SNE, PCA, UMAP)
   - Scatter plot: 2D projection, colored by fault class
   - Interaction: Hover point → shows signal ID, fault type
   - Compute Button: "Calculate Projection" (takes 30 sec, shows spinner)
   - Cache: Store projection in Redis for 10 minutes

3. **Bottom Panel: Data Table**
   - AG-Grid table: All signals matching filters
   - Columns: ID, Fault Type, Severity, RMS, Kurtosis, Timestamp
   - Features: Sortable, filterable, pagination (50 rows/page)
   - Action: Click row → navigate to Signal Viewer with that signal

**Data Flow:**
1. User selects dataset → callback fires
2. Callback calls `data_service.load_dataset(dataset_id, filters)`
3. Service loads from HDF5 cache (Phase 0 output)
4. Service computes statistics (if not cached)
5. Callback formats data for Plotly figures
6. Dash updates all charts simultaneously

**Performance Considerations:**
- **Problem:** Loading 1,430 signals (102,400 samples each) = 146M data points
- **Solution:** 
  - Load metadata only (from database) for table
  - Load full signals on-demand (when user clicks row)
  - Precompute statistics during data generation (Phase 0)
  - Cache computed features (Redis, 10 min TTL)

---

### Page 3: Signal Viewer

**Purpose:** Detailed inspection of individual signals

**Layout Zones:**

1. **Signal Selection Panel (top, 100px height)**
   - **Signal ID Input:** Text input or dropdown (autocomplete)
   - **Random Signal Button:** Load random signal from dataset
   - **Upload Button:** Upload custom signal (.npy, .csv, .mat)
   - **Navigation Arrows:** Previous/Next signal (in current dataset)

2. **Visualization Area (3 panels, vertically stacked)**

   **Panel 1: Time Domain (height: 300px)**
   - Line plot: Amplitude vs. Time (0-5 seconds)
   - Overlay: Fault severity marker (if applicable)
   - Interaction: Click-drag to zoom, double-click to reset
   - Toolbar: Download as PNG, Export data as CSV

   **Panel 2: Frequency Domain (height: 300px)**
   - Line plot: Power Spectral Density vs. Frequency (0-10 kHz)
   - Annotations: Mark characteristic frequencies (1X, 2X, 3X, sub-sync)
   - Comparison Mode Toggle: Show expected spectrum for fault type (dotted line)
   - Interaction: Hover → show exact frequency and amplitude

   **Panel 3: Time-Frequency (height: 400px)**
   - Heatmap: Spectrogram (frequency vs. time, color = intensity)
   - Controls: 
     - Method selector: STFT, CWT, Wigner-Ville
     - Window size slider (for STFT): 128, 256, 512, 1024
     - Colormap selector: Viridis, Plasma, Hot, Jet
   - Compute Button: Generate spectrogram (1-2 sec, show spinner)

3. **Metadata Panel (right sidebar, 300px width)**
   - **Signal Information:**
     - Fault Type: [Badge with color]
     - Severity: [Badge]
     - Sampling Rate: 20,480 Hz
     - Duration: 5.0 seconds
     - Samples: 102,400
   
   - **Extracted Features (Phase 1):**
     - Collapsible sections:
       - Time Domain (7 features): RMS, Kurtosis, Skewness, etc.
       - Frequency Domain (12 features): Spectral Centroid, Entropy, etc.
       - Envelope (4 features)
     - Display: Feature name, value, unit
   
   - **Actions:**
     - "Predict Fault" button (runs inference, shows result)
     - "Export Signal" button (download as .npy)
     - "Add to Comparison" button (multi-signal comparison in Phase 11C)

**Data Flow:**
1. User enters signal ID → callback fires
2. Service loads signal from storage (HDF5)
3. Service extracts features (if not cached)
4. Service generates time/frequency plots
5. Callback returns 3 figures + metadata dict
6. Dash updates all panels

**Edge Cases:**
- Signal ID not found → show error toast notification
- Invalid signal format → show validation error with hints
- Large file upload (>10MB) → reject with message "Max 10MB"

---

### Page 4: Dataset Manager

**Purpose:** Create, view, delete datasets (Phase 0 integration)

**Layout Zones:**

1. **Action Panel (top)**
   - **Create New Dataset Button:** Opens modal dialog
   - **Upload Dataset Button:** Upload pre-generated signals
   - **Refresh Button:** Reload dataset list

2. **Dataset List (main area)**
   - Card grid (3 columns)
   - Each card shows:
     - Dataset name
     - Number of signals
     - Fault classes included
     - Creation date
     - Creator (user)
     - Actions dropdown: View, Download, Delete
   
   - Interaction:
     - Click card → navigate to Data Explorer with that dataset selected
     - Click "Delete" → confirmation modal → remove from database + storage

3. **Create Dataset Modal (triggered by button)**
   - **Form Fields:**
     - Dataset Name: Text input (required)
     - Number of Signals per Fault: Slider (50-200, default: 100)
     - Fault Types to Include: Checklist (11 options, default: all)
     - Severity Levels: Checklist (incipient, mild, moderate, severe, default: all)
     - Data Augmentation: Toggle switch (enable/disable)
     - Noise Configuration: Expandable section with 7 checkboxes (Phase 0 noise types)
   
   - **Validation Rules:**
     - Name must be unique
     - At least 1 fault type selected
     - At least 1 severity level selected
   
   - **Submit Button: "Generate Dataset"**
     - Action: Creates Celery task (background job)
     - Returns task_id
     - Modal closes, shows progress notification
     - Dataset appears in list when complete

**Data Flow:**
1. User fills form → clicks Generate
2. Callback validates inputs
3. Callback creates Celery task: `tasks.generate_dataset_task.delay(config)`
4. Celery worker calls `integrations.phase0_adapter.generate_dataset(config)`
5. Phase 0 code runs (3-5 minutes)
6. On completion, worker saves to storage + database
7. Dashboard shows notification: "Dataset ready!"

**Important Design Decision:**
- **DO NOT re-implement Phase 0 logic in Dash**
- Dash only provides UI for configuring parameters
- All generation logic stays in Phase 0 code
- Adapter pattern ensures clean separation

---

### Page 5: System Health

**Purpose:** Monitor system resources and health

**Layout Zones:**

1. **Resource Usage (top row, 4 cards)**
   - Card 1: CPU Usage (gauge 0-100%)
   - Card 2: RAM Usage (gauge 0-100%)
   - Card 3: GPU Usage (gauge 0-100%, N/A if no GPU)
   - Card 4: Disk Space (gauge 0-100%)

2. **Database Status (second row)**
   - Table: Connection pool status, active connections, query performance
   - Redis: Cache hit rate, memory usage

3. **Recent Logs (third row)**
   - Scrollable log viewer (last 100 lines)
   - Filter by level: All, Error, Warning, Info
   - Auto-refresh every 5 seconds

4. **Health Checks (bottom row)**
   - Checklist with status indicators:
     - ✅ Database connection
     - ✅ Redis connection
     - ✅ Celery workers (N running)
     - ✅ File storage accessible
     - ✅ Phase 0-10 modules importable

**Data Sources:**
- System: `psutil` library (CPU, RAM, disk)
- GPU: `pynvml` (NVIDIA) or `None`
- Database: SQLAlchemy connection pool stats
- Redis: `redis.info()` command
- Logs: Read from `app.log` file (tail -n 100)

**Auto-Refresh:**
- Interval component: 5 seconds
- Updates all metrics automatically
- Can be paused with toggle button

---

## 11A.5 CRITICAL IMPLEMENTATION GUIDELINES

### Guideline 1: Callback Organization

**Problem:** Dash allows multiple callbacks to update same output → conflicts

**Rule:** One output per callback (strict enforcement)

**Pattern:**
```
Good:
- Callback A: Updates chart
- Callback B: Updates table
- Callback C: Updates notification

Bad:
- Callback A: Updates chart + table (violates rule)
- Callback B: Also updates table (conflict!)
```

**Exception:** Multiple inputs can trigger same callback (that's fine)

---

### Guideline 2: Error Handling Strategy

**Every Callback Must:**
1. Wrap ALL logic in try-except
2. Log detailed error to file (with stack trace)
3. Return user-friendly message to UI
4. Never expose internal paths, stack traces to user

**Example Pattern:**
```
Try:
  - Validate inputs
  - Call service function
  - Format output
  - Return success
Except SpecificError:
  - Log error with context
  - Return: "Dataset not found. Please check the name and try again."
Except Exception:
  - Log full stack trace
  - Return: "An unexpected error occurred. Please contact support."
```

---

### Guideline 3: Performance Optimization

**Expensive Operations (>1 second):**
- Feature extraction (36 features from signal)
- t-SNE projection (1,430 points)
- Spectrogram generation (CWT is slow)
- Model training (15 minutes - Phase 11B)

**Solutions:**
1. **Precompute:** During data generation (Phase 0), extract features immediately
2. **Cache:** Store results in Redis with TTL
3. **Background Tasks:** Use Celery for >10 second operations
4. **Lazy Loading:** Don't load all 1,430 signals at once (pagination)
5. **Progress Indicators:** Show spinner during 1-10 sec operations, progress bar for >10 sec

**Cache Key Design:**
```
Pattern: f"{operation}:{hash(inputs)}"

Examples:
- f"tsne:{dataset_id}:{method}:{perplexity}"
- f"features:{signal_id}"
- f"spectrogram:{signal_id}:{method}:{window_size}"
```

---

### Guideline 4: Testing Strategy

**What to Test:**

1. **Service Layer (priority 1)**
   - Unit tests for every service function
   - Mock external dependencies (database, Redis, Phase 0 code)
   - Aim: 90% coverage

2. **Integration Adapters (priority 2)**
   - Test Phase 0-10 integration
   - Use real Phase code (not mocked)
   - Verify outputs match expected format

3. **Callbacks (priority 3)**
   - Test callback logic (not Dash rendering)
   - Mock Dash context
   - Focus on edge cases (null inputs, empty datasets)

**What NOT to Test:**
- Dash internal rendering (trust the framework)
- Plotly figure generation (trust the library)
- CSS styling (visual QA, not automated)

**Test Data:**
- Use small synthetic datasets (10 signals, not 1,430)
- Store in `tests/fixtures/`
- Version control test data (committed to repo)

---

### Guideline 5: Configuration Management

**Environment-Specific Settings:**

Development:
- DEBUG = True
- Database: localhost:5432
- Redis: localhost:6379
- File storage: local filesystem
- Logging: console + file (DEBUG level)

Production:
- DEBUG = False
- Database: production URL (environment variable)
- Redis: production URL (environment variable)
- File storage: MinIO/S3
- Logging: file only (INFO level)

**Implementation:**
- All settings in `config.py`
- Load from environment variables (12-factor app)
- `.env.example` file documents all required variables
- Never commit real credentials to Git

---

## 11A.6 ACCEPTANCE CRITERIA (Phase 11A Complete When)

✅ **Architecture Validated**
- Three-layer architecture implemented (Presentation, Service, Data)
- All Phase 0-10 integrations use adapter pattern
- Zero duplication of Phase logic in Dash app

✅ **Pages Functional**
- Home Dashboard: Shows stats, navigates to other pages
- Data Explorer: Filters work, charts render, table loads
- Signal Viewer: Loads signals, displays 3 views (time/freq/spectrogram)
- Dataset Manager: Creates datasets (background task), lists existing
- System Health: Shows real-time metrics, auto-refreshes

✅ **Database Operational**
- PostgreSQL schema created (Alembic migrations)
- Can store datasets, signals, experiments
- Seed data loaded (11 fault classes, 1 sample dataset)

✅ **Caching Working**
- Redis connected and caching expensive operations
- Cache hit rate >50% for common queries (t-SNE, features)
- Cache invalidation works (when dataset deleted)

✅ **Multi-User Support**
- 5 users can browse simultaneously without conflicts
- Sessions isolated (User A's filters don't affect User B)
- No global state variables used

✅ **Performance Targets Met**
- Page load time: <2 seconds (after first load)
- Filter change response: <500ms
- Signal load: <1 second
- No memory leaks (tested with 100 page reloads)

✅ **Testing Coverage**
- Service layer: >85% coverage
- Integration adapters: 100% (critical path)
- Callbacks: >70% (edge cases covered)

✅ **Documentation Complete**
- README: Setup instructions (database, Redis, dependencies)
- Architecture diagram (layers, data flow)
- API documentation (service functions)
- Environment variable documentation (.env.example)

---

## 11A.7 RISKS & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Phase 0-10 integration breaks** | Medium | High | Use adapter pattern, extensive integration tests, version pin dependencies |
| **Performance issues with 1,430 signals** | High | Medium | Implement caching, lazy loading, pagination from day 1 |
| **Callback conflicts (multiple outputs)** | Medium | Medium | Strict rule: one output per callback, code review enforcement |
| **Redis/Celery setup complexity** | Medium | Low | Provide docker-compose.yml, clear setup docs, health check page |
| **Database schema changes break app** | Low | High | Use Alembic migrations, never modify schema directly |

---