# PHASE 11B: ML PIPELINE ORCHESTRATION & TRAINING MONITOR

**Duration:** 3 weeks  
**Objective:** Enable users to configure, launch, and monitor ML training experiments (Phases 1-8) through the dashboard, with real-time progress tracking and comprehensive result visualization.

---

## 11B.1 PRE-DEVELOPMENT DECISIONS

### Decision 1: Training Execution Architecture

**Challenge:** Training takes 15-45 minutes. Web requests timeout after 30-60 seconds.

**Solution: Asynchronous Task Queue Architecture**

```
USER INTERACTION FLOW:

Step 1: User configures experiment in UI
  ↓
Step 2: Click "Start Training" button
  ↓
Step 3: Dash callback creates Celery task
  ↓
Step 4: Callback returns immediately with task_id
  ↓
Step 5: UI switches to "Training Monitor" page
  ↓
Step 6: JavaScript polls /api/task-status/{task_id} every 2 seconds
  ↓
Step 7: Celery worker executes training in background
  ↓
Step 8: Worker updates progress in Redis (epoch, loss, accuracy)
  ↓
Step 9: UI displays live progress bar + metrics chart
  ↓
Step 10: Training completes → Worker saves results to database
  ↓
Step 11: UI shows completion notification + results link
```

**Key Components:**

1. **Celery Worker Pool**
   - Configuration: 2-4 workers (depending on GPU availability)
   - Queue Priority: High (inference), Normal (training), Low (batch jobs)
   - Timeout: 2 hours (safety limit for training tasks)

2. **Progress Tracking Mechanism**
   - Redis key: `task:{task_id}:progress`
   - Data structure: JSON
     ```
     {
       "status": "running",  # pending, running, completed, failed
       "progress": 0.45,     # 0.0 to 1.0
       "current_epoch": 45,
       "total_epochs": 100,
       "train_loss": 0.0234,
       "val_loss": 0.0412,
       "val_accuracy": 0.969,
       "eta_seconds": 503,
       "message": "Training epoch 45/100..."
     }
     ```
   - TTL: 24 hours (auto-cleanup)

3. **REST API Endpoints (New)**
   - `GET /api/task-status/{task_id}` - Get task progress
   - `POST /api/task-cancel/{task_id}` - Cancel running task
   - `GET /api/task-logs/{task_id}` - Stream training logs (Server-Sent Events)

**Why Not WebSockets?**
- Simpler implementation (no WebSocket infrastructure)
- Polling every 2 seconds is acceptable for training (not real-time chat)
- Easier to debug (HTTP requests visible in browser DevTools)
- Works through corporate proxies (WebSockets often blocked)

**Fallback Option:** If polling proves insufficient, Phase 11D adds WebSockets.

---

### Decision 2: Integration Strategy with Phases 1-8

**Principle:** Dashboard orchestrates existing training code, doesn't reimplement it.

**Integration Layers:**

```
Layer 1: UI (Configuration Forms)
  ↓ collects parameters
  
Layer 2: Validation Service
  ↓ validates inputs (ranges, compatibility)
  
Layer 3: Task Dispatcher
  ↓ creates Celery task with config
  
Layer 4: Training Adapter (per phase)
  ├─ Phase 1: Classical ML Adapter
  ├─ Phase 2: 1D CNN Adapter
  ├─ Phase 3: ResNet Adapter
  ├─ Phase 4: Transformer Adapter
  ├─ Phase 6: PINN Adapter
  └─ Phase 8: Ensemble Adapter
  ↓ each adapter wraps existing training code
  
Layer 5: Existing Training Code (Phases 1-8)
  ↓ runs unchanged
  
Layer 6: Callback Hooks (NEW - injected into training loops)
  ↓ report progress to Redis every N steps
  
Layer 7: Result Saver
  ↓ stores to database + file storage
```

**Critical Design Principle:**

**DO:** Add progress callbacks to training loops (minimal changes)
```python
# Existing Phase 3 training code (trainer.py):
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    # NEW: Progress callback (injected if provided)
    if progress_callback:
        progress_callback({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
```

**DON'T:** Copy entire training code into Dash app (violates DRY principle).

---

### Decision 3: Experiment Configuration Strategy

**Problem:** Different models have different hyperparameters.
- CNN: kernel_size, num_filters, dropout
- ResNet: depth (18, 34, 50), width_multiplier
- Transformer: num_heads, num_layers, d_model
- PINN: lambda_physics

**Solution: Template-Based Configuration**

**Approach:**

1. **Presets (Recommended Configurations)**
   - "Quick Test" (10 epochs, small model)
   - "Standard Training" (100 epochs, default hyperparameters)
   - "Maximum Accuracy" (200 epochs, large model, heavy augmentation)
   - "Fast Iteration" (50 epochs, aggressive early stopping)

2. **Expert Mode (Full Control)**
   - Expandable sections for each hyperparameter category
   - Tooltips explaining each parameter
   - Validation with helpful error messages

3. **Configuration Inheritance**
   - "Clone from Experiment #47" (copy all settings)
   - "Resume from Checkpoint" (continue failed training)

**User Flow:**
```
Option A (Beginners):
  Select model type → Choose preset → Click Train
  (3 clicks, 30 seconds)

Option B (Experts):
  Select model type → Switch to Expert Mode → 
  Configure 20+ parameters → Click Train
  (5-10 minutes configuration)

Option C (Iterate):
  View Experiment #47 → Click "Clone & Modify" →
  Change learning rate → Click Train
  (2 minutes)
```

**Config Storage:**
- Every experiment saves exact config as YAML file
- UI loads YAML to populate form (clone feature)
- Version control: Git commit hash stored with config (reproducibility)

---

### Decision 4: Result Visualization Strategy

**Challenge:** Different models produce different outputs.
- Classical ML: Feature importance, decision boundaries
- CNN: Training curves, confusion matrix
- Transformer: Attention maps
- PINN: Physics loss curves

**Solution: Modular Result Renderers**

**Architecture:**
```
TrainingResult (database model):
  ├─ experiment_id
  ├─ model_type  # "cnn", "resnet", "transformer", etc.
  ├─ metrics (JSON): {"accuracy": 0.963, "f1": 0.957, ...}
  ├─ plots (references): ["confusion_matrix.png", "roc_curves.png"]
  └─ artifacts (references): ["model.pth", "model.onnx"]

ResultRenderer (service):
  ├─ render_common_results()  # All models: accuracy, conf matrix
  ├─ render_cnn_results()     # CNN-specific: filter visualizations
  ├─ render_transformer_results()  # Attention maps
  └─ render_pinn_results()    # Physics loss, frequency consistency
```

**Display Strategy:**
```
Results Page Layout:

[Top Section - Common Metrics]
  - Accuracy, Precision, Recall, F1 (cards)
  - Confusion Matrix (heatmap)
  - ROC Curves (line chart)
  - Training History (loss/acc curves)

[Middle Section - Model-Specific]
  - If CNN: Filter visualizations, activation maps
  - If Transformer: Attention weight heatmaps
  - If PINN: Physics loss, Sommerfeld consistency
  - If Ensemble: Member contributions, diversity metrics

[Bottom Section - Artifacts]
  - Download model (.pth, .onnx)
  - Download config (YAML)
  - Download predictions (CSV)
  - Export report (PDF)
```

---

## 11B.2 FILE STRUCTURE ADDITIONS (32 new files)

**New directories and files added to Phase 11A structure:**

```
dash_app/
│
├── layouts/                        # ADD 4 new pages
│   ├── experiment_config.py        # NEW: Configure training experiments
│   ├── training_monitor.py         # NEW: Live training progress
│   ├── experiment_results.py       # NEW: Detailed results view
│   └── experiment_history.py       # NEW: All experiments table
│
├── callbacks/                      # ADD 4 callback files
│   ├── experiment_config_callbacks.py   # Form validation, preset loading
│   ├── training_monitor_callbacks.py    # Progress polling, log streaming
│   ├── experiment_results_callbacks.py  # Plot interactions, exports
│   └── experiment_history_callbacks.py  # Filtering, comparison
│
├── services/                       # ADD 5 services
│   ├── training_service.py         # Training orchestration
│   ├── experiment_service.py       # Experiment CRUD operations
│   ├── model_service.py            # Model loading, inference
│   ├── evaluation_service.py       # Metrics calculation, plotting
│   └── export_service.py           # PDF reports, model exports
│
├── integrations/                   # ADD 7 adapters (one per phase)
│   ├── phase1_classical_adapter.py # Random Forest, SVM training
│   ├── phase2_cnn_adapter.py       # 1D CNN training
│   ├── phase3_resnet_adapter.py    # ResNet training
│   ├── phase4_transformer_adapter.py  # Transformer training
│   ├── phase5_spectrogram_adapter.py  # 2D CNN on spectrograms
│   ├── phase6_pinn_adapter.py      # PINN training
│   └── phase8_ensemble_adapter.py  # Ensemble building
│
├── models/                         # ADD 2 database models
│   ├── experiment.py               # Experiment metadata
│   └── training_run.py             # Training run details (epochs, losses)
│
├── tasks/                          # ADD 3 Celery tasks
│   ├── training_tasks.py           # Main training task
│   ├── evaluation_tasks.py         # Post-training evaluation
│   └── export_tasks.py             # Generate reports (async)
│
├── api/                            # NEW directory: REST API
│   ├── __init__.py
│   ├── routes.py                   # Flask blueprint for API endpoints
│   ├── task_status.py              # Task status endpoint
│   └── middleware.py               # CORS, authentication (Phase 11D)
│
├── templates/                      # NEW directory: Configuration templates
│   ├── presets/
│   │   ├── quick_test.yaml
│   │   ├── standard_training.yaml
│   │   ├── maximum_accuracy.yaml
│   │   └── fast_iteration.yaml
│   └── schemas/                    # JSON schemas for validation
│       ├── cnn_config_schema.json
│       ├── resnet_config_schema.json
│       └── transformer_config_schema.json
│
└── tests/                          # ADD 3 test files
    ├── test_training_service.py
    ├── test_experiment_service.py
    └── test_phase_adapters.py
```

**Total files added:** 32  
**Total files (11A + 11B):** 58 + 32 = **90 files**

---

## 11B.3 DETAILED PAGE SPECIFICATIONS

### Page 1: Experiment Configuration (`layouts/experiment_config.py`)

**Purpose:** Configure and launch new training experiments

**URL:** `/experiment/new` or `/experiment/clone/{experiment_id}`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  🧪 NEW EXPERIMENT                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Step Indicator: 1●──2○──3○──4○]                          │
│   Select Model  Configure  Review  Launch                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  STEP 1: SELECT MODEL TYPE                                  │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Phase 1 │  │ Phase 2 │  │ Phase 3 │  │ Phase 4 │      │
│  │ Classical│  │ 1D CNN  │  │ ResNet  │  │Transform│      │
│  │    ML    │  │         │  │         │  │   er    │      │
│  │ [Select] │  │ [Select]│  │ [Select]│  │ [Select]│      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│  │ Phase 5 │  │ Phase 6 │  │ Phase 8 │                    │
│  │ Spectro │  │  PINN   │  │Ensemble │                    │
│  │  gram   │  │         │  │         │                    │
│  │ [Select] │  │ [Select]│  │ [Select]│                    │
│  └─────────┘  └─────────┘  └─────────┘                    │
│                                                             │
│                               [Next: Configure Parameters →]│
└─────────────────────────────────────────────────────────────┘
```

**Step-by-Step Wizard:**

**STEP 1: Model Selection**
- 7 cards (one per phase)
- Each card shows:
  - Icon/thumbnail
  - Model name
  - Brief description (1 sentence)
  - Typical accuracy range
  - Training time estimate
  - "Select" button
- Interaction: Click card → highlights, enables Next button

**STEP 2: Configuration Mode**
- Toggle: "Use Preset" vs. "Expert Mode"

**Option A: Use Preset**
```
┌─────────────────────────────────────────────┐
│ Choose Configuration Preset                 │
├─────────────────────────────────────────────┤
│ ○ Quick Test (10 epochs, ~2 min)          │
│   Purpose: Fast sanity check               │
│   Expected accuracy: 85-90%                │
│                                             │
│ ● Standard Training (100 epochs, ~15 min)  │
│   Purpose: Good balance (RECOMMENDED)      │
│   Expected accuracy: 95-97%                │
│                                             │
│ ○ Maximum Accuracy (200 epochs, ~30 min)   │
│   Purpose: Best possible results           │
│   Expected accuracy: 97-98%                │
│                                             │
│ ○ Fast Iteration (50 epochs, ~8 min)       │
│   Purpose: Rapid experimentation           │
│   Expected accuracy: 93-95%                │
└─────────────────────────────────────────────┘

[← Back]              [Next: Review Config →]
```

**Option B: Expert Mode**
```
┌───────────────────────────────────────────────────────────┐
│ 📋 DATA CONFIGURATION                                     │
├───────────────────────────────────────────────────────────┤
│ Dataset:              [Dropdown: Select dataset_______▼] │
│ Train/Val/Test Split: [70%] [15%] [15%]                  │
│ Data Augmentation:    [☑] Enable                         │
│   ├─ [☑] Time Shift (±500 samples)                      │
│   ├─ [☑] Amplitude Scale (0.8-1.2×)                     │
│   ├─ [☑] Add Gaussian Noise (SNR: 20-30 dB)            │
│   └─ [☑] MixUp (alpha: 0.2)                             │
├───────────────────────────────────────────────────────────┤
│ 🧠 MODEL ARCHITECTURE (ResNet-18 specific)               │
├───────────────────────────────────────────────────────────┤
│ ResNet Variant:       [○ ResNet-18 ● ResNet-34 ○ ResNet-50] │
│ Input Channels:       [1] (grayscale signal)             │
│ Number of Classes:    [11] (fault types)                 │
│ Dropout Rate:         [0.3] ◄─────────► (0.0-0.5)       │
│ Pretrained Weights:   [☐] Use ImageNet initialization    │
├───────────────────────────────────────────────────────────┤
│ 🎯 TRAINING CONFIGURATION                                 │
├───────────────────────────────────────────────────────────┤
│ Batch Size:           [32▼] (16, 32, 64, 128)           │
│ Number of Epochs:     [100] ◄─────────► (10-200)        │
│ Learning Rate:        [0.001] (scientific notation OK)   │
│ Optimizer:            [Adam▼] (Adam, SGD, AdamW)        │
│ LR Scheduler:         [☑] Cosine Annealing w/ Warmup    │
│   ├─ Warmup Epochs:   [5]                                │
│   └─ Min LR:          [1e-6]                             │
│ Early Stopping:       [☑] Enable                         │
│   ├─ Patience:        [10] epochs                        │
│   └─ Min Delta:       [0.001] (minimum improvement)     │
├───────────────────────────────────────────────────────────┤
│ 💾 CHECKPOINT CONFIGURATION                               │
├───────────────────────────────────────────────────────────┤
│ Save Frequency:       [○ Every epoch ● Best only]       │
│ Save Last N:          [3] checkpoints                    │
│ Export ONNX:          [☑] Yes (for deployment)          │
└───────────────────────────────────────────────────────────┘

[Show Advanced Options ▼]  (GPU settings, mixed precision, etc.)

[← Back]  [Save as Preset]  [Next: Review Config →]
```

**Validation Rules:**
- Real-time validation on every field change
- Red border + error message for invalid values
- Examples:
  - Batch size must be ≤ dataset size
  - Epochs must be > 0
  - Learning rate must be > 0 and < 1
  - Train + Val + Test must sum to 100%

**STEP 3: Review Configuration**
```
┌─────────────────────────────────────────────────────────────┐
│ 📋 EXPERIMENT SUMMARY                                        │
├─────────────────────────────────────────────────────────────┤
│ Experiment Name: [ResNet34_Standard_2025_06_15_____________]│
│                  (auto-generated, editable)                  │
│                                                              │
│ Model Type:       ResNet-34                                  │
│ Dataset:          BearingFaults_1430signals_v2               │
│ Training Config:  Standard preset (100 epochs)               │
│                                                              │
│ Estimated Time:   15-20 minutes                             │
│ Estimated Cost:   $0.12 GPU-hours (if cloud)                │
│                                                              │
│ [View Full Config (YAML) ▼]                                 │
│   (Collapsible section showing complete YAML)               │
│                                                              │
│ ⚠️  WARNINGS:                                               │
│   • Training will use GPU 0 (currently 45% utilized)        │
│   • Checkpoint size: ~180 MB per save                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘

[← Back to Edit]    [💾 Save Config Only]    [🚀 Start Training]
```

**STEP 4: Launch**
- Click "Start Training" → Modal confirmation
- Modal shows:
  - "Training will start in background"
  - "You can close this page safely"
  - "Notification when complete"
  - Checkbox: "Navigate to Training Monitor"
- Click "Confirm" → Creates Celery task → Redirects to Training Monitor

---

### Page 2: Training Monitor (`layouts/training_monitor.py`)

**Purpose:** Real-time monitoring of running training job

**URL:** `/experiment/{experiment_id}/monitor`

**Auto-Navigate:** Yes (from Step 4 of config page)

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  ⏳ TRAINING IN PROGRESS                                    │
│  Experiment: ResNet34_Standard_2025_06_15                   │
│  Started: 2025-06-15 14:32:11    Elapsed: 00:08:42         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PROGRESS                                                   │
│  Epoch: 47/100  ████████████░░░░░░░░░░░░  47%             │
│  ETA: 8 minutes 23 seconds                                  │
│                                                             │
│  [⏸ Pause]  [⏹ Stop]  [📊 View Logs]                      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  CURRENT METRICS                                            │
│  ┌──────────────┬──────────────┬──────────────┐           │
│  │ Train Loss   │ Train Acc    │ Val Loss     │           │
│  │   0.0234 ↓  │   97.8% ↑   │   0.0412 ↓  │           │
│  └──────────────┴──────────────┴──────────────┘           │
│  ┌──────────────┬──────────────┐                           │
│  │ Val Acc      │ Best Val Acc │                           │
│  │   96.9% ↑   │   97.1% @42  │                           │
│  └──────────────┴──────────────┘                           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TRAINING CURVES (LIVE UPDATE)                              │
│  ┌─────────────────────────────────────────────────────────┤
│  │                                                          │
│  │  Loss                      Accuracy                      │
│  │  0.5┤                     100%┤          ╱────           │
│  │     │╲                        │        ╱                 │
│  │     │ ╲___                    │      ╱                   │
│  │  0.0└──────────────        50%└────────────────          │
│  │       0   25  50  75 100       0   25  50  75 100       │
│  │                                                          │
│  │  [Download Data (CSV)]  [Export Plot (PNG)]             │
│  └──────────────────────────────────────────────────────────┘
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM RESOURCES                                            │
│  GPU 0: 94% ████████████████████░  Mem: 6.2/8.0 GB         │
│  CPU:   32% ████████░░░░░░░░░░░░  RAM: 12.3/32.0 GB        │
│  Disk:  I/O: 45 MB/s (writing checkpoints)                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  RECENT EVENTS                                               │
│  14:40:53  Epoch 47 complete. Val acc: 96.9%                │
│  14:40:51  Saving checkpoint (best model so far)            │
│  14:39:12  Epoch 46 complete. Val acc: 96.8%                │
│  14:37:33  Epoch 45 complete. Val acc: 96.7%                │
│  [Show Full Logs ▼]                                         │
└─────────────────────────────────────────────────────────────┘

Auto-refresh: Every 2 seconds  [⏸ Pause Updates]
```

**Key Features:**

1. **Real-Time Updates**
   - Polling: Every 2 seconds (via dcc.Interval)
   - Endpoint: `/api/task-status/{task_id}`
   - Updates: Progress bar, metrics cards, charts

2. **Interactive Controls**
   - **Pause Button:** 
     - Sends signal to Celery task: "pause after current epoch"
     - Task saves checkpoint, enters paused state
     - Button changes to "Resume"
   - **Stop Button:**
     - Confirmation modal: "Stop training? Progress will be saved."
     - Sends SIGTERM to Celery task
     - Task saves final checkpoint, marks as "stopped"
   - **View Logs Button:**
     - Opens modal with scrollable log viewer
     - Server-Sent Events (SSE) for live log streaming
     - Auto-scroll to bottom

3. **Training Curves**
   - Line charts update in real-time
   - X-axis: Epoch number
   - Y-axis: Loss (left) and Accuracy (right)
   - Dual y-axes (loss scale 0-1, accuracy 0-100%)
   - Hover: Show exact values
   - Zoom: Click-drag to zoom into region

4. **System Resources**
   - GPU utilization (from nvidia-smi)
   - Memory usage (GPU RAM)
   - CPU and system RAM
   - Disk I/O (checkpoint writes are visible as spikes)

5. **Event Timeline**
   - Last 10 events shown
   - Format: [timestamp] [message]
   - Types: Epoch complete, Checkpoint saved, Early stopping triggered
   - "Show Full Logs" expands to show all events

**Edge Cases:**

- **Task Fails:** 
  - Progress bar turns red
  - Shows error message: "Training failed at epoch 23. Error: [error message]"
  - Buttons: "View Logs", "Retry", "Go to Results"
  
- **Task Completes:**
  - Progress bar turns green
  - Confetti animation (optional, celebratory UX)
  - Auto-redirect to Results page after 5 seconds (with countdown)
  - Button: "View Results Now"

- **User Closes Page:**
  - Training continues in background (Celery task unaffected)
  - User can return to this page anytime (bookmark URL)
  - Notification in dashboard: "Training in progress (47%)"

---

### Page 3: Experiment Results (`layouts/experiment_results.py`)

**Purpose:** Comprehensive visualization of completed experiment

**URL:** `/experiment/{experiment_id}/results`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  ✅ EXPERIMENT RESULTS                                      │
│  Experiment: ResNet34_Standard_2025_06_15                   │
│  Status: Completed    Duration: 14m 32s    Completed: Just now │
├─────────────────────────────────────────────────────────────┤
│  [Overview] [Detailed Metrics] [Visualizations] [Artifacts]│
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB 1: OVERVIEW                                            │
│                                                             │
│  ⭐ KEY METRICS                                             │
│  ┌────────────┬────────────┬────────────┬────────────┐    │
│  │  Accuracy  │ Precision  │   Recall   │     F1     │    │
│  │   96.8%    │   96.5%    │   96.7%    │   96.6%    │    │
│  │   ⭐⭐⭐⭐  │            │            │            │    │
│  └────────────┴────────────┴────────────┴────────────┘    │
│                                                             │
│  📊 TRAINING SUMMARY                                        │
│  • Total Epochs: 100 (no early stopping triggered)         │
│  • Best Epoch: 87 (val_acc: 97.1%)                         │
│  • Final Train Loss: 0.0123 | Final Val Loss: 0.0389      │
│  • Training Time: 14m 32s | Avg: 8.7 sec/epoch            │
│  • GPU Utilization: 92% (good efficiency)                  │
│                                                             │
│  ⚖️ COMPARISON TO BASELINE                                 │
│  vs. Phase 1 Random Forest (95.3%):  +1.5% ✅             │
│  vs. Phase 2 CNN (94.2%):            +2.6% ✅             │
│  vs. Best Previous (ResNet-18, 96.2%): +0.6% ✅           │
│                                                             │
│  🎯 RECOMMENDATIONS                                         │
│  ✅ Model ready for deployment (accuracy > 96%)            │
│  ⚠️ Consider: Ensemble with Transformer (potential +1-2%) │
│  💡 Tip: Oil whirl class has lower recall (92%) - review │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB 2: DETAILED METRICS                                    │
│                                                             │
│  📋 PER-CLASS PERFORMANCE                                   │
│  ┌─────────────────┬──────┬────────┬────────┬──────┐      │
│  │ Fault Class     │ Prec │ Recall │   F1   │ Supp │      │
│  ├─────────────────┼──────┼────────┼────────┼──────┤      │
│  │ Normal          │ 98.5 │  99.2  │  98.8  │ 130  │      │
│  │ Misalignment    │ 97.3 │  96.8  │  97.0  │ 130  │      │
│  │ Imbalance       │ 96.9 │  97.5  │  97.2  │ 130  │      │
│  │ Oil Whirl       │ 94.2 │  92.3  │  93.2  │ 130  │ ⚠️   │
│  │ Oil Whip        │ 95.8 │  96.1  │  95.9  │ 130  │      │
│  │ ... (11 rows)   │      │        │        │      │      │
│  └─────────────────┴──────┴────────┴────────┴──────┘      │
│                                                             │
│  🔍 CONFUSION MATRIX (Interactive)                          │
│  [Heatmap: Predicted vs True, hover shows counts]          │
│  [Normalized ○ Absolute ● ]                                │
│                                                             │
│  📈 ROC CURVES (One-vs-Rest)                                │
│  [11 curves, one per class, with AUC scores]               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB 3: VISUALIZATIONS                                      │
│                                                             │
│  📊 TRAINING HISTORY                                        │
│  [Line charts: Loss and Accuracy over 100 epochs]          │
│                                                             │
│  🔥 FILTER VISUALIZATIONS (ResNet-specific)                 │
│  [Grid: First layer conv filters, 64 filters shown]        │
│                                                             │
│  🧠 ACTIVATION MAPS                                         │
│  [Select signal from dropdown]                              │
│  [Show: Input signal → Layer activations → Output]         │
│                                                             │
│  ⚠️ FAILURE CASE ANALYSIS                                  │
│  [Table: Top 10 misclassified signals]                     │
│  [Click row → View signal + Grad-CAM explanation]          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  TAB 4: ARTIFACTS & EXPORT                                  │
│                                                             │
│  💾 MODEL FILES                                             │
│  • model_epoch_87.pth (182 MB)        [Download]           │
│  • model_epoch_87.onnx (181 MB)       [Download]           │
│  • config.yaml (3 KB)                 [Download]           │
│                                                             │
│  📊 RESULTS & DATA                                          │
│  • predictions_test_set.csv (45 KB)   [Download]           │
│  • confusion_matrix.png               [Download]           │
│  • roc_curves.png                     [Download]           │
│  • training_history.json              [Download]           │
│                                                             │
│  📄 REPORTS                                                 │
│  • experiment_report.pdf (12 pages)   [Generate & Download]│
│  • tensorboard_logs.zip (234 MB)      [Download]           │
│                                                             │
│  🚀 DEPLOYMENT OPTIONS                                      │
│  • [Deploy to Production]  (Phase 9 integration)           │
│  • [Add to Ensemble]       (Phase 8 integration)           │
│  • [Register in Model Registry]                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

[← Back to Experiments]  [🔄 Clone & Retrain]  [🗑️ Delete Experiment]
```

**Key Features:**

1. **Multi-Tab Organization**
   - Overview: Quick summary for stakeholders
   - Detailed Metrics: Engineers deep-dive
   - Visualizations: Visual analysis
   - Artifacts: Downloads and deployment

2. **Interactive Confusion Matrix**
   - Plotly heatmap
   - Hover: Shows "Predicted: X, True: Y, Count: 12"
   - Click cell: Filters failure analysis table to show those samples
   - Toggle: Normalized (0-1) vs. Absolute counts

3. **Failure Case Analysis**
   - Automatically identifies top 10 worst predictions
   - Table columns: Signal ID, True Class, Predicted Class, Confidence, Error Type
   - Click row: Opens modal with:
     - Signal visualization
     - Grad-CAM heatmap (what model looked at)
     - Predicted probabilities bar chart
     - "Why did it fail?" analysis (e.g., "Signal has mixed characteristics")

4. **Model-Specific Sections**
   - If CNN/ResNet: Show filter visualizations
   - If Transformer: Show attention maps
   - If PINN: Show physics consistency plots
   - If Ensemble: Show member contributions

5. **PDF Report Generation**
   - Async task (Celery)
   - Includes: All charts, metrics table, config, recommendations
   - Template: Professional LaTeX template
   - Takes 10-30 seconds to generate

---

### Page 4: Experiment History (`layouts/experiment_history.py`)

**Purpose:** Browse and compare all past experiments

**URL:** `/experiments`

**Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  🧪 EXPERIMENT HISTORY                                      │
├─────────────────────────────────────────────────────────────┤
│  [Filters & Search]                                         │
│  Search: [___________________________] 🔍                   │
│  Model Type: [All ▼] Status: [All ▼] Date: [Last 30 days ▼]│
│  Sort By: [Accuracy ▼]  Order: [Descending ▼]              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  EXPERIMENTS TABLE                                           │
│  [Pagination: 1 2 3 ... 10]  Showing 1-50 of 472           │
│  [☐] Select All  [Actions ▼: Compare, Delete, Export]      │
│                                                             │
│  ┌──┬────────────┬────────┬──────┬─────────┬────────┬────┐│
│  │☐│    Date    │  Name  │ Model│   Acc   │Duration│ ⚙️ ││
│  ├──┼────────────┼────────┼──────┼─────────┼────────┼────┤│
│  │☐│ 2025-06-15 │ResNet34│ResNet│ 96.8% ✅│ 14m 32s│ ⚙️ ││
│  │☐│ 2025-06-14 │TransV2 │Transf│ 96.5% ✅│ 22m 11s│ ⚙️ ││
│  │☐│ 2025-06-14 │CNN_Fast│ CNN  │ 94.2%   │  8m 45s│ ⚙️ ││
│  │☐│ 2025-06-13 │ResNet50│ResNet│ FAILED ❌│  3m 12s│ ⚙️ ││
│  │☐│ 2025-06-12 │PINN_v1 │ PINN │ 97.1% ⭐│ 18m 03s│ ⚙️ ││
│  │☐│ ... (50 rows per page)                               ││
│  └──┴────────────┴────────┴──────┴─────────┴────────┴────┘│
│                                                             │
│  Click row → View details      ⚙️ → Quick actions menu     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  COMPARISON MODE (when 2+ experiments selected)             │
│  [Compare Selected (3 experiments)]                         │
│                                                             │
│  Opens modal with side-by-side comparison:                  │
│  • Metrics table (accuracy, F1, etc.)                       │
│  • Training curves overlay (3 lines on same plot)           │
│  • Hyperparameter diff (highlights differences)             │
│  • Statistical significance test (McNemar's test)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

1. **Advanced Filtering**
   - Full-text search (experiment name, notes)
   - Multi-select filters (model type, status, user)
   - Date range picker
   - Saved filter presets ("My experiments", "Failed runs", "Top performers")

2. **Bulk Actions**
   - Select multiple experiments
   - Actions: Compare, Delete, Export to CSV, Generate summary report

3. **Quick Actions Menu (⚙️ dropdown)**
   - View Details
   - Clone & Modify
   - Download Model
   - Add to Ensemble
   - Delete
   - Add Note/Tag

4. **Comparison Feature**
   - Select 2-10 experiments
   - Side-by-side metrics
   - Statistical tests (is difference significant?)
   - Hyperparameter diff (what changed?)
   - Winner recommendation

5. **Table Features**
   - Sortable columns (click header)
   - Resizable columns (drag border)
   - Column visibility toggle (hide/show columns)
   - Export to CSV
   - Pagination (50/100/200 rows per page)

---

## 11B.4 CRITICAL IMPLEMENTATION GUIDELINES

### Guideline 1: Progress Tracking Implementation

**Challenge:** Phase 1-8 training code doesn't have progress callbacks.

**Solution: Minimal Intrusion Pattern**

**Step 1: Add Progress Callback Parameter (Optional)**
```python
# In existing training code (e.g., training/trainer.py):

class Trainer:
    def __init__(self, ..., progress_callback=None):
        self.progress_callback = progress_callback  # NEW
    
    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss, val_acc = self._validate()
            
            # NEW: Call progress callback if provided
            if self.progress_callback:
                self.progress_callback({
                    'epoch': epoch + 1,
                    'total_epochs': self.num_epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'progress': (epoch + 1) / self.num_epochs
                })
```

**Step 2: Adapter Provides Callback**
```python
# In integrations/phase3_resnet_adapter.py:

def train_resnet(config, task_id):
    """Adapter function called by Celery task."""
    
    # Define progress callback
    def update_progress(metrics):
        # Update Redis with progress
        redis_client.setex(
            f"task:{task_id}:progress",
            86400,  # 24 hour TTL
            json.dumps({
                'status': 'running',
                'progress': metrics['progress'],
                'current_epoch': metrics['epoch'],
                'total_epochs': metrics['total_epochs'],
                'train_loss': metrics['train_loss'],
                'val_loss': metrics['val_loss'],
                'val_accuracy': metrics['val_accuracy'],
                'eta_seconds': estimate_eta(metrics),
                'message': f"Training epoch {metrics['epoch']}/{metrics['total_epochs']}..."
            })
        )
    
    # Initialize trainer with callback
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        progress_callback=update_progress  # Injected here
    )
    
    # Train (existing code, unchanged)
    results = trainer.train()
    
    return results
```

**Key Insight:** Only 3 lines added to existing training code (if statement). All dashboard complexity hidden in adapter.

---

### Guideline 2: Task State Management

**States:** pending, running, paused, completed, failed, cancelled

**State Transitions:**
```
pending → running → completed
                 → failed
                 → cancelled (by user)
                 → paused → running (resume)
```

**Implementation:**
```python
# In tasks/training_tasks.py:

@celery_app.task(bind=True)
def train_model_task(self, config):
    """Celery task for model training."""
    task_id = self.request.id
    
    try:
        # Set initial state
        update_task_state(task_id, 'running', progress=0)
        
        # Call adapter (which calls Phase code)
        adapter = get_adapter(config['model_type'])
        results = adapter.train(config, task_id)
        
        # Success
        update_task_state(task_id, 'completed', progress=1.0, results=results)
        save_results_to_database(task_id, results)
        send_notification(config['user_id'], f"Training complete! Accuracy: {results['accuracy']:.2%}")
        
        return results
        
    except Exception as e:
        # Failure
        update_task_state(task_id, 'failed', error=str(e))
        log_exception(e)
        send_notification(config['user_id'], f"Training failed: {str(e)}")
        raise
```

**Pause/Resume Mechanism:**
```python
# In training loop (existing code):
for epoch in range(num_epochs):
    # Check for pause signal
    if check_pause_signal(task_id):
        save_checkpoint(model, f"paused_epoch_{epoch}.pth")
        update_task_state(task_id, 'paused')
        wait_for_resume_signal(task_id)  # Blocks here
        update_task_state(task_id, 'running')
    
    # Regular training
    train_epoch(model, train_loader)
    ...
```

---

### Guideline 3: Configuration Validation

**Multi-Layer Validation:**

**Layer 1: Client-Side (JavaScript)**
- Real-time as user types
- Instant feedback (red border, error message)
- Prevents form submission if invalid

**Layer 2: Dash Callback (Python)**
- Server-side validation before task creation
- Checks: Value ranges, consistency (e.g., train+val+test=100%)
- Returns: List of errors or "Valid"

**Layer 3: Service Layer (Python)**
- Validates against database constraints
- Example: Dataset exists, user has permission
- Returns: Validated config or raises exception

**Layer 4: Training Code (Python)**
- Final sanity checks before training
- Example: Model can load on available GPU, dataset not corrupted
- Fail fast: Better to fail in 2 seconds than after 10 minutes

**Example Validation Rules:**

| Parameter | Validation | Error Message |
|-----------|------------|---------------|
| Batch size | Power of 2, ≤ dataset size | "Batch size must be power of 2 and ≤ 1430" |
| Learning rate | 1e-6 ≤ lr ≤ 1.0 | "Learning rate must be between 0.000001 and 1.0" |
| Epochs | 1 ≤ epochs ≤ 500 | "Epochs must be between 1 and 500" |
| Train/Val/Test | Sum = 100% | "Split percentages must sum to 100%" |
| Dropout | 0.0 ≤ dropout < 1.0 | "Dropout must be between 0.0 and 1.0 (exclusive)" |

---

### Guideline 4: Result Storage Strategy

**Challenge:** Training produces many artifacts (model files, plots, logs).

**Storage Schema:**

```
Database (PostgreSQL):
  experiments table:
    - id, name, model_type, status, created_at, user_id
    - config (JSON), hyperparameters (JSON)
    - metrics (JSON): {accuracy, f1, precision, recall, ...}
    - best_epoch, total_epochs, duration_seconds
  
  training_runs table (one row per epoch):
    - experiment_id, epoch, train_loss, val_loss, val_accuracy
    - timestamp, checkpoint_path

File Storage:
  storage/experiments/{experiment_id}/
    ├── checkpoints/
    │   ├── epoch_10.pth
    │   ├── epoch_20.pth
    │   └── best_model.pth
    ├── plots/
    │   ├── confusion_matrix.png
    │   ├── roc_curves.png
    │   ├── training_history.png
    │   └── filter_vis.png
    ├── exports/
    │   ├── model.onnx
    │   ├── predictions.csv
    │   └── report.pdf
    ├── logs/
    │   └── training.log
    └── config.yaml
```

**Access Pattern:**
- Metadata (accuracy, duration): Database (fast queries, filtering)
- Binary files (models, plots): File storage (referenced by path in database)
- Logs: File storage (streamed via SSE endpoint)

**Cleanup Policy:**
- Keep best checkpoint forever
- Delete intermediate checkpoints after 30 days (configurable)
- Delete failed experiments after 7 days (unless marked "keep")
- Archive old experiments to cold storage (S3 Glacier) after 1 year

---

### Guideline 5: Notification System

**Notification Types:**

1. **In-App (Toast Messages)**
   - Success: "Training started! (Experiment #123)"
   - Warning: "Model accuracy below baseline (92% < 95%)"
   - Error: "Training failed: CUDA out of memory"
   - Duration: 5 seconds (auto-dismiss) or user-dismiss

2. **Browser Notifications**
   - Triggered: When training completes (if user navigated away)
   - Requires: User permission (request on first experiment)
   - Content: "Training complete! Accuracy: 96.8%"

3. **Email (Phase 11D)**
   - Triggered: Experiment completes or fails
   - Content: Summary + link to results page
   - Frequency: Configurable (immediate, daily digest, never)

4. **Slack/Teams (Phase 11D)**
   - Webhook integration
   - Posts: Experiment summary to team channel
   - Format: Rich message (image, metrics, link)

**Implementation:**
```python
# In services/notification_service.py:

def notify_training_complete(experiment_id, results):
    """Send notifications for completed training."""
    experiment = db.query(Experiment).get(experiment_id)
    
    # In-app notification (stored in session)
    create_toast(
        user_id=experiment.user_id,
        type='success',
        message=f"Training complete! Accuracy: {results['accuracy']:.2%}",
        link=f"/experiment/{experiment_id}/results"
    )
    
    # Browser notification (if user away)
    if user_is_away(experiment.user_id):
        send_browser_notification(
            user_id=experiment.user_id,
            title="Experiment Complete",
            body=f"{experiment.name}: {results['accuracy']:.2%} accuracy",
            icon="/assets/logo.png"
        )
    
    # Email (if enabled in user preferences)
    if user_preferences.email_enabled:
        send_email(
            to=experiment.user.email,
            subject=f"[ML Dashboard] {experiment.name} Complete",
            template='training_complete',
            context={'experiment': experiment, 'results': results}
        )
```

---

## 11B.5 INTEGRATION WITH PHASES 1-8 (Detailed)

### Integration Pattern for Each Phase

**Phase 1: Classical ML (Random Forest, SVM)**

**Adapter:** `integrations/phase1_classical_adapter.py`

**Key Points:**
- Training is fast (<2 minutes) → may not need progress callbacks
- Uses Phase 1 feature extraction (36 features)
- No checkpoints (models are small, ~5 MB)

**Adapter Functions:**
```
train_random_forest(config, task_id)
train_svm(config, task_id)
get_feature_importance(model)  # For visualization
```

**Config Parameters:**
- n_estimators (RF), C (SVM), kernel (SVM)
- Feature selection method (all, top-k, correlation-based)
- Cross-validation folds

**Unique Visualizations:**
- Feature importance bar chart
- Decision tree (first tree of RF, simplified)
- Support vectors visualization (SVM)

---

**Phase 2: 1D CNN**

**Adapter:** `integrations/phase2_cnn_adapter.py`

**Key Points:**
- Moderate training time (~10 minutes)
- Requires progress callbacks (add to Phase 2 Trainer)
- Standard checkpointing

**Config Parameters:**
- Number of conv layers (1-5)
- Kernel sizes ([3, 5, 7, 11] for each layer)
- Number of filters ([64, 128, 256])
- Pooling type (max, average)
- FC layer sizes

**Unique Visualizations:**
- Learned filter kernels (1D, show as line plots)
- Activation maps (for selected signal)

---

**Phase 3: ResNet**

**Adapter:** `integrations/phase3_resnet_adapter.py`

**Key Points:**
- Long training time (15-25 minutes)
- Progress callbacks essential
- Large checkpoints (180 MB for ResNet-50)

**Config Parameters:**
- Architecture: ResNet-18, 34, 50, 101
- Pretrained: ImageNet initialization (yes/no)
- Width multiplier (scale number of channels)

**Unique Visualizations:**
- Residual connection analysis (gradient flow)
- Layer-wise activation statistics

---

**Phase 4: Transformer**

**Adapter:** `integrations/phase4_transformer_adapter.py`

**Key Points:**
- Very long training (20-30 minutes)
- Memory-intensive (large batch sizes problematic)
- Special handling for attention maps

**Config Parameters:**
- Number of layers (4, 6, 8, 12)
- Number of attention heads (4, 8, 16)
- d_model (embedding dimension: 128, 256, 512)
- Positional encoding type

**Unique Visualizations:**
- Attention weight heatmaps (which time steps attend to which)
- Attention head specialization analysis
- Positional encoding visualization

---

**Phase 5: Spectrogram (2D CNN)**

**Adapter:** `integrations/phase5_spectrogram_adapter.py`

**Key Points:**
- Preprocessing: Generate spectrograms (can be slow)
- Training time similar to Phase 3
- Dual-stream model option (time + frequency)

**Config Parameters:**
- Spectrogram method (STFT, CWT, WVD)
- Window size (STFT: 128, 256, 512)
- 2D CNN architecture (ResNet-2D, EfficientNet-2D)

**Unique Visualizations:**
- Spectrogram input examples
- 2D filter visualizations
- Spectrogram-specific Grad-CAM

---

**Phase 6: PINN (Physics-Informed Neural Network)**

**Adapter:** `integrations/phase6_pinn_adapter.py`

**Key Points:**
- Requires metadata (load, speed, temp) → validate availability
- Dual loss (classification + physics)
- Special evaluation metrics (frequency consistency)

**Config Parameters:**
- Physics loss weight (lambda_physics: 0.1-1.0)
- Sommerfeld/Reynolds number inclusion
- Knowledge graph (enable/disable)

**Unique Visualizations:**
- Physics loss curve (separate from classification loss)
- Frequency consistency plot (predicted vs. expected)
- Sommerfeld number correlation

---

**Phase 8: Ensemble**

**Adapter:** `integrations/phase8_ensemble_adapter.py`

**Key Points:**
- No training (just combines existing models)
- Fast operation (<1 minute)
- Requires selecting base models

**Config Parameters:**
- Base model selection (choose 3-7 models from registry)
- Ensemble method (soft voting, stacking, boosting)
- Weights (equal vs. optimized)

**Unique Visualizations:**
- Member contribution plot (which model contributed to each prediction)
- Diversity metrics (agreement matrix)
- Error correlation heatmap

---

## 11B.6 ACCEPTANCE CRITERIA (Phase 11B Complete When)

✅ **Configuration Pages Functional**
- All 7 model types have configuration forms
- Presets load correctly and are editable
- Expert mode exposes all hyperparameters
- Validation catches 100% of invalid inputs

✅ **Training Execution Working**
- Click "Start Training" → Celery task created
- Task runs in background (web request returns immediately)
- Progress tracking works for all model types
- Pause/Resume functionality tested

✅ **Training Monitor Real-Time**
- Progress bar updates every 2 seconds
- Metrics cards show current values
- Training curves update live
- System resources displayed accurately

✅ **Results Visualization Complete**
- All tabs render (Overview, Metrics, Visualizations, Artifacts)
- Model-specific visualizations work (attention maps, filters, etc.)
- Failure case analysis identifies worst 10 predictions
- PDF report generation successful

✅ **Experiment History Operational**
- Table loads all experiments from database
- Filtering and sorting work correctly
- Comparison feature compares 2-10 experiments
- Bulk actions (delete, export) functional

✅ **Integration with Phases 1-8 Validated**
- All 7 adapters implemented and tested
- Progress callbacks working in training loops
- Results correctly saved to database + storage
- No modifications to core Phase code (only callback additions)

✅ **Performance Targets Met**
- Config page loads in <1 second
- Progress updates have <500ms latency
- Results page loads in <3 seconds (with caching)
- Can handle 10 concurrent training jobs

✅ **Error Handling Robust**
- Training failures logged and displayed to user
- Task crashes don't break dashboard (graceful degradation)
- Out-of-memory errors handled (suggestion to reduce batch size)
- Network interruptions don't lose training progress

✅ **Notifications Working**
- Toast messages appear for key events
- Browser notifications work (if user permission granted)
- Notifications link to relevant pages

✅ **Testing Coverage**
- Training service: >85% coverage
- All 7 adapters: 100% coverage (critical path)
- Callbacks: >70% coverage
- End-to-end: Train → Monitor → Results tested for each model type

✅ **Documentation Complete**
- User guide: "How to Train Models"
- Developer guide: "Adding New Model Types"
- Troubleshooting: Common training issues
- Video tutorial: 5-minute walkthrough

---

## 11B.7 RISKS & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Training code changes break adapter** | Medium | High | Comprehensive integration tests, adapter versioning |
| **Celery worker crashes during training** | Medium | Medium | Checkpoint every epoch, auto-retry failed tasks |
| **Progress tracking overhead slows training** | Low | Medium | Minimize callback frequency (once per epoch), async Redis writes |
| **Large checkpoint files fill disk** | High | Medium | Implement cleanup policy, alert at 80% disk usage |
| **User closes browser, loses monitoring** | High | Low | Task continues in background, email notification on completion |
| **Multiple experiments compete for GPU** | High | Medium | Task queue priority, GPU allocation manager (Phase 11D) |
| **Config validation misses edge case** | Medium | Medium | Extensive unit tests, fuzzing with random configs |

---

## 11B.8 PHASE 11B DELIVERABLES SUMMARY

**4 New Pages:**
1. Experiment Configuration (multi-step wizard)
2. Training Monitor (real-time progress)
3. Experiment Results (comprehensive visualization)
4. Experiment History (browse, filter, compare)

**7 Integration Adapters:**
- Phase 1: Classical ML
- Phase 2: 1D CNN
- Phase 3: ResNet
- Phase 4: Transformer
- Phase 5: Spectrogram
- Phase 6: PINN
- Phase 8: Ensemble

**Key Services:**
- Training orchestration
- Experiment management
- Model evaluation
- PDF report generation

**Infrastructure:**
- Celery task queue
- Redis progress tracking
- REST API endpoints (task status, logs)
- File storage organization

**Testing:**
- 85%+ service layer coverage
- 100% adapter coverage
- End-to-end tests for all model types

---