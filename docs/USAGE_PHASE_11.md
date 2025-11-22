# Phase 11: Enterprise Dashboard - Usage Guide

**Status**: ✅ Operational (50% feature coverage)
**Last Updated**: 2024-11-22
**Version**: Phase 11C (Post-XAI Integration)

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Dashboard Architecture](#dashboard-architecture)
- [Available Features](#available-features)
- [Feature Status Matrix](#feature-status-matrix)
- [Usage Guides by Feature](#usage-guides-by-feature)
- [Missing Features](#missing-features)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **Enterprise Dashboard** is a production-grade Plotly Dash application that provides a comprehensive web interface for managing the entire LSTM_PFD system. It enables users to perform all operations through an intuitive GUI without writing code.

### What Can You Do?

✅ **Data Management** (Phase 0)
- Generate synthetic bearing vibration signals
- Import MAT files from existing datasets
- Browse and explore datasets
- Visualize signals in time and frequency domains

✅ **Model Training** (Phases 1-8)
- Create and configure experiments
- Monitor training progress in real-time
- Compare multiple experiments
- View detailed results and metrics

✅ **Explainability** (Phase 7)
- Generate SHAP explanations
- Generate LIME explanations
- Integrated Gradients attribution
- Grad-CAM visualization

⚠️ **Partially Available**
- Model comparison (basic)
- Settings and preferences (incomplete)

❌ **Not Yet Available** (See [Missing Features](#missing-features))
- HPO campaigns
- Model deployment and quantization
- System monitoring
- API management
- Testing dashboard

> **📖 For a complete gap analysis**, see [DASHBOARD_GAPS.md](DASHBOARD_GAPS.md)

---

## Getting Started

### Prerequisites

1. **Database Setup**
   ```bash
   # Run all migrations (including latest XAI migration)
   cd dash_app
   python database/run_migration.py
   ```

2. **Celery Worker** (Required for async tasks)
   ```bash
   # Terminal 1: Start Celery worker
   cd dash_app
   celery -A tasks worker --loglevel=info
   ```

3. **Redis** (Message broker for Celery)
   ```bash
   # Make sure Redis is running
   redis-server
   ```

### Starting the Dashboard

```bash
# Terminal 2: Start dashboard
cd dash_app
python app.py
```

**Access**: http://localhost:8050

**Default Login**: Dashboard doesn't require authentication by default

---

## Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PLOTLY DASH APPLICATION                   │
│                     (dash_app/app.py)                        │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐   ┌──────▼──────┐   ┌───────▼───────┐
│   LAYOUTS     │   │  CALLBACKS  │   │   SERVICES    │
│  (UI Pages)   │   │  (Logic)    │   │  (Business)   │
│               │   │             │   │               │
│ • home.py     │   │ • data_     │   │ • xai_        │
│ • data_       │   │   generation│   │   service     │
│   generation  │   │ • xai_      │   │ • notification│
│ • experiments │   │   callbacks │   │ • monitoring  │
│ • xai_        │   │ • experiment│   │               │
│   dashboard   │   │   _wizard   │   │               │
└───────┬───────┘   └──────┬──────┘   └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                ┌──────────▼──────────┐
                │    CELERY TASKS     │
                │  (Background Jobs)  │
                │                     │
                │ • data_generation   │
                │ • mat_import        │
                │ • xai_explanation   │
                │ • training_tasks    │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐   ┌──────▼──────┐   ┌──────▼──────┐
│  POSTGRESQL   │   │    REDIS    │   │  FILE SYS.  │
│  (Metadata)   │   │ (Message Q) │   │  (Models,   │
│               │   │             │   │   Datasets) │
└───────────────┘   └─────────────┘   └─────────────┘
```

### Technology Stack

- **Frontend**: Plotly Dash, Dash Bootstrap Components
- **Backend**: Flask (via Dash), Celery
- **Database**: PostgreSQL
- **Message Queue**: Redis
- **Deep Learning**: PyTorch
- **XAI**: SHAP, LIME, Captum
- **Visualization**: Plotly

---

## Available Features

### 🏠 Home Dashboard

**Route**: `/`

**Description**: Landing page with system overview

**Features**:
- Recent experiments summary
- System status
- Quick actions
- Navigation overview

---

### 📊 Data Generation (Phase 0)

**Route**: `/data-generation`

**Description**: Generate synthetic vibration signals or import MAT files

#### Tab 1: Generate Synthetic Data

**What It Does**: Creates physics-based bearing vibration signals with configurable faults

**Usage**:

1. **Configure Dataset**
   - Dataset Name: e.g., "bearing_faults_v1"
   - Number of signals per fault: 10-1000 (default: 100)

2. **Select Fault Types** (11 available)
   - Normal (baseline)
   - Ball Fault, Inner Race, Outer Race
   - Combined Fault
   - Imbalance, Misalignment
   - Oil Whirl, Cavitation, Looseness, Oil Deficiency

3. **Choose Severity Levels** (4 available)
   - Incipient (early stage)
   - Mild
   - Moderate
   - Severe

4. **Configure Noise Layers** (7-layer model)
   - Sensor noise
   - Quantization noise
   - Environmental noise
   - Electromagnetic interference
   - Thermal drift
   - Aliasing noise
   - Random walk drift

5. **Operating Conditions**
   - RPM: 1000-3600
   - Load: 0-100%
   - Temperature: 20-80°C

6. **Data Augmentation** (optional)
   - Time shift: ±10%
   - Amplitude scaling: ±20%
   - Noise injection: 0-10%

7. **Output Options**
   - Format: MAT, HDF5, or both
   - Save location: `data/datasets/`

8. **Click "Generate Dataset"**

**Output**:
- Signals saved to disk
- Database record created
- Progress shown in real-time
- Email notification on completion

**Example**:
```
Dataset Name: bearing_test_v1
Signals per fault: 100
Fault types: Normal, Ball Fault, Inner Race, Outer Race
Severity: Mild, Moderate
Output: HDF5
Total signals: 400 (4 faults × 100 signals)
```

---

#### Tab 2: Import MAT Files

**What It Does**: Imports existing MATLAB vibration data files

**Usage**:

1. **Upload Files**
   - Drag and drop MAT files
   - Or click "Browse" to select
   - Multiple files supported

2. **Configure Import**
   - Dataset Name: e.g., "cwru_bearing_data"
   - Auto-normalize: Yes/No
   - Target length: 102400 samples (or custom)

3. **Validation Options**
   - Check for zeros: Yes
   - Check for NaNs: Yes
   - Minimum length: 10000 samples

4. **Output Format**
   - HDF5 (recommended for fast access)
   - MAT (copy original files)
   - Both

5. **Click "Import MAT Files"**

**Output**:
- Signals validated and processed
- Saved to `data/datasets/`
- Database record created
- Failed files reported

**Supported MAT Structures**:
```matlab
% Structure 1: Direct arrays
signal_data = [1x102400 double]
label = 'ball_fault'

% Structure 2: Nested structure
data.signal = [1x102400 double]
data.fault_type = 'inner_race'
data.severity = 'moderate'
```

---

### 📂 Data Explorer

**Route**: `/data-explorer`

**Description**: Browse and visualize datasets

**Features**:
- Dataset listing with metadata
- Signal count and distribution
- Fault type breakdown
- Dataset statistics

**Usage**:
1. View all datasets
2. Click dataset to see details
3. Explore signal distribution
4. View basic statistics

---

### 📈 Signal Viewer

**Route**: `/signal-viewer`

**Description**: Visualize individual signals

**Features**:
- Time-domain waveform
- Frequency spectrum (FFT)
- Signal statistics (RMS, kurtosis, peak)
- Signal metadata display

**Usage**:
1. Select dataset
2. Choose signal
3. View visualizations
4. Export plots (optional)

---

### 🧪 New Experiment

**Route**: `/experiment/new`

**Description**: Configure and launch training experiments

**Usage**:

1. **Experiment Configuration**
   - Name: e.g., "CNN_baseline_v1"
   - Description: Purpose and notes
   - Tags: For organization

2. **Dataset Selection**
   - Choose training dataset
   - Train/validation split: 70/30 or 80/20

3. **Model Selection** (20+ architectures)
   - **Classical ML**: SVM, Random Forest, GradientBoosting, MLP
   - **1D CNNs**: CNN1D, MultiScale CNN, Residual CNN
   - **Advanced CNNs**: ResNet, EfficientNet, MobileNet
   - **Transformers**: Attention-based models
   - **Time-Frequency**: Spectrogram CNN, CWT CNN
   - **PINN**: Physics-Informed Neural Networks
   - **Ensemble**: Voting, Stacking, Mixture of Experts

4. **Hyperparameters**
   - Learning rate: 1e-5 to 1e-2
   - Batch size: 16, 32, 64, 128
   - Epochs: 50-200
   - Optimizer: Adam, AdamW, SGD
   - Scheduler: ReduceLROnPlateau, CosineAnnealing

5. **Training Options**
   - Early stopping: Patience 10-20 epochs
   - Checkpointing: Save best model
   - Logging interval: Every N batches

6. **Click "Launch Experiment"**

**Output**:
- Experiment created in database
- Training starts in background (Celery)
- Redirected to training monitor

---

### 📊 Experiments

**Route**: `/experiments`

**Description**: List all experiments with filtering and search

**Features**:
- Experiment table with status
- Filter by status (Pending, Running, Completed, Failed)
- Search by name
- Sort by date, accuracy
- Quick actions (view, compare, delete)

**Status Indicators**:
- 🟡 **Pending**: Queued, not started
- 🔵 **Running**: Training in progress
- 🟢 **Completed**: Successfully finished
- 🔴 **Failed**: Error during training
- ⏸️ **Paused**: Manually paused
- ⛔ **Cancelled**: Manually stopped

---

### 📉 Training Monitor

**Route**: `/experiment/{id}/monitor`

**Description**: Real-time training progress monitoring

**Features**:
- **Loss Curves**: Training and validation loss over epochs
- **Accuracy Curves**: Training and validation accuracy
- **Learning Rate**: LR schedule visualization
- **Progress Bar**: Current epoch / total epochs
- **Time Estimates**: Remaining time and ETA
- **Resource Usage**: GPU memory, CPU usage

**Live Updates**: Auto-refreshes every 5 seconds

**Actions**:
- Pause training
- Resume training
- Stop training
- View logs

---

### 📋 Experiment Results

**Route**: `/experiment/{id}/results`

**Description**: Detailed results and metrics for completed experiments

**Features**:
- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Per-class metrics
  - Macro and weighted averages

- **Confusion Matrix**:
  - Interactive heatmap
  - Class-wise breakdown
  - Misclassification patterns

- **Training History**:
  - Loss and accuracy curves
  - Best epoch highlighting
  - Learning rate schedule

- **Model Information**:
  - Architecture details
  - Parameter count
  - Model size on disk
  - Training duration

- **Actions**:
  - Download model
  - Download results (JSON)
  - Compare with other experiments
  - Generate XAI explanation

---

### 🔍 Experiment Comparison

**Route**: `/compare?ids=1,2,3`

**Description**: Side-by-side comparison of multiple experiments

**Features**:
- Metrics comparison table
- Training curves overlay
- Confusion matrix comparison
- Parameter comparison
- Performance ranking

**Usage**:
1. From Experiments page, select experiments to compare
2. Click "Compare Selected"
3. View side-by-side comparison

---

### 🧠 XAI Dashboard

**Route**: `/xai`

**Description**: Generate and visualize model explanations

**Supported Methods**:
1. **SHAP** (SHapley Additive exPlanations)
2. **LIME** (Local Interpretable Model-agnostic Explanations)
3. **Integrated Gradients**
4. **Grad-CAM** (Gradient-weighted Class Activation Mapping)

#### SHAP Explanations

**What It Does**: Explains predictions using Shapley values from game theory

**Usage**:

1. **Select Model**
   - Choose completed experiment
   - Model accuracy shown

2. **Select Signal**
   - Choose signal from test set
   - Signal metadata displayed

3. **Select Method**: SHAP

4. **Configure Parameters**
   - SHAP Method: Gradient (fast), Deep (accurate), Kernel (slow)
   - Background Samples: 50-200 (default: 100)
   - Higher = more accurate but slower

5. **Click "Generate Explanation"**

**First Time**: 15-60 seconds (background task)
**Cached**: <1 second (instant retrieval)

**Output**:
- **Signal with Attribution**:
  - Original signal (blue line)
  - SHAP values overlaid (green = positive contribution, red = negative)
  - Dual-axis plot

- **Feature Importance Waterfall**:
  - Top 20 most important time steps
  - Contribution to prediction
  - Base value vs actual prediction

- **Explanation Details**:
  - SHAP method used
  - Base value (expected output)
  - Mean absolute SHAP value
  - Max positive/negative contributions

**Interpretation**:
- **Green regions**: Time steps that increased the predicted class probability
- **Red regions**: Time steps that decreased the predicted class probability
- **Height of bars**: Magnitude of contribution

**Example Use Case**:
```
Model predicts: Ball Fault (95% confidence)
SHAP shows: High-frequency impulses at regular intervals
Interpretation: Model correctly identifies ball defect signature
```

---

#### LIME Explanations

**What It Does**: Explains predictions by perturbing signal segments

**Usage**:

1. Select Model and Signal (same as SHAP)
2. Select Method: LIME
3. **Configure Parameters**:
   - Number of Segments: 20 (signal divided into 20 parts)
   - Perturbations: 1000 (number of random samples)
   - Higher = more accurate but slower

4. Click "Generate Explanation"

**Output**:
- **Signal with Segments**:
  - Colored segments showing importance
  - Green = positive contribution
  - Red = negative contribution
  - Alpha = magnitude of contribution

- **Segment Importance Bar Chart**:
  - Top 15 most important segments
  - Ranked by absolute importance

- **Explanation Details**:
  - Number of segments
  - Segment weights
  - Most positive/negative segments

**Interpretation**:
- **Positive weight**: Segment supports predicted class
- **Negative weight**: Segment opposes predicted class
- Identifies which parts of signal drive the prediction

---

#### Integrated Gradients

**What It Does**: Attribution method using gradient integration

**Usage**:
1. Select Model and Signal
2. Select Method: Integrated Gradients
3. **Configure Parameters**:
   - Integration Steps: 50 (more = smoother)

4. Click "Generate Explanation"

**Output**:
- Attribution plot showing which time steps are important
- Similar to SHAP but uses different algorithm
- Typically faster than SHAP

---

#### Grad-CAM

**What It Does**: Visualizes CNN layer activations (CNN models only)

**Usage**:
1. Select **CNN model** (e.g., ResNet, CNN1D)
2. Select Signal
3. Select Method: Grad-CAM
4. Click "Generate Explanation"

**Output**:
- Activation heatmap overlaid on signal
- Shows which regions CNN "looks at"
- Automatically uses last convolutional layer

**Note**: Only works with CNN-based models

---

### Cached Explanations

The XAI Dashboard caches all explanations to the database for instant retrieval.

**Benefits**:
- First generation: 15-60 seconds
- Cached retrieval: <1 second
- No need to regenerate

**Viewing Cached Explanations**:
- Scroll to "Cached Explanations" section
- Click "Load" to instantly view
- Shows recent 10 explanations

---

## Feature Status Matrix

| Feature | Status | Route | Files |
|---------|--------|-------|-------|
| **Data Management** | | | |
| Generate Synthetic Data | ✅ Complete | `/data-generation` | `layouts/data_generation.py`, `callbacks/data_generation_callbacks.py` |
| Import MAT Files | ✅ Complete | `/data-generation` | `callbacks/mat_import_callbacks.py` |
| Dataset Explorer | ✅ Complete | `/data-explorer` | `layouts/data_explorer.py` |
| Signal Viewer | ✅ Complete | `/signal-viewer` | `layouts/signal_viewer.py` |
| Dataset Management | ❌ Missing | `/datasets` | None (404) |
| **Training** | | | |
| Experiment Wizard | ✅ Complete | `/experiment/new` | `layouts/experiment_wizard.py` |
| Training Monitor | ✅ Complete | `/experiment/{id}/monitor` | `layouts/training_monitor.py` |
| Experiment Results | ✅ Complete | `/experiment/{id}/results` | `layouts/experiment_results.py` |
| Experiments List | ✅ Complete | `/experiments` | `layouts/experiments.py` |
| Experiment Comparison | ✅ Complete | `/compare` | `layouts/experiment_comparison.py` |
| HPO Campaigns | ⚠️ UI Only | `/hpo/campaigns` | `layouts/hpo_campaigns.py` (NO callbacks) |
| **Explainability** | | | |
| XAI Dashboard | ✅ Complete | `/xai` | `layouts/xai_dashboard.py`, `callbacks/xai_callbacks.py` |
| SHAP | ✅ Complete | `/xai` | `services/xai_service.py` |
| LIME | ✅ Complete | `/xai` | `services/xai_service.py` |
| Integrated Gradients | ✅ Complete | `/xai` | `services/xai_service.py` |
| Grad-CAM | ✅ Complete | `/xai` | `services/xai_service.py` |
| **Deployment** | | | |
| Model Quantization | ❌ Missing | N/A | Code: `deployment/quantization.py` |
| ONNX Export | ❌ Missing | N/A | Code: `deployment/onnx_export.py` |
| Model Optimization | ❌ Missing | N/A | Code: `deployment/model_optimization.py` |
| Deployment Dashboard | ❌ Missing | N/A | No UI |
| **Monitoring** | | | |
| System Health | ⚠️ Link Only | `/system-health` | No UI (service exists) |
| API Dashboard | ❌ Missing | N/A | No UI (API exists) |
| **Testing** | | | |
| Test Execution | ❌ Missing | N/A | No UI |
| Coverage Viewer | ❌ Missing | N/A | No UI |
| Benchmark Dashboard | ❌ Missing | N/A | No UI |
| **Analysis** | | | |
| ROC Curves | ❌ Missing | N/A | Code exists |
| Error Analysis | ❌ Missing | N/A | Code exists |
| Architecture Comparison | ❌ Missing | N/A | Code exists |
| Ensemble Evaluation | ❌ Missing | N/A | Code exists |
| **Other** | | | |
| Settings | ⚠️ Partial | `/settings` | Incomplete |
| Notification Management | ❌ Missing | N/A | Backend ready |
| User Management | ❌ Missing | N/A | No authentication |

**Legend**:
- ✅ Complete: Fully functional
- ⚠️ Partial: UI exists but limited functionality
- ❌ Missing: Not implemented in dashboard (but may exist in codebase)

---

## Missing Features

For a comprehensive analysis of missing features, see **[DASHBOARD_GAPS.md](DASHBOARD_GAPS.md)**.

### Critical Gaps (High Priority)

1. **HPO Campaigns** - UI exists but zero functionality
2. **Deployment Dashboard** - Model quantization, ONNX export, optimization
3. **System Monitoring** - Health metrics, alerts, resource usage
4. **API Monitoring** - API status, request logs, performance

### Important Gaps (Medium Priority)

5. **Enhanced Evaluation** - ROC curves, detailed error analysis
6. **Testing & QA Dashboard** - Test execution, coverage, benchmarks
7. **Dataset Management** - Dedicated datasets page
8. **Feature Engineering** - Feature extraction and selection UI

### Total Missing Features: ~40

**Dashboard Coverage**: ~50% of codebase capabilities

---

## Troubleshooting

### Issue: Dashboard Won't Start

**Error**: `ModuleNotFoundError: No module named 'dash'`

**Solution**:
```bash
pip install -r requirements.txt
```

---

### Issue: "Database Connection Error"

**Error**: `sqlalchemy.exc.OperationalError: could not connect to server`

**Solution**:
1. Ensure PostgreSQL is running
2. Check database configuration in `dash_app/config/.env`
3. Run migrations: `python database/run_migration.py`

---

### Issue: "Celery Worker Not Running"

**Error**: Celery tasks stuck in "Pending" status

**Solution**:
```bash
# Start Celery worker
cd dash_app
celery -A tasks worker --loglevel=info
```

---

### Issue: XAI Explanation Fails

**Error**: "SHAP library not installed"

**Solution**:
```bash
pip install shap captum
```

---

### Issue: Slow XAI Generation

**Symptom**: SHAP takes >2 minutes

**Solutions**:
1. Use "Gradient" method instead of "Deep" or "Kernel"
2. Reduce background samples (try 50 instead of 100)
3. Use GPU if available
4. Check if explanation is cached (should be <1 second on second attempt)

---

### Issue: 404 on Dashboard Pages

**Routes That Currently 404**:
- `/datasets` - Not implemented yet
- `/statistics/compare` - Not implemented
- `/analytics` - Not implemented
- `/system-health` - Not implemented
- `/hpo/campaigns` - UI exists but no functionality

**Solution**: These features need to be implemented (see [DASHBOARD_GAPS.md](DASHBOARD_GAPS.md))

---

## Next Steps

After using the dashboard, you can:

1. **Implement Missing Features**
   - See [DASHBOARD_GAPS.md](DASHBOARD_GAPS.md) for priority order
   - Start with HPO Campaigns (most impactful)

2. **Deploy Models**
   - Use command-line tools in `deployment/`
   - Export to ONNX: `python scripts/export_onnx.py`
   - Quantize models: `python scripts/quantize_model.py`

3. **Start API Server**
   ```bash
   cd api
   uvicorn main:app --reload
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v --cov=.
   ```

5. **Create Custom Visualizations**
   - Use `visualization/` utilities
   - Extend dashboard layouts

---

## Support

- **Documentation**: See `/docs/` directory
- **API Reference**: See `/api/README.md`
- **Codebase Issues**: Check `/docs/DASHBOARD_GAPS.md`
- **Questions**: Review Phase-specific usage guides in `/USAGE_GUIDES/`

---

**Last Updated**: November 22, 2024
**Dashboard Version**: Phase 11C (Post-XAI Integration)
**Feature Coverage**: ~50% of codebase capabilities
