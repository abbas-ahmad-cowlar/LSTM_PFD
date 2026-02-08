# Phase 11: Enterprise Plotly Dash Application - Complete Guide

**Enterprise-grade web dashboard for the LSTM PFD bearing fault diagnosis system.**

---

## Table of Contents

- [Overview](#overview)
- [Phase 11A: Foundation & Data Exploration](#phase-11a-foundation--data-exploration)
- [Phase 11B: ML Pipeline Orchestration](#phase-11b-ml-pipeline-orchestration)
- [Phase 11C: Advanced Analytics & XAI](#phase-11c-advanced-analytics--xai)
- [Phase 11D: Production Hardening](#phase-11d-production-hardening)
- [Deployment](#deployment)
- [Architecture](#architecture)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## Overview

Phase 11 delivers a **production-ready enterprise dashboard** that integrates all previous phases (0-10) into a unified web application. Built with Plotly Dash, it provides:

- **Interactive experiment management** with real-time training monitoring
- **Explainable AI dashboard** with SHAP, LIME, and Grad-CAM
- **Hyperparameter optimization campaigns**
- **Statistical model comparison**
- **User authentication** and role-based access
- **Production monitoring** and alerting
- **90%+ test coverage** with comprehensive integration tests

### Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                    Dash Frontend (UI)                        │
│  - Multi-page navigation - Real-time updates - Dashboards   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  Flask Backend (Server)                      │
│  - REST API - Authentication - Rate limiting - Security      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────┬──────────────────┬──────────────┬────────────┐
│  PostgreSQL  │  Redis (Cache)   │    Celery    │  Phase 0-10│
│  (Metadata)  │  (Performance)   │  (Training)  │(Integration)│
└──────────────┴──────────────────┴──────────────┴────────────┘
```

---

## Phase 11A: Foundation & Data Exploration

### What's Included

✅ **PostgreSQL database** for experiment metadata
✅ **Redis caching** for performance optimization
✅ **Celery task queue** for background jobs
✅ **5 core pages**: Home, Data Explorer, Signal Viewer, Dataset Manager, System Health

### Quick Start

#### Option 1: Docker Compose (Recommended)

```bash
cd dash_app
cp .env.example .env
docker-compose up
```

Access at: `http://localhost:8050`

#### Option 2: Local Development

```bash
# 1. Install dependencies
cd dash_app
pip install -r requirements.txt

# 2. Start PostgreSQL and Redis (install separately)

# 3. Set environment variables
cp .env.example .env
source .env

# 4. Initialize database
python -c "from database.connection import init_database; from database.seed_data import seed_initial_data; init_database(); seed_initial_data()"

# 5. Start Celery worker (in separate terminal)
celery -A tasks.celery_app worker --loglevel=info

# 6. Start Dash app
python app.py
```

### Features

#### 1. Home Dashboard

The home page provides an overview of the entire system:

- **Quick Stats**: Total signals, fault classes, best model accuracy, experiment count
- **Quick Actions**: Navigate to key pages
- **Recent Experiments**: View latest training runs
- **System Health**: Real-time monitoring gauges
- **Dataset Distribution**: Visualization of class balance

#### 2. Data Explorer

Explore and filter the bearing fault dataset:

- **Dataset Selection**: Choose from available datasets
- **Signal Filtering**: Filter by fault type, severity, operating conditions
- **Statistical Summary**: Mean, std dev, min, max per class
- **t-SNE Visualization**: 2D projection of signal embeddings
- **Export Options**: Download filtered subsets

#### 3. Signal Viewer

Detailed analysis of individual signals:

- **Time Domain**: Plot raw signal waveform
- **Frequency Domain**: FFT magnitude spectrum
- **Spectrogram**: Time-frequency representation (STFT, CWT, WVD)
- **Feature Statistics**: Computed time and frequency domain features
- **Zoom & Pan**: Interactive Plotly plots

#### 4. Dataset Manager

Manage datasets and create new ones:

- **Upload New Data**: Import MAT files or HDF5 caches
- **Create Synthetic Datasets**: Use Phase 0 signal generation
- **Dataset Versioning**: Track multiple dataset versions
- **Metadata Editing**: Update descriptions, tags, notes

#### 5. System Health Monitor

Real-time system monitoring:

- **Resource Usage**: CPU, memory, disk utilization
- **Database Status**: Connection count, query performance
- **Redis Status**: Cache hit rate, memory usage
- **Active Tasks**: Running Celery training jobs

---

## Phase 11B: ML Pipeline Orchestration

### What's Included

✅ **Training experiment configuration wizard**
✅ **Real-time training progress monitoring**
✅ **Comprehensive results visualization**
✅ **Experiment history and comparison**
✅ **Integration with Phases 1-8 training code**

### Training Workflow

#### Step 1: Configure Experiment

Navigate to `/experiment/new` or click "New Experiment" from the Experiments page.

**Wizard Steps:**

1. **Model Selection**: Choose from 20+ architectures (Classical ML, CNNs, Transformers, PINN, Ensemble)
2. **Dataset & Hyperparameters**: Select dataset, configure model-specific hyperparameters
3. **Training Options**: Set epochs, batch size, optimizer, scheduler, augmentation
4. **Review & Launch**: Verify configuration, name experiment, add tags/notes

**Example Configuration:**

```python
{
    "model_type": "resnet34",
    "dataset_id": 1,
    "hyperparameters": {"dropout": 0.3},
    "num_epochs": 150,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "scheduler": "plateau",
    "early_stopping_patience": 15,
    "augmentation": ["noise", "time_shift"]
}
```

#### Step 2: Monitor Training

After launching, you're redirected to `/experiment/<id>/monitor` for real-time monitoring:

- **Progress Bars**: Current epoch and overall progress with ETA
- **Current Metrics**: Train/val loss and accuracy (updates every 2 seconds)
- **Training Curves**: Loss and accuracy plots (live updates)
- **Learning Rate Schedule**: LR evolution over epochs
- **Gradient Norms**: Monitor gradient flow (if enabled)
- **Training Logs**: Scrollable console output

**Features:**

- ⏸️ **Pause/Resume**: Pause training and resume later
- ⏹️ **Stop**: Gracefully stop training
- 📊 **Auto-refresh**: Metrics update every 2 seconds without page reload
- 💾 **Persistent**: Close browser, training continues in background

#### Step 3: View Results

Navigate to `/experiment/<id>/results` after completion:

- **Key Metrics Cards**: Test accuracy, test loss, training time, best epoch
- **Training History**: Loss and accuracy curves
- **Confusion Matrix**: Heatmap with per-class performance
- **Per-Class Metrics**: Precision, recall, F1-score table
- **Model Configuration**: Full config JSON
- **Hyperparameters Table**: All training hyperparameters

**Export Options:**

- **PDF Report**: Complete training report with all plots
- **Model Weights**: Download `.pth` or `.onnx` files
- **Training Logs**: Export full logs as `.txt`

### Experiment History & Comparison

Navigate to `/experiments` to view all experiments:

**Features:**

- **Search**: Filter by name, tags, notes
- **Multi-filter**: Combine model type and status filters
- **Sortable Table**: Sort by accuracy, duration, created date
- **Multi-select**: Select 2-5 experiments for comparison
- **Status Indicators**: Color-coded rows (green=completed, red=failed, blue=running)
- **Pagination**: Handle 1000s of experiments

**Comparison:**

Select multiple experiments and click "Compare Models":

- **Side-by-side Table**: Compare all metrics
- **Accuracy Chart**: Bar chart comparison
- **Training Time Chart**: Compare training efficiency
- **Statistical Tests**: T-tests, ANOVA for significance
- **Per-class Comparison**: Radar chart of F1 scores

---

## Phase 11C: Advanced Analytics & XAI

### What's Included

✅ **SHAP explanations** for feature attribution
✅ **LIME explanations** for local interpretability
✅ **Integrated Gradients** for neural network attribution
✅ **Grad-CAM** for CNN activation visualization
✅ **Hyperparameter Optimization campaigns**
✅ **Statistical model comparison tools**

### Explainable AI Dashboard

Navigate to `/xai` for interactive explanations:

#### 1. SHAP Explanations

**What is SHAP?**
SHapley Additive exPlanations compute feature importance using game theory.

**Usage:**

1. Select trained model from dropdown
2. Select signal to explain
3. Choose "SHAP" as explanation method
4. Configure background samples (default: 100)
5. Click "Generate Explanation"

**Outputs:**

- **Signal with Attribution**: Color-coded importance overlay
- **Feature Importance Bar Chart**: Top 20 most important time points
- **Summary Statistics**: Mean |SHAP value|, max contribution, etc.

**Example:**

```python
# SHAP identifies that time points 45000-47000 are most important
# for classifying "ball_fault" - corresponds to high-frequency impacts
```

#### 2. LIME Explanations

**What is LIME?**
Local Interpretable Model-agnostic Explanations perturb input to identify important features.

**Usage:**

1. Select model and signal
2. Choose "LIME" as method
3. Set number of features to show (default: 20)
4. Set perturbations (default: 1000)
5. Generate

**Outputs:**

- **Feature Importance Table**: Ranked by absolute weight
- **Decision Boundary**: Local linear approximation
- **Confidence**: LIME model R² score

#### 3. Integrated Gradients

**What is IG?**
Computes gradients from baseline to input, attributing each feature's contribution.

**Usage:**

1. Select model (must be PyTorch model)
2. Select signal
3. Choose "Integrated Gradients"
4. Set integration steps (default: 50)
5. Generate

**Outputs:**

- **Attribution Map**: Gradient-based importance
- **Target Class**: Which class was predicted
- **Baseline**: Zero signal used as reference

#### 4. Grad-CAM

**What is Grad-CAM?**
Gradient-weighted Class Activation Mapping visualizes what CNN layers focus on.

**Usage:**

1. Select CNN model (ResNet, EfficientNet, etc.)
2. Select signal
3. Choose "Grad-CAM"
4. Optionally specify target layer (default: last conv layer)
5. Generate

**Outputs:**

- **Activation Heatmap**: Overlaid on signal
- **Layer Information**: Which layer was used
- **Class Activation**: Contribution to predicted class

### Hyperparameter Optimization

Navigate to `/hpo` to create optimization campaigns:

**Supported Methods:**

- **Bayesian Optimization** (recommended): Uses Gaussian Processes for efficient search
- **Random Search**: Random sampling from search space
- **Grid Search**: Exhaustive search over discrete grid
- **Hyperband**: Adaptive resource allocation

**Workflow:**

1. Click "New Campaign"
2. Name campaign (e.g., "resnet_optimization_v1")
3. Select model type
4. Choose optimization method (Bayesian recommended)
5. Set number of trials (e.g., 50)
6. Define search space:
   - Learning rate: [1e-5, 1e-2] (log scale)
   - Dropout: [0.0, 0.5]
   - Batch size: [16, 32, 64, 128]
7. Select metric to optimize (e.g., val_accuracy)
8. Launch campaign

**Monitoring:**

- **Progress Bar**: Completed trials / total trials
- **Best Score**: Current best validation accuracy
- **Trial History**: Table of all trials with hyperparameters
- **Optimization Curve**: Best score vs. trial number
- **Parameter Importance**: Which hyperparameters matter most

**Results:**

- **Best Configuration**: Hyperparameters that achieved highest score
- **Parallel Coordinates Plot**: Visualize hyperparameter interactions
- **Export**: Download results as CSV or JSON

---

## Phase 11D: Production Hardening

### What's Included

✅ **JWT-based authentication**
✅ **Rate limiting** (60 requests/minute per IP)
✅ **Security headers** (XSS, CSP, HSTS)
✅ **Production monitoring** (CPU, memory, disk, alerts)
✅ **90%+ test coverage**
✅ **CI/CD integration**

### Authentication

#### Creating Users

**Admin creates users:**

```python
from middleware.auth import AuthMiddleware

success, user_id, error = AuthMiddleware.create_user(
    username="john_doe",
    email="john@company.com",
    password="secure_password_123",
    role="user"  # or "admin"
)
```

#### Login Workflow

**Frontend sends credentials:**

```bash
curl -X POST http://localhost:8050/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "secure_password_123"
  }'
```

**Backend returns JWT:**

```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 86400
}
```

**Subsequent requests include token:**

```bash
curl -X GET http://localhost:8050/api/experiments \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

#### Token Expiry

- **Default**: 24 hours
- **Refresh**: Re-login to get new token
- **Revocation**: Logout invalidates token (requires Redis blacklist in production)

### Rate Limiting

**Default limits:**

- **60 requests/minute** per IP address
- **Applies to**: All `/api/*` endpoints
- **Response on exceed**: `HTTP 429 Too Many Requests`

**Example response when rate limit exceeded:**

```json
{
  "error": "Rate limit exceeded",
  "limit": 60,
  "period": "1 minute",
  "retry_after": 42
}
```

**Customizing limits:**

```python
from middleware.security import rate_limiter

@app.route('/api/heavy_endpoint')
@rate_limiter.limit(requests_per_minute=10)
def heavy_endpoint():
    # This endpoint allows only 10 requests/minute
    return {"data": "..."}
```

### Security Features

#### 1. Security Headers

Automatically added to all responses:

- **X-Content-Type-Options**: `nosniff` (prevent MIME sniffing)
- **X-Frame-Options**: `DENY` (prevent clickjacking)
- **X-XSS-Protection**: `1; mode=block` (XSS protection)
- **Strict-Transport-Security**: `max-age=31536000; includeSubDomains` (force HTTPS)
- **Content-Security-Policy**: Whitelist allowed sources

#### 2. Input Sanitization

All user inputs are sanitized:

```python
from middleware.security import SecurityMiddleware

sanitized = SecurityMiddleware.sanitize_input(user_input, max_length=1000)
# Removes: < > " ' & ; | $ `
```

#### 3. File Upload Validation

```python
from middleware.security import SecurityMiddleware

is_safe = SecurityMiddleware.validate_file_upload(
    filename="dataset.h5",
    allowed_extensions={'h5', 'hdf5', 'mat', 'csv'}
)
# Checks extension, prevents path traversal
```

### Production Monitoring

#### Starting the Monitor

```python
from services.monitoring_service import monitoring_service

# Start monitoring (runs in background thread)
monitoring_service.start_monitoring(interval_seconds=60)
```

#### Metrics Collected

**System Metrics:**

- CPU usage (%)
- Memory usage (%, GB)
- Disk usage (%, GB)

**Application Metrics:**

- Total experiments
- Running experiments
- Completed experiments
- Failed experiments

#### Alerts

**Alert thresholds:**

- CPU > 90% → `HIGH_CPU` alert
- Memory > 85% → `HIGH_MEMORY` alert
- Disk > 90% → `HIGH_DISK` alert
- Failed experiments > 10 → `HIGH_FAILURES` alert

**Alert deduplication:** Alerts are not repeated within 5 minutes

**Viewing alerts:**

```python
alerts = monitoring_service.get_recent_alerts(hours=24)
# Returns list of {type, message, timestamp, severity}
```

#### Health Check Endpoint

```bash
curl http://localhost:8050/api/health
```

**Response:**

```json
{
  "status": "healthy",
  "message": "All systems operational",
  "metrics": {
    "system": {
      "cpu_percent": 35.2,
      "memory_percent": 62.1,
      "disk_percent": 45.8
    },
    "application": {
      "total_experiments": 47,
      "running_experiments": 2,
      "completed_experiments": 42,
      "failed_experiments": 3
    }
  },
  "alerts_count": 0
}
```

### Integration Tests

**Running tests:**

```bash
cd dash_app
pytest tests/test_integration.py -v
```

**Test coverage:**

```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
open htmlcov/index.html
```

**Tests included:**

- ✅ User creation and authentication (8 tests)
- ✅ Rate limiting (4 tests)
- ✅ Database models (10 tests)
- ✅ Experiment workflows (6 tests)
- ✅ Monitoring service (5 tests)
- ✅ Security middleware (7 tests)

---

## Deployment

### Docker Production Deployment

**1. Build image:**

```bash
cd dash_app
docker build -t lstm_pfd_dashboard:latest .
```

**2. Deploy with docker-compose:**

```bash
docker-compose -f docker-compose.prod.yml up -d
```

**Services:**

- `dash_app`: Main application (port 8050)
- `postgres`: PostgreSQL database (port 5432)
- `redis`: Redis cache (port 6379)
- `celery_worker`: Background training jobs
- `nginx`: Reverse proxy with HTTPS (port 443)

### Environment Variables

**Required:**

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: Flask secret key
- `JWT_SECRET_KEY`: JWT signing key

**Optional:**

- `DEBUG`: Enable debug mode (default: False)
- `APP_HOST`: Host to bind (default: 0.0.0.0)
- `APP_PORT`: Port to bind (default: 8050)

**Example `.env`:**

```bash
DATABASE_URL=postgresql://user:password@postgres:5432/lstm_pfd
REDIS_URL=redis://redis:6379/0
SECRET_KEY=change-this-to-random-secret-key-in-production
JWT_SECRET_KEY=change-this-to-another-random-key
DEBUG=False
```

### Scaling

**Horizontal scaling:**

```bash
docker-compose up --scale celery_worker=4
```

**Load balancer (nginx):**

```nginx
upstream dash_app {
    server dash_app_1:8050;
    server dash_app_2:8050;
    server dash_app_3:8050;
}
```

---

## Architecture

### Database Schema

**Key tables:**

- `users`: User accounts and roles
- `datasets`: Dataset metadata
- `signals`: Individual signal records
- `experiments`: Training experiments
- `training_runs`: Per-epoch metrics
- `hpo_campaigns`: Hyperparameter optimization campaigns
- `explanations`: Cached XAI explanations
- `system_logs`: Application logs and alerts

### Caching Strategy

**Redis caches:**

- **Expensive computations**: t-SNE embeddings (TTL: 1 hour)
- **Spectrograms**: STFT/CWT/WVD (TTL: 6 hours)
- **SHAP explanations**: Cached per (model_id, signal_id) (TTL: 24 hours)
- **Session data**: User preferences, filter selections (TTL: 1 hour)

### Async Tasks (Celery)

**Task types:**

1. `train_model_task`: Training experiments (15-45 min)
2. `hpo_trial_task`: Single HPO trial (10-30 min)
3. `generate_xai_task`: SHAP/LIME computation (2-5 min)
4. `export_report_task`: PDF report generation (30-60 sec)

**Task monitoring:**

```bash
celery -A tasks.celery_app inspect active
celery -A tasks.celery_app inspect stats
```

---

## Troubleshooting

### Database Connection Errors

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Test connection
psql -h localhost -U lstm_user -d lstm_pfd
```

### Redis Connection Errors

```bash
# Test Redis connection
docker-compose exec redis redis-cli ping
# Should return: PONG

# Check Redis memory
docker-compose exec redis redis-cli info memory
```

### Celery Tasks Not Running

```bash
# Check worker is running
docker-compose logs celery_worker

# Restart worker
docker-compose restart celery_worker

# Purge stuck tasks
celery -A tasks.celery_app purge
```

### Page Not Loading

1. Check browser console for errors
2. Check `app.log` for backend errors
3. Verify all callbacks are registered in `callbacks/__init__.py`
4. Clear browser cache and cookies

---

## Performance Targets

- **Page load**: < 2 seconds
- **Filter response**: < 500ms
- **Signal load**: < 1 second
- **Training task spawn**: < 1 second
- **XAI generation**: < 30 seconds (SHAP)

---

## Summary

Phase 11 delivers a **professional, production-ready dashboard** that:

✅ Integrates all 10 previous phases
✅ Provides intuitive UI for ML experimentation
✅ Enables real-time training monitoring
✅ Offers explainable AI capabilities
✅ Includes robust authentication and security
✅ Achieves 90%+ test coverage
✅ Scales horizontally for enterprise deployment

**Next Steps:**

1. Deploy to staging environment
2. Conduct user acceptance testing
3. Security audit
4. Performance profiling and optimization
5. Production deployment

For issues, see the [GitHub Issues page](https://github.com/abbas-ahmad-cowlar/LSTM_PFD/issues).
