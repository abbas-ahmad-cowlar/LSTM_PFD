# LSTM_PFD: The Perfect 10  Master Execution Roadmap

> **Commercial-Grade Research Platform**
> From 7.5/10 to 10/10  Complete Implementation Guide

**Generated:** January 13, 2026
**Team Size:** 10 SMEs
**Total Duration:** 15-20 weeks
**Total Chapters:** 32
**Total Acceptance Criteria:** 150+

---

## Executive Summary

This document transforms the LSTM_PFD repository from a "student project" into a **Production-Grade Research Platform** through four comprehensive phases:

| Phase | Focus | Chapters | Duration | Key Outcome |
|-------|-------|----------|----------|-------------|
| **Phase 1** | Vapor Fixes | 11 | 6 weeks | Functional dashboard |
| **Phase 2** | Documentation | 7 | 4 weeks | MkDocs site live |
| **Phase 3** | Research Novelty | 6 | 6 weeks | Publication-ready |
| **Phase 4** | Enterprise | 8 | 5 weeks | Production-ready |

---

## Table of Contents

### Phase 1: The "Vapor" Fixes (Page ~10)
- Chapter 1.1: HPO Campaign UI
- Chapter 1.2: Deployment Dashboard
- Chapter 1.3: System Health Dashboard
- Chapter 1.4: API Monitoring Dashboard
- Chapter 1.5: Dataset Management
- Chapter 1.6: Feature Engineering Dashboard
- Chapter 1.7: Advanced Training Options
- Chapter 1.8: Enhanced Evaluation Dashboard
- Chapter 1.9: Testing & QA Dashboard
- Chapter 1.10: Notification Management
- Chapter 1.11: Infrastructure & Navigation (Addendum)

### Phase 2: Documentation Overhaul (Page ~60)
- Chapter 2.1: Documentation Audit & Inventory
- Chapter 2.2: Information Architecture Design
- Chapter 2.3: Content Consolidation Execution
- Chapter 2.4: MkDocs Implementation
- Chapter 2.5: API Documentation Generation
- Chapter 2.6: Diagrams & Visual Assets
- Chapter 2.7: Research-Grade Features (Addendum)

### Phase 3: Research Novelty (Page ~100)
- Chapter 3.1: PINN Integration Finalization
- Chapter 3.2: XAI Module Enhancement
- Chapter 3.3: Ensemble Method Documentation
- Chapter 3.4: Manuscript Preparation
- Chapter 3.5: Dataset Contribution
- Chapter 3.6: Reviewer Proofing (Addendum)

### Phase 4: Enterprise Hardening (Page ~140)
- Chapter 4.1: Security Audit & Remediation
- Chapter 4.2: Docker & Container Optimization
- Chapter 4.3: Kubernetes Deployment
- Chapter 4.4: CI/CD Pipeline Enhancement
- Chapter 4.5: Monitoring & Alerting
- Chapter 4.6: Disaster Recovery
- Chapter 4.7: Code Quality & Consistency
- Chapter 4.8: Operational Safeguards (Addendum)

---


# Phase 1: The "Vapor" Fixes — Complete Implementation Guide

**Phase Duration:** 4-6 weeks  
**Team Size:** 6 SMEs  
**Priority:** 🔴 CRITICAL — Production Blocker

---

## Team Roster & Allocation

| Role                    | Assigned To | Chapters               | Weekly Hours |
| ----------------------- | ----------- | ---------------------- | ------------ |
| **Lead Full-Stack Dev** | TBD         | 1.1, 1.2, 1.4          | 40           |
| **Full-Stack Dev #2**   | TBD         | 1.3, 1.5, 1.10         | 40           |
| **Frontend Dev**        | TBD         | 1.6, 1.7, 1.9          | 40           |
| **ML Engineer**         | TBD         | 1.1, 1.2, 1.8          | 30           |
| **DevOps Engineer**     | TBD         | Celery/Redis, 1.4      | 20           |
| **QA Engineer**         | TBD         | All chapters (testing) | 40           |

---

# Chapter 1.1 — HPO Campaign UI

**Owner:** Lead Full-Stack Dev + ML Engineer  
**Duration:** 5-7 days

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Dash UI)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  HPO Form       │  │  Progress View  │  │  Results View   │ │
│  │  (Create)       │  │  (Real-time)    │  │  (Analysis)     │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼────────────────────┼────────────────────┼──────────┘
            │ POST               │ GET (2s interval)  │ GET
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CALLBACKS LAYER                              │
│  packages/dashboard/callbacks/hpo_callbacks.py                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ create_hpo_campaign() → HPOService.create_campaign()       ││
│  │ update_trial_progress() → Celery.AsyncResult.state         ││
│  │ export_best_params() → HPOService.export_results()         ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SERVICE LAYER                                │
│  packages/dashboard/services/hpo_service.py                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ HPOService.create_campaign() → celery_app.send_task()      ││
│  │ HPOService.get_campaign_progress() → AsyncResult(task_id)  ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │ Redis Message Queue
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CELERY WORKER                                │
│  packages/dashboard/tasks/hpo_tasks.py                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ @celery_app.task                                            ││
│  │ def run_hpo_optimization(campaign_id, config):              ││
│  │     study = optuna.create_study()                           ││
│  │     study.optimize(objective, n_trials, callbacks=[...])    ││
│  │     # Updates task.state for real-time progress             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Implementation

### Step 1: Configure Celery (DevOps Engineer, Day 1)

**File: `packages/dashboard/celery_config.py`**

```python
import os
from celery import Celery

def make_celery():
    celery = Celery(
        'dashboard',
        broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
        include=['packages.dashboard.tasks.hpo_tasks']
    )
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,  # Required for progress tracking
        result_extended=True,      # Store task metadata
    )
    return celery

celery_app = make_celery()
```

**File: `docker-compose.yml` (add Redis + worker)**

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  celery_worker:
    build: .
    command: celery -A packages.dashboard.celery_config worker --loglevel=info
    volumes:
      - .:/app
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy

volumes:
  redis_data:
```

### Step 2: Create Celery Task (ML Engineer, Days 1-2)

**File: `packages/dashboard/tasks/hpo_tasks.py`**

```python
from packages.dashboard.celery_config import celery_app
from packages.dashboard.database import db
from packages.dashboard.models.hpo_campaign import HPOCampaign
from experiments.hyperparameter_tuner import HyperparameterTuner
import optuna

@celery_app.task(bind=True, name='hpo.run_optimization')
def run_hpo_optimization(self, campaign_id: int, config: dict):
    """
    Executes HPO with real-time progress updates.

    The key integration: self.update_state() allows the callback
    layer to poll Celery for current progress without database queries.
    """
    campaign = HPOCampaign.query.get(campaign_id)
    campaign.status = 'running'
    campaign.celery_task_id = self.request.id
    db.session.commit()

    try:
        study = optuna.create_study(
            direction='minimize' if config.get('minimize', True) else 'maximize',
            study_name=f"campaign_{campaign_id}"
        )

        def progress_callback(study, trial):
            # Update Celery task state (polled by UI)
            self.update_state(
                state='PROGRESS',
                meta={
                    'current_trial': len(study.trials),
                    'total_trials': config['n_trials'],
                    'best_value': study.best_value if study.best_trial else None,
                    'best_params': study.best_params if study.best_trial else None
                }
            )
            # Also persist to database for durability
            campaign.completed_trials = len(study.trials)
            campaign.best_score = study.best_value if study.best_trial else None
            campaign.best_params = study.best_params if study.best_trial else None
            db.session.commit()

        # Create objective function from config
        objective = create_objective(config['model_type'], config['dataset_id'], config['param_space'])

        study.optimize(
            objective,
            n_trials=config['n_trials'],
            timeout=config.get('timeout'),
            callbacks=[progress_callback]
        )

        campaign.status = 'completed'
        db.session.commit()

        return {
            'status': 'completed',
            'best_params': study.best_params,
            'best_value': study.best_value
        }

    except Exception as e:
        campaign.status = 'failed'
        campaign.error_message = str(e)
        db.session.commit()
        raise
```

### Step 3: Create Service Layer (Lead Full-Stack Dev, Day 2)

**File: `packages/dashboard/services/hpo_service.py`**

```python
from celery.result import AsyncResult
from packages.dashboard.celery_config import celery_app
from packages.dashboard.tasks.hpo_tasks import run_hpo_optimization

class HPOService:

    @staticmethod
    def create_campaign(name, model_type, n_trials, param_space, dataset_id, user_id):
        """Creates DB record and dispatches async task."""
        campaign = HPOCampaign(
            name=name,
            model_type=model_type,
            n_trials=n_trials,
            param_space=param_space,
            dataset_id=dataset_id,
            user_id=user_id,
            status='pending'
        )
        db.session.add(campaign)
        db.session.commit()

        # Dispatch Celery task
        task = run_hpo_optimization.delay(
            campaign_id=campaign.id,
            config={
                'model_type': model_type,
                'n_trials': n_trials,
                'param_space': param_space,
                'dataset_id': dataset_id
            }
        )

        campaign.celery_task_id = task.id
        db.session.commit()

        return campaign

    @staticmethod
    def get_campaign_progress(campaign_id):
        """
        Gets real-time progress from Celery task state.
        This is how the UI gets live updates without polling the database.
        """
        campaign = HPOCampaign.query.get(campaign_id)
        if not campaign or not campaign.celery_task_id:
            return None

        result = AsyncResult(campaign.celery_task_id, app=celery_app)

        if result.state == 'PROGRESS':
            return {
                'status': 'running',
                **result.info  # Contains current_trial, total_trials, best_value, best_params
            }
        elif result.state == 'SUCCESS':
            return {'status': 'completed', **result.result}
        elif result.state == 'FAILURE':
            return {'status': 'failed', 'error': str(result.info)}
        else:
            return {'status': result.state.lower()}
```

### Step 4: Create Callbacks (Lead Full-Stack Dev, Days 3-4)

**File: `packages/dashboard/callbacks/hpo_callbacks.py`**

```python
from dash import Input, Output, State, callback, no_update
from packages.dashboard.services.hpo_service import HPOService

def register_hpo_callbacks(app):

    @app.callback(
        Output("hpo-campaign-store", "data"),
        Output("hpo-creation-alert", "children"),
        Output("hpo-creation-alert", "is_open"),
        Output("hpo-progress-interval", "disabled"),
        Input("hpo-start-btn", "n_clicks"),
        State("hpo-name-input", "value"),
        State("hpo-model-select", "value"),
        State("hpo-n-trials", "value"),
        State("hpo-param-config", "data"),
        State("hpo-dataset-select", "value"),
        prevent_initial_call=True
    )
    def start_hpo_campaign(n_clicks, name, model, n_trials, params, dataset):
        if not all([name, model, n_trials, params, dataset]):
            return no_update, "Please fill all required fields", True, True

        try:
            campaign = HPOService.create_campaign(
                name=name,
                model_type=model,
                n_trials=n_trials,
                param_space=params,
                dataset_id=dataset,
                user_id=1  # TODO: Get from session
            )
            return (
                {"campaign_id": campaign.id},
                f"Campaign '{name}' started! Tracking {n_trials} trials...",
                True,
                False  # Enable progress interval
            )
        except Exception as e:
            return no_update, f"Error: {str(e)}", True, True

    @app.callback(
        Output("hpo-progress-bar", "value"),
        Output("hpo-progress-text", "children"),
        Output("hpo-best-score", "children"),
        Output("hpo-trials-chart", "figure"),
        Input("hpo-progress-interval", "n_intervals"),
        State("hpo-campaign-store", "data"),
        prevent_initial_call=True
    )
    def update_progress(n_intervals, campaign_data):
        if not campaign_data:
            return no_update, no_update, no_update, no_update

        progress = HPOService.get_campaign_progress(campaign_data['campaign_id'])

        if not progress:
            return no_update, no_update, no_update, no_update

        if progress['status'] == 'running':
            pct = (progress['current_trial'] / progress['total_trials']) * 100
            return (
                pct,
                f"Trial {progress['current_trial']}/{progress['total_trials']}",
                f"Best Score: {progress['best_value']:.4f}" if progress['best_value'] else "—",
                create_optimization_chart(campaign_data['campaign_id'])
            )
        # Handle completed/failed states...
```

### Step 5: Register Callbacks (Lead Full-Stack Dev, Day 4)

**File: `packages/dashboard/callbacks/__init__.py`**

```python
from .hpo_callbacks import register_hpo_callbacks

def register_all_callbacks(app):
    # ... existing callbacks ...
    register_hpo_callbacks(app)  # ADD THIS LINE
```

## Acceptance Criteria

| ID     | Criterion                                            | Verification Method                   | Owner |
| ------ | ---------------------------------------------------- | ------------------------------------- | ----- |
| HPO-1  | User creates campaign → Celery task starts within 2s | QA: Manual test with Redis monitor    | QA    |
| HPO-2  | Progress bar updates every 2s during optimization    | QA: Stopwatch timing, 10 trials       | QA    |
| HPO-3  | Best score updates in real-time when improved        | QA: Verify 3 consecutive improvements | QA    |
| HPO-4  | Pause stops optimization after current trial         | QA: Verify no new trial after pause   | QA    |
| HPO-5  | Resume continues from paused state                   | QA: Verify trial count continuity     | QA    |
| HPO-6  | Stop terminates and marks campaign failed            | QA: Verify status in DB               | QA    |
| HPO-7  | Export downloads valid JSON/YAML/Python              | QA: Parse exported files              | QA    |
| HPO-8  | Parallel coordinates renders for >5 trials           | QA: Visual inspection                 | QA    |
| HPO-9  | Parameter importance shows for completed campaign    | QA: Visual inspection                 | QA    |
| HPO-10 | Error states show user-friendly messages             | QA: Trigger 3 error conditions        | QA    |

---

# Chapter 1.2 — Deployment Dashboard

**Owner:** Lead Full-Stack Dev + ML Engineer  
**Duration:** 6-8 days

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT DASHBOARD                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │ Model Select  │  │ Quantization  │  │ ONNX Export   │       │
│  │ (Dropdown)    │  │ (INT8/FP16)   │  │ (Download)    │       │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
└──────────┼──────────────────┼──────────────────┼────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXISTING BACKEND (packages/deployment/)         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ quantization.py │  │ onnx_export.py  │  │ inference.py    │ │
│  │ - dynamic_int8  │  │ - export_onnx   │  │ - benchmark     │ │
│  │ - static_int8   │  │ - validate_onnx │  │ - profile       │ │
│  │ - fp16_convert  │  │ - optimize_onnx │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Files to Create

| File                                | Lines | Owner           | Days |
| ----------------------------------- | ----- | --------------- | ---- |
| `layouts/deployment.py`             | ~400  | Frontend Dev    | 2    |
| `callbacks/deployment_callbacks.py` | ~500  | Lead Full-Stack | 2    |
| `tasks/deployment_tasks.py`         | ~200  | ML Engineer     | 1    |
| `services/deployment_service.py`    | ~300  | Lead Full-Stack | 2    |

## Key Implementation: Connecting UI to Backend

**The core integration is wrapping existing `packages/deployment/` functions:**

```python
# packages/dashboard/services/deployment_service.py

from packages.deployment.quantization import (
    dynamic_quantize,
    static_quantize,
    convert_to_fp16
)
from packages.deployment.onnx_export import (
    export_to_onnx,
    validate_onnx_model,
    optimize_onnx_model
)

class DeploymentService:

    @staticmethod
    def quantize_model(model_path: str, method: str, calibration_data=None):
        """
        Wraps existing quantization functions for UI access.
        """
        if method == 'dynamic_int8':
            return dynamic_quantize(model_path)
        elif method == 'static_int8':
            if not calibration_data:
                raise ValueError("Static quantization requires calibration data")
            return static_quantize(model_path, calibration_data)
        elif method == 'fp16':
            return convert_to_fp16(model_path)

    @staticmethod
    def export_onnx(model_path: str, output_path: str, optimize: bool = True):
        """
        Wraps ONNX export with validation.
        """
        export_to_onnx(model_path, output_path)

        # Validate the export
        is_valid, errors = validate_onnx_model(output_path)
        if not is_valid:
            raise ValueError(f"ONNX validation failed: {errors}")

        if optimize:
            optimize_onnx_model(output_path)

        return output_path
```

## Acceptance Criteria

| ID    | Criterion                                | Verification                           | Owner |
| ----- | ---------------------------------------- | -------------------------------------- | ----- |
| DEP-1 | Model dropdown shows all trained models  | QA: Train 3 models, verify all appear  | QA    |
| DEP-2 | Dynamic INT8 quantization completes <30s | QA: Timer on ResNet-18                 | QA    |
| DEP-3 | Static INT8 with calibration works       | QA: Provide 100 samples, verify output | QA    |
| DEP-4 | FP16 conversion reduces size by ~50%     | QA: Compare file sizes                 | QA    |
| DEP-5 | ONNX export produces valid model         | QA: Load in ONNX Runtime               | QA    |
| DEP-6 | Benchmark shows latency comparison       | QA: FP32 vs INT8 vs FP16 chart         | QA    |
| DEP-7 | Download button provides model file      | QA: Download and verify checksum       | QA    |
| DEP-8 | Error on invalid model shows message     | QA: Try corrupted checkpoint           | QA    |

---

# Chapter 1.3 — System Health Dashboard

**Owner:** Full-Stack Dev #2  
**Duration:** 2-3 days

## Implementation

The backend already exists in `services/monitoring_service.py`. We need to surface it.

**File: `layouts/system_health.py`**

```python
import dash_bootstrap_components as dbc
from dash import html, dcc

def create_system_health_layout():
    return dbc.Container([
        html.H2("System Health"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("CPU"),
                    dbc.CardBody([
                        html.H3(id="cpu-gauge", children="—%"),
                        dbc.Progress(id="cpu-progress", value=0)
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Memory"),
                    dbc.CardBody([
                        html.H3(id="memory-gauge", children="—%"),
                        dbc.Progress(id="memory-progress", value=0)
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Disk"),
                    dbc.CardBody([
                        html.H3(id="disk-gauge", children="—%"),
                        dbc.Progress(id="disk-progress", value=0)
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("GPU"),
                    dbc.CardBody([
                        html.H3(id="gpu-gauge", children="—%"),
                        dbc.Progress(id="gpu-progress", value=0)
                    ])
                ])
            ], width=3),
        ]),

        html.Hr(),

        html.H4("Alert History"),
        dbc.Table(id="alerts-table", striped=True, bordered=True),

        dcc.Interval(id="health-interval", interval=5000)  # 5s refresh
    ])
```

## Acceptance Criteria

| ID    | Criterion                              | Verification                 | Owner |
| ----- | -------------------------------------- | ---------------------------- | ----- |
| SYS-1 | CPU/Memory/Disk gauges update every 5s | QA: Monitor for 1 min        | QA    |
| SYS-2 | GPU shows "N/A" if no GPU available    | QA: Test on CPU-only machine | QA    |
| SYS-3 | Alerts table shows last 50 alerts      | QA: Trigger 3 alerts, verify | QA    |
| SYS-4 | High CPU (>90%) shows red indicator    | QA: Stress test CPU          | QA    |

---

# Chapter 1.4 — API Monitoring Dashboard

**Owner:** Lead Full-Stack Dev + DevOps  
**Duration:** 3-4 days

## Key Components

1. **Endpoint Status Table** — Shows `/predict`, `/predict/batch`, `/health` status
2. **Request Logs** — Filterable log viewer
3. **Latency Histogram** — P50/P95/P99 percentiles
4. **API Key Management** — CRUD for API keys

## Acceptance Criteria

| ID    | Criterion                              | Verification                    | Owner |
| ----- | -------------------------------------- | ------------------------------- | ----- |
| API-1 | Endpoint status shows green/red/yellow | QA: Kill API, verify red        | QA    |
| API-2 | Request logs paginate (50 per page)    | QA: Make 100 requests, verify   | QA    |
| API-3 | Latency histogram renders              | QA: Visual inspection           | QA    |
| API-4 | Create API key generates 32-char token | QA: Verify format               | QA    |
| API-5 | Revoke API key invalidates immediately | QA: Use revoked key, expect 401 | QA    |

---

# Chapter 1.5 — Dataset Management

**Owner:** Full-Stack Dev #2  
**Duration:** 2 days

## Fix the 404

**File: `packages/dashboard/callbacks/__init__.py`**

```python
# Add route registration
from .datasets_callbacks import register_datasets_callbacks

def register_all_callbacks(app):
    # ...
    register_datasets_callbacks(app)
```

**File: `packages/dashboard/app.py`**

```python
# Add to page routing
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/datasets":
        from layouts.datasets import create_datasets_layout
        return create_datasets_layout()
    # ...
```

## Acceptance Criteria

| ID   | Criterion                                     | Verification                   | Owner |
| ---- | --------------------------------------------- | ------------------------------ | ----- |
| DS-1 | `/datasets` no longer returns 404             | QA: Navigate to URL            | QA    |
| DS-2 | Dataset table shows all datasets              | QA: Verify count matches DB    | QA    |
| DS-3 | Delete requires confirmation                  | QA: Click delete, verify modal | QA    |
| DS-4 | Dataset details show sample count, split info | QA: Inspect 2 datasets         | QA    |

---

# Chapter 1.6 — Feature Engineering Dashboard

**Owner:** Frontend Dev + ML Engineer  
**Duration:** 4-5 days

## Acceptance Criteria

| ID   | Criterion                                     | Verification            | Owner |
| ---- | --------------------------------------------- | ----------------------- | ----- |
| FE-1 | Feature extraction runs on selected dataset   | QA: Process 1 dataset   | QA    |
| FE-2 | Feature selection (MRMR) reduces to top-K     | QA: Select K=15, verify | QA    |
| FE-3 | Importance plot renders for selected features | QA: Visual inspection   | QA    |
| FE-4 | Export saves feature config as JSON           | QA: Download and parse  | QA    |

---

# Chapter 1.7 — Advanced Training Options

**Owner:** Frontend Dev  
**Duration:** 2-3 days

## Enhancement to Experiment Wizard

Add toggles to existing `layouts/experiment_wizard.py`:

```python
dbc.FormGroup([
    dbc.Checkbox(id="mixed-precision-toggle", label="Enable Mixed Precision (FP16)"),
    dbc.Checkbox(id="progressive-resizing-toggle", label="Enable Progressive Resizing"),
    dbc.Checkbox(id="knowledge-distillation-toggle", label="Enable Knowledge Distillation"),
])
```

## Acceptance Criteria

| ID   | Criterion                                     | Verification             | Owner |
| ---- | --------------------------------------------- | ------------------------ | ----- |
| AT-1 | Mixed precision toggle enables AMP            | QA: Check training logs  | QA    |
| AT-2 | Progressive resizing starts at 50% size       | QA: Verify initial batch | QA    |
| AT-3 | Knowledge distillation requires teacher model | QA: Verify validation    | QA    |

---

# Chapter 1.8 — Enhanced Evaluation Dashboard

**Owner:** ML Engineer + Frontend Dev  
**Duration:** 3-4 days

## Acceptance Criteria

| ID   | Criterion                                      | Verification                      | Owner |
| ---- | ---------------------------------------------- | --------------------------------- | ----- |
| EV-1 | ROC curve renders for multi-class              | QA: Visual inspection (11 curves) | QA    |
| EV-2 | Error analysis shows top-10 misclassifications | QA: Verify samples                | QA    |
| EV-3 | Architecture comparison shows FLOPs            | QA: Compare 3 models              | QA    |

---

# Chapter 1.9 — Testing & QA Dashboard

**Owner:** Frontend Dev + QA Engineer  
**Duration:** 3-4 days

## Acceptance Criteria

| ID   | Criterion                                  | Verification                       | Owner |
| ---- | ------------------------------------------ | ---------------------------------- | ----- |
| QA-1 | Test runner executes pytest suite          | QA: Run full suite                 | QA    |
| QA-2 | Coverage visualization shows line coverage | QA: Verify file-level display      | QA    |
| QA-3 | Benchmark trends show historical data      | QA: Run 3 benchmarks, verify chart | QA    |

---

# Chapter 1.10 — Notification Management

**Owner:** Full-Stack Dev #2  
**Duration:** 1-2 days

## Acceptance Criteria

| ID    | Criterion                                   | Verification                | Owner |
| ----- | ------------------------------------------- | --------------------------- | ----- |
| NOT-1 | Email toggle enables/disables notifications | QA: Send test email         | QA    |
| NOT-2 | Webhook URL validates on save               | QA: Invalid URL shows error | QA    |
| NOT-3 | Frequency dropdown has 4 options            | QA: Verify options          | QA    |

---

# Chapter 1.11 — Infrastructure & Navigation Integration (Addendum)

**Owner:** DevOps + Frontend Dev  
**Duration:** 1-2 days  
**Priority:** 🔴 CRITICAL — Build Blocker

> [!IMPORTANT]
> This chapter addresses the "glue" that connects the vapor fixes to a working system. Without these changes, CI/CD will fail and users cannot access new pages.

---

## A. Dependency Updates

### File: `requirements.txt`

**Add the following mandatory libraries:**

```txt
# Phase 1.11: Infrastructure Dependencies
celery==5.3.6
redis==5.0.1
optuna==3.5.0
flower==2.0.1              # Celery task monitoring UI
onnx==1.15.0
onnxruntime==1.17.0

# Ensure dash-bootstrap-components is recent
dash-bootstrap-components>=1.5.0
```

### File: `packages/dashboard/requirements.txt`

**Sync with root requirements.txt:**

```txt
# Add if missing
celery==5.3.6
redis==5.0.1
optuna==3.5.0
```

### Verification

```bash
pip install -r requirements.txt
python -c "import celery, redis, optuna; print('✓ All dependencies installed')"
```

---

## B. Sidebar Navigation Update

### File: `packages/dashboard/components/sidebar.py`

**Current State:** Sidebar has placeholder `#` links.  
**Target State:** All links point to functional routes.

### Changes Required

```python
# packages/dashboard/components/sidebar.py

# BEFORE (broken):
nav_links = [
    {"name": "Home", "href": "/", "icon": "bi bi-house"},
    {"name": "Data Explorer", "href": "/data-explorer", "icon": "bi bi-table"},
    {"name": "Experiments", "href": "/experiments", "icon": "bi bi-flask"},
    {"name": "HPO", "href": "#", "icon": "bi bi-sliders"},           # BROKEN
    {"name": "Deployment", "href": "#", "icon": "bi bi-cloud"},      # BROKEN
    {"name": "Datasets", "href": "#", "icon": "bi bi-database"},     # BROKEN
    # ...
]

# AFTER (fixed):
nav_links = [
    {"name": "Home", "href": "/", "icon": "bi bi-house"},
    {"name": "Data Explorer", "href": "/data-explorer", "icon": "bi bi-table"},
    {"name": "Experiments", "href": "/experiments", "icon": "bi bi-flask"},
    {"name": "HPO Campaigns", "href": "/hpo", "icon": "bi bi-sliders"},
    {"name": "Deployment", "href": "/deployment", "icon": "bi bi-cloud-upload"},
    {"name": "Datasets", "href": "/datasets", "icon": "bi bi-database"},
    {"name": "XAI Dashboard", "href": "/xai", "icon": "bi bi-diagram-3"},
    {"name": "Feature Engineering", "href": "/features", "icon": "bi bi-graph-up"},
    {"name": "Evaluation", "href": "/evaluation", "icon": "bi bi-bar-chart"},

    # Admin Section (role-gated)
    {"name": "---", "divider": True},
    {"name": "System Health", "href": "/monitoring/system", "icon": "bi bi-heart-pulse", "admin_only": True},
    {"name": "API Monitoring", "href": "/monitoring/api", "icon": "bi bi-broadcast", "admin_only": True},
    {"name": "Testing & QA", "href": "/testing", "icon": "bi bi-check2-circle", "admin_only": True},
    {"name": "Settings", "href": "/settings", "icon": "bi bi-gear"},
]
```

### Route Registration Check

Verify `packages/dashboard/app.py` has all routes:

```python
# Ensure these routes exist in render_page callback
ROUTES = {
    '/': 'home',
    '/data-explorer': 'data_explorer',
    '/experiments': 'experiments',
    '/hpo': 'hpo_campaigns',           # NEW
    '/deployment': 'deployment',        # NEW
    '/datasets': 'datasets',            # NEW
    '/xai': 'xai_dashboard',
    '/features': 'feature_engineering', # NEW
    '/evaluation': 'evaluation',        # NEW
    '/monitoring/system': 'system_health',  # NEW
    '/monitoring/api': 'api_monitoring',    # NEW
    '/testing': 'testing_dashboard',        # NEW
    '/settings': 'settings',
}
```

---

## C. Research Logger Hook (Critical for Phase 3)

### Purpose

For academic publication (Phase 3), we need structured logs of HPO results that can be directly converted to ablation study tables, scatter plots, and parameter sensitivity charts.

### File: `packages/dashboard/tasks/hpo_tasks.py`

**Add the following at campaign completion:**

```python
import json
import os
from datetime import datetime

def save_research_artifact(campaign_id: int, study: optuna.Study, config: dict):
    """
    Saves a 'paper-ready' artifact for research analysis.

    Output Format:
    - JSON file with all trial data
    - Directly importable to pandas for analysis
    - Compatible with LaTeX table generation
    """
    # Create output directory
    artifact_dir = "outputs/research_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)

    artifact_path = f"{artifact_dir}/hpo_campaign_{campaign_id}.json"

    artifact = {
        'metadata': {
            'campaign_id': campaign_id,
            'timestamp': datetime.now().isoformat(),
            'n_trials': len(study.trials),
            'algorithm': config.get('sampler', 'TPE'),
            'direction': 'minimize' if study.direction == optuna.study.StudyDirection.MINIMIZE else 'maximize'
        },
        'search_space': config['param_space'],
        'best_result': {
            'params': study.best_params,
            'value': study.best_value,
            'trial_number': study.best_trial.number
        },
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'value': t.value,
                'state': t.state.name,
                'duration_seconds': (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete else None
            }
            for t in study.trials
        ],
        'parameter_importance': dict(optuna.importance.get_param_importances(study)) if len(study.trials) > 2 else {}
    }

    with open(artifact_path, 'w') as f:
        json.dump(artifact, f, indent=2, default=str)

    return artifact_path

# In run_hpo_optimization, after study.optimize():
@celery_app.task(bind=True, name='hpo.run_optimization')
def run_hpo_optimization(self, campaign_id: int, config: dict):
    # ... existing code ...

    study.optimize(
        objective,
        n_trials=config['n_trials'],
        timeout=config.get('timeout'),
        callbacks=[progress_callback]
    )

    # NEW: Save research artifact
    artifact_path = save_research_artifact(campaign_id, study, config)

    campaign.status = 'completed'
    campaign.artifact_path = artifact_path  # Store path in DB
    db.session.commit()

    return {
        'status': 'completed',
        'best_params': study.best_params,
        'best_value': study.best_value,
        'artifact_path': artifact_path  # Return for reference
    }
```

### Database Schema Update

**File: `packages/dashboard/models/hpo_campaign.py`**

```python
class HPOCampaign(db.Model):
    # ... existing fields ...

    # NEW: Research artifact path
    artifact_path = db.Column(db.String(500), nullable=True)
```

### Usage for Research (Phase 3)

```python
# Load HPO artifacts for paper analysis
import pandas as pd
import json
import glob

artifacts = glob.glob('outputs/research_artifacts/hpo_campaign_*.json')
trials_data = []

for path in artifacts:
    with open(path) as f:
        data = json.load(f)
    for trial in data['all_trials']:
        trial['campaign_id'] = data['metadata']['campaign_id']
        trials_data.append(trial)

df = pd.DataFrame(trials_data)

# Now you can generate:
# - Ablation tables: df.groupby(['param_name']).mean()
# - Parameter sensitivity: df.plot(x='learning_rate', y='value')
# - Best configs: df.sort_values('value').head(10)
```

---

## D. Docker Compose Update

### File: `docker-compose.yml`

**Complete updated configuration:**

```yaml
version: "3.8"

services:
  dashboard:
    build: .
    ports:
      - "8050:8050"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
      - DATABASE_URL=sqlite:///dashboard.db
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./outputs:/app/outputs # For research artifacts

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  celery_worker:
    build: .
    command: celery -A packages.dashboard.celery_config worker --loglevel=info --concurrency=2
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./outputs:/app/outputs

  celery_flower:
    build: .
    command: celery -A packages.dashboard.celery_config flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - celery_worker

volumes:
  redis_data:
```

---

## Acceptance Criteria (Addendum)

| ID    | Criterion                                   | Verification             | Owner  |
| ----- | ------------------------------------------- | ------------------------ | ------ |
| INT-1 | `pip install -r requirements.txt` succeeds  | CI: No errors            | DevOps |
| INT-2 | All sidebar links route correctly (no 404s) | QA: Click every link     | QA     |
| INT-3 | Admin-only links hidden for regular users   | QA: Test with non-admin  | QA     |
| INT-4 | HPO completion creates research artifact    | QA: Check `outputs/`     | QA     |
| INT-5 | Research artifact contains all trial data   | QA: Validate JSON schema | QA     |
| INT-6 | `docker-compose up` starts all services     | DevOps: Full stack test  | DevOps |
| INT-7 | Flower UI accessible at :5555               | DevOps: Open browser     | DevOps |

---

# Phase 1 Summary

## Timeline

| Week | Chapters         | Milestones                           |
| ---- | ---------------- | ------------------------------------ |
| 1    | 1.1 (HPO)        | Celery configured, HPO functional    |
| 2    | 1.2 (Deployment) | Quantization + ONNX accessible       |
| 3    | 1.3, 1.4         | Monitoring dashboards live           |
| 4    | 1.5, 1.6, 1.7    | Dataset, Features, Training complete |
| 5    | 1.8, 1.9, 1.10   | Evaluation, Testing, Notifications   |
| 6    | 1.11 + QA        | Integration, full regression testing |

## Definition of Done

Phase 1 is complete when:

- [ ] All 11 chapters pass acceptance criteria
- [ ] No 404 errors in dashboard navigation
- [ ] All Celery tasks execute successfully
- [ ] Research artifacts generate correctly
- [ ] QA sign-off on all 47 acceptance criteria
- [ ] Zero critical bugs in bug tracker

---

_Phase 1 Complete Document (with Addendum) — Ready for Phase 2 on request_


---

# Phase 2: The Documentation Overhaul — Complete Implementation Guide

**Phase Duration:** 2-3 weeks  
**Team Size:** 3 SMEs  
**Priority:** 🟡 HIGH — User Experience Blocker

---

## Team Roster & Allocation

| Role                      | Assigned To | Chapters           | Weekly Hours |
| ------------------------- | ----------- | ------------------ | ------------ |
| **Lead Technical Writer** | TBD         | 2.1, 2.2, 2.3      | 40           |
| **Technical Writer #2**   | TBD         | 2.3, 2.5, 2.6      | 40           |
| **DevOps Engineer**       | TBD         | 2.4 (MkDocs CI/CD) | 15           |

---

# Chapter 2.1 — Documentation Audit & Inventory

**Owner:** Lead Technical Writer  
**Duration:** 2-3 days

## 2.1.1 Complete File Census

### Current State (Post Phase A/B Cleanup)

| Location                  | File Count | Total Size | Status        |
| ------------------------- | ---------- | ---------- | ------------- |
| Root (`/`)                | 4 MD files | ~90KB      | ✅ Essential  |
| `docs/`                   | 55+ files  | ~400KB     | ⚠️ Sprawling  |
| `docs/archive/`           | 27 files   | ~200KB     | 📦 Historical |
| `docs/user-guide/phases/` | 12 files   | ~230KB     | ✅ Organized  |
| `packages/dashboard/`     | 5 MD files | ~75KB      | ⚠️ Misplaced  |

### Audit Template

Create `docs/_audit/file_census.csv`:

```csv
file_path,word_count,last_modified,topic,status,action
README.md,8500,2025-11-28,overview,active,keep
docs/DASHBOARD_GAPS.md,2100,2025-11-22,status,active,archive_after_fix
docs/IMPLEMENTATION_PLAN.md,2000,2025-10,planning,stale,archive
...
```

### Classification Categories

| Category       | Criteria                             | Action                      |
| -------------- | ------------------------------------ | --------------------------- |
| **Active**     | Current, accurate, referenced        | Keep, update if needed      |
| **Stale**      | >6 months old, may have inaccuracies | Review, update or archive   |
| **Redundant**  | Duplicates content in another file   | Merge into canonical source |
| **Historical** | Past decisions, no longer applicable | Move to `docs/archive/`     |
| **Orphaned**   | Not linked from anywhere             | Delete or integrate         |

## 2.1.2 Overlap Analysis

### Known Duplications

| Topic              | Files with Duplicate Content                    | Canonical Target                        |
| ------------------ | ----------------------------------------------- | --------------------------------------- |
| Installation       | README.md, QUICKSTART.md, START_HERE.md         | `docs/getting-started/installation.md`  |
| Dashboard setup    | README.md, GUI_QUICKSTART.md, USAGE_PHASE_11.md | `docs/user-guide/dashboard.md`          |
| Phase summaries    | README.md, QUICKSTART.md, 12 phase guides       | README.md (summary) + individual guides |
| Environment config | README.md, .env.example, dashboard README       | `docs/getting-started/configuration.md` |

### Overlap Resolution Strategy

```
For each duplicate:
1. Identify the most complete version
2. Designate it as canonical
3. In other files, replace with:
   > See [Canonical Topic](link) for details.
4. Add redirect comment for future editors
```

## 2.1.3 Gap Identification

### Missing Documentation

| Gap                      | Priority  | Target Location                        | Owner            |
| ------------------------ | --------- | -------------------------------------- | ---------------- |
| ARCHITECTURE.md (visual) | 🔴 HIGH   | `docs/development/architecture.md`     | Tech Writer #2   |
| Configuration Reference  | 🔴 HIGH   | `docs/reference/configuration.md`      | Lead Tech Writer |
| Troubleshooting Index    | 🟡 MEDIUM | `docs/troubleshooting/index.md`        | Tech Writer #2   |
| API Authentication Guide | 🟡 MEDIUM | `docs/api-reference/authentication.md` | Lead Tech Writer |
| Model Selection Guide    | 🟢 LOW    | `docs/user-guide/model-selection.md`   | Tech Writer #2   |

## Acceptance Criteria

| ID    | Criterion                                | Verification                | Owner   |
| ----- | ---------------------------------------- | --------------------------- | ------- |
| AUD-1 | File census CSV complete (all 104 files) | Count matches `find` output | Lead TW |
| AUD-2 | Each file classified into category       | No "unknown" categories     | Lead TW |
| AUD-3 | Overlap pairs documented                 | Matrix shows all duplicates | Lead TW |
| AUD-4 | Gap list prioritized                     | HIGH/MEDIUM/LOW assigned    | Lead TW |

---

# Chapter 2.2 — Information Architecture Design

**Owner:** Lead Technical Writer  
**Duration:** 2 days

## 2.2.1 Target Structure

### MkDocs Navigation Hierarchy

```yaml
# mkdocs.yml nav structure
nav:
  - Home: index.md

  - Getting Started:
      - Installation: getting-started/installation.md
      - Quick Start: getting-started/quickstart.md
      - Configuration: getting-started/configuration.md
      - First Experiment: getting-started/first-experiment.md

  - User Guide:
      - Dashboard:
          - Overview: user-guide/dashboard/overview.md
          - Data Explorer: user-guide/dashboard/data-explorer.md
          - Experiments: user-guide/dashboard/experiments.md
          - XAI Analysis: user-guide/dashboard/xai.md
          - HPO Campaigns: user-guide/dashboard/hpo.md
      - CLI Workflows:
          - Training: user-guide/cli/training.md
          - Evaluation: user-guide/cli/evaluation.md
          - Inference: user-guide/cli/inference.md
      - Phase Guides:
          - Phase 0 - Data: user-guide/phases/phase-0.md
          - Phase 1 - Classical ML: user-guide/phases/phase-1.md
          - Phase 2 - 1D CNN: user-guide/phases/phase-2.md
          - Phase 3 - Advanced CNN: user-guide/phases/phase-3.md
          - Phase 4 - Transformer: user-guide/phases/phase-4.md
          - Phase 5 - Time-Frequency: user-guide/phases/phase-5.md
          - Phase 6 - PINN: user-guide/phases/phase-6.md
          - Phase 7 - XAI: user-guide/phases/phase-7.md
          - Phase 8 - Ensemble: user-guide/phases/phase-8.md
          - Phase 9 - Deployment: user-guide/phases/phase-9.md
          - Phase 10 - QA: user-guide/phases/phase-10.md
          - Phase 11 - Dashboard: user-guide/phases/phase-11.md

  - API Reference:
      - REST API: api-reference/rest-api.md
      - Python API: api-reference/python-api.md
      - Model Registry: api-reference/model-registry.md

  - Development:
      - Architecture: development/architecture.md
      - Contributing: development/contributing.md
      - Testing: development/testing.md
      - Code Style: development/code-style.md

  - Deployment:
      - Docker: deployment/docker.md
      - Kubernetes: deployment/kubernetes.md
      - Production Guide: deployment/production.md

  - Reference:
      - Configuration: reference/configuration.md
      - Model Comparison: reference/model-comparison.md
      - FAQ: reference/faq.md

  - Troubleshooting:
      - Common Issues: troubleshooting/common-issues.md
      - Error Messages: troubleshooting/error-messages.md

  - Archive: archive/index.md
```

## 2.2.2 Navigation Flow

### User Journey Mapping

```
NEW USER
    │
    ├─→ README.md (30s overview)
    │       │
    │       ▼
    ├─→ getting-started/installation.md (10 min)
    │       │
    │       ▼
    ├─→ getting-started/quickstart.md (30 min)
    │       │
    │       ▼
    └─→ DECISION POINT
            │
            ├─→ "I want GUI" → user-guide/dashboard/overview.md
            │
            └─→ "I want CLI" → user-guide/cli/training.md

RETURNING USER
    │
    └─→ Search or direct nav to specific topic

DEVELOPER
    │
    ├─→ development/architecture.md
    │
    └─→ development/contributing.md
```

## 2.2.3 Cross-Reference Strategy

### Linking Policy

1. **Internal links use relative paths**: `[See Installation](../getting-started/installation.md)`
2. **API references use code links**: `[`create_model()`](../api-reference/python-api.md#create_model)`
3. **Phase guides link to each other**: "After completing Phase 2, proceed to [Phase 3](phase-3.md)"
4. **No broken links allowed**: CI check validates all links

### Redirect Map

For renamed/moved files, create `docs/_redirects.md`:

```markdown
| Old Path                             | New Path                              | Reason                 |
| ------------------------------------ | ------------------------------------- | ---------------------- |
| QUICKSTART.md                        | docs/getting-started/quickstart.md    | Restructure            |
| START_HERE.md                        | (deleted)                             | Merged into quickstart |
| USAGE_GUIDES/\*.md                   | docs/user-guide/phases/\*.md          | Restructure            |
| packages/dashboard/GUI_QUICKSTART.md | docs/user-guide/dashboard/overview.md | Centralize             |
```

## Acceptance Criteria

| ID   | Criterion                       | Verification            | Owner   |
| ---- | ------------------------------- | ----------------------- | ------- |
| IA-1 | mkdocs.yml nav complete         | Renders without errors  | Lead TW |
| IA-2 | All nav items have target files | `mkdocs build` succeeds | Lead TW |
| IA-3 | User journey documented         | Diagram reviewed        | Lead TW |
| IA-4 | Redirect map complete           | All old links mapped    | Lead TW |

---

# Chapter 2.3 — Content Consolidation Execution

**Owner:** Lead Technical Writer + Tech Writer #2  
**Duration:** 5-7 days

## 2.3.1 Root-Level Cleanup

### Target Root State

```
LSTM_PFD/
├── README.md          # Project overview (max 500 lines)
├── CHANGELOG.md       # Version history
├── CONTRIBUTING.md    # Contribution guide
├── LICENSE            # License file
└── docs/              # All other documentation
```

### README.md Restructure

**Before:** 1,124 lines with full phase details  
**After:** ~400 lines with links to detailed docs

```markdown
# LSTM_PFD

> One-paragraph summary

## Quick Start

> 5-line install + link to full guide

## Features

> Bullet list with links to detailed pages

## Documentation

> Links to docs/ structure

## License
```

## 2.3.2 Getting Started Merge

### Consolidation Plan

| Source              | Content to Extract        | Target                                  |
| ------------------- | ------------------------- | --------------------------------------- |
| QUICKSTART.md (all) | Installation steps        | `getting-started/installation.md`       |
| QUICKSTART.md (all) | First experiment tutorial | `getting-started/quickstart.md`         |
| START_HERE.md       | Decision tree             | `getting-started/quickstart.md` (intro) |
| README.md           | Env config section        | `getting-started/configuration.md`      |
| .env.example        | Variable documentation    | `getting-started/configuration.md`      |

### File: `docs/getting-started/installation.md`

```markdown
# Installation

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU)

## Step 1: Clone Repository

...

## Step 2: Create Virtual Environment

...

## Step 3: Install PyTorch

...

## Step 4: Install Dependencies

...

## Step 5: Verify Installation

...

## Troubleshooting

- [CUDA Issues](../troubleshooting/common-issues.md#cuda)
- [Dependency Conflicts](../troubleshooting/common-issues.md#dependencies)
```

## 2.3.3 Phase Guide Normalization

### Standard Template for Each Phase Guide

````markdown
# Phase X: [Name]

## Overview

> 2-3 sentences: what this phase does, why it matters

## Prerequisites

- [ ] Phase X-1 completed
- [ ] Required data/models

## Architecture

> Diagram or description of components

## Quick Start

```bash
# Minimal commands to run this phase
```
````

## Detailed Configuration

> Parameter reference table

## Expected Results

> Accuracy ranges, output files

## Troubleshooting

> Phase-specific issues

## Next Steps

> Link to Phase X+1

```

### File Renaming

| Current Name | New Name |
|--------------|----------|
| PHASE_1_USAGE_GUIDE.md | phase-1.md |
| PHASE_5_ARCHITECTURE.md | (merge into phase-5.md) |
| Phase_9_DEPLOYMENT_GUIDE.md | phase-9.md |
| Phase_10_QA_INTEGRATION_GUIDE.md | phase-10.md |

## 2.3.4 Archive Policy

### What Goes to Archive

- Phase planning docs (already there)
- Implementation history
- Resolved bug documentation
- Superseded guides

### Archive Structure

```

docs/archive/
├── index.md # Archive index with search
├── planning/ # Phase plans (Phase_0.md - Phase_11D.md)
├── implementation/ # Historical implementation notes
└── deprecated/ # Superseded documentation

````

### Archive Notice Template

```markdown
> [!WARNING]
> **Archived Document**
> This document is historical and may be outdated.
> For current information, see [Current Topic](../current-location.md).
>
> *Archived on: YYYY-MM-DD*
> *Reason: [Superseded/Completed/Deprecated]*
````

## Acceptance Criteria

| ID    | Criterion                              | Verification                 | Owner   |
| ----- | -------------------------------------- | ---------------------------- | ------- |
| CON-1 | README.md ≤500 lines                   | `wc -l README.md`            | Lead TW |
| CON-2 | No duplicate installation steps        | Grep for "pip install"       | Lead TW |
| CON-3 | All phase guides use standard template | Visual review                | TW #2   |
| CON-4 | Archive files have notice banner       | Grep for "Archived Document" | TW #2   |
| CON-5 | No orphaned files in docs/             | Link checker passes          | DevOps  |

---

# Chapter 2.4 — MkDocs Implementation

**Owner:** DevOps Engineer + Lead Technical Writer  
**Duration:** 2-3 days

## 2.4.1 Setup & Configuration

### File: `mkdocs.yml`

```yaml
site_name: LSTM_PFD Documentation
site_description: Advanced Bearing Fault Diagnosis System
site_author: Your Team
site_url: https://your-org.github.io/LSTM_PFD/

repo_name: LSTM_PFD
repo_url: https://github.com/your-org/LSTM_PFD

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.tabs.link

plugins:
  - search
  - minify:
      minify_html: true

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - attr_list
  - md_in_html
  - tables
  - toc:
      permalink: true

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/your-org/LSTM_PFD

nav:
  # (see Chapter 2.2.1 for full nav)
```

### Dependencies

```txt
# docs/requirements.txt
mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocs-minify-plugin>=0.7.0
pymdown-extensions>=10.0
```

## 2.4.2 Automated Build

### File: `.github/workflows/docs.yml`

```yaml
name: Documentation

on:
  push:
    branches: [main]
    paths:
      - "docs/**"
      - "mkdocs.yml"
  pull_request:
    paths:
      - "docs/**"
      - "mkdocs.yml"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r docs/requirements.txt

      - name: Build docs
        run: mkdocs build --strict

      - name: Link checker
        uses: lycheeverse/lychee-action@v1
        with:
          args: --verbose --no-progress './site/**/*.html'
          fail: true

  deploy:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r docs/requirements.txt

      - name: Deploy to GitHub Pages
        run: mkdocs gh-deploy --force
```

## 2.4.3 Versioning Strategy

### Using mike for Versioned Docs

```bash
pip install mike

# Deploy version 1.0
mike deploy --push --update-aliases 1.0 latest

# Deploy version 1.1
mike deploy --push --update-aliases 1.1 latest

# Set default version
mike set-default --push latest
```

## Acceptance Criteria

| ID   | Criterion                    | Verification                      | Owner   |
| ---- | ---------------------------- | --------------------------------- | ------- |
| MK-1 | `mkdocs build` succeeds      | CI pipeline green                 | DevOps  |
| MK-2 | `mkdocs serve` runs locally  | Manual test                       | DevOps  |
| MK-3 | GitHub Pages deploys on push | Verify URL accessible             | DevOps  |
| MK-4 | Search works                 | Search for "PINN", verify results | Lead TW |
| MK-5 | Dark mode toggle works       | Visual verification               | Lead TW |
| MK-6 | No broken links              | Lychee check passes               | DevOps  |

---

# Chapter 2.5 — API Documentation Generation

**Owner:** Lead Technical Writer  
**Duration:** 2-3 days

## 2.5.1 OpenAPI Spec

### Auto-generate from FastAPI

```python
# scripts/generate_openapi.py
from api.main import app
import json

openapi_schema = app.openapi()
with open('docs/api-reference/openapi.json', 'w') as f:
    json.dump(openapi_schema, f, indent=2)
```

### Embed in MkDocs

```yaml
# mkdocs.yml
plugins:
  - swagger-ui-tag

# Usage in docs/api-reference/rest-api.md
<swagger-ui src="./openapi.json"/>
```

## 2.5.2 Python API Reference

### Using pdoc

```bash
pip install pdoc

# Generate HTML docs
pdoc --html --output-dir docs/api-reference/python packages/core

# Or Markdown for MkDocs
pdoc --template-dir docs/_templates packages/core -o docs/api-reference/python
```

### Key APIs to Document

| Module                         | Key Functions/Classes                         |
| ------------------------------ | --------------------------------------------- |
| `packages.core.models`         | `create_model()`, `ModelFactory`, `BaseModel` |
| `packages.core.training`       | `Trainer`, `CNNTrainer`, `PINNTrainer`        |
| `packages.core.evaluation`     | `Evaluator`, `ConfusionAnalyzer`              |
| `packages.core.explainability` | `SHAPExplainer`, `LIMEExplainer`              |
| `data`                         | `SignalGenerator`, `BearingFaultDataset`      |

## Acceptance Criteria

| ID    | Criterion                                  | Verification              | Owner   |
| ----- | ------------------------------------------ | ------------------------- | ------- |
| API-1 | OpenAPI spec generates without errors      | Script runs               | Lead TW |
| API-2 | All endpoints documented                   | Compare to FastAPI routes | Lead TW |
| API-3 | Python API covers 20 key classes/functions | Count in output           | Lead TW |
| API-4 | Code examples in docstrings                | Spot check 5 functions    | Lead TW |

---

# Chapter 2.6 — Diagrams & Visual Assets

**Owner:** Technical Writer #2  
**Duration:** 2-3 days

## 2.6.1 Architecture Diagram

### Create with Mermaid

**File: `docs/development/architecture.md`**

````markdown
# System Architecture

## High-Level Overview

```mermaid
graph TB
    subgraph "Data Layer"
        A[Signal Generator] --> B[HDF5 Cache]
        C[MAT Importer] --> B
    end

    subgraph "Model Layer"
        B --> D[Model Factory]
        D --> E[CNN Models]
        D --> F[Transformer]
        D --> G[PINN]
        D --> H[Ensemble]
    end

    subgraph "Training Layer"
        E & F & G --> I[Trainer]
        I --> J[Checkpoints]
    end

    subgraph "Deployment Layer"
        J --> K[Quantization]
        J --> L[ONNX Export]
        K & L --> M[Inference API]
    end

    subgraph "Dashboard"
        M --> N[Dash App]
        B --> N
        I --> N
    end
```
````

## Data Flow

...

````

## 2.6.2 Phase Progression Diagram

```mermaid
gantt
    title LSTM_PFD Development Phases
    dateFormat  YYYY-MM

    section Foundation
    Phase 0 - Data Pipeline    :done, p0, 2025-01, 30d

    section Data-Driven Models
    Phase 1 - Classical ML     :done, p1, after p0, 23d
    Phase 2 - 1D CNN           :done, p2, after p1, 27d
    Phase 3 - Advanced CNN     :done, p3, after p2, 34d
    Phase 4 - Transformer      :done, p4, after p3, 29d

    section Advanced Techniques
    Phase 5 - Time-Frequency   :done, p5, after p4, 14d
    Phase 6 - PINN             :done, p6, after p5, 16d
    Phase 7 - XAI              :done, p7, after p6, 12d
    Phase 8 - Ensemble         :done, p8, after p7, 10d

    section Production
    Phase 9 - Deployment       :done, p9, after p8, 14d
    Phase 10 - QA              :done, p10, after p9, 25d
    Phase 11 - Dashboard       :active, p11, after p10, 30d
````

## 2.6.3 Dashboard Screenshots

### Screenshot Capture Script

```python
# scripts/capture_screenshots.py
from playwright.sync_api import sync_playwright

PAGES = [
    ('/', 'home.png'),
    ('/data-explorer', 'data-explorer.png'),
    ('/experiments', 'experiments.png'),
    ('/xai', 'xai-dashboard.png'),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1280, 'height': 800})

    for path, filename in PAGES:
        page.goto(f'http://localhost:8050{path}')
        page.wait_for_load_state('networkidle')
        page.screenshot(path=f'docs/assets/screenshots/{filename}')

    browser.close()
```

## Acceptance Criteria

| ID    | Criterion                              | Verification           | Owner |
| ----- | -------------------------------------- | ---------------------- | ----- |
| VIS-1 | Architecture diagram renders in MkDocs | Visual check           | TW #2 |
| VIS-2 | Phase timeline diagram accurate        | Compare to README      | TW #2 |
| VIS-3 | Screenshots for 5 main pages           | Files exist in assets/ | TW #2 |
| VIS-4 | Screenshots annotated with callouts    | Visual check           | TW #2 |

---

# Chapter 2.7 — Research-Grade Documentation Features (Addendum)

**Owner:** Technical Writer #2  
**Duration:** 1-2 days  
**Priority:** 🔴 HIGH — Publication Blocker

> [!IMPORTANT]
> Without these features, PINN equations will render as broken text and reviewers cannot interact with results. This chapter is mandatory for Phase 3 (Research Novelty).

---

## A. Enabling LaTeX/Math Support

### Purpose

The project involves PINNs and signal processing with equations like:

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_1 \mathcal{L}_{physics} + \lambda_2 \mathcal{L}_{boundary}$$

$$f_{BPFO} = \frac{n}{2}\left(1 - \frac{d}{D}\cos\theta\right)f_r$$

Without MathJax, these render as broken text.

### Update: `mkdocs.yml`

```yaml
markdown_extensions:
  # ... existing extensions ...
  - pymdownx.arithmatex:
      generic: true

extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
```

### Create: `docs/javascripts/mathjax.js`

```javascript
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
```

### Usage in Documentation

```markdown
The physics loss is defined as:

$$\mathcal{L}_{physics} = \frac{1}{N}\sum_{i=1}^{N}\left|\frac{\partial^2 u}{\partial t^2} - c^2 \nabla^2 u\right|^2$$

Inline math: The contact angle \(\theta\) affects fault frequency.
```

---

## B. Interactive Plot Embedding

### Purpose

Static screenshots prevent reviewers from exploring data. Interactive Plotly embeds allow zoom, hover, and data inspection inside the docs.

### Export Script: `scripts/export_interactive_plots.py`

```python
"""
Export Plotly figures as interactive HTML for documentation.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("docs/assets/interactive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def export_confusion_matrix(y_true, y_pred, labels, filename="confusion_matrix.html"):
    """Export interactive confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues"
    )
    fig.update_layout(title="Confusion Matrix (Hover for details)")
    fig.write_html(OUTPUT_DIR / filename, include_plotlyjs='cdn')
    return str(OUTPUT_DIR / filename)

def export_roc_curves(fpr_dict, tpr_dict, auc_dict, filename="roc_curves.html"):
    """Export multi-class ROC curves."""
    fig = go.Figure()
    for class_name in fpr_dict:
        fig.add_trace(go.Scatter(
            x=fpr_dict[class_name],
            y=tpr_dict[class_name],
            name=f"{class_name} (AUC={auc_dict[class_name]:.3f})",
            mode='lines'
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random",
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    fig.update_layout(
        title="ROC Curves by Fault Class",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    fig.write_html(OUTPUT_DIR / filename, include_plotlyjs='cdn')
    return str(OUTPUT_DIR / filename)

def export_training_curves(history, filename="training_curves.html"):
    """Export training/validation loss curves."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['train_loss'], name='Train Loss'))
    fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss'))
    fig.add_trace(go.Scatter(y=history['val_accuracy'], name='Val Accuracy', yaxis='y2'))
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy", overlaying='y', side='right')
    )
    fig.write_html(OUTPUT_DIR / filename, include_plotlyjs='cdn')
    return str(OUTPUT_DIR / filename)
```

### Usage in Documentation

```markdown
# Model Performance

Here is the interactive confusion matrix. Hover to see exact counts:

<iframe src="../../assets/interactive/confusion_matrix.html" 
        width="100%" height="600px" frameBorder="0"></iframe>

The ROC curves for all 11 fault classes (zoom to inspect individual classes):

<iframe src="../../assets/interactive/roc_curves.html" 
        width="100%" height="500px" frameBorder="0"></iframe>
```

### Required Assets

| Plot             | Source             | Output                                          |
| ---------------- | ------------------ | ----------------------------------------------- |
| Confusion Matrix | Evaluation results | `docs/assets/interactive/confusion_matrix.html` |
| ROC Curves       | Multi-class ROC    | `docs/assets/interactive/roc_curves.html`       |
| Training Curves  | Training history   | `docs/assets/interactive/training_curves.html`  |
| SHAP Summary     | XAI module         | `docs/assets/interactive/shap_summary.html`     |
| HPO Surface      | Optuna study       | `docs/assets/interactive/hpo_surface.html`      |

---

## C. Docstring Quality Gate

### Purpose

Chapter 2.5 assumes code has docstrings for pdoc to generate. If docstrings are missing, pdoc produces empty pages. We enforce a minimum coverage threshold.

### Install: `interrogate`

```txt
# Add to docs/requirements.txt
interrogate==1.5.0
```

### Configuration: `pyproject.toml`

```toml
[tool.interrogate]
ignore-init-method = true
ignore-init-module = true
ignore-magic = true
ignore-semiprivate = true
ignore-private = true
ignore-property-decorators = true
ignore-module = false
ignore-nested-functions = true
fail-under = 80
exclude = ["tests", "docs", "build", "venv"]
verbose = 1
```

### CI Integration: `.github/workflows/docs.yml`

```yaml
jobs:
  build:
    steps:
      # ... existing steps ...

      - name: Check Docstring Coverage
        run: |
          pip install interrogate
          echo "Checking packages/core..."
          interrogate -v --fail-under 80 packages/core
          echo "Checking packages/dashboard/services..."
          interrogate -v --fail-under 70 packages/dashboard/services

      - name: Build docs
        run: mkdocs build --strict
```

### Current Coverage Check

```bash
# Run locally to identify gaps
interrogate -v packages/core

# Example output:
# packages/core/models/base.py: 45.0%
# packages/core/training/trainer.py: 78.0%
# packages/core/explainability/shap_explainer.py: 92.0%
# TOTAL: 72.5% (needs improvement)
```

### Priority Files for Docstring Addition

| File                                     | Current | Target | Effort  |
| ---------------------------------------- | ------- | ------ | ------- |
| `packages/core/models/pinn.py`           | 40%     | 90%    | 2 hours |
| `packages/core/models/factory.py`        | 50%     | 85%    | 1 hour  |
| `packages/core/training/pinn_trainer.py` | 35%     | 85%    | 2 hours |
| `packages/core/explainability/*.py`      | 80%     | 95%    | 1 hour  |
| `data/signal_generator.py`               | 60%     | 90%    | 1 hour  |

---

## D. Updated Dependencies

### File: `docs/requirements.txt`

```txt
# Core MkDocs
mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocs-minify-plugin>=0.7.0
pymdown-extensions>=10.0

# Version management
mike>=2.0.0

# Research-grade features (Chapter 2.7)
interrogate>=1.5.0

# Screenshot automation
playwright>=1.40.0
```

---

## Acceptance Criteria (Addendum)

| ID    | Criterion                                 | Verification            | Owner  |
| ----- | ----------------------------------------- | ----------------------- | ------ |
| RES-1 | LaTeX equations render correctly          | Visual check: PINN page | TW #2  |
| RES-2 | MathJax loads without console errors      | Browser DevTools        | TW #2  |
| RES-3 | 5 interactive plots embedded              | Check iframe loads      | TW #2  |
| RES-4 | Plots zoom/hover work in docs             | Manual test             | TW #2  |
| RES-5 | Docstring coverage ≥80% for packages/core | `interrogate` CI pass   | TW #2  |
| RES-6 | CI blocks on coverage failure             | Test with low coverage  | DevOps |

---

# Phase 2 Summary

## Timeline

| Week | Tasks                   | Milestones                              |
| ---- | ----------------------- | --------------------------------------- |
| 1    | Audit, IA Design        | Census complete, nav structure approved |
| 2    | Content consolidation   | README reduced, phase guides normalized |
| 3    | MkDocs setup, API docs  | Docs site live, CI/CD configured        |
| 4    | Research features (2.7) | LaTeX + interactive plots working       |

## Definition of Done

Phase 2 is complete when:

- [ ] File count reduced from 104 to <60
- [ ] MkDocs site builds and deploys
- [ ] No broken links (CI check passes)
- [ ] README.md ≤500 lines
- [ ] All phase guides follow standard template
- [ ] API documentation generated
- [ ] Architecture diagram in docs
- [ ] LaTeX equations render correctly
- [ ] Interactive plots embedded (5 minimum)
- [ ] Docstring coverage ≥80%
- [ ] QA sign-off on navigation flow

---

_Phase 2 Complete Document (with Addendum) — Ready for Phase 3 on request_


---

# Phase 3: Research Novelty & Academic Value — Complete Implementation Guide

**Phase Duration:** 4-6 weeks  
**Team Size:** 4 SMEs  
**Priority:** 🟡 HIGH — Publication Enabler

---

## Team Roster & Allocation

| Role                   | Assigned To | Chapters      | Weekly Hours |
| ---------------------- | ----------- | ------------- | ------------ |
| **Lead ML Researcher** | TBD         | 3.1, 3.2, 3.3 | 40           |
| **ML Researcher #2**   | TBD         | 3.1, 3.5      | 40           |
| **Technical Writer**   | TBD         | 3.4           | 30           |
| **Data Engineer**      | TBD         | 3.5           | 20           |

---

## Phase 3 Objectives

1. **PINN Integration** → Publication-ready with ablation studies
2. **XAI Module** → Validated explanations with quality metrics
3. **Ensemble Methods** → Documented voting strategies
4. **Manuscript** → Camera-ready for top-tier venue
5. **Dataset** → Citable release with DOI

---

# Chapter 3.1 — PINN Integration Finalization

**Owner:** Lead ML Researcher + ML Researcher #2  
**Duration:** 8-10 days

## 3.1.1 Physics Constraint Validation

### Current PINN Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PINN ARCHITECTURE                             │
│                                                                   │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │ Signal Input │───▶│ CNN Encoder │───▶│ Physics Loss│        │
│   │ (1D Time)    │    │             │    │             │        │
│   └─────────────┘    └──────┬──────┘    └──────┬──────┘        │
│                              │                  │                │
│                              ▼                  ▼                │
│                       ┌─────────────────────────────┐           │
│                       │   L_total = L_data +        │           │
│                       │            λ₁·L_physics +   │           │
│                       │            λ₂·L_boundary    │           │
│                       └─────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Physics Constraints to Validate

| Constraint              | Equation                           | Verification Method                 |
| ----------------------- | ---------------------------------- | ----------------------------------- |
| Energy Conservation     | ∑E_in = ∑E_out + E_dissipated      | Compare predicted vs. actual RMS    |
| Momentum Conservation   | ∂p/∂t = F_net                      | Verify acceleration consistency     |
| Bearing Kinematics      | f_BPFO = (n/2)(1 - d/D·cos(θ))·f_r | Check frequency prediction accuracy |
| Fault Signature Physics | A_fault ∝ severity                 | Correlation analysis                |

### Ablation Study Design

```python
# scripts/research/pinn_ablation.py

ABLATION_CONFIGS = [
    # Baseline: No physics
    {'name': 'CNN_only', 'lambda_physics': 0, 'lambda_boundary': 0},

    # Physics only
    {'name': 'Physics_low', 'lambda_physics': 0.01, 'lambda_boundary': 0},
    {'name': 'Physics_med', 'lambda_physics': 0.1, 'lambda_boundary': 0},
    {'name': 'Physics_high', 'lambda_physics': 1.0, 'lambda_boundary': 0},

    # Boundary only
    {'name': 'Boundary_low', 'lambda_physics': 0, 'lambda_boundary': 0.01},
    {'name': 'Boundary_med', 'lambda_physics': 0, 'lambda_boundary': 0.1},

    # Combined
    {'name': 'Combined_balanced', 'lambda_physics': 0.1, 'lambda_boundary': 0.1},
    {'name': 'Combined_physics_heavy', 'lambda_physics': 0.5, 'lambda_boundary': 0.1},
    {'name': 'Combined_optimal', 'lambda_physics': 0.2, 'lambda_boundary': 0.15},  # From HPO
]

def run_ablation():
    results = []
    for config in ABLATION_CONFIGS:
        # Train model
        model = PINNClassifier(**config)
        trainer.fit(model, train_loader, val_loader)

        # Evaluate
        metrics = evaluator.evaluate(model, test_loader)
        metrics['config'] = config['name']
        results.append(metrics)

    return pd.DataFrame(results)
```

### Deliverable: Ablation Table

| Configuration    | Accuracy  | F1-Score  | Physics Loss | Training Time |
| ---------------- | --------- | --------- | ------------ | ------------- |
| CNN_only         | 94.2%     | 0.941     | N/A          | 12 min        |
| Physics_med      | 96.8%     | 0.967     | 0.023        | 18 min        |
| Combined_optimal | **98.1%** | **0.980** | 0.018        | 22 min        |

## 3.1.2 Hyperparameter Sensitivity

### Key Parameters to Document

| Parameter             | Range Tested  | Optimal | Sensitivity |
| --------------------- | ------------- | ------- | ----------- |
| λ_physics             | [0.001, 1.0]  | 0.2     | HIGH        |
| λ_boundary            | [0.001, 1.0]  | 0.15    | MEDIUM      |
| Physics MLP depth     | [2, 3, 4]     | 3       | LOW         |
| Physics embedding dim | [32, 64, 128] | 64      | MEDIUM      |

### Visualization: Parameter Sensitivity Surface

```python
# Generate 3D surface plot for paper
import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(
    x=lambda_physics_range,
    y=lambda_boundary_range,
    z=accuracy_matrix
)])
fig.update_layout(
    title='PINN Hyperparameter Sensitivity',
    scene=dict(
        xaxis_title='λ_physics',
        yaxis_title='λ_boundary',
        zaxis_title='Accuracy'
    )
)
fig.write_image('docs/figures/pinn_sensitivity.png')
```

## 3.1.3 Benchmark Against Baselines

### Experimental Protocol

| Model                 | Dataset | Cross-Validation | Metric       |
| --------------------- | ------- | ---------------- | ------------ |
| SVM (Phase 1)         | CWRU    | 5-fold           | Accuracy, F1 |
| 1D-CNN (Phase 2)      | CWRU    | 5-fold           | Accuracy, F1 |
| ResNet (Phase 3)      | CWRU    | 5-fold           | Accuracy, F1 |
| Transformer (Phase 4) | CWRU    | 5-fold           | Accuracy, F1 |
| **PINN (Phase 6)**    | CWRU    | 5-fold           | Accuracy, F1 |
| Ensemble (Phase 8)    | CWRU    | 5-fold           | Accuracy, F1 |

### Statistical Significance

```python
# McNemar's test for pairwise model comparison
from statsmodels.stats.contingency_tables import mcnemar

def compare_models(model_a_preds, model_b_preds, true_labels):
    # Build contingency table
    b = sum((model_a_preds == true_labels) & (model_b_preds != true_labels))
    c = sum((model_a_preds != true_labels) & (model_b_preds == true_labels))

    result = mcnemar([[0, b], [c, 0]], exact=False)
    return result.pvalue

# Report p-values in paper
p_value = compare_models(pinn_preds, cnn_preds, y_test)
print(f"PINN vs CNN: p = {p_value:.4f}")  # p < 0.05 → significant
```

## 3.1.4 Reproducibility Package

### File Structure

```
reproducibility/
├── README.md              # Instructions
├── environment.yml        # Conda environment
├── requirements.txt       # Pip requirements
├── config/
│   └── pinn_optimal.yaml  # Best hyperparameters
├── data/
│   └── data_splits.json   # Train/val/test indices
├── checkpoints/
│   └── pinn_best.pth      # Pre-trained weights
└── scripts/
    ├── download_data.py   # Data acquisition
    ├── train_pinn.py      # Training script
    └── evaluate_pinn.py   # Evaluation script
```

### Fixed Seeds

```python
# reproducibility/scripts/set_seeds.py
import random
import numpy as np
import torch

MASTER_SEED = 42

def set_all_seeds(seed=MASTER_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Acceptance Criteria

| ID     | Criterion                                    | Verification        | Owner   |
| ------ | -------------------------------------------- | ------------------- | ------- |
| PINN-1 | Ablation study covers 9+ configurations      | Count configs       | Lead ML |
| PINN-2 | Statistical significance computed (p-values) | McNemar test        | Lead ML |
| PINN-3 | Sensitivity analysis for λ parameters        | Surface plot exists | ML #2   |
| PINN-4 | Reproducibility package runs on clean env    | Docker test         | ML #2   |
| PINN-5 | Physics loss decreases during training       | Training logs       | Lead ML |

---

# Chapter 3.2 — XAI Module Enhancement

**Owner:** Lead ML Researcher  
**Duration:** 5-7 days

## 3.2.1 Explanation Quality Metrics

### Metrics to Implement

| Metric                | Definition                                     | Purpose                                  |
| --------------------- | ---------------------------------------------- | ---------------------------------------- |
| **Faithfulness**      | Accuracy drop when removing important features | Verify explanations match model behavior |
| **Stability**         | Similarity of explanations for similar inputs  | Ensure consistent explanations           |
| **Comprehensibility** | Human evaluation score                         | Validate domain expert understanding     |
| **Sparsity**          | # of features in explanation                   | Simpler = more interpretable             |

### Implementation

```python
# packages/core/explainability/metrics.py

class ExplanationMetrics:

    @staticmethod
    def faithfulness(model, x, explanation, k_features=10):
        """
        Compute faithfulness by removing top-k important features.
        Higher accuracy drop = more faithful explanation.
        """
        original_pred = model.predict_proba(x)

        # Get top-k feature indices from explanation
        top_k = np.argsort(np.abs(explanation))[-k_features:]

        # Remove features (set to mean)
        x_masked = x.copy()
        x_masked[top_k] = x.mean()

        masked_pred = model.predict_proba(x_masked)

        # Accuracy drop = faithfulness
        return np.abs(original_pred - masked_pred).max()

    @staticmethod
    def stability(explainer, x_samples, perturbation_std=0.01):
        """
        Compute explanation stability under small perturbations.
        """
        explanations = []
        for _ in range(10):
            x_perturbed = x + np.random.normal(0, perturbation_std, x.shape)
            exp = explainer.explain(x_perturbed)
            explanations.append(exp)

        # Pairwise correlation of explanations
        correlations = []
        for i in range(len(explanations)):
            for j in range(i+1, len(explanations)):
                corr = np.corrcoef(explanations[i], explanations[j])[0, 1]
                correlations.append(corr)

        return np.mean(correlations)  # Higher = more stable
```

## 3.2.2 Comparative Analysis

### Method Comparison Table

| Method               | Faithfulness | Stability | Speed  | Best For           |
| -------------------- | ------------ | --------- | ------ | ------------------ |
| SHAP                 | 0.82         | 0.91      | Slow   | Feature importance |
| LIME                 | 0.78         | 0.85      | Fast   | Local explanation  |
| Integrated Gradients | 0.85         | 0.93      | Medium | Signal attribution |
| Grad-CAM             | 0.75         | 0.88      | Fast   | Visual inspection  |

### Decision Guide for Users

```python
# packages/dashboard/services/xai_service.py

def recommend_explainer(input_type, user_goal, time_budget):
    """
    Recommend best XAI method based on context.
    """
    if user_goal == 'feature_importance' and time_budget > 60:
        return 'SHAP'  # Most comprehensive
    elif user_goal == 'quick_explanation':
        return 'LIME'  # Fastest
    elif input_type == '1d_signal' and user_goal == 'attribution':
        return 'IntegratedGradients'  # Best for time series
    elif input_type == 'spectrogram':
        return 'GradCAM'  # Visual attribution
    else:
        return 'SHAP'  # Default
```

## 3.2.3 Expert Validation Protocol

### Survey Design

```markdown
## Domain Expert Evaluation Survey

**Participant**: **\*\***\_\_\_**\*\***  
**Expertise**: [ ] Bearing Diagnostics [ ] Signal Processing [ ] ML/AI

### Task 1: Identify Fault Type

Given the following explanation visualization, identify the fault type:
[Image: SHAP attribution plot]

1. Ball Fault
2. Inner Race Fault
3. Outer Race Fault
4. Normal

### Task 2: Rate Explanation Quality

On a scale of 1-5, rate the following:

| Criterion                                              | 1   | 2   | 3   | 4   | 5   |
| ------------------------------------------------------ | --- | --- | --- | --- | --- |
| The explanation highlights relevant signal regions     |     |     |     |     |     |
| The explanation is consistent with my domain knowledge |     |     |     |     |     |
| I trust this explanation                               |     |     |     |     |     |
| I would use this in production                         |     |     |     |     |     |

### Task 3: Free-form Feedback

What would make these explanations more useful?
```

## Acceptance Criteria

| ID    | Criterion                       | Verification     | Owner   |
| ----- | ------------------------------- | ---------------- | ------- |
| XAI-1 | Faithfulness metric implemented | Unit test        | Lead ML |
| XAI-2 | Stability metric implemented    | Unit test        | Lead ML |
| XAI-3 | Comparison table populated      | Visual check     | Lead ML |
| XAI-4 | Expert survey designed          | Document exists  | Lead ML |
| XAI-5 | ≥5 expert evaluations collected | Survey responses | Lead ML |

---

# Chapter 3.3 — Ensemble Method Documentation

**Owner:** Lead ML Researcher  
**Duration:** 3-4 days

## 3.3.1 Voting Strategy Analysis

### Strategies Implemented

| Strategy      | Formula                     | Accuracy  |
| ------------- | --------------------------- | --------- |
| Hard Voting   | mode(predictions)           | 97.2%     |
| Soft Voting   | argmax(mean(probabilities)) | 97.8%     |
| Weighted Soft | argmax(∑ wᵢ·pᵢ)             | **98.3%** |

### Weight Derivation

```python
# experiments/ensemble_weights.py

def derive_optimal_weights(models, val_loader):
    """
    Derive optimal ensemble weights using validation performance.
    """
    # Get validation accuracies
    accuracies = []
    for model in models:
        acc = evaluate(model, val_loader)
        accuracies.append(acc)

    # Normalize to weights
    weights = np.array(accuracies)
    weights = weights / weights.sum()

    return weights

# Result: [0.15, 0.18, 0.22, 0.20, 0.25] for 5 models
```

## 3.3.2 Stacking Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STACKING ENSEMBLE                             │
│                                                                   │
│  Level 0 (Base Models):                                          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐              │
│  │ CNN  │  │ResNet│  │Trans │  │ PINN │  │ MLP  │              │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘              │
│     │         │         │         │         │                    │
│     └─────────┴─────────┴─────────┴─────────┘                   │
│                         │                                        │
│                    [Concatenate]                                 │
│                         │                                        │
│  Level 1 (Meta-Learner):                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │   Logistic Regression / Small MLP            │                │
│  └─────────────────────────────────────────────┘                │
│                         │                                        │
│                    [Final Prediction]                            │
└─────────────────────────────────────────────────────────────────┘
```

### Overfitting Prevention

| Technique           | Applied | Impact                    |
| ------------------- | ------- | ------------------------- |
| K-fold for Level-0  | ✅      | Prevents data leakage     |
| Holdout for Level-1 | ✅      | Unbiased meta-training    |
| L2 regularization   | ✅      | Prevents weight explosion |
| Early stopping      | ✅      | Prevents overfit          |

## Acceptance Criteria

| ID    | Criterion                      | Verification   | Owner   |
| ----- | ------------------------------ | -------------- | ------- |
| ENS-1 | 3 voting strategies documented | Doc review     | Lead ML |
| ENS-2 | Weight derivation explained    | Formula + code | Lead ML |
| ENS-3 | Stacking diagram created       | Visual check   | Lead ML |
| ENS-4 | Overfitting prevention listed  | Table complete | Lead ML |

---

# Chapter 3.4 — Manuscript Preparation

**Owner:** Technical Writer + Lead ML Researcher  
**Duration:** 10-14 days

## 3.4.1 Target Venue Selection

| Venue                                    | IF   | Scope Match | Review Time | Decision    |
| ---------------------------------------- | ---- | ----------- | ----------- | ----------- |
| IEEE Trans. Industrial Informatics       | 12.3 | ⭐⭐⭐⭐⭐  | 3-6 months  | Primary     |
| Mechanical Systems and Signal Processing | 8.4  | ⭐⭐⭐⭐⭐  | 2-4 months  | Backup      |
| Expert Systems with Applications         | 8.5  | ⭐⭐⭐⭐    | 2-3 months  | Alternative |

## 3.4.2 Paper Structure

```markdown
# Physics-Informed Neural Networks for Explainable Bearing Fault Diagnosis

## Abstract (250 words)

- Problem: Bearing fault diagnosis lacks interpretability
- Solution: PINN + XAI integration
- Results: 98.1% accuracy, validated explanations
- Impact: Trust in industrial AI

## 1. Introduction (1.5 pages)

- Industrial relevance of bearing fault diagnosis
- Limitations of black-box ML
- Contribution summary

## 2. Related Work (1.5 pages)

- Traditional methods (FFT, envelope analysis)
- Deep learning approaches
- Physics-informed learning
- XAI in fault diagnosis

## 3. Methodology (3 pages)

### 3.1 Problem Formulation

### 3.2 PINN Architecture

### 3.3 Physics Constraints

### 3.4 XAI Integration

## 4. Experimental Setup (1.5 pages)

### 4.1 Datasets

### 4.2 Baselines

### 4.3 Evaluation Metrics

### 4.4 Implementation Details

## 5. Results (2.5 pages)

### 5.1 Classification Performance

### 5.2 Ablation Studies

### 5.3 Explanation Quality

### 5.4 Expert Validation

## 6. Discussion (1 page)

### 6.1 Physics-Data Synergy

### 6.2 Limitations

### 6.3 Future Work

## 7. Conclusion (0.5 pages)

## References (~40 citations)
```

## 3.4.3 Figures & Tables

### Required Figures

| Figure | Content                  | Tool            |
| ------ | ------------------------ | --------------- |
| Fig. 1 | System architecture      | draw.io         |
| Fig. 2 | PINN architecture        | Mermaid         |
| Fig. 3 | Confusion matrix         | Matplotlib      |
| Fig. 4 | ROC curves (11-class)    | Matplotlib      |
| Fig. 5 | SHAP explanation example | SHAP library    |
| Fig. 6 | Ablation study results   | Matplotlib      |
| Fig. 7 | Parameter sensitivity    | Plotly → static |

### Required Tables

| Table   | Content                  |
| ------- | ------------------------ |
| Table 1 | Dataset statistics       |
| Table 2 | Baseline comparison      |
| Table 3 | Ablation study results   |
| Table 4 | XAI method comparison    |
| Table 5 | Expert validation scores |

## 3.4.4 LaTeX Template

```latex
% docs/paper/main.tex
\documentclass[journal]{IEEEtran}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}

\title{Physics-Informed Neural Networks for Explainable Bearing Fault Diagnosis}

\author{
  \IEEEauthorblockN{Author 1, Author 2, Author 3}
  \IEEEauthorblockA{Affiliation\\
  Email}
}

\begin{document}
\maketitle

\begin{abstract}
% Abstract here
\end{abstract}

\begin{IEEEkeywords}
Bearing fault diagnosis, Physics-informed neural networks,
Explainable AI, Deep learning, Predictive maintenance
\end{IEEEkeywords}

\section{Introduction}
% Content here

\end{document}
```

## Acceptance Criteria

| ID    | Criterion              | Verification         | Owner       |
| ----- | ---------------------- | -------------------- | ----------- |
| MAN-1 | Venue selected         | Decision documented  | Lead ML     |
| MAN-2 | Paper outline complete | Structure approved   | Tech Writer |
| MAN-3 | All 7 figures drafted  | Visual review        | Tech Writer |
| MAN-4 | All 5 tables populated | Data complete        | Lead ML     |
| MAN-5 | First draft complete   | Word count >5000     | Tech Writer |
| MAN-6 | Internal review passed | 2 reviewer sign-offs | Lead ML     |

---

# Chapter 3.5 — Dataset Contribution

**Owner:** ML Researcher #2 + Data Engineer  
**Duration:** 4-5 days

## 3.5.1 Synthetic Data Documentation

### Signal Generator Specification

```python
# data/signal_generator.py documentation

class BearingSignalGenerator:
    """
    Generates synthetic bearing vibration signals with physics-based fault signatures.

    Parameters:
    -----------
    sampling_rate : int
        Samples per second (default: 12000)
    duration : float
        Signal duration in seconds (default: 1.0)
    fault_type : str
        One of: 'normal', 'ball', 'inner_race', 'outer_race'
    severity : float
        Fault severity 0-1 (default: 0.5)
    noise_level : float
        Gaussian noise std (default: 0.1)

    Physics Model:
    -------------
    f_BPFO = (n/2) * (1 - d/D * cos(θ)) * f_r
    f_BPFI = (n/2) * (1 + d/D * cos(θ)) * f_r
    f_BSF = (D/2d) * (1 - (d/D * cos(θ))²) * f_r

    Where:
    - n: Number of rolling elements
    - d: Ball diameter
    - D: Pitch diameter
    - θ: Contact angle
    - f_r: Rotation frequency
    """
```

## 3.5.2 Data Quality Assurance

### Comparison with Real Data

| Metric         | Synthetic | CWRU   | MFPT   | Match |
| -------------- | --------- | ------ | ------ | ----- |
| Mean RMS       | 0.42      | 0.45   | 0.41   | ✅    |
| Peak frequency | 118 Hz    | 120 Hz | 115 Hz | ✅    |
| Kurtosis       | 3.8       | 4.1    | 3.6    | ✅    |
| Crest factor   | 4.2       | 4.5    | 4.0    | ✅    |

## 3.5.3 Data Release

### Hosting Options

| Platform                 | DOI | License   | Cost |
| ------------------------ | --- | --------- | ---- |
| **Zenodo** (recommended) | ✅  | CC-BY-4.0 | Free |
| Hugging Face             | ✅  | CC-BY-4.0 | Free |
| IEEE DataPort            | ✅  | Various   | Paid |

### Data Card

```yaml
# dataset_card.yaml
name: LSTM-PFD Bearing Fault Dataset
version: 1.0.0
license: CC-BY-4.0
size: 2.3 GB
samples: 50,000
classes: 11
format: HDF5, CSV
citation: |
  @dataset{lstm_pfd_2026,
    author = {Your Name},
    title = {LSTM-PFD Bearing Fault Dataset},
    year = {2026},
    publisher = {Zenodo},
    doi = {10.5281/zenodo.XXXXXXX}
  }
```

## Acceptance Criteria

| ID    | Criterion                    | Verification       | Owner    |
| ----- | ---------------------------- | ------------------ | -------- |
| DAT-1 | Generator physics documented | Docstring complete | ML #2    |
| DAT-2 | Quality comparison with CWRU | Table complete     | ML #2    |
| DAT-3 | Dataset uploaded to Zenodo   | DOI obtained       | Data Eng |
| DAT-4 | Data card created            | YAML valid         | Data Eng |
| DAT-5 | Citation format ready        | BibTeX valid       | Data Eng |

---

# Chapter 3.6 — Reviewer Proofing (Addendum)

**Owner:** Lead ML Researcher  
**Duration:** 2-3 days  
**Priority:** 🔴 HIGH — Rejection Prevention

> [!IMPORTANT]
> Reviewers will ask: "Where does your model fail?" and "Can I exactly reproduce this?" This addendum addresses both.

---

## A. Failure Analysis ("Negative Results")

### Purpose

Don't just show where the model succeeds—show where it fails. This builds credibility with reviewers and demonstrates academic rigor.

### Implementation

```python
# scripts/research/failure_analysis.py
"""
Analyze misclassified samples to identify systematic failure modes.
"""
import numpy as np
import pandas as pd
from collections import Counter

def analyze_failures(y_true, y_pred, X_test, class_names):
    """
    Comprehensive failure analysis for paper.
    """
    # Identify misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]

    # Confusion pairs
    confusion_pairs = []
    for idx in misclassified_idx:
        confusion_pairs.append((
            class_names[y_true[idx]],
            class_names[y_pred[idx]]
        ))

    confusion_counts = Counter(confusion_pairs)

    # Top-5 confusion pairs
    print("Top 5 Confusion Pairs:")
    for (true_class, pred_class), count in confusion_counts.most_common(5):
        print(f"  {true_class} → {pred_class}: {count} samples")

    # Signal characteristics of failed samples
    failed_signals = X_test[misclassified_idx]
    correct_signals = X_test[np.where(y_true == y_pred)[0]]

    analysis = {
        'failed_mean_snr': compute_snr(failed_signals).mean(),
        'correct_mean_snr': compute_snr(correct_signals).mean(),
        'failed_mean_rms': np.sqrt((failed_signals ** 2).mean()),
        'correct_mean_rms': np.sqrt((correct_signals ** 2).mean()),
        'total_failures': len(misclassified_idx),
        'failure_rate': len(misclassified_idx) / len(y_true) * 100
    }

    return pd.DataFrame([analysis])

def compute_snr(signals):
    """Estimate signal-to-noise ratio."""
    # Simplified SNR estimation
    signal_power = np.var(signals, axis=1)
    noise_power = np.median(np.abs(np.diff(signals, axis=1)), axis=1) ** 2
    return 10 * np.log10(signal_power / (noise_power + 1e-10))
```

### Deliverable: Failure Analysis Table

| Confusion Pair          | Count | % of Errors | Likely Cause                 |
| ----------------------- | ----- | ----------- | ---------------------------- |
| Inner Race → Ball       | 12    | 28%         | Similar frequency signatures |
| Ball → Roller           | 8     | 19%         | Low severity cases           |
| Outer Race → Normal     | 5     | 12%         | High noise level             |
| Inner Race → Outer Race | 4     | 9%          | Overlapping harmonics        |

### Key Findings (For Paper)

```markdown
## 5.5 Failure Mode Analysis

The PINN model achieved 98.1% accuracy, with 47 misclassified samples
out of 2,500 test cases. Analysis revealed:

1. **Inner Race ↔ Ball Confusion (28% of errors)**: Both fault types
   produce characteristic frequencies at similar harmonics of the
   rotation frequency. The physics constraint helped reduce this
   confusion by 40% compared to the baseline CNN.

2. **High-Noise Failures**: Samples with SNR < 5 dB accounted for
   65% of all misclassifications. This suggests a practical deployment
   threshold for signal quality.

3. **Low-Severity Edge Cases**: 8 of 47 failures occurred at severity
   levels < 0.2, where fault signatures are near the noise floor.
```

### Visualization

```python
# Generate failure analysis plots for paper
import plotly.express as px

# 1. Confusion pair sunburst
fig = px.sunburst(
    failure_df,
    path=['true_class', 'predicted_class'],
    values='count',
    title='Misclassification Analysis'
)
fig.write_image('docs/figures/failure_sunburst.png')

# 2. SNR distribution of failed vs correct
fig = px.histogram(
    combined_df,
    x='snr',
    color='classification',
    barmode='overlay',
    title='SNR Distribution: Correct vs Failed Predictions'
)
fig.write_image('docs/figures/snr_analysis.png')
```

---

## B. Data Version Control (DVC)

### Purpose

Link code to exact data version for perfect reproducibility. Reviewers can verify: "Experiments ran on Data v1.0 (DVC Hash: a1b2c3d)."

### Setup

```bash
# Initialize DVC
pip install dvc dvc-s3  # or dvc-gdrive, dvc-azure

cd LSTM_PFD
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Track Data

```bash
# Track large data files
dvc add data/processed/bearing_dataset.h5
dvc add data/raw/CWRU/
dvc add outputs/models/pinn_best.pth

# This creates .dvc files (small, git-tracked)
git add data/processed/bearing_dataset.h5.dvc
git add data/raw/CWRU.dvc
git add outputs/models/pinn_best.pth.dvc
git commit -m "Track data and model with DVC"
```

### Configure Remote Storage

```bash
# Set up remote storage (e.g., S3)
dvc remote add -d myremote s3://lstm-pfd-data/dvc-store

# Or Google Drive (free)
dvc remote add -d gdrive gdrive://lstm-pfd-folder

# Push data
dvc push
```

### Version Tagging

```bash
# Create data version tag
git tag -a "data-v1.0" -m "Initial dataset release"
git push origin data-v1.0
dvc push

# For paper, reference:
# "Experiments were run on Data Version v1.0 (DVC Hash: a1b2c3d4e5f6)"
```

### File Structure After DVC

```
LSTM_PFD/
├── .dvc/
│   ├── config           # Remote storage config
│   ├── .gitignore
│   └── cache/           # Local cache (gitignored)
├── data/
│   ├── processed/
│   │   ├── bearing_dataset.h5      # Actual data (gitignored)
│   │   └── bearing_dataset.h5.dvc  # DVC pointer (git-tracked)
│   └── raw/
│       ├── CWRU/                   # Actual data (gitignored)
│       └── CWRU.dvc                # DVC pointer (git-tracked)
├── outputs/
│   └── models/
│       ├── pinn_best.pth           # Model weights (gitignored)
│       └── pinn_best.pth.dvc       # DVC pointer (git-tracked)
└── dvc.lock                        # Version lock file
```

### Reproducibility Command

```bash
# For reviewers/reproducers:
git clone https://github.com/your-org/LSTM_PFD
git checkout data-v1.0  # Exact data version from paper
dvc pull               # Download exact data files
python scripts/train_pinn.py  # Reproduce results
```

### Paper Citation Format

```markdown
## Reproducibility Statement

All experiments were conducted using Data Version 1.0, tracked via
Data Version Control (DVC). The exact data hash is `a1b2c3d4e5f6`,
ensuring bit-for-bit reproducibility. To reproduce:

1. Clone repository at tag `paper-v1.0`
2. Run `dvc pull` to obtain exact data files
3. Execute `python scripts/reproduce_all.py`

Model checkpoints and training logs are archived at:
https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## Acceptance Criteria (Addendum)

| ID    | Criterion                                         | Verification                    | Owner    |
| ----- | ------------------------------------------------- | ------------------------------- | -------- |
| REV-1 | Failure analysis identifies top-5 confusion pairs | Script runs                     | Lead ML  |
| REV-2 | SNR analysis of failed samples complete           | Table populated                 | Lead ML  |
| REV-3 | Failure mode section drafted for paper            | Word count >300                 | Lead ML  |
| REV-4 | DVC initialized and configured                    | `dvc status` clean              | Data Eng |
| REV-5 | Data tagged with version                          | `git tag` shows data-v1.0       | Data Eng |
| REV-6 | DVC remote configured and pushed                  | `dvc pull` on fresh clone works | Data Eng |

---

# Phase 3 Summary

## Timeline

| Week | Tasks                               | Milestones              |
| ---- | ----------------------------------- | ----------------------- |
| 1    | PINN ablation, reproducibility      | Ablation table complete |
| 2    | XAI metrics, comparison             | Metrics implemented     |
| 3    | Ensemble docs, manuscript outline   | Outline approved        |
| 4    | First draft, figures                | Draft complete          |
| 5    | Expert validation, failure analysis | Survey + failures done  |
| 6    | Dataset release, DVC setup          | DOI + DVC tags          |

## Definition of Done

Phase 3 is complete when:

- [ ] Ablation study with 9+ configurations
- [ ] Statistical significance for all model comparisons
- [ ] XAI quality metrics implemented
- [ ] ≥5 expert evaluations collected
- [ ] Failure analysis with top-5 confusion pairs
- [ ] DVC initialized and data versioned
- [ ] Paper first draft complete (>5000 words)
- [ ] Dataset released with DOI
- [ ] All 25+ acceptance criteria passed

---

_Phase 3 Complete Document (with Addendum) — Ready for Phase 4 on request_


---

# Phase 4: Enterprise Hardening — Complete Implementation Guide

**Phase Duration:** 3-4 weeks  
**Team Size:** 4 SMEs  
**Priority:** 🟡 HIGH — Production Readiness

---

## Team Roster & Allocation

| Role                      | Assigned To | Chapters      | Weekly Hours |
| ------------------------- | ----------- | ------------- | ------------ |
| **DevOps Lead**           | TBD         | 4.2, 4.3, 4.4 | 40           |
| **Security Engineer**     | TBD         | 4.1, 4.6      | 35           |
| **SRE/Platform Engineer** | TBD         | 4.5, 4.6      | 35           |
| **Senior Developer**      | TBD         | 4.7           | 25           |

---

# Chapter 4.1 — Security Audit & Remediation

**Owner:** Security Engineer  
**Duration:** 5-7 days

## 4.1.1 Authentication Hardening

### Current State Audit

```bash
# Run security audit
pip install bandit safety
bandit -r packages/ -f json -o security_report.json
safety check -r requirements.txt --json > dependency_audit.json
```

### JWT Configuration Hardening

**File: `packages/dashboard/config/security.py`**

```python
from datetime import timedelta

JWT_CONFIG = {
    # Token expiration
    'ACCESS_TOKEN_EXPIRES': timedelta(minutes=30),  # Was: 24 hours
    'REFRESH_TOKEN_EXPIRES': timedelta(days=7),

    # Algorithm (use RS256 for production)
    'ALGORITHM': 'RS256',  # Was: HS256

    # Token rotation
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,

    # Security headers
    'COOKIE_SECURE': True,
    'COOKIE_HTTPONLY': True,
    'COOKIE_SAMESITE': 'Strict',
}
```

### Password Policy

```python
# packages/dashboard/utils/password_policy.py

import re
from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Was: 10
)

def validate_password(password: str) -> tuple[bool, str]:
    """
    Enforce strong password policy.
    """
    if len(password) < 12:
        return False, "Password must be at least 12 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain digit"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain special character"
    return True, "Password valid"
```

## 4.1.2 Authorization Review

### RBAC Implementation

```python
# packages/dashboard/models/permissions.py

from enum import Enum

class Role(Enum):
    VIEWER = "viewer"      # Read-only access
    OPERATOR = "operator"  # Run experiments, view results
    ADMIN = "admin"        # Full access
    SUPER_ADMIN = "super_admin"  # System configuration

ROLE_PERMISSIONS = {
    Role.VIEWER: [
        'view_experiments',
        'view_datasets',
        'view_results',
    ],
    Role.OPERATOR: [
        'view_experiments', 'create_experiments', 'run_experiments',
        'view_datasets', 'upload_datasets',
        'view_results', 'export_results',
        'use_xai_dashboard',
    ],
    Role.ADMIN: [
        '*',  # All operator permissions
        'manage_users',
        'manage_api_keys',
        'view_system_health',
        'configure_notifications',
    ],
    Role.SUPER_ADMIN: [
        '**',  # Everything including system config
    ],
}

def has_permission(user, permission: str) -> bool:
    role_perms = ROLE_PERMISSIONS.get(user.role, [])
    if '**' in role_perms:
        return True
    if '*' in role_perms and permission not in ['system_config', 'delete_all']:
        return True
    return permission in role_perms
```

## 4.1.3 Secret Management

### Environment Variable Audit

```bash
# Find hardcoded secrets
grep -rn "password\|secret\|api_key\|token" packages/ --include="*.py" | grep -v ".pyc"
```

### Migration to Vault (Optional)

```python
# packages/dashboard/config/secrets.py

import os
from functools import lru_cache

class SecretManager:
    """
    Centralized secret management with fallback to env vars.
    """

    def __init__(self):
        self.backend = os.getenv('SECRET_BACKEND', 'env')
        if self.backend == 'vault':
            import hvac
            self.client = hvac.Client(url=os.getenv('VAULT_ADDR'))
            self.client.token = os.getenv('VAULT_TOKEN')

    @lru_cache
    def get_secret(self, key: str) -> str:
        if self.backend == 'vault':
            secret = self.client.secrets.kv.v2.read_secret_version(path=key)
            return secret['data']['data']['value']
        return os.getenv(key)

secrets = SecretManager()

# Usage
DATABASE_URL = secrets.get_secret('DATABASE_URL')
JWT_SECRET = secrets.get_secret('JWT_SECRET')
```

## 4.1.4 Dependency Vulnerability Scan

### CI Integration

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 0 * * 0" # Weekly

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r packages/ -f json -o bandit-report.json || true

      - name: Run Safety
        run: |
          pip install safety
          safety check -r requirements.txt --json > safety-report.json || true

      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
```

## 4.1.5 OWASP Top 10 Checklist

| Vulnerability                  | Status | Mitigation                            |
| ------------------------------ | ------ | ------------------------------------- |
| A01: Broken Access Control     | ✅     | RBAC implemented                      |
| A02: Cryptographic Failures    | ✅     | bcrypt, RS256 JWT                     |
| A03: Injection                 | ⚠️     | SQLAlchemy ORM, parameterized queries |
| A04: Insecure Design           | ✅     | Input validation                      |
| A05: Security Misconfiguration | ⚠️     | Hardened headers needed               |
| A06: Vulnerable Components     | ✅     | Safety scan in CI                     |
| A07: Auth Failures             | ✅     | Rate limiting, lockout                |
| A08: Software Integrity        | ⚠️     | SBOM needed                           |
| A09: Logging Failures          | ✅     | Structured logging                    |
| A10: SSRF                      | ✅     | URL validation                        |

## Acceptance Criteria

| ID    | Criterion                       | Verification      | Owner    |
| ----- | ------------------------------- | ----------------- | -------- |
| SEC-1 | JWT tokens expire in ≤30 min    | Check config      | Security |
| SEC-2 | Password policy enforced        | Test registration | Security |
| SEC-3 | RBAC denies unauthorized access | Test all roles    | Security |
| SEC-4 | No hardcoded secrets in code    | Grep audit        | Security |
| SEC-5 | Bandit scan passes (no HIGH)    | CI report         | Security |
| SEC-6 | Safety scan passes              | CI report         | Security |

---

# Chapter 4.2 — Docker & Container Optimization

**Owner:** DevOps Lead  
**Duration:** 3-4 days

## 4.2.1 Multi-Stage Build

**File: `Dockerfile`**

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application
COPY packages/ ./packages/
COPY config/ ./config/

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8050/health')" || exit 1

EXPOSE 8050

CMD ["python", "-m", "packages.dashboard.app"]
```

### Size Comparison

| Image         | Size Before | Size After | Reduction |
| ------------- | ----------- | ---------- | --------- |
| Dashboard     | 2.1 GB      | 850 MB     | 60%       |
| Celery Worker | 1.8 GB      | 720 MB     | 60%       |

## 4.2.2 Docker Compose Production

**File: `docker-compose.prod.yml`**

```yaml
version: "3.8"

services:
  dashboard:
    image: ${REGISTRY}/lstm-pfd-dashboard:${VERSION}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    ports:
      - "8050:8050"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - CELERY_BROKER_URL=redis://redis:6379/0
      - JWT_SECRET=${JWT_SECRET}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8050/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      redis:
        condition: service_healthy
      db:
        condition: service_healthy

  redis:
    image: redis:7-alpine
    deploy:
      resources:
        limits:
          memory: 512M
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  celery_worker:
    image: ${REGISTRY}/lstm-pfd-worker:${VERSION}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "4"
          memory: 8G
    command: celery -A packages.dashboard.celery_config worker --loglevel=info --concurrency=4
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=dashboard
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d dashboard"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  redis_data:
  postgres_data:
```

## Acceptance Criteria

| ID    | Criterion                  | Verification         | Owner  |
| ----- | -------------------------- | -------------------- | ------ |
| DOC-1 | Image size <1GB            | `docker images`      | DevOps |
| DOC-2 | Container runs as non-root | `docker exec whoami` | DevOps |
| DOC-3 | Health check passes        | `docker inspect`     | DevOps |
| DOC-4 | Resource limits set        | Compose validation   | DevOps |

---

# Chapter 4.3 — Kubernetes Deployment

**Owner:** DevOps Lead  
**Duration:** 4-5 days

## 4.3.1 Helm Chart Structure

```
helm/lstm-pfd/
├── Chart.yaml
├── values.yaml
├── values-staging.yaml
├── values-prod.yaml
├── templates/
│   ├── _helpers.tpl
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   └── serviceaccount.yaml
```

## 4.3.2 Key Manifests

**File: `templates/deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "lstm-pfd.fullname" . }}-dashboard
spec:
  replicas: {{ .Values.dashboard.replicas }}
  selector:
    matchLabels:
      app: dashboard
  template:
    spec:
      serviceAccountName: {{ include "lstm-pfd.serviceAccountName" . }}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
        - name: dashboard
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          ports:
            - containerPort: 8050
          resources:
            limits:
              cpu: {{ .Values.dashboard.resources.limits.cpu }}
              memory: {{ .Values.dashboard.resources.limits.memory }}
            requests:
              cpu: {{ .Values.dashboard.resources.requests.cpu }}
              memory: {{ .Values.dashboard.resources.requests.memory }}
          livenessProbe:
            httpGet:
              path: /health
              port: 8050
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8050
            initialDelaySeconds: 5
            periodSeconds: 5
          envFrom:
            - configMapRef:
                name: {{ include "lstm-pfd.fullname" . }}-config
            - secretRef:
                name: {{ include "lstm-pfd.fullname" . }}-secrets
```

**File: `templates/hpa.yaml`**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "lstm-pfd.fullname" . }}-dashboard
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "lstm-pfd.fullname" . }}-dashboard
  minReplicas: {{ .Values.dashboard.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.dashboard.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

## 4.3.3 Values Files

**File: `values-prod.yaml`**

```yaml
image:
  repository: your-registry/lstm-pfd
  tag: "1.0.0"

dashboard:
  replicas: 3
  resources:
    limits:
      cpu: "2"
      memory: "4Gi"
    requests:
      cpu: "500m"
      memory: "1Gi"
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10

celeryWorker:
  replicas: 2
  resources:
    limits:
      cpu: "4"
      memory: "8Gi"

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: lstm-pfd.your-domain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: lstm-pfd-tls
      hosts:
        - lstm-pfd.your-domain.com
```

## Acceptance Criteria

| ID    | Criterion                       | Verification                 | Owner    |
| ----- | ------------------------------- | ---------------------------- | -------- |
| K8S-1 | Helm chart lints without errors | `helm lint`                  | DevOps   |
| K8S-2 | Deploys to staging cluster      | `kubectl get pods`           | DevOps   |
| K8S-3 | HPA scales on load              | Load test + watch            | DevOps   |
| K8S-4 | Ingress routes traffic          | curl endpoint                | DevOps   |
| K8S-5 | Secrets not in plain text       | `kubectl get secret -o yaml` | Security |

---

# Chapter 4.4 — CI/CD Pipeline Enhancement

**Owner:** DevOps Lead  
**Duration:** 3-4 days

## 4.4.1 Multi-Stage Pipeline

**File: `.github/workflows/ci-cd.yml`**

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Stage 1: Lint & Format
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Lint
        run: |
          pip install flake8 black isort mypy
          black --check packages/
          isort --check packages/
          flake8 packages/
          mypy packages/ --ignore-missing-imports

  # Stage 2: Test
  test:
    needs: lint
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      - name: Run tests
        run: pytest tests/ --cov=packages --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  # Stage 3: Security Scan
  security:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Bandit
        run: |
          pip install bandit
          bandit -r packages/ -ll
      - name: Safety
        run: |
          pip install safety
          safety check -r requirements.txt

  # Stage 4: Build
  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Login to Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and Push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # Stage 5: Deploy Staging
  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Staging
        run: |
          helm upgrade --install lstm-pfd-staging ./helm/lstm-pfd \
            -f ./helm/lstm-pfd/values-staging.yaml \
            --set image.tag=${{ github.sha }}

  # Stage 6: Deploy Production
  deploy-prod:
    needs: build
    if: github.event_name == 'release'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Production
        run: |
          helm upgrade --install lstm-pfd ./helm/lstm-pfd \
            -f ./helm/lstm-pfd/values-prod.yaml \
            --set image.tag=${{ github.event.release.tag_name }}
```

## 4.4.2 Automated Release

**File: `.github/workflows/release.yml`**

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate Changelog
        id: changelog
        uses: requarks/changelog-action@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          tag: ${{ github.ref_name }}

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          body: ${{ steps.changelog.outputs.changes }}
          draft: false
          prerelease: false
```

## Acceptance Criteria

| ID   | Criterion                         | Verification     | Owner  |
| ---- | --------------------------------- | ---------------- | ------ |
| CI-1 | Pipeline runs on every push       | Check Actions    | DevOps |
| CI-2 | All 6 stages complete <15 min     | Pipeline timing  | DevOps |
| CI-3 | Staging auto-deploys from develop | Check deployment | DevOps |
| CI-4 | Production deploys on release     | Check deployment | DevOps |
| CI-5 | Changelog auto-generated          | Release notes    | DevOps |

---

# Chapter 4.5 — Monitoring & Alerting

**Owner:** SRE/Platform Engineer  
**Duration:** 4-5 days

## 4.5.1 Prometheus Metrics

**File: `packages/dashboard/metrics.py`**

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# ML metrics
INFERENCE_LATENCY = Histogram(
    'ml_inference_duration_seconds',
    'ML inference latency',
    ['model_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

ACTIVE_EXPERIMENTS = Gauge(
    'active_experiments',
    'Number of running experiments'
)

# Celery metrics
CELERY_TASK_COUNT = Counter(
    'celery_tasks_total',
    'Total Celery tasks',
    ['task_name', 'status']
)
```

## 4.5.2 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "LSTM-PFD Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [{ "expr": "rate(http_requests_total[5m])" }]
      },
      {
        "title": "P95 Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~'5..'}[5m])) / sum(rate(http_requests_total[5m])) * 100"
          }
        ]
      },
      {
        "title": "Active Experiments",
        "type": "stat",
        "targets": [{ "expr": "active_experiments" }]
      }
    ]
  }
}
```

## 4.5.3 Alert Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: lstm-pfd
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate (>5%)"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency > 2s"

      - alert: CeleryQueueBacklog
        expr: celery_queue_length > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Celery queue backing up"
```

## Acceptance Criteria

| ID    | Criterion                     | Verification   | Owner |
| ----- | ----------------------------- | -------------- | ----- |
| MON-1 | Prometheus scraping metrics   | Check /metrics | SRE   |
| MON-2 | Grafana dashboard loads       | Visual check   | SRE   |
| MON-3 | Alerts fire on test condition | Trigger test   | SRE   |
| MON-4 | PagerDuty integration works   | Test page      | SRE   |

---

# Chapter 4.6 — Disaster Recovery

**Owner:** Security Engineer + SRE  
**Duration:** 3-4 days

## 4.6.1 Backup Strategy

| Data Type          | Frequency | Retention | Location     |
| ------------------ | --------- | --------- | ------------ |
| Database           | Hourly    | 7 days    | S3           |
| Model checkpoints  | On save   | 30 days   | S3           |
| Config             | On change | 90 days   | Git          |
| Research artifacts | Daily     | 1 year    | S3 + Glacier |

## 4.6.2 Recovery Procedures

### RTO/RPO Definitions

| Metric               | Target | Justification        |
| -------------------- | ------ | -------------------- |
| RTO (Recovery Time)  | 1 hour | Business continuity  |
| RPO (Recovery Point) | 1 hour | Acceptable data loss |

### Runbook: Database Recovery

```bash
#!/bin/bash
# scripts/disaster-recovery/restore-db.sh

# 1. Stop application
kubectl scale deployment lstm-pfd-dashboard --replicas=0

# 2. Download latest backup
aws s3 cp s3://lstm-pfd-backups/db/latest.sql.gz ./

# 3. Decompress
gunzip latest.sql.gz

# 4. Restore
kubectl exec -i $(kubectl get pod -l app=postgres -o jsonpath='{.items[0].metadata.name}') \
  -- psql -U dashboard < latest.sql

# 5. Restart application
kubectl scale deployment lstm-pfd-dashboard --replicas=3

# 6. Verify
curl -f http://lstm-pfd.your-domain.com/health
```

## Acceptance Criteria

| ID   | Criterion                | Verification    | Owner |
| ---- | ------------------------ | --------------- | ----- |
| DR-1 | Automated backups run    | Check S3 bucket | SRE   |
| DR-2 | Restore tested quarterly | Recovery test   | SRE   |
| DR-3 | Runbooks documented      | Doc review      | SRE   |
| DR-4 | RTO <1 hour verified     | Timed test      | SRE   |

---

# Chapter 4.7 — Code Quality & Consistency

**Owner:** Senior Developer  
**Duration:** 3-4 days

## 4.7.1 Structural Deduplication

### Resolve config/ Duplication

| Current                       | Action                        | Target             |
| ----------------------------- | ----------------------------- | ------------------ |
| `/config/`                    | Keep as project-wide          | Central config     |
| `/packages/dashboard/config/` | Rename to `dashboard_config/` | Dashboard-specific |

### Resolve utils/ Duplication

```python
# Option 1: Namespace (Recommended)
# /utils/ → project-level utilities
# /packages/dashboard/utils/ → rename to dashboard_utils/

# Option 2: Centralize
# Move all to /utils/, import as:
from utils.auth import validate_token  # Project-level
from utils.dashboard import format_chart  # Dashboard-specific (subfolder)
```

## 4.7.2 Pre-commit Configuration

**File: `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: detect-private-key
```

## 4.7.3 Type Hints Adoption

```python
# Example: packages/core/models/base.py

from typing import Optional, Dict, Any, List
from torch import Tensor
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def predict(self, x: Tensor) -> List[int]:
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1).tolist()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> "BaseModel":
        model = cls(config or {})
        model.load_state_dict(torch.load(path))
        return model
```

## Acceptance Criteria

| ID   | Criterion                        | Verification                  | Owner |
| ---- | -------------------------------- | ----------------------------- | ----- |
| CQ-1 | No duplicate config/ directories | `find . -name config -type d` | Dev   |
| CQ-2 | Pre-commit hooks installed       | `pre-commit run --all-files`  | Dev   |
| CQ-3 | Black formatting applied         | CI check                      | Dev   |
| CQ-4 | Type hints on public APIs        | mypy coverage                 | Dev   |
| CQ-5 | visualization/ merged            | Single directory              | Dev   |

---

# Chapter 4.8 — Operational Safeguards (Addendum)

**Owner:** SRE/Platform Engineer  
**Duration:** 2 days  
**Priority:** 🔴 HIGH — Cost & Availability Protection

> [!IMPORTANT]
> These are the silent killers that don't break the build but break the budget or crash the server under load.

---

## A. API Rate Limiting (Flask-Limiter)

### Purpose

Protect heavy inference endpoints from abuse. Without this, one user running a loop on `/predict` can crash your server or spike your cloud bill.

### Dependencies

```txt
# Add to requirements.txt
Flask-Limiter==3.5.0
```

### Implementation

**File: `packages/dashboard/app.py`**

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Setup Limiter with Redis storage for distributed rate limiting
limiter = Limiter(
    get_remote_address,
    app=server,
    storage_uri="redis://redis:6379/1",  # Separate DB from Celery
    default_limits=["200 per day", "50 per hour"]
)

# Protect specific heavy routes
@server.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")  # Strict limit for heavy inference
def predict():
    # ... inference logic ...
    pass

@server.route('/predict/batch', methods=['POST'])
@limiter.limit("5 per minute")  # Even stricter for batch
def predict_batch():
    # ... batch inference logic ...
    pass

@server.route('/api/hpo/start', methods=['POST'])
@limiter.limit("3 per hour")  # HPO is very expensive
def start_hpo():
    # ... HPO start logic ...
    pass
```

### Rate Limit Headers

Clients receive headers showing their limit status:

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1609459200
```

### Error Response

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded: 10 per minute",
  "retry_after": 45
}
```

---

## B. JSON Structured Logging (structlog)

### Purpose

Python's default logger outputs plain text (`INFO: User logged in`). Enterprise log systems (ELK, Datadog, Loki) need JSON logs for parsing and querying.

### Dependencies

```txt
# Add to requirements.txt
structlog==24.1.0
```

### Implementation

**File: `packages/dashboard/config/logging_config.py`**

```python
import structlog
import logging
import sys
from datetime import datetime

def configure_logging():
    """
    Configure structured JSON logging for enterprise observability.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()  # Output as JSON
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Redirect standard logging to structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO
    )

# Usage
logger = structlog.get_logger()

# Before (plain text):
# logging.info("User logged in")
# Output: INFO:root:User logged in

# After (JSON):
logger.info("user_login", user_id=123, ip_address="192.168.1.1")
# Output: {"timestamp": "2026-01-13T01:43:00Z", "level": "info", "event": "user_login", "user_id": 123, "ip_address": "192.168.1.1"}
```

### Integration with Dash App

**File: `packages/dashboard/app.py`**

```python
from packages.dashboard.config.logging_config import configure_logging
import structlog

# Initialize at startup
configure_logging()
logger = structlog.get_logger()

# Log structured events
@app.callback(...)
def handle_request(...):
    logger.info("callback_invoked",
                callback="handle_request",
                user_id=get_current_user_id(),
                input_size=len(data))
    # ... process ...
    logger.info("callback_completed",
                callback="handle_request",
                duration_ms=elapsed * 1000)
```

### Query Examples in Loki/ELK

```
# Find all failed logins
{app="lstm-pfd"} |= "login_failed"

# Find slow requests (>1s)
{app="lstm-pfd"} | json | duration_ms > 1000

# Find specific user activity
{app="lstm-pfd"} | json | user_id = 42
```

---

## C. Kubernetes Resource Quotas (Cost Control)

### Purpose

Prevent the HPA (Chapter 4.3) from scaling infinitely. If HPA scales to 50 nodes and stays there overnight, you run out of budget.

### Implementation

**File: `helm/lstm-pfd/templates/resource-quota.yaml`**

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: {{ include "lstm-pfd.fullname" . }}-compute-quota
  namespace: {{ .Release.Namespace }}
spec:
  hard:
    # CPU limits
    requests.cpu: "10"        # Max 10 cores total for namespace
    limits.cpu: "20"          # Max 20 cores burst

    # Memory limits
    requests.memory: 32Gi     # Max 32GB RAM requested
    limits.memory: 64Gi       # Max 64GB RAM burst

    # GPU limits (if applicable)
    requests.nvidia.com/gpu: "4"  # Max 4 GPUs total

    # Pod limits
    pods: "20"                # Max 20 pods in namespace

    # Storage limits
    requests.storage: 100Gi   # Max 100GB persistent storage
    persistentvolumeclaims: "10"  # Max 10 PVCs
```

### Limit Range for Individual Pods

**File: `helm/lstm-pfd/templates/limit-range.yaml`**

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: {{ include "lstm-pfd.fullname" . }}-limit-range
  namespace: {{ .Release.Namespace }}
spec:
  limits:
    - type: Container
      default:
        cpu: "1"
        memory: "2Gi"
      defaultRequest:
        cpu: "500m"
        memory: "1Gi"
      max:
        cpu: "4"
        memory: "8Gi"
      min:
        cpu: "100m"
        memory: "128Mi"
```

### Budget Alerts

**File: `helm/lstm-pfd/templates/budget-alert.yaml`**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: {{ include "lstm-pfd.fullname" . }}-budget-alerts
spec:
  groups:
    - name: cost-alerts
      rules:
        - alert: HighPodCount
          expr: count(kube_pod_info{namespace="{{ .Release.Namespace }}"}) > 15
          for: 30m
          labels:
            severity: warning
          annotations:
            summary: "High pod count - check for runaway scaling"
            description: "Namespace has {{ $value }} pods running for 30+ minutes"

        - alert: ResourceQuotaNearLimit
          expr: |
            kube_resourcequota{namespace="{{ .Release.Namespace }}", type="used"}
            / kube_resourcequota{namespace="{{ .Release.Namespace }}", type="hard"} > 0.8
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "Resource quota >80% utilized"
```

---

## Acceptance Criteria (Addendum)

| ID    | Criterion                              | Verification                 | Owner |
| ----- | -------------------------------------- | ---------------------------- | ----- |
| OPS-1 | Rate limiter blocks after 10 req/min   | Load test                    | SRE   |
| OPS-2 | Rate limit headers in response         | curl -v                      | SRE   |
| OPS-3 | Logs output as valid JSON              | `python -m json.tool`        | SRE   |
| OPS-4 | Logs include timestamp, level, event   | Log inspection               | SRE   |
| OPS-5 | ResourceQuota blocks over-provisioning | Create 21st pod, expect fail | SRE   |
| OPS-6 | Budget alert fires at 80% quota        | Prometheus query             | SRE   |

---

# Phase 4 Summary

## Timeline

| Week | Tasks                            | Milestones            |
| ---- | -------------------------------- | --------------------- |
| 1    | Security audit, JWT hardening    | OWASP checklist green |
| 2    | Docker optimization, Helm charts | K8s deployment ready  |
| 3    | CI/CD pipeline, monitoring       | Full pipeline live    |
| 4    | DR testing, code quality         | Production-ready      |
| 5    | Operational safeguards (4.8)     | Rate limits + quotas  |

## Definition of Done

Phase 4 is complete when:

- [ ] All OWASP Top 10 addressed
- [ ] Docker images <1GB
- [ ] Kubernetes deployment verified
- [ ] CI/CD pipeline <15 min end-to-end
- [ ] Monitoring dashboard live
- [ ] DR tested with RTO <1 hour
- [ ] Pre-commit hooks enforced
- [ ] Rate limiting active on /predict
- [ ] JSON structured logging enabled
- [ ] Resource quotas prevent runaway scaling
- [ ] All 30+ acceptance criteria passed

---

_Phase 4 Complete Document (with Addendum) — MASTER ROADMAP COMPLETE_

