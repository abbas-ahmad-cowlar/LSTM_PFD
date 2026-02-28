# IDB 2.4 - Async Tasks Best Practices

**Curator**: AI Agent  
**Date**: 2026-01-23  
**Source**: `packages/dashboard/tasks/` (11 task modules, 23 Celery tasks)

---

## 1. Task Definition Patterns

### 1.1 Use Bound Tasks for State Access

**Pattern**: Always use `bind=True` to access the task instance for progress updates and task metadata.

```python
@celery_app.task(bind=True)
def train_model_task(self, config: dict):
    task_id = self.request.id  # Access task ID
    self.update_state(state='PROGRESS', meta={'progress': 0.5})
    ...
```

**Why**: Enables real-time progress tracking, task metadata access, and proper state management.

---

### 1.2 Typed Configuration Parameters

**Pattern**: Accept configuration as typed dictionaries with clear parameter documentation.

```python
@celery_app.task(bind=True)
def generate_dataset_task(self, config: dict):
    """
    Celery task for dataset generation.

    Args:
        config: Generation configuration
            - name: Dataset name
            - generation_id: Database record ID
            - num_signals_per_fault: Signals per fault type
            - fault_types: List of fault types to generate
            - output_format: 'mat', 'hdf5', or 'both'
            - random_seed: Random seed for reproducibility
    """
```

**Why**: Self-documenting API, flexible extensibility, single serialization point.

---

### 1.3 Centralized Celery App with Autodiscovery

**Pattern**: Initialize Celery once with explicit module inclusion.

```python
# tasks/__init__.py
celery_app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'tasks.training_tasks',
        'tasks.hpo_tasks',
        'tasks.deployment_tasks',
        # ... all task modules
    ]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
```

**Why**: Predictable task discovery, centralized configuration, security (JSON-only).

---

### 1.4 Eager Mode for Testing

**Pattern**: Support eager execution via environment variable for local development.

```python
if os.getenv('CELERY_ALWAYS_EAGER', 'False').lower() == 'true':
    celery_app.conf.update(task_always_eager=True)
```

**Why**: Run tasks synchronously during development/testing without worker infrastructure.

---

## 2. Result Handling Conventions

### 2.1 Consistent Return Structure

**Pattern**: All tasks return a dictionary with `success` boolean and contextual data.

```python
# Success case
return {
    "success": True,
    "experiment_id": experiment_id,
    "metrics": {"accuracy": 0.95, "loss": 0.12},
    "duration_seconds": 3600,
    "task_id": task_id
}

# Failure case
return {
    "success": False,
    "error": str(e),
    "traceback": traceback.format_exc(),
    "task_id": task_id
}
```

**Why**: Uniform API for callers, predictable error handling, debugging support.

---

### 2.2 Progress Metadata Updates

**Pattern**: Use `self.update_state()` with structured progress metadata.

```python
self.update_state(
    state='PROGRESS',
    meta={
        'progress': progress,      # 0-100 or 0.0-1.0
        'current': current,        # Current item
        'total': total,            # Total items
        'status': 'Processing...'  # Human-readable status
    }
)
```

**Why**: Real-time progress in UI, debugging visibility, user feedback.

---

### 2.3 Database Status Synchronization

**Pattern**: Update database entity status alongside task state for persistence.

```python
# On task start
with get_db_session() as session:
    experiment = session.query(Experiment).filter_by(id=experiment_id).first()
    if experiment:
        experiment.status = ExperimentStatus.RUNNING
        experiment.config["celery_task_id"] = task_id
        session.commit()

# On task completion
with get_db_session() as session:
    experiment = session.query(Experiment).filter_by(id=experiment_id).first()
    if experiment:
        experiment.status = ExperimentStatus.COMPLETED
        experiment.metrics = results
        session.commit()
```

**Why**: Task state survives worker restarts, queryable history, UI consistency.

---

## 3. Retry and Error Handling Patterns

### 3.1 Comprehensive Try/Except with Logging

**Pattern**: Wrap entire task body with exception handling and structured logging.

```python
@celery_app.task(bind=True)
def my_task(self, config: dict):
    task_id = self.request.id
    logger.info(f"Starting task {task_id}")

    try:
        # Task logic
        result = do_work(config)
        logger.info(f"Task {task_id} completed successfully")
        return {"success": True, "result": result}

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
```

**Why**: No silent failures, debugging context, consistent error reporting.

---

### 3.2 Database Status Update in Exception Handler

**Pattern**: Update entity status to FAILED in exception handler before returning.

```python
except Exception as e:
    logger.error(f"Task {task_id} failed: {e}", exc_info=True)

    # Update database status
    try:
        with get_db_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                session.commit()
    except Exception as update_error:
        logger.error(f"Failed to update status: {update_error}")

    # Update Celery state
    self.update_state(state='FAILURE', meta={'error': str(e)})
    raise  # Re-raise for Celery tracking
```

**Why**: Database reflects accurate state regardless of Celery result backend.

---

### 3.3 Graceful Nested Exception Handling

**Pattern**: Isolate notification/cleanup failures from main task failure.

```python
# Send notification (don't let failure affect task result)
try:
    NotificationService.emit_event(
        event_type=EventType.TRAINING_COMPLETE,
        user_id=config.get('user_id', 1),
        data={...}
    )
except Exception as e:
    logger.error(f"Failed to send notification: {e}")
    # Continue - notification failure shouldn't fail the task
```

**Why**: Non-critical operations don't cascade failures.

---

## 4. Monitoring Conventions

### 4.1 Task ID Logging

**Pattern**: Include task ID in all log messages for traceability.

```python
task_id = self.request.id
logger.info(f"Starting training task {task_id} for experiment {experiment_id}")
logger.info(f"Training task {task_id} completed in {duration:.2f}s")
logger.error(f"Training task {task_id} failed: {e}", exc_info=True)
```

**Why**: Correlate logs across distributed workers, debugging production issues.

---

### 4.2 Progress Callback Pattern

**Pattern**: Define inner callback functions for streaming progress to both Celery and database.

```python
def progress_callback(current, total, fault_type=None):
    """Update progress during generation."""
    progress = int((current / total) * 100)
    status_msg = f"Generating {fault_type} ({current}/{total})"

    # Update Celery state
    self.update_state(
        state='PROGRESS',
        meta={'progress': progress, 'status': status_msg}
    )

    # Update database
    try:
        with get_db_session() as session:
            entity = session.query(Entity).filter_by(id=entity_id).first()
            if entity:
                entity.progress = progress
                session.commit()
    except Exception as e:
        logger.error(f"Failed to update progress: {e}")

# Pass to adapter/service
results = Adapter.generate(config, progress_callback=progress_callback)
```

**Why**: Decouples progress tracking from business logic, reusable pattern.

---

### 4.3 Duration Tracking

**Pattern**: Track and report task duration for performance monitoring.

```python
import time

start_time = time.time()
results = do_expensive_work()
duration = time.time() - start_time

logger.info(f"Task completed in {duration:.2f}s")

return {
    "success": True,
    "duration_seconds": duration,
    "results": results
}
```

**Why**: Performance visibility, capacity planning, SLA monitoring.

---

## 5. Task Chaining Patterns

### 5.1 Direct Task Invocation for Batches

**Pattern**: For batch operations, iterate and invoke child tasks directly.

```python
@celery_app.task(bind=True)
def generate_batch_explanations_task(self, config: dict):
    signal_ids = config.get('signal_ids', [])
    results = []

    for i, signal_id in enumerate(signal_ids):
        self.update_state(
            state='PROGRESS',
            meta={'current': i + 1, 'total': len(signal_ids)}
        )

        individual_config = {
            'experiment_id': config['experiment_id'],
            'signal_id': signal_id,
            'method': config['method'],
        }

        try:
            result = generate_explanation_task(individual_config)
            results.append({'signal_id': signal_id, 'success': True, 'result': result})
        except Exception as e:
            results.append({'signal_id': signal_id, 'success': False, 'error': str(e)})

    return {'success': True, 'results': results}
```

**Why**: Maintains parent task context, aggregated results, progress tracking.

---

### 5.2 HPO Trial Orchestration

**Pattern**: For optimization loops, define objective function inline.

```python
@celery_app.task(bind=True)
def run_hpo_campaign_task(self, campaign_id: int):
    study = optuna.create_study(direction=direction)

    def objective(trial):
        """Optuna objective function."""
        # Suggest hyperparameters
        params = {
            'learning_rate': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'hidden_size': trial.suggest_int('hidden', 64, 512),
        }

        # Train and evaluate
        results = train_with_params(params)

        # Update parent task progress
        self.update_state(
            state='PROGRESS',
            meta={'trial': trial.number, 'best': study.best_value}
        )

        return results['metric']

    study.optimize(objective, n_trials=num_trials)
```

**Why**: Leverages Optuna's optimization, maintains task progress visibility.

---

### 5.3 Adapter Pattern for External Systems

**Pattern**: Use adapter classes to integrate with external training/generation systems.

```python
# Determine adapter based on model type
model_type = config["model_type"]

if model_type in ["rf", "svm", "gbm"]:
    from integrations.phase1_adapter import Phase1Adapter
    results = Phase1Adapter.train(config, progress_callback=progress_callback)
else:
    from integrations.deep_learning_adapter import DeepLearningAdapter
    results = DeepLearningAdapter.train(config, progress_callback=progress_callback)
```

**Why**: Decouples task orchestration from implementation, supports multiple backends.

---

## Quick Reference

| Category       | Pattern                            | Key Benefit          |
| -------------- | ---------------------------------- | -------------------- |
| **Definition** | `bind=True`                        | Access task instance |
| **Definition** | JSON serialization                 | Security             |
| **Results**    | `{"success": bool, ...}`           | Uniform API          |
| **Results**    | Database sync                      | Persistent state     |
| **Errors**     | `logger.error(..., exc_info=True)` | Full traceback       |
| **Errors**     | Nested try/except for cleanup      | Graceful degradation |
| **Monitoring** | Task ID in all logs                | Traceability         |
| **Monitoring** | Progress callbacks                 | Real-time UI         |
| **Chaining**   | Adapter pattern                    | Backend flexibility  |

---

## Anti-Patterns to Avoid

| Anti-Pattern              | Problem              | Better Approach             |
| ------------------------- | -------------------- | --------------------------- |
| Bare `except: pass`       | Silent failures      | Log and handle specifically |
| Hardcoded URLs            | Wrong in production  | Use config variables        |
| No time limits            | Workers hang forever | Set `time_limit`            |
| No result expiration      | Redis OOM            | Set `result_expires`        |
| Progress as 0-100 AND 0-1 | Inconsistent UI      | Pick one convention         |
