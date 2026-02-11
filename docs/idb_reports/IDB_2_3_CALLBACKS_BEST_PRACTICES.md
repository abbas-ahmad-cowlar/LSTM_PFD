# IDB 2.3: Callbacks Best Practices

**IDB ID**: 2.3  
**Domain**: Dashboard Platform  
**Extraction Date**: 2026-01-23

---

## 1. Callback Organization Patterns

### 1.1 Registration Pattern ✓

All callbacks follow a consistent registration pattern that enables modular loading:

```python
# callbacks/xai_callbacks.py
def register_xai_callbacks(app):
    """Register all XAI dashboard callbacks."""

    @app.callback(
        Output('xai-model-dropdown', 'options'),
        Input('url', 'pathname')
    )
    def load_available_models(pathname):
        # Implementation
        pass
```

**Benefits**:

- Single entry point per module
- Easy to enable/disable features
- Testable registration

### 1.2 Central Registration with Graceful Fallbacks ✓

```python
# callbacks/__init__.py
try:
    from callbacks.xai_callbacks import register_xai_callbacks
    register_xai_callbacks(app)
except ImportError as e:
    print(f"Warning: Could not import xai_callbacks: {e}")
```

**Adoption**: Use this pattern for all new callback modules to prevent single failures from breaking the entire dashboard.

### 1.3 Related Callback Co-Registration ✓

When callbacks are tightly coupled, register them together:

```python
# callbacks/data_generation_callbacks.py
def register_data_generation_callbacks(app):
    """Register all data generation and import callbacks."""

    # Register MAT import callbacks (related functionality)
    from callbacks.mat_import_callbacks import register_mat_import_callbacks
    register_mat_import_callbacks(app)

    # Data generation callbacks follow...
```

---

## 2. Input/Output Conventions

### 2.1 Page-Scoped Callbacks ✓

Use pathname filtering to prevent callbacks from firing on wrong pages:

```python
@app.callback(
    Output('xai-model-dropdown', 'options'),
    Input('url', 'pathname')
)
def load_available_models(pathname):
    if pathname != '/xai':
        raise PreventUpdate
    # ... proceed with data loading
```

### 2.2 Multi-Output Pattern ✓

Group related outputs in a single callback to reduce round-trips:

```python
@app.callback(
    [
        Output('xai-prediction-display', 'children'),
        Output('xai-results-store', 'data'),
    ],
    Input('generate-xai-btn', 'n_clicks'),
    [
        State('xai-model-dropdown', 'value'),
        State('xai-signal-dropdown', 'value'),
        # ... additional states
    ],
    prevent_initial_call=True
)
def generate_explanation(n_clicks, experiment_id, signal_id, ...):
    return prediction_display, results_data
```

### 2.3 `prevent_initial_call` for Action Callbacks ✓

Always use `prevent_initial_call=True` for callbacks triggered by user actions:

```python
@app.callback(
    Output('webhook-form-message', 'children'),
    Input('save-webhook-btn', 'n_clicks'),
    [...],
    prevent_initial_call=True  # ✓ Prevents firing on page load
)
def save_webhook(n_clicks, ...):
    if not n_clicks:
        return ""
```

### 2.4 Pattern-Based IDs for Dynamic Components ✓

Use `ALL`, `MATCH` for dynamic component lists:

```python
from dash import Input, Output, State, ALL, MATCH, ctx

@app.callback(
    Output('webhooks-table', 'children', allow_duplicate=True),
    Input({'type': 'toggle-webhook-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def toggle_webhook_handler(n_clicks):
    if not ctx.triggered or not any(n_clicks):
        return html.Div()

    trigger_id = ctx.triggered[0]['prop_id']
    button_id = json.loads(trigger_id.split('.')[0])
    webhook_id = button_id['index']
    # ... handle specific webhook
```

---

## 3. State Management Patterns

### 3.1 Store Components for Cross-Callback Data ✓

Use `dcc.Store` for application state:

```python
# In layout
dcc.Store(id='active-generation-id', data=None),
dcc.Store(id='wizard-config', data={}),
dcc.Store(id='xai-results-store', data={}),

# In callback - write to store
@app.callback(
    Output('active-generation-id', 'data'),
    Input('start-generation-btn', 'n_clicks'),
    ...
)
def start_generation(n_clicks, ...):
    # ... create generation
    return generation_id

# In callback - read from store
@app.callback(
    Output('generation-status', 'children'),
    Input('generation-poll-interval', 'n_intervals'),
    State('active-generation-id', 'data'),  # ✓ Read via State
)
def poll_generation_status(n_intervals, generation_id):
    if not generation_id:
        raise PreventUpdate
```

### 3.2 `no_update` for Selective Output Updates ✓

Use `no_update` when only some outputs should change:

```python
from dash import no_update

@app.callback(
    [Output('field1', 'value'), Output('field2', 'value'), Output('modal', 'is_open')],
    Input('upload-config-file', 'contents'),
)
def load_configuration(contents):
    if error_condition:
        return (
            no_update,  # Don't change field1
            no_update,  # Don't change field2
            True,       # Keep modal open
        )
    return value1, value2, False
```

### 3.3 Polling Pattern for Async Tasks ✓

Use interval components for progress polling:

```python
# Layout
dcc.Interval(id='generation-poll-interval', interval=1000, disabled=True),

# Start polling
@app.callback(
    Output('generation-poll-interval', 'disabled'),
    Input('start-generation-btn', 'n_clicks'),
)
def start_generation(n_clicks):
    # ... launch task
    return False  # Enable polling

# Poll status
@app.callback(
    [
        Output('generation-status', 'children'),
        Output('generation-poll-interval', 'disabled', allow_duplicate=True),
    ],
    Input('generation-poll-interval', 'n_intervals'),
    prevent_initial_call=True
)
def poll_status(n_intervals):
    if task_complete:
        return status_content, True  # Disable polling
    return progress_content, False   # Continue polling
```

---

## 4. Error Handling in Callbacks

### 4.1 Comprehensive Try-Except with User Feedback ✓

```python
@app.callback(...)
def save_webhook(n_clicks, name, provider, url, ...):
    if not n_clicks:
        return "", None

    try:
        # Validate inputs first
        name = validate_required(name, "Webhook name")
        url = validate_url(url, "Webhook URL")

        # Perform operation
        webhook = WebhookService.create_webhook(...)

        return dbc.Alert([
            html.I(className="bi bi-check-circle me-2"),
            f"Webhook '{name}' created successfully!"
        ], color="success"), None

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return dbc.Alert(str(e), color="danger"), webhook_id

    except ValueError as e:
        logger.warning(f"Value error: {e}")
        return dbc.Alert(str(e), color="danger"), webhook_id

    except Exception as e:
        logger.error(f"Error saving webhook: {e}", exc_info=True)
        return dbc.Alert(
            "An unexpected error occurred while saving the webhook",
            color="danger"
        ), webhook_id
```

### 4.2 Graceful Empty State Handling ✓

```python
@app.callback(...)
def load_available_models(pathname):
    if pathname != '/xai':
        raise PreventUpdate

    try:
        with get_db_session() as session:
            experiments = session.query(Experiment)...

            if not experiments:  # ✓ Handle empty case
                return []

            options = [...]
            return options

    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        return []  # ✓ Return empty list, not error
```

### 4.3 Input Validation Pattern ✓

```python
from utils.validation import (
    validate_required,
    validate_string_length,
    validate_url,
    validate_list_not_empty,
    ValidationError
)

@app.callback(...)
def save_webhook(n_clicks, name, provider, url, events, ...):
    try:
        # Validate all inputs upfront
        name = validate_required(name, "Webhook name")
        name = validate_string_length(name, 100, "Webhook name")
        url = validate_required(url, "Webhook URL")
        url = validate_url(url, "Webhook URL")
        events = validate_list_not_empty(events, "Event selection")

        # ... proceed with valid data
    except ValidationError as e:
        return dbc.Alert(str(e), color="danger"), webhook_id
```

---

## 5. Performance Optimization Patterns

### 5.1 Cache-First Pattern ✓

Check cache before expensive operations:

```python
@app.callback(...)
def generate_explanation(n_clicks, experiment_id, signal_id, method, ...):
    # Check cache first
    cached = ExplanationCache.get_explanation(experiment_id, signal_id, method)

    if cached and cached.get('success'):
        logger.info(f"Using cached {method} explanation")
        cached['status'] = 'ready'
        cached['from_cache'] = True
        return _create_prediction_display(cached), cached

    # Not in cache - launch async task
    task = generate_explanation_task.delay(config)
    return prediction_display, {'task_id': task.id, 'status': 'generating'}
```

### 5.2 Limit Query Results ✓

Always limit database queries in callbacks:

```python
@app.callback(...)
def load_available_models(pathname):
    with get_db_session() as session:
        experiments = session.query(Experiment)\
            .filter(Experiment.status == ExperimentStatus.COMPLETED)\
            .order_by(Experiment.created_at.desc())\
            .limit(50)  # ✓ Always limit
            .all()
```

### 5.3 Async Task Delegation ✓

For long-running operations, delegate to Celery:

```python
from tasks.xai_tasks import generate_explanation_task

@app.callback(...)
def generate_explanation(n_clicks, ...):
    # Quick validation - runs immediately
    if not experiment_id or not signal_id:
        raise PreventUpdate

    # Heavy work - delegated to Celery
    task = generate_explanation_task.delay(config)

    # Return immediately with task ID
    return prediction_display, {
        'task_id': task.id,
        'status': 'generating'
    }
```

---

## 6. Testing Callback Patterns

### 6.1 Testable Helper Functions ✓

Extract complex logic to testable helper functions:

```python
# In callbacks/data_generation_callbacks.py

# Helper functions (easily testable)
def _estimate_generation_time(num_signals):
    """Estimate generation time based on number of signals."""
    return max(1, round(num_signals / SIGNALS_PER_MINUTE_GENERATION))

def _create_generation_stats(generation):
    """Create statistics display for generation."""
    return [
        html.Div([html.Strong("Signals: "), html.Span(f"{generation.num_signals:,}")]),
        html.Div([html.Strong("Fault Types: "), html.Span(f"{generation.num_faults}")]),
    ]

def _get_status_icon(status):
    """Get Font Awesome icon for status."""
    icons = {
        DatasetGenerationStatus.PENDING: "fas fa-clock",
        DatasetGenerationStatus.RUNNING: "fas fa-spinner fa-spin",
        DatasetGenerationStatus.COMPLETED: "fas fa-check-circle",
        DatasetGenerationStatus.FAILED: "fas fa-exclamation-triangle",
    }
    return icons.get(status, "fas fa-question-circle")
```

### 6.2 Type-Annotated Helpers ✓

Use type hints for complex return types:

```python
from typing import Dict, Any

def _create_prediction_display(result: Dict[str, Any]) -> html.Div:
    """Create prediction display card."""
    predicted_class = result.get('predicted_class', 0)
    confidence = result.get('confidence', 0)
    # ...
    return html.Div([...])

def _create_shap_details(result: Dict[str, Any]) -> html.Div:
    """Create SHAP explanation details panel."""
    # ...
```

### 6.3 Docstrings with Args/Returns ✓

Document callback functions thoroughly:

```python
@app.callback(...)
def generate_explanation(n_clicks, experiment_id, signal_id, method,
                        num_features, bg_samples, perturbations):
    """
    Generate XAI explanation (check cache first, then launch async task).

    Args:
        n_clicks: Button click count
        experiment_id: Selected experiment ID
        signal_id: Selected signal ID
        method: XAI method ('shap', 'lime', 'integrated_gradients', 'gradcam')
        num_features: Number of features to show
        bg_samples: Number of background samples (SHAP)
        perturbations: Number of perturbations (LIME)

    Returns:
        Tuple of (prediction display, results store data)
    """
```

---

## Quick Reference: Pattern Adoption Checklist

| Pattern                           | Files Using            | Adopt?               |
| --------------------------------- | ---------------------- | -------------------- |
| `register_*_callbacks(app)`       | All 28 modules         | ✅ Yes               |
| Try-except with graceful fallback | 25 modules             | ✅ Yes               |
| `prevent_initial_call=True`       | 20+ callbacks          | ✅ Yes               |
| `dcc.Store` for state             | 15 layouts             | ✅ Yes               |
| Cache-first pattern               | `xai_callbacks.py`     | ✅ Yes               |
| Polling pattern                   | 3 modules              | ✅ Yes (async tasks) |
| Typed helper functions            | 5 modules              | ✅ Yes               |
| Docstrings with Args/Returns      | `xai_callbacks.py`     | ✅ Expand            |
| Input validation utilities        | `webhook_callbacks.py` | ✅ Expand            |
| Service-layer delegation          | 20+ modules            | ✅ Expand            |

---

## Anti-Patterns to Avoid

| Anti-Pattern                               | Alternative                      |
| ------------------------------------------ | -------------------------------- |
| Silent `raise PreventUpdate` in except     | Return user-facing error message |
| Direct DB queries in callbacks             | Call service layer methods       |
| Unbounded query results                    | Always use `.limit()`            |
| Hardcoded timeouts                         | Import from `utils.constants`    |
| Business logic in callbacks                | Extract to services/helpers      |
| Missing `prevent_initial_call` for buttons | Always add for action callbacks  |
