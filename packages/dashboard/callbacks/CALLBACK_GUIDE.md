# Callback Developer Guide

> How to write, test, and maintain Dash callbacks in the LSTM-PFD Dashboard.

## How to Create a New Callback

### Step 1: Create the Callback Module

Create a new file in `packages/dashboard/callbacks/` following the naming convention `{feature}_callbacks.py`:

```python
"""
{Feature} callbacks.
{One-line description of what these callbacks handle.}
"""
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from utils.logger import setup_logger

logger = setup_logger(__name__)


def register_{feature}_callbacks(app):
    """Register all {feature} callbacks."""

    @app.callback(
        Output('{feature}-output', 'children'),
        [Input('{feature}-trigger', 'n_clicks')],
        [State('{feature}-input', 'value')],
        prevent_initial_call=True
    )
    def handle_action(n_clicks, input_value):
        """Handle user action."""
        if not n_clicks:
            raise PreventUpdate

        try:
            # Business logic here
            result = "Success"
            return dbc.Alert(result, color="success")
        except Exception as e:
            logger.error(f"Error in {feature}: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    logger.info("{Feature} callbacks registered")
```

### Step 2: Register in `__init__.py`

Add a `try/except` import block to `register_all_callbacks()`:

```python
# Import and register {Feature} callbacks
try:
    from callbacks.{feature}_callbacks import register_{feature}_callbacks
    register_{feature}_callbacks(app)
except ImportError as e:
    print(f"Warning: Could not import {feature}_callbacks: {e}")
```

### Step 3: Add URL Route (if needed)

If your callback powers a new page, add a route in the `display_page` callback:

```python
elif pathname == '/{feature}':
    return create_{feature}_layout()
```

And add the corresponding layout import at the top of `__init__.py`.

## Input/Output Naming Conventions

The codebase uses consistent naming patterns for Dash component IDs:

| Pattern                         | Example                       | Usage                                          |
| ------------------------------- | ----------------------------- | ---------------------------------------------- |
| `{page}-{element}`              | `home-total-signals`          | Page-scoped elements                           |
| `{feature}-{action}-btn`        | `generate-roc-btn`            | Action buttons                                 |
| `{feature}-{data}-store`        | `comparison-data-store`       | `dcc.Store` components                         |
| `{feature}-{element}-select`    | `eval-experiment-select`      | Dropdown selectors                             |
| `{feature}-{element}-table`     | `experiments-table`           | Data tables                                    |
| `{feature}-{element}-container` | `experiments-table-container` | Wrapper divs                                   |
| `{feature}-{name}-modal`        | `share-link-modal`            | Modal dialogs                                  |
| `url`                           | `url`                         | Global URL (always `Input('url', 'pathname')`) |

### Output Multiplicity

Callbacks may return multiple outputs as a tuple:

```python
@app.callback(
    [
        Output('output-1', 'children'),
        Output('output-2', 'figure'),
        Output('output-3', 'children'),
    ],
    [Input('url', 'pathname')]
)
def multi_output(pathname):
    return (component_1, figure_2, component_3)
```

### Pattern-Matching Callbacks

Some modules use `ALL` or `MATCH` for dynamic component IDs:

```python
from dash import ALL, MATCH

# Fires when ANY button with type "revoke-btn" is clicked
@app.callback(
    Output(...),
    Input({'type': 'revoke-btn', 'index': ALL}, 'n_clicks')
)
```

Used in: `api_key_callbacks.py`, `webhook_callbacks.py`, `notification_callbacks.py`, `hpo_callbacks.py`, `email_digest_callbacks.py`.

## State Management Patterns

### URL-Based State (Primary)

Most page-level callbacks trigger on URL changes:

```python
@app.callback(
    Output('content', 'children'),
    [Input('url', 'pathname')]
)
def on_page_load(pathname):
    if pathname != '/my-page':
        raise PreventUpdate
    # Load data for this page
```

### `dcc.Store` for Cross-Callback Data

When data needs to flow between callbacks on the same page:

```python
# Callback 1: write to store
@app.callback(
    Output('data-store', 'data'),
    [Input('fetch-btn', 'n_clicks')]
)
def fetch_data(n_clicks):
    return {'key': 'value'}

# Callback 2: read from store
@app.callback(
    Output('display', 'children'),
    [Input('data-store', 'data')]
)
def display_data(data):
    return str(data)
```

Used in: `comparison_callbacks.py` (`comparison-data-store`), `experiments_callbacks.py` (`selected-experiments-store`), `xai_callbacks.py` (results store).

### Interval-Based Polling

Long-running operations use `dcc.Interval` for status polling:

```python
@app.callback(
    Output('progress', 'children'),
    [Input('poll-interval', 'n_intervals')],
    [State('task-id-store', 'data')]
)
def poll_status(n_intervals, task_id):
    if not task_id:
        raise PreventUpdate
    result = AsyncResult(task_id)
    # Update progress display
```

Used in: `data_generation_callbacks.py`, `mat_import_callbacks.py`, `training_monitor_callbacks.py`, `email_digest_callbacks.py`, `system_health_callbacks.py`.

## Error Handling Patterns

### Standard Try/Except → Alert

Every callback that performs I/O wraps its body in a try/except:

```python
try:
    with get_db_session() as session:
        # database operations
    return dbc.Alert("Success!", color="success")
except Exception as e:
    logger.error(f"Error message: {e}", exc_info=True)
    return dbc.Alert(f"Error: {str(e)}", color="danger")
```

### PreventUpdate for No-Op Cases

Guard clauses at the top of every callback prevent unnecessary execution:

```python
def my_callback(n_clicks, pathname):
    if not n_clicks:
        raise PreventUpdate
    if pathname != '/expected-page':
        raise PreventUpdate
```

### Graceful Fallback Returns

Callbacks that return multiple outputs provide safe defaults on error:

```python
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    return (
        html.P("Error loading data"),
        {},    # empty figure
        {},    # empty figure
        "Error", "Error", "Error", "Error"
    )
```

## Performance Considerations

### Use `PreventUpdate` Aggressively

Every callback should check whether it actually needs to run. The URL-based pattern `if pathname != '/my-page': raise PreventUpdate` is critical because **all page-level callbacks fire on every URL change**.

### Use `prevent_initial_call=True`

For button-triggered callbacks, set `prevent_initial_call=True` to avoid running on page load:

```python
@app.callback(
    Output(...), Input('button', 'n_clicks'),
    prevent_initial_call=True
)
```

### Avoid Heavy Computation in Callbacks

Delegate heavy work to Celery tasks (via `.delay()`) and poll for results. Examples:

- `evaluation_callbacks.py` → `generate_roc_analysis_task.delay()`
- `testing_callbacks.py` → `run_tests_task.delay()`
- `xai_callbacks.py` → `generate_explanation_task.delay()`

### Database Session Scope

Always use the context manager to ensure sessions are closed:

```python
with get_db_session() as session:
    # queries here
# session is automatically closed
```

If objects need to be used after the session closes, call `session.expunge(obj)` first (see `experiments_callbacks.py`).

### Identify the Trigger with `callback_context`

When a callback has multiple `Input` triggers, use `callback_context` to determine which one fired:

```python
from dash import callback_context

trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
if trigger_id == 'button-a':
    # handle button A
elif trigger_id == 'button-b':
    # handle button B
```

## Testing Callbacks

### Unit Testing Approach

Callbacks are plain Python functions. Test them by calling the inner function directly:

```python
def test_load_experiments():
    """Test that load_experiments returns valid component."""
    # The inner function is accessible as a closure,
    # but the practical approach is to test the service layer
    # and verify callback wiring via integration tests.
    pass
```

### Integration Testing

Use the Dash testing framework (`dash.testing`):

```python
from dash.testing.application_runners import import_app

def test_home_page(dash_duo):
    app = import_app("app")
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal("#home-total-signals", "0", timeout=10)
```

### Manual Verification

Run the dashboard locally and verify:

1. Each page loads without errors
2. Buttons trigger the expected actions
3. Polling callbacks update correctly
4. Error states display appropriate alerts

## Common Pitfalls

### Circular Output Conflicts

Two callbacks cannot write to the same `Output`. If you need this, use `allow_duplicate=True`:

```python
@app.callback(
    Output('toast-container', 'children', allow_duplicate=True),
    Input('button', 'n_clicks'),
    prevent_initial_call=True
)
```

Used in: `comparison_callbacks.py`, `experiments_callbacks.py`.

### Missing `prevent_initial_call`

Forgetting `prevent_initial_call=True` on button-triggered callbacks causes them to fire on page load with `n_clicks=None`, which may produce errors unless there is a guard clause.

### Session Object Access After Close

SQLAlchemy lazy-loaded attributes are inaccessible after the session closes. Either:

- Access all needed attributes inside the `with` block, or
- Call `session.expunge(obj)` before exiting

### Import Order in `__init__.py`

Layout imports are at module level, but callback imports are inside `register_all_callbacks()`. This avoids circular imports since callbacks import from services which may import from models that share dependencies with layouts.
