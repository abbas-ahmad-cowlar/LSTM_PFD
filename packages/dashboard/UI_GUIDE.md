# Dashboard UI Guide

> Practical guide for working with the LSTM PFD Dashboard frontend.

## Architecture Overview

The dashboard is a **Plotly Dash** application built on:

| Layer      | Technology                | Purpose                                  |
| ---------- | ------------------------- | ---------------------------------------- |
| Framework  | Plotly Dash 2.x           | Reactive web framework (Python → HTML)   |
| Server     | Flask                     | HTTP server, API blueprints, sessions    |
| UI Library | dash-bootstrap-components | Bootstrap 5 layout and widgets           |
| Icons      | Font Awesome 6            | Icon set loaded from CDN                 |
| Routing    | `dcc.Location` + callback | Client-side URL routing                  |
| State      | `dcc.Store`               | Session-scoped client state              |
| Styling    | CSS (Bootstrap + custom)  | `assets/theme.css` + `assets/custom.css` |

## Project Structure

```
packages/dashboard/
├── app.py                  # Entry point — creates Dash app, assembles shell
├── dashboard_config.py     # Configuration from env vars
├── assets/
│   ├── theme.css           # Core theme (colors, typography, layout)
│   └── custom.css          # Overrides (sidebar, cards, dropdowns, responsive)
├── components/             # Reusable shell + widgets
│   ├── sidebar.py          # Left nav (collapsible, icon groups)
│   ├── header.py           # Top navbar + mobile hamburger
│   ├── footer.py           # Footer with version
│   ├── cards.py            # Stat/info card factories
│   ├── skeleton.py         # Skeleton loaders
│   └── tag_manager.py      # Tag input/display/filter
├── layouts/                # Page layout modules (one per page)
│   ├── home.py
│   ├── experiments.py
│   └── ... (22 files)
├── callbacks/              # Callback registration (IDB 2.3)
│   └── __init__.py         # Routing + callback imports
├── services/               # Backend services (IDB 2.2)
├── tasks/                  # Celery async tasks (IDB 2.4)
├── models/                 # SQLAlchemy models (IDB 4.1)
├── database/               # DB connection + migrations (IDB 4.1)
├── api/                    # Flask API blueprints
├── templates/              # Jinja2 email templates
└── utils/                  # Utilities + constants
```

## Adding a New Page

### Step 1: Create the Layout

Create `layouts/my_new_page.py`:

```python
"""
My new page layout.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_my_new_page_layout():
    """Create my new page layout."""
    return dbc.Container([
        html.H2("My New Page", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Section Title"),
                    dbc.CardBody([
                        html.P("Content goes here."),
                        dcc.Graph(id="my-new-chart", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], width=12),
        ]),
    ], fluid=True)
```

### Step 2: Register the Route

In `callbacks/__init__.py`, add the import and route:

```python
# At the top, with other layout imports
from layouts.my_new_page import create_my_new_page_layout

# Inside display_page(), add a new elif branch
elif pathname == '/my-new-page':
    return create_my_new_page_layout()
```

### Step 3: Add to Sidebar Navigation

In `components/sidebar.py`, add an entry to the appropriate section in `NAV_ITEMS`:

```python
NAV_ITEMS = {
    # ...
    'Analytics': [
        # ... existing items ...
        {'id': 'nav-my-page', 'icon': 'fas fa-star', 'label': 'My Page', 'href': '/my-new-page'},
    ],
}
```

Optional: add `'admin_only': True` to restrict visibility.

### Step 4: Register Callbacks (if interactive)

Create `callbacks/my_new_page_callbacks.py`:

```python
from dash import Input, Output

def register_my_new_page_callbacks(app):
    @app.callback(
        Output('my-new-chart', 'figure'),
        Input('some-trigger', 'n_intervals')
    )
    def update_chart(n):
        # Build and return a Plotly figure
        ...
```

Then register it in `callbacks/__init__.py`:

```python
try:
    from callbacks.my_new_page_callbacks import register_my_new_page_callbacks
    register_my_new_page_callbacks(app)
except ImportError as e:
    print(f"Warning: Could not import my_new_page_callbacks: {e}")
```

## Creating Reusable Components

Place new components in `components/` and follow these conventions:

1. **Module docstring** explaining the component purpose
2. **Factory functions** named `create_*()` that return Dash component trees
3. **Type hints** on all parameters
4. **No side effects** — components should be pure functions
5. **Export from `__init__.py`** if the component is used across multiple layouts

Example:

```python
"""
Alert banner component.
"""
import dash_bootstrap_components as dbc
from dash import html


def create_alert_banner(message: str, color: str = "warning", dismissable: bool = True):
    """Create a dismissable alert banner."""
    return dbc.Alert(
        [html.I(className="fas fa-exclamation-triangle me-2"), message],
        color=color,
        dismissable=dismissable,
        className="mb-3"
    )
```

## Asset Management

### CSS Files

Dash automatically serves all files in the `assets/` directory. Files are loaded in **alphabetical order**.

| File         | Purpose                                                                    | Size     |
| ------------ | -------------------------------------------------------------------------- | -------- |
| `custom.css` | Sidebar styles, card hover, dropdown z-index fixes, responsive breakpoints | ~1.7 KB  |
| `theme.css`  | Core theme — colors, typography, layout rules, skeleton animations         | ~28.8 KB |

### Adding New CSS

Create a new `.css` file in `assets/`. Dash will auto-detect and include it. Use a naming convention like `custom-{feature}.css`.

### Adding Static Assets

Place images, fonts, or JS files in `assets/`. Reference via `/assets/filename.ext` in your layout code.

## Theming and Styling

### Bootstrap Theme

The app uses `dbc.themes.BOOTSTRAP` as the base theme. All Bootstrap 5 utility classes are available:

- **Spacing:** `mb-4`, `py-3`, `me-2`
- **Colors:** `text-primary`, `text-muted`, `bg-dark`
- **Layout:** `d-flex`, `justify-content-between`, `w-100`

### Component ID Conventions

Use descriptive, hyphenated IDs scoped by page:

| Pattern              | Example                 | Used For               |
| -------------------- | ----------------------- | ---------------------- |
| `{page}-{element}`   | `home-total-signals`    | Page-specific elements |
| `{feature}-{action}` | `sidebar-toggle-btn`    | Shell components       |
| `{entity}-{field}`   | `experiment-name-input` | Form fields            |

Unique IDs are required since Dash enforces global ID uniqueness across all registered callbacks.

### Responsive Design

- **Mobile hamburger:** `header.py` includes a clientside callback toggling the sidebar on mobile
- **Sidebar:** Hidden below `768px` via CSS media query, toggled via hamburger button
- **Grid:** `dbc.Row` / `dbc.Col` with Bootstrap responsive width props

## Stores and Intervals

The app layout (in `app.py`) includes these global Dash stores and intervals:

| Component      | ID                       | Type    | Purpose                             |
| -------------- | ------------------------ | ------- | ----------------------------------- |
| `dcc.Location` | `url`                    | —       | URL routing                         |
| `dcc.Store`    | `session-store`          | session | User session state                  |
| `dcc.Store`    | `comparison-cart`        | session | Selected experiments for comparison |
| `dcc.Interval` | `refresh-interval`       | —       | 5-second polling for dynamic data   |
| `dcc.Interval` | `system-health-interval` | —       | 5-second polling for health metrics |

## Running the Dashboard

```bash
cd packages/dashboard
pip install -r requirements.txt
python app.py
```

The server starts on `http://localhost:8050` by default (configurable via `APP_HOST` / `APP_PORT` env vars).

## Related Documentation

- [Layouts README](layouts/README.md) — Page catalog with all routes
- [Components README](components/README.md) — Component catalog and API reference
- [Services (IDB 2.2)](services/) — Backend service layer
- [Callbacks (IDB 2.3)](callbacks/) — Callback registration
