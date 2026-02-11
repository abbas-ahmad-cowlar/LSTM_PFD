# IDB 2.1: Frontend/UI Best Practices

**Domain:** Dashboard Platform  
**IDB ID:** 2.1  
**Date:** 2026-01-23

---

## 1. Layout Organization Patterns

### 1.1 File Naming Convention

```
# Layout files use snake_case with descriptive names
experiment_wizard.py      ✅ Good - clear purpose
data_explorer.py          ✅ Good - clear purpose
xai_dashboard.py          ✅ Good - includes "dashboard" suffix
```

### 1.2 Function Naming Convention

```python
# Main layout function
def create_xxx_layout():
    """Create the main page layout."""
    return dbc.Container([...], fluid=True)

# Tab content functions
def create_xxx_tab():
    """Create tab content."""
    return html.Div([...])

# Card/section functions
def create_xxx_card():
    """Create a card component."""
    return dbc.Card([...])

# Helper functions (private)
def _create_helper():
    """Internal helper - prefixed with underscore."""
    pass
```

### 1.3 Layout Structure Template

```python
"""
Module docstring describing the layout purpose.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from utils.constants import RELEVANT_CONSTANTS

def create_page_layout():
    """Create page layout with standard structure."""
    return dbc.Container([
        # 1. Page Header
        dbc.Row([
            dbc.Col([
                html.H2("Page Title", className="mb-3"),
                html.P("Description", className="text-muted")
            ])
        ], className="mb-4"),

        # 2. Main Content
        dbc.Row([...]),

        # 3. Hidden Stores (at bottom)
        dcc.Store(id='page-data-store', data=None),

    ], fluid=True, className="py-4")
```

### 1.4 Recommended File Size Limits

| Category            | Max Lines | Max Size |
| ------------------- | --------- | -------- |
| Simple layout       | 150       | 5KB      |
| Medium layout       | 300       | 12KB     |
| Complex layout      | 500       | 15KB     |
| **Split threshold** | >500      | >15KB    |

---

## 2. Component Design Conventions

### 2.1 Component Function Signature

```python
def create_component(
    component_id: str,           # Required: unique ID
    title: str = "Default",      # Optional with default
    variant: str = "primary",    # Optional with enum-like values
    **kwargs                     # Allow extensibility
) -> html.Div:
    """
    Create a reusable component.

    Args:
        component_id: Unique identifier for callbacks
        title: Display title
        variant: Style variant (primary, secondary, success, danger)

    Returns:
        Dash HTML component
    """
    return html.Div([...], id=component_id)
```

### 2.2 Component ID Naming

```python
# Pattern: {page}-{component}-{element}
id="settings-api-key-table"        ✅ Hierarchical
id="home-total-signals"            ✅ Clear ownership
id="xai-model-dropdown"            ✅ Descriptive

# Avoid
id="table1"                        ❌ Not descriptive
id="btn"                           ❌ Too generic
```

### 2.3 Store Pattern for State

```python
# Always place stores at the end of layouts
dcc.Store(id='page-data-store', data=None),           # Page data
dcc.Store(id='page-selected-id', data=None),          # Selection state
dcc.Store(id='page-filter-state', data={}),           # Filter state
```

### 2.4 Loading State Pattern

```python
# Wrap data-dependent components in Loading
dcc.Loading(
    id="loading-component-name",
    type="default",                    # Options: default, circle, dot
    children=[
        html.Div(id='dynamic-content')
    ]
)
```

---

## 3. CSS/Styling Patterns

### 3.1 CSS Variables (Custom Properties)

```css
:root {
  /* Brand Colors */
  --brand-primary: #2196f3;
  --brand-primary-dark: #1976d2;
  --brand-secondary: #ff9800;

  /* Semantic Colors */
  --color-success: #4caf50;
  --color-warning: #ff9800;
  --color-danger: #f44336;

  /* Surfaces */
  --surface-primary: #ffffff;
  --surface-secondary: #f7fafc;
  --surface-hover: #edf2f7;

  /* Typography */
  --font-family-base: "Inter", system-ui, sans-serif;
  --font-size-base: 1rem;

  /* Spacing */
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;

  /* Border Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
}
```

### 3.2 Dark Mode Pattern

```css
/* Light theme (default) */
:root,
[data-theme="light"] {
  --bg-primary: #ffffff;
  --text-primary: #1a202c;
}

/* Dark theme */
[data-theme="dark"] {
  --bg-primary: #0d1117;
  --text-primary: #e6edf3;
}

/* Apply variables */
body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
}
```

### 3.3 Bootstrap Class Usage

```python
# Spacing classes
className="mb-4"           # margin-bottom: 1.5rem
className="py-4"           # padding-y: 1.5rem
className="mt-3 mb-3"      # margin-top/bottom: 1rem

# Flexbox utilities
className="d-flex align-items-center gap-2"
className="justify-content-between"

# Text utilities
className="text-muted"     # Secondary text
className="text-primary"   # Brand color text
className="text-center"    # Centered text

# Card styling
className="shadow-sm"      # Subtle shadow
className="border rounded" # With border
```

### 3.4 Animation Keyframes

```css
/* Skeleton shimmer */
@keyframes shimmer {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}

.skeleton {
  background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
}
```

---

## 4. Asset Management Conventions

### 4.1 CSS File Organization

```
assets/
├── theme.css        # Main theme with CSS variables, components
└── custom.css       # Project-specific overrides (keep minimal)
```

### 4.2 CSS Import Order

```css
/* 1. CSS Variables (custom properties) */
/* 2. Base/Reset styles */
/* 3. Layout styles */
/* 4. Component styles */
/* 5. Utility classes */
/* 6. Media queries (mobile-first) */
```

### 4.3 Icon Usage

```python
# FontAwesome icons with consistent sizing
html.I(className="fas fa-database fa-2x text-primary")
html.I(className="fas fa-cog me-2")  # With margin-end

# Icon patterns
className="fas fa-xxx"     # Solid icons
className="far fa-xxx"     # Regular (outline) icons
className="fab fa-xxx"     # Brand icons
```

---

## 5. Accessibility Requirements

### 5.1 ARIA Labels (Required)

```python
# Icon-only buttons MUST have aria-label
dbc.Button(
    html.I(className="fas fa-trash"),
    id="delete-btn",
    color="danger",
    **{"aria-label": "Delete item"}  # Required
)

# Form inputs MUST have labels
dbc.Label("Email Address", html_for="email-input"),
dbc.Input(id="email-input", type="email")
```

### 5.2 Semantic Structure

```python
# Use proper heading hierarchy
html.H1("Page Title")           # One per page
html.H2("Section Heading")      # Major sections
html.H3("Subsection Heading")   # Subsections

# Use semantic containers
html.Main([...])                # Main content
html.Nav([...])                 # Navigation
html.Article([...])             # Self-contained content
html.Section([...])             # Thematic grouping
```

### 5.3 Focus Management

```python
# Modals should trap focus
dbc.Modal(
    [...],
    id="modal",
    is_open=False,
    keyboard=True,          # Allow Esc to close
    backdrop="static"       # Prevent click-outside close
)
```

### 5.4 Color Contrast

```css
/* Minimum contrast ratios (WCAG AA) */
/* Normal text: 4.5:1 */
/* Large text (18px+): 3:1 */
/* UI components: 3:1 */

--text-primary: #1a202c; /* On white: 12.6:1 ✓ */
--text-secondary: #4a5568; /* On white: 7.0:1 ✓ */
--text-tertiary: #718096; /* On white: 4.5:1 ✓ */
```

---

## 6. Responsive Design Patterns

### 6.1 Breakpoints

```css
/* Mobile-first breakpoints */
@media (max-width: 480px) {
  /* Small mobile */
}
@media (max-width: 768px) {
  /* Mobile/Tablet */
}
@media (max-width: 992px) {
  /* Tablet */
}
@media (max-width: 1200px) {
  /* Desktop */
}
```

### 6.2 Grid System Usage

```python
# Responsive columns
dbc.Row([
    dbc.Col([...], xs=12, md=6, lg=4),  # Full → Half → Third
    dbc.Col([...], xs=12, md=6, lg=4),
    dbc.Col([...], xs=12, md=12, lg=4),
])

# Auto-width columns
dbc.Col([...], width="auto")  # Shrink to content
```

### 6.3 Mobile Navigation Pattern

```css
/* Hide sidebar on mobile */
@media (max-width: 768px) {
  .sidebar {
    transform: translateX(-100%);
    position: fixed;
    z-index: 1050;
  }

  .sidebar.mobile-open {
    transform: translateX(0);
  }

  .hamburger-btn {
    display: flex;
  }
}
```

### 6.4 Touch-Friendly Targets

```css
/* Minimum 44x44px touch targets */
@media (max-width: 768px) {
  .btn,
  .nav-link {
    min-height: 44px;
    min-width: 44px;
  }
}
```

---

## 7. Callback Best Practices

### 7.1 Clientside Callbacks for Performance

```python
# Use clientside callbacks for DOM manipulation
from dash import clientside_callback, Input, Output

clientside_callback(
    """
    function(n_clicks, current_state) {
        if (!n_clicks) return window.dash_clientside.no_update;

        const element = document.getElementById('target');
        element.classList.toggle('active');
        return !current_state;
    }
    """,
    Output('state-store', 'data'),
    Input('toggle-btn', 'n_clicks'),
    State('state-store', 'data'),
    prevent_initial_call=True
)
```

### 7.2 Prevent Initial Call

```python
# Always use prevent_initial_call for user-triggered actions
@callback(
    Output('output', 'children'),
    Input('button', 'n_clicks'),
    prevent_initial_call=True  # Prevents firing on page load
)
def handle_click(n_clicks):
    return f"Clicked {n_clicks} times"
```

---

## 8. Code Quality Checklist

### Pre-Commit Checklist

- [ ] All functions have docstrings
- [ ] Component IDs are unique and descriptive
- [ ] Loading wrappers around dynamic content
- [ ] No hardcoded colors (use CSS variables)
- [ ] ARIA labels on icon buttons
- [ ] Responsive columns defined
- [ ] Stores placed at end of layout
- [ ] No unused imports
- [ ] File under 500 lines

### Component Checklist

- [ ] Accepts `id` parameter
- [ ] Has type hints
- [ ] Returns single root element
- [ ] Uses semantic HTML where applicable
- [ ] Follows naming conventions

---

## Quick Reference Card

| Pattern         | Example                              |
| --------------- | ------------------------------------ |
| Layout function | `create_xxx_layout()`                |
| Tab function    | `create_xxx_tab()`                   |
| Card function   | `create_xxx_card()`                  |
| Component ID    | `{page}-{component}-{element}`       |
| Store ID        | `{page}-{type}-store`                |
| CSS variable    | `var(--brand-primary)`               |
| Spacing class   | `mb-4`, `py-3`, `gap-2`              |
| Icon class      | `fas fa-xxx me-2`                    |
| Loading wrap    | `dcc.Loading(id="loading-xxx", ...)` |

---

_End of Best Practices Document_
