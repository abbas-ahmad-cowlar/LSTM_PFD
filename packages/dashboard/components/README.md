# Dashboard Components

> Reusable UI components shared across layout pages.

## Overview

The `components/` directory contains 6 Python modules that provide the shared UI shell and reusable widgets. All layout pages are rendered inside the shell defined by `sidebar.py`, `header.py`, and `footer.py` (assembled in `app.py`).

## Component Catalog

| Component       | File             | Functions Exported                                                                                                                                                     | Description                                                                                                      |
| --------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Sidebar**     | `sidebar.py`     | `create_sidebar()`, `create_sidebar_styles()`                                                                                                                          | Collapsible left navigation with icon-only mode, section groups, tooltips, and admin-only item flags             |
| **Header**      | `header.py`      | `create_header()`                                                                                                                                                      | Top navbar with brand logo, mobile hamburger button, nav links (Home, Experiments, Analytics), and user dropdown |
| **Footer**      | `footer.py`      | `create_footer()`                                                                                                                                                      | Simple footer with version string and documentation/GitHub links                                                 |
| **Cards**       | `cards.py`       | `create_stat_card()`, `create_info_card()`                                                                                                                             | Reusable card widgets for stat display and info panels                                                           |
| **Skeleton**    | `skeleton.py`    | `skeleton_text()`, `skeleton_card()`, `skeleton_table()`, `skeleton_chart()`, `skeleton_metric_card()`, `skeleton_list_item()`, `skeleton_page()`, `loading_wrapper()` | Content-aware skeleton loaders replacing generic spinners                                                        |
| **Tag Manager** | `tag_manager.py` | `create_single_tag()`, `create_tag_display()`, `create_tag_input()`, `create_color_picker()`, `create_tag_filter()`, `create_bulk_tag_bar()`                           | Tag input with autocomplete, color picker, display, filter pills, and bulk operations                            |

## Shell Architecture

The app shell is assembled in [`app.py`](../app.py):

```
┌─────────────────────────────────────────────────┐
│  create_sidebar()          main-wrapper          │
│  ┌──────────┐   ┌──────────────────────────────┐ │
│  │ Sidebar  │   │  create_header()             │ │
│  │ (fixed)  │   ├──────────────────────────────┤ │
│  │          │   │  page-content (from routing) │ │
│  │          │   │                              │ │
│  │          │   ├──────────────────────────────┤ │
│  │          │   │  create_footer()             │ │
│  └──────────┘   └──────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## API Reference

### `create_stat_card(title, value, icon, color)`

Renders a Bootstrap card with an icon float-right and large value text.

| Parameter | Type  | Default           | Description                                    |
| --------- | ----- | ----------------- | ---------------------------------------------- |
| `title`   | `str` | required          | Card heading (muted text)                      |
| `value`   | `str` | required          | Large display value                            |
| `icon`    | `str` | `"fa-chart-line"` | Font Awesome icon class (without `fas` prefix) |
| `color`   | `str` | `"primary"`       | Bootstrap color variant                        |

### `create_info_card(title, content, icon)`

Renders a card with a header containing an icon + title and arbitrary body content.

| Parameter | Type           | Default            | Description             |
| --------- | -------------- | ------------------ | ----------------------- |
| `title`   | `str`          | required           | Card header text        |
| `content` | Dash component | required           | Card body content       |
| `icon`    | `str`          | `"fa-info-circle"` | Font Awesome icon class |

### Skeleton Loaders

All skeleton functions accept a `dark_mode: bool = True` parameter.

| Function                                                            | Key Parameters                                           | Description                              |
| ------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------- |
| `skeleton_text(width, is_title)`                                    | `width`: `'short'`/`'medium'`/`'long'`                   | Single animated text line                |
| `skeleton_card(n_cards, n_lines, show_title)`                       | —                                                        | Card placeholders with text lines        |
| `skeleton_table(n_rows, n_cols)`                                    | —                                                        | Table placeholder with header + rows     |
| `skeleton_chart(chart_type, height)`                                | `chart_type`: `'bar'`/`'line'`/`'pie'`                   | Chart placeholder                        |
| `skeleton_metric_card(n_cards)`                                     | —                                                        | Dashboard metric card placeholders       |
| `skeleton_list_item(n_items, with_avatar)`                          | —                                                        | List with optional avatar circles        |
| `skeleton_page(page_type)`                                          | `page_type`: `'dashboard'`/`'table'`/`'form'`/`'detail'` | Full page skeleton                       |
| `loading_wrapper(loading_component, content_component, is_loading)` | —                                                        | Toggle between skeleton and real content |

### Sidebar Navigation Items

The sidebar defines navigation groups via the `NAV_ITEMS` dict in `sidebar.py`:

| Group       | Items                                                        | Admin-Only Items        |
| ----------- | ------------------------------------------------------------ | ----------------------- |
| main        | Home                                                         | —                       |
| Data        | Generate Data, Data Explorer, Signal Viewer, Datasets        | —                       |
| ML Pipeline | New Experiment, Experiments, Training Monitor, HPO Campaigns | —                       |
| Analytics   | Evaluation, XAI Dashboard, Visualizations                    | —                       |
| Production  | Deployment, API Monitoring, Testing                          | API Monitoring, Testing |
| System      | System Health, Settings                                      | System Health           |

Items marked `admin_only: True` are conditionally rendered based on the user's role.

## Styling

Components are styled through:

- **`dash-bootstrap-components`** — Bootstrap 5 utility classes
- **`assets/custom.css`** — Sidebar styles, card hover effects, dropdown z-index fixes, responsive breakpoints
- **`assets/theme.css`** — Core theme with color variables, typography, and layout rules
- **Inline CSS in `create_sidebar_styles()`** — 290-line CSS block for expanded/collapsed sidebar, scroll, transitions

## Dependencies

- **Requires:** `dash`, `dash-bootstrap-components`, `utils.constants`
- **Provides:** Shell components and widgets consumed by `app.py` and layout modules

## Related Documentation

- [Layouts README](../layouts/README.md) — Page catalog
- [UI Guide](../UI_GUIDE.md) — How to create new components
