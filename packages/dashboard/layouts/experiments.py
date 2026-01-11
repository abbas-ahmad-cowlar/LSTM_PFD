"""
Experiments list layout (Phase 11B).
Full experiment history with filtering, sorting, and comparison.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_experiments_layout():
    """Create comprehensive experiments list layout."""
    return dbc.Container([
        html.H2("Experiment History", className="mb-4"),

        # Filters and actions row
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dcc.Dropdown(
                        id="saved-searches-dropdown",
                        placeholder="📌 Saved searches...",
                        options=[],  # Populated by callback
                        className="flex-grow-0",
                        style={'minWidth': '180px'}
                    ),
                    dbc.Input(
                        id="experiment-search",
                        placeholder="Search by name, tags, or notes...",
                        type="text",
                        debounce=True
                    ),
                    dbc.Button(
                        html.I(className="fas fa-bookmark"),
                        id="save-search-btn",
                        color="secondary",
                        outline=True,
                        size="sm",
                        title="Save current search"
                    ),
                ], className="mb-0"),
            ], width=3),
            dbc.Col([
                dcc.Dropdown(
                    id="experiment-tag-filter",
                    placeholder="Filter by tags",
                    options=[],  # Populated by callback
                    multi=True
                ),
            ], width=2),
            dbc.Col([
                dcc.Dropdown(
                    id="experiment-model-filter",
                    placeholder="Filter by model",
                    options=[],  # Populated by callback
                    multi=True
                ),
            ], width=2),
            dbc.Col([
                dcc.Dropdown(
                    id="experiment-status-filter",
                    placeholder="Filter by status",
                    options=[
                        {"label": "Pending", "value": "pending"},
                        {"label": "Running", "value": "running"},
                        {"label": "Completed", "value": "completed"},
                        {"label": "Failed", "value": "failed"},
                        {"label": "Cancelled", "value": "cancelled"},
                    ],
                    multi=True
                ),
            ], width=2),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([html.I(className="fas fa-plus me-2"), "New"],
                               href="/experiment/new", color="primary", size="sm"),
                    dbc.Button([html.I(className="fas fa-tags me-2"), "Manage Tags"],
                               id="manage-tags-btn", color="success", size="sm", disabled=True),
                    dbc.Button([html.I(className="fas fa-chart-bar me-2"), "Compare"],
                               id="compare-experiments-btn", color="info", size="sm", disabled=True),
                ], className="float-end")
            ], width=3),
        ], className="mb-4"),

        # Summary statistics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id="experiments-summary")
                    ])
                ], className="shadow-sm mb-4")
            ])
        ]),

        # Experiments table
        dbc.Card([
            dbc.CardBody([
                html.Div(id="experiments-table-container", children=[
                    html.P("Loading experiments...", className="text-muted text-center")
                ])
            ])
        ], className="shadow-sm mb-4"),

        # Comparison cart
        dbc.Offcanvas(
            id="comparison-offcanvas",
            title="Experiment Comparison",
            is_open=False,
            placement="end",
            style={"width": "600px"},
            children=[
                html.Div(id="comparison-cart-content"),
                html.Hr(),
                dbc.Button([html.I(className="fas fa-chart-line me-2"), "View Comparison"],
                           id="view-comparison-btn", color="primary", className="w-100"),
            ]
        ),

        # Tag Management Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Manage Tags")),
            dbc.ModalBody([
                html.P(id="selected-experiments-count", className="text-muted mb-3"),

                html.H6("Add Tags", className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id="tag-autocomplete",
                            placeholder="Type to search tags or create new...",
                            options=[],  # Populated by callback
                            multi=True,
                            className="mb-3"
                        ),
                    ], width=9),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-plus")],
                            id="add-tags-btn",
                            color="success",
                            className="w-100"
                        ),
                    ], width=3),
                ]),
                html.P("Popular tags:", className="small text-muted mb-2"),
                html.Div(id="popular-tags-chips", className="mb-4"),

                html.Hr(),

                html.H6("Current Tags", className="mb-2"),
                html.P("Tags on selected experiments:", className="small text-muted mb-2"),
                html.Div(id="current-tags-display", className="mb-3"),

                html.Div(id="tag-operation-status", className="mt-3"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-tag-modal-btn", color="secondary")
            ])
        ], id="tag-management-modal", is_open=False, size="lg"),

        # Save Search Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("💾 Save Search Query")),
            dbc.ModalBody([
                html.P("Save this search for quick access later:", className="text-muted mb-3"),

                # Show current search query
                html.Div([
                    html.Label("Current Search:", className="fw-bold small"),
                    html.Pre(
                        id="current-search-query-display",
                        className="bg-light p-2 rounded small",
                        style={'whiteSpace': 'pre-wrap'}
                    ),
                ], className="mb-3"),

                # Save search name
                dbc.Label("Name *", html_for="save-search-name-input"),
                dbc.Input(
                    id="save-search-name-input",
                    placeholder="e.g., High Accuracy Baselines, Recent ResNets...",
                    type="text",
                    className="mb-3"
                ),
                dbc.FormText("Give your search a descriptive name"),

                # Pin option
                dbc.Checkbox(
                    id="pin-search-checkbox",
                    label="📌 Pin to top (show first in saved searches)",
                    value=False,
                    className="mb-3"
                ),

                # Status message
                html.Div(id="save-search-status", className="mt-3"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-save-search-btn", color="secondary", className="me-2"),
                dbc.Button("Save Search", id="confirm-save-search-btn", color="primary")
            ])
        ], id="save-search-modal", is_open=False),

        # Delete Search Confirmation Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("🗑️ Delete Saved Search?")),
            dbc.ModalBody([
                html.P("Are you sure you want to delete this saved search?"),
                html.Div(id="delete-search-info"),
                html.P([
                    html.Strong("This action cannot be undone."),
                ], className="text-warning small mt-2"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-delete-search-btn", color="secondary", className="me-2"),
                dbc.Button("Delete", id="confirm-delete-search-btn", color="danger")
            ])
        ], id="delete-search-modal", is_open=False),

        # Hidden stores
        dcc.Store(id="selected-experiments-store", data=[]),
        dcc.Store(id="selected-saved-search-id", data=None),

    ], fluid=True)


def create_experiments_table(experiments):
    """Create interactive experiments table."""
    if not experiments:
        return html.P("No experiments found", className="text-muted text-center")

    # Import here to avoid circular imports
    from database.connection import get_session
    from models.tag import ExperimentTag
    from sqlalchemy.orm import joinedload

    # Prepare table data with tags
    session = get_session()

    # Bulk load all tags for all experiments to avoid N+1 query (CRITICAL OPTIMIZATION!)
    # This reduces queries from N+1 to just 2 queries total
    experiment_ids = [exp.id for exp in experiments]
    experiment_tags_map = {}  # experiment_id -> list of tag names

    if experiment_ids:
        # Single query with eager loading to get all tags for all experiments
        experiment_tag_mappings = session.query(ExperimentTag).options(
            joinedload(ExperimentTag.tag)  # Eager load Tag relationship
        ).filter(
            ExperimentTag.experiment_id.in_(experiment_ids)
        ).all()

        # Build mapping: experiment_id -> [tag names]
        for exp_tag in experiment_tag_mappings:
            if exp_tag.tag:  # Ensure tag exists
                if exp_tag.experiment_id not in experiment_tags_map:
                    experiment_tags_map[exp_tag.experiment_id] = []
                experiment_tags_map[exp_tag.experiment_id].append(exp_tag.tag.name)

    table_data = []
    for exp in experiments:
        # Get tags from pre-loaded mapping (no additional query!)
        tags = experiment_tags_map.get(exp.id, [])
        tag_names = ', '.join(tags) if tags else ""

        table_data.append({
            "id": exp.id,
            "name": exp.name,
            "model_type": exp.model_type,
            "tags": tag_names,
            "status": exp.status.value,
            "accuracy": f"{exp.metrics.get('test_accuracy', 0) * 100:.2f}%" if exp.metrics else "N/A",
            "epochs": f"{exp.best_epoch}/{exp.total_epochs}" if exp.total_epochs else "N/A",
            "duration": format_duration(exp.duration_seconds) if exp.duration_seconds else "N/A",
            "created_at": exp.created_at.strftime("%Y-%m-%d %H:%M") if exp.created_at else "N/A",
        })
    session.close()

    return dash_table.DataTable(
        id="experiments-table",
        columns=[
            {"name": "ID", "id": "id", "type": "numeric"},
            {"name": "Name", "id": "name"},
            {"name": "Model", "id": "model_type"},
            {"name": "Tags", "id": "tags"},
            {"name": "Status", "id": "status"},
            {"name": "Accuracy", "id": "accuracy"},
            {"name": "Epochs", "id": "epochs"},
            {"name": "Duration", "id": "duration"},
            {"name": "Created", "id": "created_at"},
        ],
        data=table_data,
        sort_action="native",
        filter_action="native",
        row_selectable="multi",
        selected_rows=[],
        page_action="native",
        page_size=20,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': '14px',
        },
        style_cell_conditional=[
            {
                'if': {'column_id': 'tags'},
                'fontSize': '12px',
                'color': '#666'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{status} = "completed"'},
                'backgroundColor': 'rgba(40, 167, 69, 0.1)'
            },
            {
                'if': {'filter_query': '{status} = "failed"'},
                'backgroundColor': 'rgba(220, 53, 69, 0.1)'
            },
            {
                'if': {'filter_query': '{status} = "running"'},
                'backgroundColor': 'rgba(0, 123, 255, 0.1)'
            },
        ],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
    )


def format_duration(seconds):
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
