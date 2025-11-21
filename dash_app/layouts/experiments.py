"""
Experiments list layout (Phase 11B).
Full experiment history with filtering, sorting, and comparison.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table


def create_experiments_layout():
    """Create comprehensive experiments list layout."""
    return dbc.Container([
        html.H2("Experiment History", className="mb-4"),

        # Filters and actions row
        dbc.Row([
            dbc.Col([
                dbc.Input(
                    id="experiment-search",
                    placeholder="Search by name, tags, or notes...",
                    type="text",
                    debounce=True
                ),
            ], width=4),
            dbc.Col([
                dcc.Dropdown(
                    id="experiment-model-filter",
                    placeholder="Filter by model type",
                    options=[],  # Populated by callback
                    multi=True
                ),
            ], width=3),
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

        # Hidden stores
        dcc.Store(id="selected-experiments-store", data=[]),

    ], fluid=True)


def create_experiments_table(experiments):
    """Create interactive experiments table."""
    if not experiments:
        return html.P("No experiments found", className="text-muted text-center")

    # Prepare table data
    table_data = []
    for exp in experiments:
        table_data.append({
            "id": exp.id,
            "name": exp.name,
            "model_type": exp.model_type,
            "status": exp.status.value,
            "accuracy": f"{exp.metrics.get('test_accuracy', 0) * 100:.2f}%" if exp.metrics else "N/A",
            "epochs": f"{exp.best_epoch}/{exp.total_epochs}" if exp.total_epochs else "N/A",
            "duration": format_duration(exp.duration_seconds) if exp.duration_seconds else "N/A",
            "created_at": exp.created_at.strftime("%Y-%m-%d %H:%M") if exp.created_at else "N/A",
        })

    return dash_table.DataTable(
        id="experiments-table",
        columns=[
            {"name": "ID", "id": "id", "type": "numeric"},
            {"name": "Name", "id": "name"},
            {"name": "Model", "id": "model_type"},
            {"name": "Status", "id": "status"},
            {"name": "Test Accuracy", "id": "accuracy"},
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
