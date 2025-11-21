"""
Model Comparison Dashboard (Phase 11C).
Statistical comparison of multiple trained models.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_model_comparison_layout():
    """Create model comparison dashboard."""
    return dbc.Container([
        html.H2("Model Comparison", className="mb-4"),

        # Model selection
        dbc.Card([
            dbc.CardBody([
                html.H5("Select Models to Compare", className="card-title mb-3"),

                dbc.Label("Choose 2-5 experiments:"),
                dcc.Dropdown(
                    id="comparison-experiments-dropdown",
                    placeholder="Select experiments to compare...",
                    multi=True,
                    options=[],  # Populated by callback
                    className="mb-3"
                ),

                dbc.Button([html.I(className="fas fa-chart-bar me-2"), "Compare Models"],
                           id="run-comparison-btn", color="primary", disabled=True),
            ])
        ], className="shadow-sm mb-4"),

        # Comparison results
        html.Div(id="comparison-results", children=[]),

    ], fluid=True)


def create_comparison_results(experiments_data):
    """Create comparison results visualization."""
    if not experiments_data or len(experiments_data) < 2:
        return dbc.Alert("Please select at least 2 experiments to compare", color="warning")

    return html.Div([
        # Summary table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Comparison"),
                    dbc.CardBody([
                        html.Div(id="comparison-table")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Metric comparisons
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Accuracy Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id="comparison-accuracy-chart")
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Training Time Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id="comparison-time-chart")
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Statistical tests
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Statistical Significance Tests"),
                    dbc.CardBody([
                        html.Div(id="statistical-tests-results")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Per-class comparison
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Per-Class F1 Scores"),
                    dbc.CardBody([
                        dcc.Graph(id="comparison-per-class-chart")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Export options
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([html.I(className="fas fa-download me-2"), "Export PDF Report"],
                               id="export-pdf-btn", color="primary"),
                    dbc.Button([html.I(className="fas fa-file-csv me-2"), "Export CSV"],
                               id="export-csv-btn", color="secondary"),
                ])
            ])
        ])
    ])
