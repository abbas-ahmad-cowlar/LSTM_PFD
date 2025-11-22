"""
Enhanced Evaluation Dashboard Layout.
ROC curves, error analysis, architecture comparison.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_evaluation_dashboard_layout():
    """Create enhanced evaluation dashboard layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-chart-bar me-3"),
                    "Enhanced Model Evaluation"
                ], className="mb-1"),
                html.P("ROC curves, error analysis, and architecture comparison", className="text-muted mb-4")
            ])
        ]),

        # Experiment Selection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Experiments"),
                    dbc.CardBody([
                        dbc.Label("Experiment"),
                        dcc.Dropdown(
                            id="eval-experiment-select",
                            placeholder="Select a completed experiment...",
                            className="mb-3"
                        ),
                        dbc.Label("Compare Multiple (optional)"),
                        dcc.Dropdown(
                            id="eval-compare-experiments",
                            placeholder="Select 2-5 experiments to compare...",
                            multi=True
                        )
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Evaluation Tabs
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Evaluation Analysis"),
                    dbc.CardBody([
                        dbc.Tabs([
                            # ROC Analysis Tab
                            dbc.Tab(
                                label="ROC Curves",
                                tab_id="roc-tab",
                                children=[
                                    html.Div([
                                        dbc.Button(
                                            [html.I(className="fas fa-play me-2"), "Generate ROC Analysis"],
                                            id="generate-roc-btn",
                                            color="primary",
                                            className="my-3"
                                        ),
                                        html.Div(id="roc-analysis-content")
                                    ], className="p-3")
                                ]
                            ),

                            # Error Analysis Tab
                            dbc.Tab(
                                label="Error Analysis",
                                tab_id="error-tab",
                                children=[
                                    html.Div([
                                        dbc.Button(
                                            [html.I(className="fas fa-search me-2"), "Analyze Errors"],
                                            id="analyze-errors-btn",
                                            color="warning",
                                            className="my-3"
                                        ),
                                        html.Div(id="error-analysis-content")
                                    ], className="p-3")
                                ]
                            ),

                            # Architecture Comparison Tab
                            dbc.Tab(
                                label="Architecture Comparison",
                                tab_id="comparison-tab",
                                children=[
                                    html.Div([
                                        dbc.Button(
                                            [html.I(className="fas fa-balance-scale me-2"), "Compare Architectures"],
                                            id="compare-architectures-btn",
                                            color="info",
                                            className="my-3"
                                        ),
                                        html.Div(id="architecture-comparison-content")
                                    ], className="p-3")
                                ]
                            ),

                        ], id="evaluation-tabs", active_tab="roc-tab")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Storage for task IDs
        dcc.Store(id='roc-task-id'),
        dcc.Store(id='error-task-id'),

        # Polling interval for task status
        dcc.Interval(
            id='evaluation-task-polling',
            interval=2*1000,  # Check every 2 seconds
            n_intervals=0,
            disabled=True
        ),

    ], fluid=True)
