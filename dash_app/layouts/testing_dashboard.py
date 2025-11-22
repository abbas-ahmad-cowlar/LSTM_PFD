"""
Testing & QA Dashboard Layout.
Run tests, view coverage, and analyze benchmarks.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_testing_dashboard_layout():
    """Create testing & QA dashboard layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-vial me-3"),
                    "Testing & QA Dashboard"
                ], className="mb-1"),
                html.P("Run tests, analyze coverage, and monitor code quality", className="text-muted mb-4")
            ])
        ]),

        # Quick Actions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Quick Actions"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="fas fa-play me-2"), "Run All Tests"],
                                    id="run-all-tests-btn",
                                    color="primary",
                                    className="w-100"
                                )
                            ], width=3),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="fas fa-chart-pie me-2"), "Generate Coverage"],
                                    id="run-coverage-btn",
                                    color="success",
                                    className="w-100"
                                )
                            ], width=3),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="fas fa-tachometer-alt me-2"), "Run Benchmarks"],
                                    id="run-benchmarks-btn",
                                    color="info",
                                    className="w-100"
                                )
                            ], width=3),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="fas fa-check-circle me-2"), "Quality Checks"],
                                    id="run-quality-btn",
                                    color="warning",
                                    className="w-100"
                                )
                            ], width=3),
                        ])
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Tabs
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Tabs([
                            # Test Execution Tab
                            dbc.Tab(
                                label="Test Execution",
                                tab_id="tests-tab",
                                children=[
                                    html.Div([
                                        # Test Configuration
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Test Path"),
                                                dbc.Input(
                                                    id="test-path-input",
                                                    placeholder="tests/",
                                                    value="tests/"
                                                )
                                            ], width=6),
                                            dbc.Col([
                                                dbc.Label("Markers (optional)"),
                                                dbc.Input(
                                                    id="test-markers-input",
                                                    placeholder="e.g., unit, integration"
                                                )
                                            ], width=6),
                                        ], className="mb-3"),

                                        # Test Results
                                        html.Div(id="test-results-content"),

                                        # Test Output
                                        html.Div([
                                            html.H5("Test Output", className="mt-3"),
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.Pre(
                                                        id="test-output-text",
                                                        style={
                                                            'maxHeight': '400px',
                                                            'overflow': 'auto',
                                                            'backgroundColor': '#f8f9fa',
                                                            'padding': '10px',
                                                            'fontSize': '12px'
                                                        }
                                                    )
                                                ])
                                            ])
                                        ])
                                    ], className="p-3")
                                ]
                            ),

                            # Coverage Tab
                            dbc.Tab(
                                label="Coverage Analysis",
                                tab_id="coverage-tab",
                                children=[
                                    html.Div([
                                        # Coverage Configuration
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Test Path"),
                                                dbc.Input(
                                                    id="coverage-test-path",
                                                    placeholder="tests/",
                                                    value="tests/"
                                                )
                                            ], width=4),
                                            dbc.Col([
                                                dbc.Label("Source Path"),
                                                dbc.Input(
                                                    id="coverage-source-path",
                                                    placeholder=".",
                                                    value="."
                                                )
                                            ], width=4),
                                            dbc.Col([
                                                dbc.Label("Min Coverage (%)"),
                                                dbc.Input(
                                                    id="coverage-threshold",
                                                    type="number",
                                                    value=80,
                                                    min=0,
                                                    max=100
                                                )
                                            ], width=4),
                                        ], className="mb-3"),

                                        # Coverage Results
                                        html.Div(id="coverage-results-content"),

                                        # Coverage Chart
                                        dcc.Graph(id="coverage-chart")
                                    ], className="p-3")
                                ]
                            ),

                            # Benchmarks Tab
                            dbc.Tab(
                                label="Performance Benchmarks",
                                tab_id="benchmarks-tab",
                                children=[
                                    html.Div([
                                        # Benchmark Configuration
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Model Path (optional)"),
                                                dbc.Input(
                                                    id="benchmark-model-path",
                                                    placeholder="path/to/model.pth"
                                                )
                                            ], width=6),
                                            dbc.Col([
                                                dbc.Label("API URL (optional)"),
                                                dbc.Input(
                                                    id="benchmark-api-url",
                                                    placeholder="http://localhost:8000"
                                                )
                                            ], width=6),
                                        ], className="mb-3"),

                                        # Benchmark Results
                                        html.Div(id="benchmark-results-content"),

                                        # Benchmark Charts
                                        dbc.Row([
                                            dbc.Col([
                                                dcc.Graph(id="benchmark-latency-chart")
                                            ], width=6),
                                            dbc.Col([
                                                dcc.Graph(id="benchmark-throughput-chart")
                                            ], width=6),
                                        ])
                                    ], className="p-3")
                                ]
                            ),

                            # Quality Tab
                            dbc.Tab(
                                label="Code Quality",
                                tab_id="quality-tab",
                                children=[
                                    html.Div([
                                        # Quality Configuration
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Path to Check"),
                                                dbc.Input(
                                                    id="quality-path",
                                                    placeholder=".",
                                                    value="."
                                                )
                                            ], width=12),
                                        ], className="mb-3"),

                                        # Quality Results
                                        html.Div(id="quality-results-content")
                                    ], className="p-3")
                                ]
                            ),

                        ], id="testing-tabs", active_tab="tests-tab")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Task Storage
        dcc.Store(id='test-task-id'),
        dcc.Store(id='coverage-task-id'),
        dcc.Store(id='benchmark-task-id'),
        dcc.Store(id='quality-task-id'),

        # Polling interval for task status
        dcc.Interval(
            id='testing-task-polling',
            interval=2*1000,  # Check every 2 seconds
            n_intervals=0,
            disabled=True
        ),

    ], fluid=True)
