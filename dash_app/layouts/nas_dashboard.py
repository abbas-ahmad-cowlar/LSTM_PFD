"""
Neural Architecture Search (NAS) Dashboard Layout (Phase 4, Feature 3/3).
Provides UI for creating and monitoring NAS campaigns.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_nas_dashboard_layout():
    """
    Create the NAS dashboard layout.

    Features:
        - NAS campaign creation
        - Campaign list and monitoring
        - Architecture visualization
        - Results analysis
    """
    return dbc.Container([
        # Page Header
        dbc.Row([
            dbc.Col([
                html.H2("🔬 Neural Architecture Search", className="mb-3"),
                html.P(
                    "Automatically discover optimal neural architectures for your fault classification task.",
                    className="text-muted"
                )
            ])
        ], className="mb-4"),

        # Campaign Creation Card
        dbc.Card([
            dbc.CardHeader([
                html.H5("Create NAS Campaign", className="mb-0 d-inline"),
                dbc.Button(
                    [html.I(className="bi bi-info-circle me-2"), "Help"],
                    id='nas-help-btn',
                    color="link",
                    size="sm",
                    className="float-end"
                ),
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Campaign Name:", className="fw-bold"),
                        dbc.Input(
                            id="nas-campaign-name",
                            placeholder="e.g., CNN-Search-1",
                            type="text",
                            className="mb-3"
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Dataset:", className="fw-bold"),
                        dcc.Dropdown(
                            id="nas-dataset-select",
                            placeholder="Choose a dataset...",
                            className="mb-3"
                        )
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Search Algorithm:", className="fw-bold"),
                        dbc.RadioItems(
                            id="nas-search-algorithm",
                            options=[
                                {"label": "Random Search (Fast, baseline)", "value": "random"},
                                {"label": "Bayesian Optimization (Coming soon)", "value": "bayesian", "disabled": True},
                            ],
                            value="random",
                            className="mb-3"
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Number of Trials:", className="fw-bold"),
                        dbc.Input(
                            id="nas-num-trials",
                            type="number",
                            value=20,
                            min=5,
                            max=100,
                            className="mb-3"
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Epochs per Trial:", className="fw-bold"),
                        dbc.Input(
                            id="nas-epochs-per-trial",
                            type="number",
                            value=10,
                            min=5,
                            max=50,
                            className="mb-3"
                        )
                    ], width=4),
                ]),

                # Search Space Configuration (Accordion)
                html.Hr(),
                html.H6("Search Space Configuration", className="mb-3"),
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Checklist(
                            id="nas-operations",
                            options=[
                                {"label": " Conv 3x1", "value": "conv_3"},
                                {"label": " Conv 5x1", "value": "conv_5"},
                                {"label": " Conv 7x1", "value": "conv_7"},
                                {"label": " Separable Conv 3x1", "value": "sep_conv_3"},
                                {"label": " Max Pool 3x1", "value": "max_pool_3"},
                                {"label": " Avg Pool 3x1", "value": "avg_pool_3"},
                                {"label": " Skip Connection", "value": "skip_connect"},
                            ],
                            value=["conv_3", "conv_5", "max_pool_3", "skip_connect"],
                            className="mb-3"
                        ),
                    ], title="Operations"),
                    dbc.AccordionItem([
                        dbc.Checklist(
                            id="nas-channel-sizes",
                            options=[
                                {"label": " 32 channels", "value": 32},
                                {"label": " 64 channels", "value": 64},
                                {"label": " 128 channels", "value": 128},
                                {"label": " 256 channels", "value": 256},
                                {"label": " 512 channels", "value": 512},
                            ],
                            value=[32, 64, 128, 256],
                            className="mb-3"
                        ),
                    ], title="Channel Sizes"),
                    dbc.AccordionItem([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Min Layers:"),
                                dbc.Input(id="nas-min-layers", type="number", value=3, min=1, max=20)
                            ], width=6),
                            dbc.Col([
                                html.Label("Max Layers:"),
                                dbc.Input(id="nas-max-layers", type="number", value=10, min=1, max=20)
                            ], width=6),
                        ])
                    ], title="Network Depth"),
                ], start_collapsed=False, className="mb-3"),

                # Launch button and status
                html.Div(id='nas-creation-status', className="mb-3"),
                dbc.Button(
                    [html.I(className="bi bi-rocket me-2"), "Launch NAS Campaign"],
                    id="launch-nas-btn",
                    color="primary",
                    size="lg",
                    className="w-100"
                ),
            ])
        ], className="mb-4"),

        # Campaigns List
        html.H4("NAS Campaigns", className="mt-4 mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh"],
                    id='reload-nas-campaigns-btn',
                    color="secondary",
                    size="sm",
                    className="mb-3"
                ),
            ])
        ]),
        dcc.Loading(
            id="loading-nas-campaigns",
            children=[html.Div(id='nas-campaigns-table')],
            type="default"
        ),

        # Campaign Details Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id='nas-campaign-modal-title')),
            dbc.ModalBody([
                # Campaign info
                html.Div(id="nas-campaign-info"),

                html.Hr(),

                # Trials table
                html.H5("Trials", className="mt-3 mb-3"),
                html.Div(id="nas-trials-table"),

                html.Hr(),

                # Best architecture
                html.H5("Best Architecture", className="mt-3 mb-3"),
                html.Div(id="nas-best-architecture"),

                # Architecture visualization
                dcc.Loading(
                    dcc.Graph(id="nas-architecture-graph", style={'height': '400px'}),
                    type="default"
                ),

            ], style={'maxHeight': '80vh', 'overflowY': 'auto'}),
            dbc.ModalFooter([
                dbc.Button("Export Best Architecture", id="export-nas-architecture-btn", color="primary", className="me-2"),
                dbc.Button("Close", id='close-nas-modal-btn', color="secondary")
            ])
        ], id='nas-campaign-modal', is_open=False, size="xl"),

        # Architecture Export Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Export Architecture")),
            dbc.ModalBody([
                html.Label("Export Format:", className="fw-bold mb-2"),
                dbc.RadioItems(
                    id="nas-export-format",
                    options=[
                        {"label": "PyTorch Code", "value": "pytorch"},
                        {"label": "JSON Configuration", "value": "json"},
                    ],
                    value="pytorch",
                    className="mb-3"
                ),
                html.Hr(),
                html.Pre(id='nas-exported-code', style={'backgroundColor': '#f5f5f5', 'padding': '1rem', 'overflowX': 'auto'})
            ]),
            dbc.ModalFooter([
                dbc.Button("Copy to Clipboard", id="copy-nas-code-btn", color="primary", className="me-2"),
                dbc.Button("Close", id='close-export-modal-btn', color="secondary")
            ])
        ], id='nas-export-modal', is_open=False, size="lg"),

        # Store for selected campaign ID
        dcc.Store(id='selected-nas-campaign-id', data=None),

        # Auto-refresh interval
        dcc.Interval(
            id='nas-auto-refresh',
            interval=10*1000,  # 10 seconds
            n_intervals=0,
            disabled=False
        ),

    ], fluid=True, className="py-4")
