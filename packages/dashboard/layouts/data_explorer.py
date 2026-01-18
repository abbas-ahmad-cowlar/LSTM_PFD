"""
Data Explorer layout.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from utils.constants import FAULT_CLASSES, SEVERITY_LEVELS


def create_data_explorer_layout():
    """Create data explorer layout."""
    return dbc.Container([
        html.H2("Data Explorer", className="mb-4"),

        dbc.Row([
            # Filter panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters"),
                    dbc.CardBody([
                        html.Label("Dataset"),
                        dcc.Dropdown(
                            id="dataset-selector",
                            placeholder="Select dataset",
                            className="mb-3"
                        ),

                        html.Label("Fault Classes"),
                        dcc.Checklist(
                            id="fault-filter",
                            options=[{"label": fault.replace("_", " ").title(), "value": fault}
                                     for fault in FAULT_CLASSES],
                            value=FAULT_CLASSES,
                            className="mb-3"
                        ),

                        html.Label("Severity Levels"),
                        dcc.Checklist(
                            id="severity-filter",
                            options=[{"label": sev.title(), "value": sev}
                                     for sev in SEVERITY_LEVELS],
                            value=SEVERITY_LEVELS,
                            className="mb-3"
                        ),

                        dbc.Button("Apply Filters", id="apply-filters-btn", color="primary", className="w-100 mb-2"),
                        dbc.Button("Reset", id="reset-filters-btn", color="secondary", className="w-100"),
                    ])
                ], className="shadow-sm sticky-top", style={"top": "20px"})
            ], width=3),

            # Main visualization area
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Overview", tab_id="overview", children=[
                        html.Div([
                            html.H5("Summary Statistics", className="mt-3 mb-3"),
                            html.Div(id="summary-stats-table"),

                            html.H5("Class Distribution", className="mt-4 mb-3"),
                            dcc.Graph(id="class-distribution-chart")
                        ])
                    ]),

                    dbc.Tab(label="Feature Distributions", tab_id="features", children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Feature", className="mt-3"),
                                    dcc.Dropdown(
                                        id="feature-selector",
                                        placeholder="Select feature",
                                        className="mb-3"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Chart Type", className="mt-3"),
                                    dcc.Dropdown(
                                        id="chart-type-selector",
                                        options=[
                                            {"label": "🎻 Violin Plot", "value": "violin"},
                                            {"label": "🔵 Strip/Jitter", "value": "strip"},
                                            {"label": "📈 KDE Lines", "value": "kde"},
                                            {"label": "🏔️ Ridge Plot", "value": "ridge"},
                                            {"label": "📊 Histogram", "value": "histogram"},
                                        ],
                                        value="violin",
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ], width=6),
                            ]),
                            dcc.Loading(
                                id="feature-plot-loading",
                                type="circle",
                                children=dcc.Graph(id="feature-distribution-plot")
                            )
                        ])
                    ]),

                    dbc.Tab(label="Dimensionality Reduction", tab_id="dimred", children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Method", className="mt-3"),
                                    dcc.Dropdown(
                                        id="dimred-method",
                                        options=[
                                            {"label": "t-SNE", "value": "tsne"},
                                            {"label": "PCA", "value": "pca"},
                                            {"label": "UMAP", "value": "umap"}
                                        ],
                                        value="tsne",
                                        className="mb-3"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        dbc.Button("Calculate Projection", id="compute-dimred-btn",
                                                   color="primary", className="mt-4")
                                    ])
                                ], width=6),
                            ]),
                            dcc.Loading(
                                id="dimred-loading",
                                type="default",
                                children=dcc.Graph(id="dimred-plot")
                            )
                        ])
                    ]),

                    # =========================================================
                    # NEW: Spectral Analysis Tab (PSD, Envelope, Cepstrum)
                    # =========================================================
                    dbc.Tab(label="📊 Spectral Analysis", tab_id="spectral", children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Analysis Type", className="mt-3"),
                                    dcc.Dropdown(
                                        id="spectral-analysis-type",
                                        options=[
                                            {"label": "📈 Power Spectral Density (PSD)", "value": "psd"},
                                            {"label": "🌊 Envelope Spectrum", "value": "envelope"},
                                            {"label": "🔄 Cepstrum Analysis", "value": "cepstrum"},
                                        ],
                                        value="psd",
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Fault Class to Analyze", className="mt-3"),
                                    dcc.Dropdown(
                                        id="spectral-fault-selector",
                                        options=[{"label": f.replace("_", " ").title(), "value": f}
                                                 for f in FAULT_CLASSES],
                                        value="normal",
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Rotational Frequency (Hz)", className="mt-2"),
                                    dcc.Input(
                                        id="rotational-freq-input",
                                        type="number",
                                        value=60,
                                        min=1,
                                        max=500,
                                        step=1,
                                        className="form-control mb-3",
                                        style={"width": "150px"}
                                    ),
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        dbc.Checklist(
                                            id="show-harmonics-toggle",
                                            options=[{"label": " Show Harmonic Markers (1X, 2X, 3X)", "value": "show"}],
                                            value=["show"],
                                            className="mt-4"
                                        )
                                    ])
                                ], width=8),
                            ]),
                            dcc.Loading(
                                id="spectral-loading",
                                type="circle",
                                children=dcc.Graph(id="spectral-analysis-plot")
                            ),
                            # Info card
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("About This Analysis", className="card-title"),
                                    html.Div(id="spectral-info-text", className="small text-muted")
                                ])
                            ], className="mt-3", style={"backgroundColor": "#f8f9fa"})
                        ])
                    ]),

                    # =========================================================
                    # NEW: Feature Comparison Tab (Spider, Heatmap, Importance)
                    # =========================================================
                    dbc.Tab(label="🔬 Feature Comparison", tab_id="comparison", children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Comparison Type", className="mt-3"),
                                    dcc.Dropdown(
                                        id="comparison-chart-type",
                                        options=[
                                            {"label": "🕸️ Spider/Radar Chart", "value": "spider"},
                                            {"label": "🔥 Feature Heatmap", "value": "heatmap"},
                                            {"label": "📊 Feature Importance", "value": "importance"},
                                        ],
                                        value="spider",
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Normalize Values", className="mt-3"),
                                    dbc.Checklist(
                                        id="normalize-features-toggle",
                                        options=[{"label": " Normalize to 0-1 range", "value": "normalize"}],
                                        value=["normalize"],
                                        className="mt-2"
                                    )
                                ], width=6),
                            ]),
                            dcc.Loading(
                                id="comparison-loading",
                                type="circle",
                                children=dcc.Graph(id="feature-comparison-plot", style={"height": "550px"})
                            ),
                            # Legend/description
                            html.Div(id="comparison-description", className="mt-3 small text-muted text-center")
                        ])
                    ]),

                ], id="data-explorer-tabs", active_tab="overview"),

                html.Hr(),

                html.H5("Signals Table", className="mt-4 mb-3"),
                html.Div(id="signals-table")
            ], width=9),
        ])
    ], fluid=True)
