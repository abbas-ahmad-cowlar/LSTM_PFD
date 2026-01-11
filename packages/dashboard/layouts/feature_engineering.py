"""
Feature Engineering Dashboard Layout.
Extract, select, and analyze features.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_feature_engineering_layout():
    """Create feature engineering dashboard."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-magic me-3"),
                    "Feature Engineering"
                ], className="mb-1"),
                html.P("Extract, select, and analyze features from your datasets", className="text-muted mb-4")
            ])
        ]),

        # Tabs
        dbc.Card([
            dbc.CardBody([
                dbc.Tabs([
                    # Feature Extraction Tab
                    dbc.Tab(
                        label="Feature Extraction",
                        tab_id="extraction-tab",
                        children=[
                            html.Div([
                                # Dataset Selection
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Select Dataset"),
                                        dcc.Dropdown(
                                            id="fe-dataset-select",
                                            placeholder="Choose a dataset...",
                                            className="mb-3"
                                        )
                                    ], width=12),
                                ]),

                                # Domain Selection
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Feature Domain"),
                                        dbc.RadioItems(
                                            id="fe-domain",
                                            options=[
                                                {"label": "All Features (36 features)", "value": "all"},
                                                {"label": "Time Domain (7 features)", "value": "time"},
                                                {"label": "Frequency Domain (12 features)", "value": "frequency"},
                                                {"label": "Wavelet Domain (7 features)", "value": "wavelet"},
                                                {"label": "Bispectrum (6 features)", "value": "bispectrum"},
                                                {"label": "Envelope Analysis (4 features)", "value": "envelope"}
                                            ],
                                            value="all",
                                            className="mb-3"
                                        )
                                    ], width=12),
                                ]),

                                # Extract Button
                                dbc.Button(
                                    [html.I(className="fas fa-play me-2"), "Extract Features"],
                                    id="extract-features-btn",
                                    color="primary",
                                    size="lg",
                                    className="mb-4"
                                ),

                                # Results
                                html.Div(id="fe-extraction-results")

                            ], className="p-4")
                        ]
                    ),

                    # Feature Importance Tab
                    dbc.Tab(
                        label="Feature Importance",
                        tab_id="importance-tab",
                        children=[
                            html.Div([
                                # Dataset and Domain Selection
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Dataset"),
                                        dcc.Dropdown(
                                            id="fi-dataset-select",
                                            placeholder="Choose a dataset..."
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Feature Domain"),
                                        dcc.Dropdown(
                                            id="fi-domain",
                                            options=[
                                                {"label": "All Features", "value": "all"},
                                                {"label": "Time Domain", "value": "time"},
                                                {"label": "Frequency Domain", "value": "frequency"},
                                                {"label": "Wavelet Domain", "value": "wavelet"},
                                                {"label": "Bispectrum", "value": "bispectrum"},
                                                {"label": "Envelope", "value": "envelope"}
                                            ],
                                            value="all"
                                        )
                                    ], width=6),
                                ], className="mb-3"),

                                # Method Selection
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Importance Method"),
                                        dcc.Dropdown(
                                            id="fi-method",
                                            options=[
                                                {"label": "Mutual Information (Fast)", "value": "mutual_info"},
                                                {"label": "Random Forest Gini", "value": "random_forest"},
                                                {"label": "Permutation Importance (Slow)", "value": "permutation"}
                                            ],
                                            value="mutual_info"
                                        )
                                    ], width=12),
                                ], className="mb-3"),

                                # Compute Button
                                dbc.Button(
                                    [html.I(className="fas fa-chart-bar me-2"), "Compute Importance"],
                                    id="compute-importance-btn",
                                    color="primary",
                                    size="lg",
                                    className="mb-4"
                                ),

                                # Importance Chart
                                html.Div(id="fi-results")

                            ], className="p-4")
                        ]
                    ),

                    # Feature Selection Tab
                    dbc.Tab(
                        label="Feature Selection",
                        tab_id="selection-tab",
                        children=[
                            html.Div([
                                # Dataset and Domain Selection
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Dataset"),
                                        dcc.Dropdown(
                                            id="fs-dataset-select",
                                            placeholder="Choose a dataset..."
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Feature Domain"),
                                        dcc.Dropdown(
                                            id="fs-domain",
                                            options=[
                                                {"label": "All Features", "value": "all"},
                                                {"label": "Time Domain", "value": "time"},
                                                {"label": "Frequency Domain", "value": "frequency"},
                                                {"label": "Wavelet Domain", "value": "wavelet"},
                                                {"label": "Bispectrum", "value": "bispectrum"},
                                                {"label": "Envelope", "value": "envelope"}
                                            ],
                                            value="all"
                                        )
                                    ], width=6),
                                ], className="mb-3"),

                                # Selection Method
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Selection Method"),
                                        dcc.Dropdown(
                                            id="fs-method",
                                            options=[
                                                {"label": "MRMR (Minimum Redundancy Maximum Relevance)", "value": "mrmr"},
                                                {"label": "Variance Threshold", "value": "variance"},
                                                {"label": "Mutual Information", "value": "mutual_info"},
                                                {"label": "Recursive Feature Elimination (RFE)", "value": "rfe"}
                                            ],
                                            value="mrmr"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Number of Features"),
                                        dbc.Input(
                                            id="fs-num-features",
                                            type="number",
                                            value=15,
                                            min=1,
                                            max=36
                                        )
                                    ], width=6),
                                ], className="mb-3"),

                                # Select Button
                                dbc.Button(
                                    [html.I(className="fas fa-filter me-2"), "Select Features"],
                                    id="select-features-btn",
                                    color="primary",
                                    size="lg",
                                    className="mb-4"
                                ),

                                # Selected Features Results
                                html.Div(id="fs-results")

                            ], className="p-4")
                        ]
                    ),

                    # Correlation Matrix Tab
                    dbc.Tab(
                        label="Correlation Analysis",
                        tab_id="correlation-tab",
                        children=[
                            html.Div([
                                # Dataset and Domain Selection
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Dataset"),
                                        dcc.Dropdown(
                                            id="corr-dataset-select",
                                            placeholder="Choose a dataset..."
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Feature Domain"),
                                        dcc.Dropdown(
                                            id="corr-domain",
                                            options=[
                                                {"label": "All Features", "value": "all"},
                                                {"label": "Time Domain", "value": "time"},
                                                {"label": "Frequency Domain", "value": "frequency"},
                                                {"label": "Wavelet Domain", "value": "wavelet"},
                                                {"label": "Bispectrum", "value": "bispectrum"},
                                                {"label": "Envelope", "value": "envelope"}
                                            ],
                                            value="all"
                                        )
                                    ], width=6),
                                ], className="mb-3"),

                                # Compute Button
                                dbc.Button(
                                    [html.I(className="fas fa-project-diagram me-2"), "Compute Correlation"],
                                    id="compute-correlation-btn",
                                    color="primary",
                                    size="lg",
                                    className="mb-4"
                                ),

                                # Correlation Heatmap
                                html.Div(id="corr-results")

                            ], className="p-4")
                        ]
                    ),

                ], id="feature-tabs", active_tab="extraction-tab")
            ])
        ], className="shadow-sm"),

        # Storage for extracted features
        dcc.Store(id='extracted-features-store'),
        dcc.Store(id='feature-importance-store'),
        dcc.Store(id='selected-features-store'),

    ], fluid=True)
