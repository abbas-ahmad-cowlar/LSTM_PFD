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
                            html.Label("Select Feature", className="mt-3"),
                            dcc.Dropdown(
                                id="feature-selector",
                                placeholder="Select feature",
                                className="mb-3"
                            ),
                            dcc.Graph(id="feature-distribution-plot")
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
                ], id="data-explorer-tabs", active_tab="overview"),

                html.Hr(),

                html.H5("Signals Table", className="mt-4 mb-3"),
                html.Div(id="signals-table")
            ], width=9),
        ])
    ], fluid=True)
