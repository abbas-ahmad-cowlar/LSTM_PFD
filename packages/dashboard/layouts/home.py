"""
Home dashboard layout.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from components.cards import create_stat_card
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_home_layout():
    """Create home dashboard layout."""
    return dbc.Container([
        html.H2("Dashboard Overview", className="mb-4"),

        # Quick stats row - with IDs for dynamic updates
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-database fa-2x text-primary"),
                    ], className="float-end"),
                    html.H5("Total Signals", className="card-title text-muted"),
                    html.H2(id="home-total-signals", children="Loading...", className="text-primary"),
                ])
            ], className="shadow-sm"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-tags fa-2x text-success"),
                    ], className="float-end"),
                    html.H5("Fault Classes", className="card-title text-muted"),
                    html.H2(id="home-fault-classes", children="Loading...", className="text-success"),
                ])
            ], className="shadow-sm"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-trophy fa-2x text-warning"),
                    ], className="float-end"),
                    html.H5("Best Accuracy", className="card-title text-muted"),
                    html.H2(id="home-best-accuracy", children="Loading...", className="text-warning"),
                ])
            ], className="shadow-sm"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-flask fa-2x text-info"),
                    ], className="float-end"),
                    html.H5("Experiments", className="card-title text-muted"),
                    html.H2(id="home-total-experiments", children="Loading...", className="text-info"),
                ])
            ], className="shadow-sm"), width=3),
        ], className="mb-4"),


        dbc.Row([
            # Quick actions
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Quick Actions"),
                    dbc.CardBody([
                        dbc.Button([
                            html.I(className="fas fa-database me-2"),
                            "Explore Datasets"
                        ], href="/data-explorer", color="primary", className="w-100 mb-2"),
                        dbc.Button([
                            html.I(className="fas fa-signal me-2"),
                            "View Signals"
                        ], href="/signal-viewer", color="secondary", className="w-100 mb-2"),
                        dbc.Button([
                            html.I(className="fas fa-flask me-2"),
                            "Train Model"
                        ], href="/experiment/new", color="success", className="w-100 mb-2"),
                        dbc.Button([
                            html.I(className="fas fa-chart-line me-2"),
                            "View Analytics"
                        ], href="/visualization", color="info", className="w-100"),
                    ])
                ], className="shadow-sm")
            ], width=4),

            # Recent experiments
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Experiments"),
                    dbc.CardBody([
                        html.Div(id="recent-experiments-list", children=[
                            html.P("No recent experiments", className="text-muted")
                        ])
                    ])
                ], className="shadow-sm")
            ], width=8),
        ], className="mb-4"),

        # Charts row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Health"),
                    dbc.CardBody([
                        dcc.Graph(id="system-health-gauge", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Dataset Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="dataset-distribution-chart", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], width=6),
        ])
    ], fluid=True)
