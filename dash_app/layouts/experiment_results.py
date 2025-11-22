"""
Experiment results visualization dashboard (Phase 11B).
Comprehensive results analysis for completed experiments.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_experiment_results_layout(experiment_id):
    """Create detailed results visualization for an experiment."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2(id="results-experiment-title", className="mb-2"),
                html.P(id="results-experiment-subtitle", className="text-muted mb-4"),
            ], width=10),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([html.I(className="fas fa-download me-2"), "Export"],
                               id="export-results-btn", color="primary", size="sm"),
                    dbc.Button([html.I(className="fas fa-share me-2"), "Share"],
                               id="share-results-btn", color="info", size="sm"),
                ], className="float-end")
            ], width=2),
        ]),

        # Key metrics cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Test Accuracy", className="text-muted"),
                        html.H3(id="results-test-accuracy", className="text-success"),
                    ])
                ], className="shadow-sm text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Test Loss", className="text-muted"),
                        html.H3(id="results-test-loss", className="text-warning"),
                    ])
                ], className="shadow-sm text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Training Time", className="text-muted"),
                        html.H3(id="results-training-time", className="text-info"),
                    ])
                ], className="shadow-sm text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Best Epoch", className="text-muted"),
                        html.H3(id="results-best-epoch", className="text-primary"),
                    ])
                ], className="shadow-sm text-center")
            ], width=3),
        ], className="mb-4"),

        # Training history plots
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Training & Validation Loss"),
                    dbc.CardBody([
                        dcc.Graph(id="results-loss-plot")
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Training & Validation Accuracy"),
                    dbc.CardBody([
                        dcc.Graph(id="results-accuracy-plot")
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Confusion matrix and metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Confusion Matrix"),
                    dbc.CardBody([
                        dcc.Graph(id="results-confusion-matrix")
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Per-Class Metrics"),
                    dbc.CardBody([
                        html.Div(id="results-per-class-metrics")
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Model architecture and hyperparameters
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Configuration"),
                    dbc.CardBody([
                        html.Pre(id="results-model-config", className="bg-light p-3",
                                 style={"fontSize": "12px", "maxHeight": "400px", "overflow": "auto"})
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Training Hyperparameters"),
                    dbc.CardBody([
                        html.Div(id="results-hyperparameters")
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Hidden stores
        dcc.Store(id="results-experiment-id", data=experiment_id),

    ], fluid=True)
