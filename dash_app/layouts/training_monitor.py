"""
Real-time training progress monitoring (Phase 11B).
Displays live training metrics and progress.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_training_monitor_layout(experiment_id):
    """
    Create training monitoring layout for a specific experiment.

    Args:
        experiment_id: ID of the experiment to monitor
    """
    return dbc.Container([
        # Header with experiment info
        dbc.Row([
            dbc.Col([
                html.H2(id="experiment-title", className="mb-2"),
                html.P(id="experiment-subtitle", className="text-muted mb-4"),
            ], width=8),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button([html.I(className="fas fa-pause me-2"), "Pause"],
                               id="pause-training-btn", color="warning", outline=True),
                    dbc.Button([html.I(className="fas fa-stop me-2"), "Stop"],
                               id="stop-training-btn", color="danger", outline=True),
                ], className="float-end mt-2")
            ], width=4),
        ]),

        # Progress indicator
        dbc.Card([
            dbc.CardBody([
                html.H5("Training Progress", className="card-title"),
                dbc.Row([
                    dbc.Col([
                        html.Div(id="epoch-progress-text", className="mb-2"),
                        dbc.Progress(id="epoch-progress-bar", className="mb-3", striped=True, animated=True),
                    ], width=6),
                    dbc.Col([
                        html.Div(id="overall-progress-text", className="mb-2"),
                        dbc.Progress(id="overall-progress-bar", className="mb-3"),
                    ], width=6),
                ]),

                # Status badges
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(id="status-badge", className="me-2"),
                        dbc.Badge(id="time-elapsed-badge", color="secondary", className="me-2"),
                        dbc.Badge(id="eta-badge", color="info"),
                    ])
                ])
            ])
        ], className="mb-4 shadow-sm"),

        # Real-time metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Current Epoch Metrics", className="card-title"),
                        html.Div(id="current-metrics-display")
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Best Metrics", className="card-title"),
                        html.Div(id="best-metrics-display")
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Training curves
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Loss Curves"),
                    dbc.CardBody([
                        dcc.Graph(id="loss-curve", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Accuracy Curves"),
                    dbc.CardBody([
                        dcc.Graph(id="accuracy-curve", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Learning rate and other metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Learning Rate Schedule"),
                    dbc.CardBody([
                        dcc.Graph(id="lr-schedule", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Gradient Norm"),
                    dbc.CardBody([
                        dcc.Graph(id="gradient-norm", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Per-Class F1 Scores"),
                    dbc.CardBody([
                        dcc.Graph(id="per-class-f1", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], width=4),
        ], className="mb-4"),

        # Training logs
        dbc.Card([
            dbc.CardHeader([
                html.H5("Training Logs", className="d-inline"),
                dbc.Button([html.I(className="fas fa-download me-2"), "Download"],
                           id="download-logs-btn", color="primary", size="sm", className="float-end")
            ]),
            dbc.CardBody([
                html.Pre(id="training-logs", className="bg-dark text-light p-3",
                         style={"maxHeight": "300px", "overflow": "auto", "fontSize": "12px"})
            ])
        ], className="mb-4 shadow-sm"),

        # Hidden stores and intervals
        dcc.Store(id="experiment-id-store", data=experiment_id),
        dcc.Interval(id="training-update-interval", interval=2000, n_intervals=0),  # Update every 2 seconds
        dcc.Store(id="training-data-store", data={}),

    ], fluid=True)


def format_time_duration(seconds):
    """Format seconds into human-readable duration."""
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


def create_metric_display(label, value, format_str=".4f", color="primary"):
    """Helper to create a metric display element."""
    return html.Div([
        html.Small(label, className="text-muted d-block"),
        html.H4(f"{value:{format_str}}", className=f"text-{color} mb-0")
    ], className="mb-2")
