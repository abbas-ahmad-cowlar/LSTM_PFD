"""
Footer component.
"""
import dash_bootstrap_components as dbc
from dash import html
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_footer():
    """Create application footer."""
    return dbc.Container([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P([
                    "LSTM PFD Dashboard v1.0.0 | ",
                    html.A("Documentation", href="/docs", className="text-decoration-none"),
                    " | ",
                    html.A("GitHub", href="https://github.com/abbas-ahmad-cowlar/LSTM_PFD", target="_blank", className="text-decoration-none"),
                ], className="text-muted small text-center")
            ], width=12)
        ]),
    ], fluid=True, className="mt-4 mb-3")
