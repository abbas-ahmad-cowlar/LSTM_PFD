"""
Reusable card components.
"""
import dash_bootstrap_components as dbc
from dash import html
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_stat_card(title: str, value: str, icon: str = "fa-chart-line", color: str = "primary"):
    """Create a statistic card."""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas {icon} fa-2x text-{color}"),
            ], className="float-end"),
            html.H5(title, className="card-title text-muted"),
            html.H2(value, className=f"text-{color}"),
        ])
    ], className="shadow-sm")


def create_info_card(title: str, content, icon: str = "fa-info-circle"):
    """Create an information card."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className=f"fas {icon} me-2"),
            title
        ]),
        dbc.CardBody(content)
    ], className="shadow-sm")
