"""
Left sidebar navigation component.
"""
import dash_bootstrap_components as dbc
from dash import html
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_sidebar():
    """Create sidebar navigation."""
    return html.Div([
        html.H5("Navigation", className="mt-3 mb-3"),
        dbc.Nav([
            dbc.NavLink([
                html.I(className="fas fa-home me-2"),
                "Home"
            ], href="/", active="exact"),

            html.Hr(),

            html.P("Data", className="text-muted small mb-1"),
            dbc.NavLink([
                html.I(className="fas fa-cog me-2"),
                "Generate Data"
            ], href="/data-generation", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-database me-2"),
                "Data Explorer"
            ], href="/data-explorer", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-signal me-2"),
                "Signal Viewer"
            ], href="/signal-viewer", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-folder me-2"),
                "Datasets"
            ], href="/datasets", active="exact"),

            html.Hr(),

            html.P("Training", className="text-muted small mb-1"),
            dbc.NavLink([
                html.I(className="fas fa-flask me-2"),
                "New Experiment"
            ], href="/experiment/new", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-list me-2"),
                "Experiments"
            ], href="/experiments", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-search me-2"),
                "HPO Campaigns"
            ], href="/hpo/campaigns", active="exact"),

            html.Hr(),

            html.P("Analysis", className="text-muted small mb-1"),
            dbc.NavLink([
                html.I(className="fas fa-brain me-2"),
                "XAI Explorer"
            ], href="/xai/explain", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-chart-bar me-2"),
                "Statistics"
            ], href="/statistics/compare", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-chart-line me-2"),
                "Analytics"
            ], href="/analytics", active="exact"),

            html.Hr(),

            dbc.NavLink([
                html.I(className="fas fa-heartbeat me-2"),
                "System Health"
            ], href="/system-health", active="exact"),
        ], vertical=True, pills=True),
    ], className="sidebar")
