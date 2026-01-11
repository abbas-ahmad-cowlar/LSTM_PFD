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
            dbc.NavLink([
                html.I(className="fas fa-project-diagram me-2"),
                "Feature Engineering"
            ], href="/feature-engineering", active="exact"),

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
            dbc.NavLink([
                html.I(className="fas fa-sitemap me-2"),
                "NAS"
            ], href="/nas", active="exact"),

            html.Hr(),

            html.P("Evaluation", className="text-muted small mb-1"),
            dbc.NavLink([
                html.I(className="fas fa-chart-bar me-2"),
                "Evaluation"
            ], href="/evaluation", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-brain me-2"),
                "XAI Dashboard"
            ], href="/xai", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-chart-area me-2"),
                "Visualizations"
            ], href="/visualization", active="exact"),

            html.Hr(),

            html.P("Production", className="text-muted small mb-1"),
            dbc.NavLink([
                html.I(className="fas fa-rocket me-2"),
                "Deployment"
            ], href="/deployment", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-server me-2"),
                "API Monitoring"
            ], href="/api-monitoring", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-vial me-2"),
                "Testing & QA"
            ], href="/testing", active="exact"),

            html.Hr(),

            html.P("System", className="text-muted small mb-1"),
            dbc.NavLink([
                html.I(className="fas fa-heartbeat me-2"),
                "System Health"
            ], href="/system-health", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-cog me-2"),
                "Settings"
            ], href="/settings", active="exact"),
        ], vertical=True, pills=True),
    ], className="sidebar")
