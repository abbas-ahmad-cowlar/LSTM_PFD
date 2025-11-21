"""
Experiments list layout (Phase 11B).
"""
import dash_bootstrap_components as dbc
from dash import html, dash_table


def create_experiments_layout():
    """Create experiments list layout."""
    return dbc.Container([
        html.H2("Experiment History", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Input(id="experiment-search", placeholder="Search experiments...", type="text"),
            ], width=6),
            dbc.Col([
                dcc.Dropdown(
                    id="experiment-model-filter",
                    placeholder="Filter by model type",
                    multi=True
                ),
            ], width=3),
            dbc.Col([
                dbc.Button([
                    html.I(className="fas fa-plus me-2"),
                    "New Experiment"
                ], href="/experiment/new", color="primary", className="float-end"),
            ], width=3),
        ], className="mb-4"),

        html.Div(id="experiments-table-container", children=[
            html.P("Loading experiments...", className="text-muted")
        ])
    ], fluid=True)


from dash import dcc
