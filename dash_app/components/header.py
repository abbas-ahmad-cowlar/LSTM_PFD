"""
Top navigation header component.
"""
import dash_bootstrap_components as dbc
from dash import html


def create_header():
    """Create application header."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.A(
                        dbc.Row([
                            dbc.Col(html.I(className="fas fa-brain me-2")),
                            dbc.Col(dbc.NavbarBrand("LSTM PFD Dashboard", className="ms-2")),
                        ], align="center", className="g-0"),
                        href="/",
                        style={"textDecoration": "none"}
                    ),
                    width="auto"
                ),
            ], align="center", className="g-0"),

            dbc.Row([
                dbc.Col(
                    dbc.NavbarToggler(id="navbar-toggler"),
                    width="auto"
                ),
            ], align="center", className="g-0 ms-auto"),

            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Home", href="/")),
                    dbc.NavItem(dbc.NavLink("Experiments", href="/experiments")),
                    dbc.NavItem(dbc.NavLink("Analytics", href="/analytics")),
                    dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("Profile", href="/profile"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Settings", href="/settings"),
                            dbc.DropdownMenuItem("Logout", href="/logout"),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="User",
                        className="ms-2"
                    ),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                is_open=False,
                navbar=True
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-3"
    )
