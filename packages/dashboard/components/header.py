"""
Top navigation header component with mobile hamburger menu.

Deficiency #34: Mobile Responsiveness - Hamburger menu button
"""
import dash_bootstrap_components as dbc
from dash import html, clientside_callback, Input, Output
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_header():
    """Create application header with mobile hamburger button."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                # Hamburger button - only visible on mobile (CSS controlled)
                dbc.Col(
                    html.Button(
                        html.I(className="fas fa-bars"),
                        id="mobile-hamburger-btn",
                        className="hamburger-btn",
                        n_clicks=0
                    ),
                    width="auto",
                    className="d-lg-none"  # Hide on large screens
                ),
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


# Clientside callback for mobile hamburger menu
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks === undefined || n_clicks === 0) {
            return window.dash_clientside.no_update;
        }
        
        const sidebar = document.getElementById('sidebar-container');
        const body = document.body;
        
        if (sidebar) {
            sidebar.classList.toggle('mobile-open');
        }
        
        if (body) {
            body.classList.toggle('mobile-sidebar-open');
        }
        
        return {};
    }
    """,
    Output('mobile-hamburger-btn', 'style'),
    [Input('mobile-hamburger-btn', 'n_clicks')],
    prevent_initial_call=True
)

