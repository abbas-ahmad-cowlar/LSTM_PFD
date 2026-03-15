"""
User Profile tab for the Settings page.
Provides UI for managing account information and preferences.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_profile_tab():
    """Create user profile tab."""
    return dbc.Container([
        html.H4("User Profile", className="mt-3 mb-3"),
        html.P(
            "Manage your account information and preferences.",
            className="text-muted mb-4"
        ),

        # Profile Information Card
        dbc.Card([
            dbc.CardHeader(html.H5("Profile Information", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Username", html_for='profile-username-display'),
                        dbc.Input(
                            id='profile-username-display',
                            type="text",
                            disabled=True,
                            className="mb-3",
                            **{"aria-label": "Username (read-only)"}
                        ),
                        dbc.FormText("Username cannot be changed after account creation"),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Role", html_for='profile-role-display'),
                        dbc.Input(
                            id='profile-role-display',
                            type="text",
                            disabled=True,
                            className="mb-3",
                            **{"aria-label": "User role (read-only)"}
                        ),
                        dbc.FormText("Your role determines your permissions"),
                    ], md=6),
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Email *", html_for='profile-email-input'),
                        dbc.Input(
                            id='profile-email-input',
                            type="email",
                            placeholder="your-email@example.com",
                            className="mb-3",
                            **{"aria-required": "true"}
                        ),
                        dbc.FormText("Used for notifications and account recovery"),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Account Status", html_for='profile-status-display'),
                        dbc.Input(
                            id='profile-status-display',
                            type="text",
                            disabled=True,
                            className="mb-3",
                            **{"aria-label": "Account status (read-only)"}
                        ),
                    ], md=6),
                ]),

                html.Div(id='profile-update-message', className="mb-3",
                         **{"aria-live": "polite"}),

                dbc.Button(
                    [html.I(className="bi bi-save me-2"), "Save Changes"],
                    id='save-profile-btn',
                    color="primary",
                    **{"aria-label": "Save profile changes"}
                ),
            ])
        ], className="mb-4"),

        # Account Info Card (Read-only)
        dbc.Card([
            dbc.CardHeader(html.H5("Account Information", className="mb-0")),
            dbc.CardBody([
                html.P([
                    html.Strong("Account Created: "),
                    html.Span(id='profile-created-at', className="text-muted")
                ], className="mb-2"),
                html.P([
                    html.Strong("Last Updated: "),
                    html.Span(id='profile-updated-at', className="text-muted")
                ], className="mb-2"),
            ])
        ], className="mb-4"),

        # Store for user data
        dcc.Store(id='profile-user-data', data=None),

    ], className="py-4")
