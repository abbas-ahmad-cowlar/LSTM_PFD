"""
Security Settings tab for the Settings page.
Provides UI for password management, 2FA, sessions, and login history.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_security_tab():
    """Create security settings tab."""
    return dbc.Container([
        html.H4("Security Settings", className="mt-3 mb-3"),
        html.P(
            "Manage your password, authentication, and security preferences.",
            className="text-muted mb-4"
        ),

        # Change Password Card
        dbc.Card([
            dbc.CardHeader(html.H5("Change Password", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Current Password *", html_for='current-password-input'),
                        dbc.Input(
                            id='current-password-input',
                            type="password",
                            placeholder="Enter current password",
                            className="mb-3",
                            **{"aria-required": "true"}
                        ),
                    ], md=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("New Password *", html_for='new-password-input'),
                        dbc.Input(
                            id='new-password-input',
                            type="password",
                            placeholder="Enter new password (min 8 characters)",
                            className="mb-3",
                            **{"aria-required": "true"}
                        ),
                        dbc.FormText("Password must be at least 8 characters with uppercase, lowercase, and numbers"),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Confirm New Password *", html_for='confirm-password-input'),
                        dbc.Input(
                            id='confirm-password-input',
                            type="password",
                            placeholder="Confirm new password",
                            className="mb-3",
                            **{"aria-required": "true"}
                        ),
                    ], md=6),
                ]),

                # Password strength indicator
                html.Div(id='password-strength-indicator', className="mb-3",
                         **{"aria-live": "polite"}),

                html.Div(id='password-change-message', className="mb-3",
                         **{"aria-live": "assertive"}),

                dbc.Button(
                    [html.I(className="bi bi-key me-2"), "Change Password"],
                    id='change-password-btn',
                    color="primary",
                    **{"aria-label": "Change password"}
                ),
            ])
        ], className="mb-4"),

        # Two-Factor Authentication Card
        dbc.Card([
            dbc.CardHeader(html.H5("Two-Factor Authentication (2FA)", className="mb-0")),
            dbc.CardBody([
                html.P(
                    "Add an extra layer of security to your account by requiring a verification code in addition to your password.",
                    className="mb-3"
                ),

                html.Div(id='2fa-status-display', children=[
                    dbc.Alert([
                        html.I(className="bi bi-info-circle me-2"),
                        "Two-factor authentication is currently disabled"
                    ], color="warning")
                ], className="mb-3", **{"aria-live": "polite"}),

                dbc.Button(
                    [html.I(className="bi bi-shield-check me-2"), "Enable 2FA"],
                    id='enable-2fa-btn',
                    color="success",
                    className="me-2",
                    **{"aria-label": "Enable two-factor authentication"}
                ),
                dbc.Button(
                    [html.I(className="bi bi-shield-x me-2"), "Disable 2FA"],
                    id='disable-2fa-btn',
                    color="danger",
                    disabled=True,
                    **{"aria-label": "Disable two-factor authentication"}
                ),
            ])
        ], className="mb-4"),

        # Active Sessions Card
        dbc.Card([
            dbc.CardHeader([
                html.H5("Active Sessions", className="mb-0 d-inline"),
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh"],
                    id='refresh-sessions-btn',
                    color="secondary",
                    size="sm",
                    className="float-end",
                    **{"aria-label": "Refresh active sessions"}
                ),
            ]),
            dbc.CardBody([
                html.P("Monitor devices and locations where your account is currently logged in.", className="text-muted mb-3"),
                dcc.Loading(
                    id="loading-sessions",
                    children=[html.Div(id='active-sessions-table', **{"aria-live": "polite"})],
                    type="default"
                ),
                html.Div(className="mt-3"),
                dbc.Button(
                    [html.I(className="bi bi-x-circle me-2"), "Logout All Other Sessions"],
                    id='logout-all-sessions-btn',
                    color="danger",
                    outline=True,
                    size="sm",
                    **{"aria-label": "Logout all other active sessions"}
                ),
            ])
        ], className="mb-4"),

        # Login History Card
        dbc.Card([
            dbc.CardHeader(html.H5("Login History", className="mb-0")),
            dbc.CardBody([
                html.P("View recent login activity for your account.", className="text-muted mb-3"),
                dcc.Loading(
                    id="loading-login-history",
                    children=[html.Div(id='login-history-table', **{"aria-live": "polite"})],
                    type="default"
                )
            ])
        ], className="mb-4"),

        # 2FA Setup Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("🔒 Enable Two-Factor Authentication")),
            dbc.ModalBody([
                html.P("Scan this QR code with your authenticator app:", className="mb-3"),
                html.Div(id='2fa-qr-code', className="text-center mb-3"),
                html.Hr(),
                html.P("Or enter this secret key manually:", className="mb-2 small text-muted"),
                html.Pre(id='2fa-secret-key', className="bg-light p-2 rounded text-center"),
                html.Hr(),
                dbc.Label("Enter verification code from your app:", className="mt-3"),
                dbc.Input(
                    id='2fa-verify-code-input',
                    type="text",
                    placeholder="6-digit code",
                    maxLength=6,
                    className="mb-3 text-center",
                    style={'fontSize': '1.5rem', 'letterSpacing': '0.5rem'},
                    **{"aria-label": "Two-factor authentication verification code",
                       "aria-required": "true"}
                ),
                html.Div(id='2fa-setup-message', className="mt-3",
                         **{"aria-live": "assertive"}),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id='cancel-2fa-setup-btn', color="secondary", className="me-2"),
                dbc.Button("Verify & Enable", id='verify-2fa-btn', color="success")
            ])
        ], id='2fa-setup-modal', is_open=False,
           **{"aria-modal": "true"}),

        # Store for security data
        dcc.Store(id='security-data', data=None),

    ], className="py-4")
