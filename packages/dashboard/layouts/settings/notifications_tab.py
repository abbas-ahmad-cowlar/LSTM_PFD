"""
Notifications Settings tab for the Settings page.
Provides UI for notification preferences, email configuration, and notification history.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_notifications_tab():
    """Create notifications settings tab."""
    return dbc.Container([
        # Section Header
        html.H4("Notification Preferences", className="mt-3 mb-3"),
        html.P([
            "Configure how you want to be notified about important events. ",
            "Choose notification channels and frequency for each event type."
        ], className="text-muted mb-4"),

        # Notification Preferences Card
        dbc.Card([
            dbc.CardHeader([
                html.H5("Event Notification Settings", className="mb-0 d-inline"),
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), "Reload"],
                    id='reload-notification-prefs-btn',
                    color="secondary",
                    size="sm",
                    className="float-end",
                    **{"aria-label": "Reload notification preferences"}
                ),
            ]),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-notification-prefs",
                    children=[html.Div(id='notification-preferences-table',
                                       **{"aria-live": "polite"})],
                    type="default"
                )
            ])
        ], className="mb-4"),

        # Email Configuration Card
        dbc.Card([
            dbc.CardHeader(html.H5("Email Configuration", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("SMTP Server", html_for='smtp-server-input'),
                        dbc.Input(
                            id='smtp-server-input',
                            placeholder="e.g., smtp.gmail.com",
                            type="text",
                            className="mb-3",
                            **{"aria-label": "SMTP server address"}
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("SMTP Port", html_for='smtp-port-input'),
                        dbc.Input(
                            id='smtp-port-input',
                            type="number",
                            value=587,
                            min=1,
                            max=65535,
                            className="mb-3",
                            **{"aria-label": "SMTP port number"}
                        ),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Username / Email", html_for='smtp-username-input'),
                        dbc.Input(
                            id='smtp-username-input',
                            placeholder="your-email@example.com",
                            type="text",
                            className="mb-3"
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Password", html_for='smtp-password-input'),
                        dbc.Input(
                            id='smtp-password-input',
                            type="password",
                            placeholder="••••••••",
                            className="mb-3"
                        ),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Checkbox(
                            id='smtp-use-tls',
                            label="Use TLS/SSL",
                            value=True,
                            className="mb-3"
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Test Email Address", html_for='test-email-input'),
                        dbc.Input(
                            id='test-email-input',
                            placeholder="your-email@example.com",
                            type="email",
                            className="mb-3"
                        ),
                    ], width=6),
                ]),
                html.Div(id='email-config-message', className="mb-3",
                         **{"aria-live": "polite"}),
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="bi bi-envelope-check me-2"), "Send Test Email"],
                        id='send-test-email-btn',
                        color="info",
                        className="me-2",
                        **{"aria-label": "Send a test email"}
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-save me-2"), "Save Configuration"],
                        id='save-email-config-btn',
                        color="primary",
                        **{"aria-label": "Save email configuration"}
                    ),
                ]),
            ])
        ], className="mb-4"),

        # Notification History Card
        dbc.Card([
            dbc.CardHeader([
                html.H5("Notification History", className="mb-0 d-inline"),
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh"],
                        id='refresh-notification-history-btn',
                        color="secondary",
                        size="sm",
                        outline=True,
                        className="me-2",
                        **{"aria-label": "Refresh notification history"}
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-download me-2"), "Export"],
                        id='export-notification-history-btn',
                        color="secondary",
                        size="sm",
                        outline=True,
                        **{"aria-label": "Export notification history"}
                    ),
                ], className="float-end")
            ]),
            dbc.CardBody([
                html.P("Search and filter email notification history", className="text-muted mb-3"),

                # Filter Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Search", className="small"),
                        dbc.Input(
                            id='email-log-search-input',
                            placeholder="Search subject or recipient...",
                            type="text",
                            debounce=True,
                            className="mb-3",
                            **{"aria-label": "Search notification history"}
                        ),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Filter by Status", className="small"),
                        dcc.Dropdown(
                            id='email-log-status-filter',
                            options=[
                                {'label': 'All Status', 'value': 'all'},
                                {'label': 'Sent', 'value': 'sent'},
                                {'label': 'Failed', 'value': 'failed'},
                                {'label': 'Pending', 'value': 'pending'},
                            ],
                            value='all',
                            clearable=False,
                            className="mb-3"
                        ),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Time Range", className="small"),
                        dcc.Dropdown(
                            id='email-log-time-filter',
                            options=[
                                {'label': 'Last Hour', 'value': 'hour'},
                                {'label': 'Last 24 Hours', 'value': 'day'},
                                {'label': 'Last 7 Days', 'value': 'week'},
                                {'label': 'Last 30 Days', 'value': 'month'},
                                {'label': 'All Time', 'value': 'all'},
                            ],
                            value='month',
                            clearable=False,
                            className="mb-3"
                        ),
                    ], md=3),
                ]),

                dcc.Loading(
                    id="loading-notification-history",
                    children=[html.Div(id='notification-history-table',
                                       **{"aria-live": "polite"})],
                    type="default"
                ),

                # Pagination
                html.Div([
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P(id='email-log-count', children="Showing 0 emails",
                                   className="text-muted small mb-0",
                                   **{"aria-live": "polite"})
                        ], md=6),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Previous", id='email-log-prev-page-btn',
                                           size="sm", outline=True, disabled=True,
                                           **{"aria-label": "Previous page"}),
                                dbc.Button("Next", id='email-log-next-page-btn',
                                           size="sm", outline=True, disabled=True,
                                           **{"aria-label": "Next page"}),
                            ], className="float-end")
                        ], md=6),
                    ])
                ]),
            ])
        ], className="mb-4"),

        # Store for pagination
        dcc.Store(id='email-log-page-number', data=1),
        dcc.Store(id='email-log-items-per-page', data=100),

        # Store for user preferences data
        dcc.Store(id='notification-prefs-data', data=None),

    ], className="py-4")
