"""
Settings page layout (Feature #1: API Keys).
Provides UI for managing API keys and user settings.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
from layouts.email_digest_management import create_email_digest_tab


def create_settings_layout():
    """
    Create the settings page layout.

    Features:
        - API Keys management tab
        - User profile tab (placeholder for future)
        - Security settings tab (placeholder for future)
    """
    return dbc.Container([
        # Page Header
        dbc.Row([
            dbc.Col([
                html.H2("⚙️ Settings", className="mb-3"),
                html.P(
                    "Manage your API keys, profile, and security settings.",
                    className="text-muted"
                )
            ])
        ], className="mb-4"),

        # Tabs
        dbc.Tabs([
            # API Keys Tab
            dbc.Tab(
                label="🔑 API Keys",
                tab_id="api-keys",
                children=create_api_keys_tab()
            ),

            # User Profile Tab (placeholder)
            dbc.Tab(
                label="👤 Profile",
                tab_id="profile",
                children=create_profile_tab()
            ),

            # Security Tab (placeholder)
            dbc.Tab(
                label="🔒 Security",
                tab_id="security",
                children=create_security_tab()
            ),

            # Notifications Tab
            dbc.Tab(
                label="🔔 Notifications",
                tab_id="notifications",
                children=create_notifications_tab()
            ),

            # Webhooks Tab
            dbc.Tab(
                label="🔗 Webhooks",
                tab_id="webhooks",
                children=create_webhooks_tab()
            ),

            # Email Digest Tab
            dbc.Tab(
                label="📧 Email Digests",
                tab_id="email-digests",
                children=create_email_digest_tab()
            ),
        ], id='settings-tabs', active_tab='api-keys'),

    ], fluid=True, className="py-4")


def create_api_keys_tab():
    """Create API Keys management tab."""
    return html.Div([
        # Section Header
        html.H4("API Keys", className="mt-3 mb-3"),
        html.P([
            "Use API keys to authenticate programmatic access to the platform. ",
            "Each key can have custom rate limits and permissions."
        ], className="text-muted mb-4"),

        # Alert for API key security
        dbc.Alert([
            html.I(className="bi bi-shield-lock me-2"),
            html.Strong("Security Notice: "),
            "API keys are shown only once at creation. Store them securely and never commit them to version control."
        ], color="warning", className="mb-4"),

        # API Keys Table
        dbc.Card([
            dbc.CardHeader([
                html.H5("Your API Keys", className="mb-0 d-inline"),
                dbc.Button(
                    [html.I(className="bi bi-plus-circle me-2"), "Generate New API Key"],
                    id='generate-key-btn',
                    color="primary",
                    size="sm",
                    className="float-end"
                ),
            ]),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-api-keys",
                    children=[html.Div(id='api-keys-table')],
                    type="default"
                )
            ])
        ], className="mb-4"),

        # API Key Usage Statistics
        dbc.Card([
            dbc.CardHeader([
                html.H5("API Usage Statistics", className="mb-0 d-inline"),
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh"],
                    id='refresh-api-stats-btn',
                    color="secondary",
                    size="sm",
                    className="float-end"
                ),
            ]),
            dbc.CardBody([
                html.P("View API usage patterns and metrics for your API keys", className="text-muted mb-4"),

                # Summary Stats Row
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4(id='total-api-requests-count', children="0", className="text-primary mb-0"),
                            html.Small("Total Requests (30 days)", className="text-muted")
                        ], className="text-center p-3 border rounded")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.H4(id='avg-response-time', children="0ms", className="text-info mb-0"),
                            html.Small("Avg Response Time", className="text-muted")
                        ], className="text-center p-3 border rounded")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.H4(id='api-success-rate', children="0%", className="text-success mb-0"),
                            html.Small("Success Rate", className="text-muted")
                        ], className="text-center p-3 border rounded")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.H4(id='active-api-keys-count', children="0", className="text-warning mb-0"),
                            html.Small("Active API Keys", className="text-muted")
                        ], className="text-center p-3 border rounded")
                    ], md=3),
                ], className="mb-4"),

                # Charts Row
                dbc.Row([
                    dbc.Col([
                        html.H6("Requests Over Time (Last 30 Days)", className="mb-3"),
                        dcc.Loading(
                            id="loading-api-usage-chart",
                            children=[dcc.Graph(id='api-usage-timeline-chart', config={'displayModeBar': False})],
                            type="default"
                        )
                    ], md=12),
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col([
                        html.H6("Top API Keys by Request Count", className="mb-3"),
                        dcc.Loading(
                            id="loading-top-keys-chart",
                            children=[dcc.Graph(id='api-top-keys-chart', config={'displayModeBar': False})],
                            type="default"
                        )
                    ], md=6),
                    dbc.Col([
                        html.H6("Requests by Endpoint", className="mb-3"),
                        dcc.Loading(
                            id="loading-endpoints-chart",
                            children=[dcc.Graph(id='api-endpoints-chart', config={'displayModeBar': False})],
                            type="default"
                        )
                    ], md=6),
                ], className="mb-4"),

                # Detailed Usage Table
                html.Hr(),
                html.H6("Detailed Usage by API Key", className="mt-4 mb-3"),
                dcc.Loading(
                    id="loading-api-usage-table",
                    children=[html.Div(id='api-usage-detail-table')],
                    type="default"
                )
            ])
        ], className="mb-4"),

        # Modal for generating new API key
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Generate New API Key")),
            dbc.ModalBody([
                # Key Name
                dbc.Label("Name *", html_for='key-name-input'),
                dbc.Input(
                    id='key-name-input',
                    placeholder="e.g., CI/CD Pipeline, Production API, Mobile App",
                    type="text",
                    className="mb-3"
                ),
                dbc.FormText("A descriptive name to help you identify this key"),

                # Environment
                dbc.Label("Environment *", html_for='key-environment-input', className="mt-3"),
                dbc.Select(
                    id='key-environment-input',
                    options=[
                        {'label': 'Live (Production)', 'value': 'live'},
                        {'label': 'Test (Development)', 'value': 'test'},
                    ],
                    value='live',
                    className="mb-3"
                ),
                dbc.FormText("Live keys are for production use, test keys are for development"),

                # Rate Limit
                dbc.Label("Rate Limit (requests/hour) *", html_for='key-rate-limit-input', className="mt-3"),
                dbc.Input(
                    id='key-rate-limit-input',
                    type="number",
                    value=1000,
                    min=1,
                    max=10000,
                    className="mb-3"
                ),
                dbc.FormText("Maximum number of API requests per hour (default: 1000)"),

                # Expiration
                dbc.Label("Expiration (days)", html_for='key-expiry-input', className="mt-3"),
                dbc.Input(
                    id='key-expiry-input',
                    type="number",
                    placeholder="Leave empty for no expiration",
                    min=1,
                    max=730,
                    className="mb-3"
                ),
                dbc.FormText("Optional: Key will expire after this many days"),

                # Scopes
                dbc.Label("Permissions *", html_for='key-scopes-input', className="mt-3"),
                dbc.Checklist(
                    id='key-scopes-input',
                    options=[
                        {'label': ' Read (view data and predictions)', 'value': 'read'},
                        {'label': ' Write (create and update resources)', 'value': 'write'},
                    ],
                    value=['read', 'write'],
                    className="mb-3"
                ),

                # Error/Success messages
                html.Div(id='key-generation-message', className="mt-3"),

                # Generated key display (shown after creation)
                html.Div(id='generated-key-display', className="mt-3")
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id='cancel-key-btn', color="secondary", className="me-2"),
                dbc.Button(
                    "Generate Key",
                    id='confirm-generate-btn',
                    color="primary",
                    disabled=False
                )
            ])
        ], id='generate-key-modal', is_open=False, size="lg"),

        # Modal for viewing key details
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("API Key Details")),
            dbc.ModalBody(id='key-details-modal-body'),
            dbc.ModalFooter([
                dbc.Button("Close", id='close-key-details-btn', color="secondary")
            ])
        ], id='key-details-modal', is_open=False, size="lg"),

        # Confirmation modal for revoking keys
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("⚠️ Revoke API Key?")),
            dbc.ModalBody([
                html.P("Are you sure you want to revoke this API key?"),
                html.P([
                    html.Strong("This action cannot be undone."),
                    " All requests using this key will be rejected immediately."
                ], className="text-danger"),
                html.Div(id='revoke-key-info')
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id='cancel-revoke-btn', color="secondary", className="me-2"),
                dbc.Button("Revoke Key", id='confirm-revoke-btn', color="danger")
            ])
        ], id='revoke-key-modal', is_open=False),

        # Store for selected key ID
        dcc.Store(id='selected-key-id', data=None),

    ])


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
                            className="mb-3"
                        ),
                        dbc.FormText("Username cannot be changed after account creation"),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Role", html_for='profile-role-display'),
                        dbc.Input(
                            id='profile-role-display',
                            type="text",
                            disabled=True,
                            className="mb-3"
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
                            className="mb-3"
                        ),
                        dbc.FormText("Used for notifications and account recovery"),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Account Status", html_for='profile-status-display'),
                        dbc.Input(
                            id='profile-status-display',
                            type="text",
                            disabled=True,
                            className="mb-3"
                        ),
                    ], md=6),
                ]),

                html.Div(id='profile-update-message', className="mb-3"),

                dbc.Button(
                    [html.I(className="bi bi-save me-2"), "Save Changes"],
                    id='save-profile-btn',
                    color="primary"
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
                            className="mb-3"
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
                            className="mb-3"
                        ),
                        dbc.FormText("Password must be at least 8 characters with uppercase, lowercase, and numbers"),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Confirm New Password *", html_for='confirm-password-input'),
                        dbc.Input(
                            id='confirm-password-input',
                            type="password",
                            placeholder="Confirm new password",
                            className="mb-3"
                        ),
                    ], md=6),
                ]),

                # Password strength indicator
                html.Div(id='password-strength-indicator', className="mb-3"),

                html.Div(id='password-change-message', className="mb-3"),

                dbc.Button(
                    [html.I(className="bi bi-key me-2"), "Change Password"],
                    id='change-password-btn',
                    color="primary"
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
                ], className="mb-3"),

                dbc.Button(
                    [html.I(className="bi bi-shield-check me-2"), "Enable 2FA"],
                    id='enable-2fa-btn',
                    color="success",
                    className="me-2"
                ),
                dbc.Button(
                    [html.I(className="bi bi-shield-x me-2"), "Disable 2FA"],
                    id='disable-2fa-btn',
                    color="danger",
                    disabled=True
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
                    className="float-end"
                ),
            ]),
            dbc.CardBody([
                html.P("Monitor devices and locations where your account is currently logged in.", className="text-muted mb-3"),
                dcc.Loading(
                    id="loading-sessions",
                    children=[html.Div(id='active-sessions-table')],
                    type="default"
                ),
                html.Div(className="mt-3"),
                dbc.Button(
                    [html.I(className="bi bi-x-circle me-2"), "Logout All Other Sessions"],
                    id='logout-all-sessions-btn',
                    color="danger",
                    outline=True,
                    size="sm"
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
                    children=[html.Div(id='login-history-table')],
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
                    style={'fontSize': '1.5rem', 'letterSpacing': '0.5rem'}
                ),
                html.Div(id='2fa-setup-message', className="mt-3"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id='cancel-2fa-setup-btn', color="secondary", className="me-2"),
                dbc.Button("Verify & Enable", id='verify-2fa-btn', color="success")
            ])
        ], id='2fa-setup-modal', is_open=False),

        # Store for security data
        dcc.Store(id='security-data', data=None),

    ], className="py-4")


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
                    className="float-end"
                ),
            ]),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-notification-prefs",
                    children=[html.Div(id='notification-preferences-table')],
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
                            className="mb-3"
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
                            className="mb-3"
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
                html.Div(id='email-config-message', className="mb-3"),
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="bi bi-envelope-check me-2"), "Send Test Email"],
                        id='send-test-email-btn',
                        color="info",
                        className="me-2"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-save me-2"), "Save Configuration"],
                        id='save-email-config-btn',
                        color="primary"
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
                        className="me-2"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-download me-2"), "Export"],
                        id='export-notification-history-btn',
                        color="secondary",
                        size="sm",
                        outline=True,
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
                            className="mb-3"
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
                    children=[html.Div(id='notification-history-table')],
                    type="default"
                ),

                # Pagination
                html.Div([
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P(id='email-log-count', children="Showing 0 emails", className="text-muted small mb-0")
                        ], md=6),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Previous", id='email-log-prev-page-btn', size="sm", outline=True, disabled=True),
                                dbc.Button("Next", id='email-log-next-page-btn', size="sm", outline=True, disabled=True),
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


def create_webhooks_tab():
    """Create webhooks management tab."""
    return dbc.Container([
        # Section Header
        html.H4("Webhook Integrations", className="mt-3 mb-3"),
        html.P([
            "Connect your experiments to Slack, Microsoft Teams, or custom webhooks. ",
            "Get instant notifications when training completes, HPO campaigns finish, or errors occur."
        ], className="text-muted mb-4"),

        # Info Alert
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            html.Strong("Getting Started: "),
            "Create a webhook URL in your Slack workspace or Teams channel, then add it here to start receiving notifications."
        ], color="info", className="mb-4"),

        # Webhooks Table
        dbc.Card([
            dbc.CardHeader([
                html.H5("Your Webhooks", className="mb-0 d-inline"),
                dbc.Button(
                    [html.I(className="bi bi-plus-circle me-2"), "Add Webhook"],
                    id='add-webhook-btn',
                    color="primary",
                    size="sm",
                    className="float-end"
                ),
            ]),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-webhooks",
                    children=[html.Div(id='webhooks-table')],
                    type="default"
                )
            ])
        ], className="mb-4"),

        # Quick Links Card
        dbc.Card([
            dbc.CardHeader(html.H5("Setup Guides", className="mb-0")),
            dbc.CardBody([
                html.P("Learn how to create webhook URLs for different platforms:", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.H6([html.I(className="bi bi-slack me-2"), "Slack"]),
                        html.P("1. Go to your Slack workspace settings", className="small"),
                        html.P("2. Navigate to 'Apps' → 'Incoming Webhooks'", className="small"),
                        html.P("3. Click 'Add to Slack' and select a channel", className="small"),
                        html.P("4. Copy the webhook URL and paste it above", className="small"),
                        html.A(
                            [html.I(className="bi bi-box-arrow-up-right me-1"), "Official Guide"],
                            href="https://api.slack.com/messaging/webhooks",
                            target="_blank",
                            className="btn btn-sm btn-outline-secondary"
                        )
                    ], md=4),
                    dbc.Col([
                        html.H6([html.I(className="bi bi-microsoft-teams me-2"), "Microsoft Teams"]),
                        html.P("1. Open Teams and select your channel", className="small"),
                        html.P("2. Click '...' → 'Connectors' → 'Incoming Webhook'", className="small"),
                        html.P("3. Name your webhook and click 'Create'", className="small"),
                        html.P("4. Copy the webhook URL and paste it above", className="small"),
                        html.A(
                            [html.I(className="bi bi-box-arrow-up-right me-1"), "Official Guide"],
                            href="https://docs.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook",
                            target="_blank",
                            className="btn btn-sm btn-outline-secondary"
                        )
                    ], md=4),
                    dbc.Col([
                        html.H6([html.I(className="bi bi-code-square me-2"), "Custom Webhook"]),
                        html.P("For custom integrations, your endpoint should:", className="small"),
                        html.Ul([
                            html.Li("Accept POST requests", className="small"),
                            html.Li("Handle JSON payloads", className="small"),
                            html.Li("Return 200 OK on success", className="small"),
                        ]),
                        html.A(
                            [html.I(className="bi bi-file-text me-1"), "Payload Docs"],
                            href="#",
                            className="btn btn-sm btn-outline-secondary"
                        )
                    ], md=4),
                ])
            ])
        ], className="mb-4"),

        # Modal for adding/editing webhook
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id='webhook-modal-title')),
            dbc.ModalBody([
                # Provider Type
                dbc.Label("Provider *", html_for='webhook-provider-select'),
                dbc.Select(
                    id='webhook-provider-select',
                    options=[
                        {'label': '🔵 Slack', 'value': 'slack'},
                        {'label': '🟦 Microsoft Teams', 'value': 'teams'},
                        {'label': '⚙️ Custom Webhook', 'value': 'webhook'},
                    ],
                    value='slack',
                    className="mb-3"
                ),

                # Name
                dbc.Label("Name *", html_for='webhook-name-input'),
                dbc.Input(
                    id='webhook-name-input',
                    placeholder="e.g., #ml-experiments, Production Alerts",
                    type="text",
                    className="mb-3"
                ),
                dbc.FormText("A descriptive name to identify this webhook"),

                # Webhook URL
                dbc.Label("Webhook URL *", html_for='webhook-url-input', className="mt-3"),
                dbc.Input(
                    id='webhook-url-input',
                    placeholder="https://hooks.slack.com/services/...",
                    type="url",
                    className="mb-3"
                ),
                dbc.FormText("The webhook URL from your provider"),

                # Description
                dbc.Label("Description", html_for='webhook-description-input', className="mt-3"),
                dbc.Textarea(
                    id='webhook-description-input',
                    placeholder="Optional: Add notes about this webhook...",
                    rows=2,
                    className="mb-3"
                ),

                # Event Selection
                dbc.Label("Events to Monitor *", html_for='webhook-events-checklist', className="mt-3"),
                html.P("Select which events should trigger this webhook:", className="small text-muted mb-2"),
                dbc.Checklist(
                    id='webhook-events-checklist',
                    options=[
                        {'label': ' Training Started', 'value': 'training.started'},
                        {'label': ' Training Completed', 'value': 'training.complete'},
                        {'label': ' Training Failed', 'value': 'training.failed'},
                        {'label': ' HPO Campaign Started', 'value': 'hpo.campaign_started'},
                        {'label': ' HPO Campaign Completed', 'value': 'hpo.campaign_complete'},
                        {'label': ' NAS Campaign Started', 'value': 'nas.campaign_started'},
                        {'label': ' NAS Campaign Completed', 'value': 'nas.campaign_complete'},
                        {'label': ' Deployment Created', 'value': 'deployment.created'},
                        {'label': ' System Alerts', 'value': 'system.alert'},
                    ],
                    value=['training.complete', 'training.failed'],
                    className="mb-3"
                ),

                # Active Status
                html.Hr(),
                dbc.Checkbox(
                    id='webhook-is-active',
                    label="Enable this webhook",
                    value=True,
                    className="mb-3"
                ),

                # Error/Success messages
                html.Div(id='webhook-form-message', className="mt-3"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Test Webhook", id='test-webhook-btn', color="info", className="me-auto"),
                dbc.Button("Cancel", id='cancel-webhook-btn', color="secondary", className="me-2"),
                dbc.Button("Save Webhook", id='save-webhook-btn', color="primary")
            ])
        ], id='webhook-modal', is_open=False, size="lg"),

        # Modal for webhook details
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Webhook Details")),
            dbc.ModalBody([
                html.Div(id='webhook-details-content'),

                html.Hr(),
                html.H6("Recent Deliveries", className="mt-3 mb-3"),
                html.P("Last 10 webhook delivery attempts:", className="text-muted small"),
                dcc.Loading(
                    id="loading-webhook-history",
                    children=[html.Div(id='webhook-delivery-history')],
                    type="default"
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id='close-webhook-details-btn', color="secondary")
            ])
        ], id='webhook-details-modal', is_open=False, size="lg"),

        # Confirmation modal for deleting webhook
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("⚠️ Delete Webhook?")),
            dbc.ModalBody([
                html.P("Are you sure you want to delete this webhook?"),
                html.P([
                    html.Strong("This action cannot be undone."),
                    " You will stop receiving notifications for this integration."
                ], className="text-warning"),
                html.Div(id='delete-webhook-info')
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id='cancel-delete-webhook-btn', color="secondary", className="me-2"),
                dbc.Button("Delete Webhook", id='confirm-delete-webhook-btn', color="danger")
            ])
        ], id='delete-webhook-modal', is_open=False),

        # Store for selected webhook ID
        dcc.Store(id='selected-webhook-id', data=None),
        # Store for edit mode flag
        dcc.Store(id='webhook-edit-mode', data=False),

    ], className="py-4")
