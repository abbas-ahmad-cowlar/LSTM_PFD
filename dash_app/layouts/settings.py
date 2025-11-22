"""
Settings page layout (Feature #1: API Keys).
Provides UI for managing API keys and user settings.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


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

        # API Key Usage Statistics (placeholder)
        html.Div(id='api-key-stats'),

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
    """Create user profile tab (placeholder)."""
    return dbc.Container([
        html.H4("User Profile", className="mt-3 mb-3"),
        dbc.Alert(
            "User profile management coming soon!",
            color="info"
        ),
        html.P("This section will allow you to update your:"),
        html.Ul([
            html.Li("Username and email"),
            html.Li("Display name and bio"),
            html.Li("Notification preferences"),
            html.Li("Profile picture"),
        ])
    ], className="py-4")


def create_security_tab():
    """Create security settings tab (placeholder)."""
    return dbc.Container([
        html.H4("Security Settings", className="mt-3 mb-3"),
        dbc.Alert(
            "Security settings coming soon!",
            color="info"
        ),
        html.P("This section will allow you to:"),
        html.Ul([
            html.Li("Change your password"),
            html.Li("Enable two-factor authentication (2FA)"),
            html.Li("View active sessions"),
            html.Li("Review login history"),
            html.Li("Configure security alerts"),
        ])
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
                dbc.Button(
                    [html.I(className="bi bi-download me-2"), "Export"],
                    id='export-notification-history-btn',
                    color="secondary",
                    size="sm",
                    className="float-end"
                ),
            ]),
            dbc.CardBody([
                html.P("View the last 50 notifications sent to you.", className="text-muted"),
                dcc.Loading(
                    id="loading-notification-history",
                    children=[html.Div(id='notification-history-table')],
                    type="default"
                )
            ])
        ], className="mb-4"),

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
