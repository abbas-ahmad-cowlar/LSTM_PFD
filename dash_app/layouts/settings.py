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
