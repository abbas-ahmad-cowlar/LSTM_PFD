"""
API Keys management tab for the Settings page.
Provides UI for generating, viewing, and revoking API keys.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


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
                    className="float-end",
                    **{"aria-label": "Generate a new API key"}
                ),
            ]),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-api-keys",
                    children=[html.Div(id='api-keys-table', **{"aria-live": "polite"})],
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
                    className="float-end",
                    **{"aria-label": "Refresh API usage statistics"}
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
                    children=[html.Div(id='api-usage-detail-table', **{"aria-live": "polite"})],
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
                    className="mb-3",
                    **{"aria-required": "true"}
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
                    className="mb-3",
                    **{"aria-required": "true"}
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
                    className="mb-3",
                    **{"aria-required": "true"}
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
                html.Div(id='key-generation-message', className="mt-3", **{"aria-live": "assertive"}),

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
        ], id='generate-key-modal', is_open=False, size="lg",
           **{"aria-modal": "true"}),

        # Modal for viewing key details
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("API Key Details")),
            dbc.ModalBody(id='key-details-modal-body'),
            dbc.ModalFooter([
                dbc.Button("Close", id='close-key-details-btn', color="secondary")
            ])
        ], id='key-details-modal', is_open=False, size="lg",
           **{"aria-modal": "true"}),

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
        ], id='revoke-key-modal', is_open=False,
           **{"aria-modal": "true"}),

        # Store for selected key ID
        dcc.Store(id='selected-key-id', data=None),

    ])
