"""
Webhooks management tab for the Settings page.
Provides UI for managing Slack, Teams, and custom webhook integrations.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


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
                    className="float-end",
                    **{"aria-label": "Add a new webhook integration"}
                ),
            ]),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-webhooks",
                    children=[html.Div(id='webhooks-table', **{"aria-live": "polite"})],
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
                            className="btn btn-sm btn-outline-secondary",
                            rel="noopener noreferrer"
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
                            className="btn btn-sm btn-outline-secondary",
                            rel="noopener noreferrer"
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
                    className="mb-3",
                    **{"aria-required": "true"}
                ),

                # Name
                dbc.Label("Name *", html_for='webhook-name-input'),
                dbc.Input(
                    id='webhook-name-input',
                    placeholder="e.g., #ml-experiments, Production Alerts",
                    type="text",
                    className="mb-3",
                    **{"aria-required": "true"}
                ),
                dbc.FormText("A descriptive name to identify this webhook"),

                # Webhook URL
                dbc.Label("Webhook URL *", html_for='webhook-url-input', className="mt-3"),
                dbc.Input(
                    id='webhook-url-input',
                    placeholder="https://hooks.slack.com/services/...",
                    type="url",
                    className="mb-3",
                    **{"aria-required": "true"}
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
                html.Div(id='webhook-form-message', className="mt-3",
                         **{"aria-live": "assertive"}),
            ]),
            dbc.ModalFooter([
                dbc.Button("Test Webhook", id='test-webhook-btn', color="info",
                           className="me-auto",
                           **{"aria-label": "Send test webhook payload"}),
                dbc.Button("Cancel", id='cancel-webhook-btn', color="secondary", className="me-2"),
                dbc.Button("Save Webhook", id='save-webhook-btn', color="primary")
            ])
        ], id='webhook-modal', is_open=False, size="lg",
           **{"aria-modal": "true"}),

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
                    children=[html.Div(id='webhook-delivery-history',
                                       **{"aria-live": "polite"})],
                    type="default"
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id='close-webhook-details-btn', color="secondary")
            ])
        ], id='webhook-details-modal', is_open=False, size="lg",
           **{"aria-modal": "true"}),

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
        ], id='delete-webhook-modal', is_open=False,
           **{"aria-modal": "true"}),

        # Store for selected webhook ID
        dcc.Store(id='selected-webhook-id', data=None),
        # Store for edit mode flag
        dcc.Store(id='webhook-edit-mode', data=False),

    ], className="py-4")
