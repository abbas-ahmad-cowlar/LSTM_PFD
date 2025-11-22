"""
Webhook Management Callbacks (Feature #4).
Handles UI interactions for webhook configuration management.
"""
import json
from dash import Input, Output, State, html, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
from datetime import datetime

from services.webhook_service import WebhookService
from utils.logger import setup_logger
from utils.validation import (
    validate_required,
    validate_string_length,
    validate_url,
    validate_list_not_empty,
    ValidationError
)

logger = setup_logger(__name__)


def register_webhook_callbacks(app):
    """
    Register all webhook management callbacks.

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output('webhooks-table', 'children'),
        [
            Input('settings-tabs', 'active_tab'),
            Input('webhook-modal', 'is_open')  # Reload when modal closes
        ],
        prevent_initial_call=False
    )
    def load_webhooks_table(active_tab, modal_is_open):
        """Load and display webhooks table."""
        if active_tab != 'webhooks':
            return html.Div()

        try:
            # For demo purposes, use a test user ID
            # In production, get this from session/JWT token
            user_id = 1  # TODO: Get from authenticated session

            # Get webhooks
            webhooks = WebhookService.list_user_webhooks(user_id, include_inactive=True)

            if not webhooks:
                return dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    "No webhooks configured yet. Click 'Add Webhook' to create one."
                ], color="info")

            # Build table
            table_header = html.Thead(html.Tr([
                html.Th("Name"),
                html.Th("Provider"),
                html.Th("Events"),
                html.Th("Status"),
                html.Th("Last Used"),
                html.Th("Success Rate"),
                html.Th("Actions")
            ]))

            table_rows = []
            for webhook in webhooks:
                # Provider icon and badge
                if webhook.provider_type == 'slack':
                    provider_badge = dbc.Badge(
                        [html.I(className="bi bi-slack me-1"), "Slack"],
                        color="primary",
                        className="me-1"
                    )
                elif webhook.provider_type == 'teams':
                    provider_badge = dbc.Badge(
                        [html.I(className="bi bi-microsoft-teams me-1"), "Teams"],
                        color="info",
                        className="me-1"
                    )
                else:
                    provider_badge = dbc.Badge(
                        [html.I(className="bi bi-code-square me-1"), "Custom"],
                        color="secondary",
                        className="me-1"
                    )

                # Status badge
                if not webhook.is_active:
                    status_badge = dbc.Badge("Disabled", color="secondary", className="me-1")
                elif webhook.consecutive_failures >= 3:
                    status_badge = dbc.Badge("Failing", color="danger", className="me-1")
                elif webhook.consecutive_failures > 0:
                    status_badge = dbc.Badge("Warning", color="warning", className="me-1")
                else:
                    status_badge = dbc.Badge("Active", color="success", className="me-1")

                # Events count
                event_count = len(webhook.enabled_events) if webhook.enabled_events else 0
                events_display = f"{event_count} event{'s' if event_count != 1 else ''}"

                # Last used
                last_used = (
                    webhook.last_used_at.strftime("%Y-%m-%d %H:%M")
                    if webhook.last_used_at else "Never"
                )

                # Success rate (from stats)
                try:
                    stats = WebhookService.get_webhook_stats(webhook.id, user_id)
                    success_rate = f"{stats.get('success_rate', 0)}%"
                except Exception as e:
                    logger.error(f"Error getting webhook stats for webhook {webhook.id}: {e}", exc_info=True)
                    success_rate = "N/A"

                # Actions
                actions = dbc.ButtonGroup([
                    dbc.Button(
                        html.I(className="bi bi-info-circle"),
                        id={'type': 'view-webhook-btn', 'index': webhook.id},
                        color="info",
                        size="sm",
                        title="View details"
                    ),
                    dbc.Button(
                        html.I(className="bi bi-pencil"),
                        id={'type': 'edit-webhook-btn', 'index': webhook.id},
                        color="primary",
                        size="sm",
                        title="Edit webhook"
                    ),
                    dbc.Button(
                        html.I(className="bi bi-send" if webhook.is_active else "bi bi-play"),
                        id={'type': 'toggle-webhook-btn', 'index': webhook.id},
                        color="warning" if webhook.is_active else "success",
                        size="sm",
                        title="Disable" if webhook.is_active else "Enable"
                    ),
                    dbc.Button(
                        html.I(className="bi bi-trash"),
                        id={'type': 'delete-webhook-btn', 'index': webhook.id},
                        color="danger",
                        size="sm",
                        title="Delete webhook"
                    )
                ], size="sm")

                table_rows.append(html.Tr([
                    html.Td(webhook.name or "Unnamed"),
                    html.Td(provider_badge),
                    html.Td(events_display),
                    html.Td(status_badge),
                    html.Td(last_used),
                    html.Td(success_rate),
                    html.Td(actions)
                ]))

            table_body = html.Tbody(table_rows)
            table = dbc.Table(
                [table_header, table_body],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True
            )

            return table

        except Exception as e:
            logger.error(f"Error loading webhooks table: {e}", exc_info=True)
            return dbc.Alert(f"Error loading webhooks: {str(e)}", color="danger")

    # Modal open/close callbacks
    @app.callback(
        [
            Output('webhook-modal', 'is_open'),
            Output('webhook-modal-title', 'children'),
            Output('webhook-edit-mode', 'data'),
            Output('webhook-name-input', 'value'),
            Output('webhook-provider-select', 'value'),
            Output('webhook-url-input', 'value'),
            Output('webhook-description-input', 'value'),
            Output('webhook-events-checklist', 'value'),
            Output('webhook-is-active', 'value'),
        ],
        [
            Input('add-webhook-btn', 'n_clicks'),
            Input({'type': 'edit-webhook-btn', 'index': ALL}, 'n_clicks'),
            Input('cancel-webhook-btn', 'n_clicks'),
            Input('save-webhook-btn', 'n_clicks'),
        ],
        [
            State('selected-webhook-id', 'data'),
            State('webhook-name-input', 'value'),
            State('webhook-provider-select', 'value'),
            State('webhook-url-input', 'value'),
            State('webhook-description-input', 'value'),
            State('webhook-events-checklist', 'value'),
            State('webhook-is-active', 'value'),
        ],
        prevent_initial_call=True
    )
    def manage_webhook_modal(add_clicks, edit_clicks, cancel_clicks, save_clicks,
                            selected_id, name, provider, url, description, events, is_active):
        """Handle webhook modal open/close and form management."""
        ctx = callback_context
        if not ctx.triggered:
            return False, "Add Webhook", False, "", "slack", "", "", ['training.complete', 'training.failed'], True

        trigger_id = ctx.triggered[0]['prop_id']

        # Add webhook button clicked
        if 'add-webhook-btn' in trigger_id:
            return True, "Add New Webhook", False, "", "slack", "", "", ['training.complete', 'training.failed'], True

        # Edit webhook button clicked
        if 'edit-webhook-btn' in trigger_id:
            # Get which button was clicked
            button_id = json.loads(trigger_id.split('.')[0])
            webhook_id = button_id['index']

            # Load webhook data
            user_id = 1  # TODO: Get from session
            webhook = WebhookService.get_webhook(webhook_id, user_id)

            if webhook:
                return (
                    True,
                    "Edit Webhook",
                    True,
                    webhook.name or "",
                    webhook.provider_type,
                    webhook.webhook_url,
                    webhook.description or "",
                    webhook.enabled_events or [],
                    webhook.is_active
                )

        # Cancel or save - close modal
        return False, "Add Webhook", False, "", "slack", "", "", ['training.complete', 'training.failed'], True

    @app.callback(
        [
            Output('webhook-form-message', 'children'),
            Output('selected-webhook-id', 'data'),
        ],
        Input('save-webhook-btn', 'n_clicks'),
        [
            State('webhook-name-input', 'value'),
            State('webhook-provider-select', 'value'),
            State('webhook-url-input', 'value'),
            State('webhook-description-input', 'value'),
            State('webhook-events-checklist', 'value'),
            State('webhook-is-active', 'value'),
            State('webhook-edit-mode', 'data'),
            State('selected-webhook-id', 'data'),
        ],
        prevent_initial_call=True
    )
    def save_webhook(n_clicks, name, provider, url, description, events, is_active, edit_mode, webhook_id):
        """Save webhook (create or update)."""
        if not n_clicks:
            return "", None

        try:
            # Validate inputs
            name = validate_required(name, "Webhook name")
            name = validate_string_length(name, 100, "Webhook name")

            url = validate_required(url, "Webhook URL")
            url = validate_url(url, "Webhook URL")

            events = validate_list_not_empty(events, "Event selection")

            user_id = 1  # TODO: Get from session

            if edit_mode and webhook_id:
                # Update existing webhook
                updated = WebhookService.update_webhook(
                    webhook_id=webhook_id,
                    user_id=user_id,
                    name=name.strip(),
                    provider_type=provider,
                    webhook_url=url.strip(),
                    description=description.strip() if description else None,
                    enabled_events=events,
                    is_active=is_active
                )

                if updated:
                    return dbc.Alert([
                        html.I(className="bi bi-check-circle me-2"),
                        "Webhook updated successfully!"
                    ], color="success"), None
                else:
                    return dbc.Alert("Failed to update webhook", color="danger"), webhook_id

            else:
                # Create new webhook
                webhook = WebhookService.create_webhook(
                    user_id=user_id,
                    provider_type=provider,
                    webhook_url=url.strip(),
                    name=name.strip(),
                    enabled_events=events,
                    description=description.strip() if description else None,
                    is_active=is_active
                )

                return dbc.Alert([
                    html.I(className="bi bi-check-circle me-2"),
                    f"Webhook '{name}' created successfully!"
                ], color="success"), None

        except ValidationError as e:
            logger.warning(f"Validation error in save_webhook: {e}")
            return dbc.Alert(str(e), color="danger"), webhook_id
        except ValueError as e:
            logger.warning(f"Value error in save_webhook: {e}")
            return dbc.Alert(str(e), color="danger"), webhook_id
        except Exception as e:
            logger.error(f"Error saving webhook: {e}", exc_info=True)
            return dbc.Alert("An unexpected error occurred while saving the webhook", color="danger"), webhook_id

    @app.callback(
        Output('webhook-form-message', 'children', allow_duplicate=True),
        Input('test-webhook-btn', 'n_clicks'),
        [
            State('webhook-name-input', 'value'),
            State('webhook-provider-select', 'value'),
            State('webhook-url-input', 'value'),
            State('webhook-events-checklist', 'value'),
            State('webhook-edit-mode', 'data'),
            State('selected-webhook-id', 'data'),
        ],
        prevent_initial_call=True
    )
    def test_webhook_handler(n_clicks, name, provider, url, events, edit_mode, webhook_id):
        """Test webhook by sending a test notification."""
        if not n_clicks:
            return ""

        try:
            user_id = 1  # TODO: Get from session

            # If editing existing webhook, test it directly
            if edit_mode and webhook_id:
                result = WebhookService.test_webhook(webhook_id, user_id)
                if result['success']:
                    return dbc.Alert([
                        html.I(className="bi bi-check-circle me-2"),
                        result['message']
                    ], color="success")
                else:
                    return dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        result['message']
                    ], color="danger")

            # For new webhooks, create a temporary one for testing
            else:
                if not url or not url.strip():
                    return dbc.Alert("Please provide a webhook URL first", color="warning")

                # Create temporary webhook
                temp_webhook = WebhookService.create_webhook(
                    user_id=user_id,
                    provider_type=provider,
                    webhook_url=url.strip(),
                    name=f"Test - {name or 'Unnamed'}",
                    enabled_events=events or ['test.webhook'],
                    is_active=False  # Don't activate automatically
                )

                # Test it
                result = WebhookService.test_webhook(temp_webhook.id, user_id)

                # Delete temporary webhook
                WebhookService.delete_webhook(temp_webhook.id, user_id)

                if result['success']:
                    return dbc.Alert([
                        html.I(className="bi bi-check-circle me-2"),
                        result['message']
                    ], color="success")
                else:
                    return dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        result['message']
                    ], color="danger")

        except Exception as e:
            logger.error(f"Error testing webhook: {e}", exc_info=True)
            return dbc.Alert(f"Test failed: {str(e)}", color="danger")

    @app.callback(
        Output('webhooks-table', 'children', allow_duplicate=True),
        Input({'type': 'toggle-webhook-btn', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def toggle_webhook_handler(n_clicks):
        """Toggle webhook active status."""
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks):
            return html.Div()

        try:
            trigger_id = ctx.triggered[0]['prop_id']
            button_id = json.loads(trigger_id.split('.')[0])
            webhook_id = button_id['index']

            user_id = 1  # TODO: Get from session

            # Get current webhook
            webhook = WebhookService.get_webhook(webhook_id, user_id)
            if not webhook:
                return html.Div()

            # Toggle status
            new_status = not webhook.is_active
            WebhookService.toggle_webhook(webhook_id, user_id, new_status)

            logger.info(f"Toggled webhook {webhook_id} to {'active' if new_status else 'inactive'}")

            # Return empty div to trigger table reload
            return html.Div()

        except Exception as e:
            logger.error(f"Error toggling webhook: {e}", exc_info=True)
            return html.Div()

    @app.callback(
        [
            Output('delete-webhook-modal', 'is_open'),
            Output('delete-webhook-info', 'children'),
            Output('selected-webhook-id', 'data', allow_duplicate=True),
        ],
        [
            Input({'type': 'delete-webhook-btn', 'index': ALL}, 'n_clicks'),
            Input('cancel-delete-webhook-btn', 'n_clicks'),
            Input('confirm-delete-webhook-btn', 'n_clicks'),
        ],
        [
            State('selected-webhook-id', 'data'),
        ],
        prevent_initial_call=True
    )
    def manage_delete_webhook_modal(delete_clicks, cancel_clicks, confirm_clicks, selected_id):
        """Handle delete webhook confirmation modal."""
        ctx = callback_context
        if not ctx.triggered:
            return False, "", None

        trigger_id = ctx.triggered[0]['prop_id']

        # Delete button clicked
        if 'delete-webhook-btn' in trigger_id:
            button_id = json.loads(trigger_id.split('.')[0])
            webhook_id = button_id['index']

            user_id = 1  # TODO: Get from session
            webhook = WebhookService.get_webhook(webhook_id, user_id)

            if webhook:
                info = html.Div([
                    html.P(f"Webhook: {webhook.name}"),
                    html.P(f"Provider: {webhook.provider_type.title()}"),
                ])
                return True, info, webhook_id

        # Confirm delete
        if 'confirm-delete-webhook-btn' in trigger_id and selected_id:
            user_id = 1  # TODO: Get from session
            WebhookService.delete_webhook(selected_id, user_id)
            logger.info(f"Deleted webhook {selected_id}")
            return False, "", None

        # Cancel
        return False, "", None

    @app.callback(
        [
            Output('webhook-details-modal', 'is_open'),
            Output('webhook-details-content', 'children'),
            Output('webhook-delivery-history', 'children'),
        ],
        [
            Input({'type': 'view-webhook-btn', 'index': ALL}, 'n_clicks'),
            Input('close-webhook-details-btn', 'n_clicks'),
        ],
        prevent_initial_call=True
    )
    def manage_webhook_details_modal(view_clicks, close_clicks):
        """Handle webhook details modal."""
        ctx = callback_context
        if not ctx.triggered:
            return False, "", ""

        trigger_id = ctx.triggered[0]['prop_id']

        # View button clicked
        if 'view-webhook-btn' in trigger_id:
            button_id = json.loads(trigger_id.split('.')[0])
            webhook_id = button_id['index']

            user_id = 1  # TODO: Get from session
            webhook = WebhookService.get_webhook(webhook_id, user_id)

            if not webhook:
                return False, "", ""

            # Get stats
            stats = WebhookService.get_webhook_stats(webhook_id, user_id)

            # Build details content
            details = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.P([html.Strong("Name: "), webhook.name or "Unnamed"]),
                        html.P([html.Strong("Provider: "), webhook.provider_type.title()]),
                        html.P([html.Strong("Status: "), "Active" if webhook.is_active else "Disabled"]),
                    ], md=6),
                    dbc.Col([
                        html.P([html.Strong("Created: "), webhook.created_at.strftime("%Y-%m-%d %H:%M")]),
                        html.P([html.Strong("Last Used: "), webhook.last_used_at.strftime("%Y-%m-%d %H:%M") if webhook.last_used_at else "Never"]),
                        html.P([html.Strong("Consecutive Failures: "), str(webhook.consecutive_failures)]),
                    ], md=6),
                ]),
                html.Hr(),
                html.H6("Events Monitored:"),
                html.Ul([html.Li(event) for event in (webhook.enabled_events or [])]),
                html.Hr(),
                html.H6("Statistics:"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(stats.get('total_deliveries', 0), className="text-center"),
                                html.P("Total Deliveries", className="text-center text-muted small mb-0")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{stats.get('success_rate', 0)}%", className="text-center text-success"),
                                html.P("Success Rate", className="text-center text-muted small mb-0")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(stats.get('last_24h', 0), className="text-center"),
                                html.P("Last 24h", className="text-center text-muted small mb-0")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(stats.get('failed', 0), className="text-center text-danger"),
                                html.P("Failed", className="text-center text-muted small mb-0")
                            ])
                        ])
                    ], md=3),
                ])
            ])

            # Get delivery history
            logs = WebhookService.get_webhook_logs(webhook_id, user_id, limit=10)

            if not logs:
                history = dbc.Alert("No delivery history yet", color="info")
            else:
                history_rows = []
                for log in logs:
                    status_badge = dbc.Badge(
                        log.status,
                        color="success" if log.status == 'sent' else "danger"
                    )

                    history_rows.append(html.Tr([
                        html.Td(log.event_type),
                        html.Td(status_badge),
                        html.Td(str(log.http_status_code) if log.http_status_code else "N/A"),
                        html.Td(log.created_at.strftime("%Y-%m-%d %H:%M:%S")),
                    ]))

                history = dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Event"),
                        html.Th("Status"),
                        html.Th("HTTP Code"),
                        html.Th("Time"),
                    ])),
                    html.Tbody(history_rows)
                ], bordered=True, size="sm", hover=True)

            return True, details, history

        # Close button
        return False, "", ""
