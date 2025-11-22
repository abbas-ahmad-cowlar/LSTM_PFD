"""
API Key Management Callbacks (Feature #1).
Handles UI interactions for API key management.
"""
from dash import Input, Output, State, html, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime

from services.api_key_service import APIKeyService
from utils.logger import setup_logger

logger = setup_logger(__name__)


def register_api_key_callbacks(app):
    """
    Register all API key management callbacks.

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output('api-keys-table', 'children'),
        Input('settings-tabs', 'active_tab'),
        prevent_initial_call=False
    )
    def load_api_keys_table(active_tab):
        """Load and display API keys table."""
        if active_tab != 'api-keys':
            return html.Div()

        try:
            # For demo purposes, use a test user ID
            # In production, get this from session/JWT token
            user_id = 1  # TODO: Get from authenticated session

            # Get API keys
            keys = APIKeyService.list_user_keys(user_id, include_inactive=True)

            if not keys:
                return dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    "No API keys yet. Click 'Generate New API Key' to create one."
                ], color="info")

            # Build table
            table_header = html.Thead(html.Tr([
                html.Th("Name"),
                html.Th("Key Prefix"),
                html.Th("Rate Limit"),
                html.Th("Scopes"),
                html.Th("Status"),
                html.Th("Last Used"),
                html.Th("Created"),
                html.Th("Actions")
            ]))

            table_rows = []
            for key in keys:
                # Status badge
                if not key.is_active:
                    status_badge = dbc.Badge("Revoked", color="danger", className="me-1")
                elif key.is_expired():
                    status_badge = dbc.Badge("Expired", color="warning", className="me-1")
                else:
                    status_badge = dbc.Badge("Active", color="success", className="me-1")

                # Scopes badges
                scope_badges = [
                    dbc.Badge(scope, color="secondary", className="me-1")
                    for scope in (key.scopes or [])
                ]

                # Last used
                last_used = (
                    key.last_used_at.strftime("%Y-%m-%d %H:%M")
                    if key.last_used_at else "Never"
                )

                # Created at
                created = key.created_at.strftime("%Y-%m-%d")

                # Actions
                actions = dbc.ButtonGroup([
                    dbc.Button(
                        html.I(className="bi bi-info-circle"),
                        id={'type': 'view-key-btn', 'index': key.id},
                        color="info",
                        size="sm",
                        title="View details"
                    ),
                    dbc.Button(
                        html.I(className="bi bi-trash"),
                        id={'type': 'revoke-key-btn', 'index': key.id},
                        color="danger",
                        size="sm",
                        disabled=not key.is_active,
                        title="Revoke key"
                    )
                ], size="sm")

                row = html.Tr([
                    html.Td(key.name),
                    html.Td(html.Code(f"{key.prefix}...")),
                    html.Td(f"{key.rate_limit:,}/hr"),
                    html.Td(scope_badges),
                    html.Td(status_badge),
                    html.Td(last_used, className="text-muted small"),
                    html.Td(created, className="text-muted small"),
                    html.Td(actions)
                ])
                table_rows.append(row)

            table_body = html.Tbody(table_rows)

            return dbc.Table(
                [table_header, table_body],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True
            )

        except Exception as e:
            logger.error(f"Error loading API keys: {e}", exc_info=True)
            return dbc.Alert(
                f"Error loading API keys: {str(e)}",
                color="danger"
            )

    @app.callback(
        Output('generate-key-modal', 'is_open'),
        [
            Input('generate-key-btn', 'n_clicks'),
            Input('cancel-key-btn', 'n_clicks'),
            Input('confirm-generate-btn', 'n_clicks')
        ],
        State('generate-key-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_generate_modal(gen_clicks, cancel_clicks, confirm_clicks, is_open):
        """Toggle generate key modal."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id in ['generate-key-btn', 'cancel-key-btn']:
            return not is_open

        # Keep open after confirm to show generated key
        if button_id == 'confirm-generate-btn':
            return is_open

        return is_open

    @app.callback(
        [
            Output('generated-key-display', 'children'),
            Output('key-generation-message', 'children'),
            Output('api-keys-table', 'children', allow_duplicate=True)
        ],
        Input('confirm-generate-btn', 'n_clicks'),
        [
            State('key-name-input', 'value'),
            State('key-environment-input', 'value'),
            State('key-rate-limit-input', 'value'),
            State('key-expiry-input', 'value'),
            State('key-scopes-input', 'value')
        ],
        prevent_initial_call=True
    )
    def generate_new_key(n_clicks, name, environment, rate_limit, expiry_days, scopes):
        """Generate a new API key."""
        if not n_clicks:
            return html.Div(), html.Div(), html.Div()

        try:
            # Validate inputs
            if not name or not name.strip():
                return (
                    html.Div(),
                    dbc.Alert("Please enter a key name", color="danger"),
                    html.Div()
                )

            if not scopes:
                return (
                    html.Div(),
                    dbc.Alert("Please select at least one permission", color="danger"),
                    html.Div()
                )

            # For demo purposes, use a test user ID
            user_id = 1  # TODO: Get from authenticated session

            # Generate key
            result = APIKeyService.generate_key(
                user_id=user_id,
                name=name.strip(),
                environment=environment,
                rate_limit=int(rate_limit or 1000),
                expires_in_days=int(expiry_days) if expiry_days else None,
                scopes=scopes
            )

            # Display generated key (SHOW ONCE!)
            key_display = dbc.Alert([
                html.H5([
                    html.I(className="bi bi-key-fill me-2"),
                    "API Key Generated Successfully!"
                ], className="alert-heading"),
                html.Hr(),
                html.P([
                    html.Strong("Your API key (copy now - shown only once):"),
                ]),
                dbc.InputGroup([
                    dbc.Input(
                        value=result['api_key'],
                        id='generated-key-value',
                        readonly=True,
                        type="text"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-clipboard")],
                        id='copy-key-btn',
                        color="secondary",
                        n_clicks=0
                    )
                ], className="mb-3"),
                html.P([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    html.Strong("Important: "),
                    "Store this key securely. You won't be able to see it again. "
                    "If you lose it, you'll need to generate a new one."
                ], className="mb-0 text-danger small")
            ], color="success")

            # Success message
            success_msg = dbc.Alert(
                "Key generated successfully! Store it securely.",
                color="success"
            )

            # Reload table
            keys = APIKeyService.list_user_keys(user_id, include_inactive=True)
            table = load_api_keys_table('api-keys')

            return key_display, success_msg, table

        except ValueError as e:
            return (
                html.Div(),
                dbc.Alert(f"Validation error: {str(e)}", color="danger"),
                html.Div()
            )
        except Exception as e:
            logger.error(f"Error generating API key: {e}", exc_info=True)
            return (
                html.Div(),
                dbc.Alert(f"Error: {str(e)}", color="danger"),
                html.Div()
            )

    @app.callback(
        [
            Output('revoke-key-modal', 'is_open'),
            Output('selected-key-id', 'data'),
            Output('revoke-key-info', 'children')
        ],
        [
            Input({'type': 'revoke-key-btn', 'index': dash.ALL}, 'n_clicks'),
            Input('cancel-revoke-btn', 'n_clicks'),
            Input('confirm-revoke-btn', 'n_clicks')
        ],
        [
            State('revoke-key-modal', 'is_open'),
            State('selected-key-id', 'data')
        ],
        prevent_initial_call=True
    )
    def handle_revoke_key(revoke_clicks, cancel_clicks, confirm_clicks, is_open, selected_key_id):
        """Handle revoke key modal and action."""
        ctx = callback_context
        if not ctx.triggered:
            return is_open, selected_key_id, html.Div()

        trigger_id = ctx.triggered[0]['prop_id']

        # Open modal for specific key
        if 'revoke-key-btn' in trigger_id:
            import json
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
            button_id = json.loads(trigger_id.split('.')[0])
            key_id = button_id['index']

            # Get key details
            user_id = 1  # TODO: Get from session
            keys = APIKeyService.list_user_keys(user_id, include_inactive=True)
            key = next((k for k in keys if k.id == key_id), None)

            if key:
                info = html.Div([
                    html.P([html.Strong("Name: "), key.name]),
                    html.P([html.Strong("Prefix: "), html.Code(f"{key.prefix}...")]),
                    html.P([html.Strong("Created: "), key.created_at.strftime("%Y-%m-%d %H:%M")])
                ])
                return True, key_id, info

        # Cancel
        if 'cancel-revoke-btn' in trigger_id:
            return False, None, html.Div()

        # Confirm revoke
        if 'confirm-revoke-btn' in trigger_id and selected_key_id:
            try:
                user_id = 1  # TODO: Get from session
                success = APIKeyService.revoke_key(selected_key_id, user_id)

                if success:
                    logger.info(f"Revoked API key {selected_key_id}")

                return False, None, html.Div()

            except Exception as e:
                logger.error(f"Error revoking key: {e}", exc_info=True)
                return False, None, html.Div()

        return is_open, selected_key_id, html.Div()

    logger.info("API key management callbacks registered")
