"""
API Key Management Callbacks (Feature #1).
Handles UI interactions for API key management.
"""
import json
from dash import Input, Output, State, html, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime

from services.api_key_service import APIKeyService
from utils.logger import setup_logger
from utils.auth_utils import get_current_user_id

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
            user_id = get_current_user_id()

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

            user_id = get_current_user_id()

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
            button_id = json.loads(trigger_id.split('.')[0])
            key_id = button_id['index']

            # Get key details
            user_id = get_current_user_id()
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
                user_id = get_current_user_id()
                success = APIKeyService.revoke_key(selected_key_id, user_id)

                if success:
                    logger.info(f"Revoked API key {selected_key_id}")

                return False, None, html.Div()

            except Exception as e:
                logger.error(f"Error revoking key: {e}", exc_info=True)
                return False, None, html.Div()

        return is_open, selected_key_id, html.Div()

    @app.callback(
        [
            Output('total-api-requests-count', 'children'),
            Output('avg-response-time', 'children'),
            Output('api-success-rate', 'children'),
            Output('active-api-keys-count', 'children'),
        ],
        [
            Input('settings-tabs', 'active_tab'),
            Input('refresh-api-stats-btn', 'n_clicks'),
        ]
    )
    def update_api_stats_summary(active_tab, refresh_clicks):
        """Update API usage summary statistics."""
        if active_tab != 'api-keys':
            return "0", "0ms", "0%", "0"

        try:
            from database.connection import get_db_session
            from models.api_key import APIKey, APIUsage
            from models.api_request_log import APIRequestLog
            from sqlalchemy import func
            from datetime import timedelta

            with get_db_session() as session:
                # Active API keys count
                user_id = 1  # TODO: Get from session
                active_keys_count = session.query(APIKey)\
                    .filter_by(user_id=user_id, is_active=True)\
                    .count()

                # Get API request stats for last 30 days
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)

                # Total requests (from APIUsage or APIRequestLog)
                total_requests = session.query(func.count(APIUsage.id))\
                    .join(APIKey, APIUsage.api_key_id == APIKey.id)\
                    .filter(APIKey.user_id == user_id)\
                    .filter(APIUsage.timestamp >= thirty_days_ago)\
                    .scalar() or 0

                # Average response time
                avg_response = session.query(func.avg(APIUsage.response_time_ms))\
                    .join(APIKey, APIUsage.api_key_id == APIKey.id)\
                    .filter(APIKey.user_id == user_id)\
                    .filter(APIUsage.timestamp >= thirty_days_ago)\
                    .scalar()

                avg_response_str = f"{int(avg_response)}ms" if avg_response else "0ms"

                # Success rate (status codes 200-299)
                successful_requests = session.query(func.count(APIUsage.id))\
                    .join(APIKey, APIUsage.api_key_id == APIKey.id)\
                    .filter(APIKey.user_id == user_id)\
                    .filter(APIUsage.timestamp >= thirty_days_ago)\
                    .filter(APIUsage.status_code >= 200)\
                    .filter(APIUsage.status_code < 300)\
                    .scalar() or 0

                success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
                success_rate_str = f"{success_rate:.1f}%"

                return (
                    f"{total_requests:,}",
                    avg_response_str,
                    success_rate_str,
                    str(active_keys_count)
                )

        except Exception as e:
            logger.error(f"Error loading API stats summary: {e}", exc_info=True)
            return "Error", "Error", "Error", "Error"

    @app.callback(
        Output('api-usage-timeline-chart', 'figure'),
        [
            Input('settings-tabs', 'active_tab'),
            Input('refresh-api-stats-btn', 'n_clicks'),
        ]
    )
    def update_api_usage_timeline(active_tab, refresh_clicks):
        """Update API usage timeline chart."""
        if active_tab != 'api-keys':
            return {}

        try:
            from database.connection import get_db_session
            from models.api_key import APIKey, APIUsage
            from sqlalchemy import func
            from datetime import timedelta
            import plotly.graph_objects as go

            with get_db_session() as session:
                user_id = 1  # TODO: Get from session
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)

                # Get daily request counts
                daily_stats = session.query(
                    func.date(APIUsage.timestamp).label('date'),
                    func.count(APIUsage.id).label('count')
                )\
                    .join(APIKey, APIUsage.api_key_id == APIKey.id)\
                    .filter(APIKey.user_id == user_id)\
                    .filter(APIUsage.timestamp >= thirty_days_ago)\
                    .group_by(func.date(APIUsage.timestamp))\
                    .order_by(func.date(APIUsage.timestamp))\
                    .all()

                if not daily_stats:
                    return {
                        'data': [],
                        'layout': {
                            'title': 'No data available',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Requests'},
                            'height': 300,
                        }
                    }

                dates = [stat.date for stat in daily_stats]
                counts = [stat.count for stat in daily_stats]

                fig = go.Figure(data=[
                    go.Scatter(
                        x=dates,
                        y=counts,
                        mode='lines+markers',
                        name='Requests',
                        line=dict(color='#0d6efd', width=2),
                        marker=dict(size=6)
                    )
                ])

                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Number of Requests',
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    hovermode='x unified'
                )

                return fig

        except Exception as e:
            logger.error(f"Error loading API usage timeline: {e}", exc_info=True)
            return {}

    @app.callback(
        Output('api-top-keys-chart', 'figure'),
        [
            Input('settings-tabs', 'active_tab'),
            Input('refresh-api-stats-btn', 'n_clicks'),
        ]
    )
    def update_top_keys_chart(active_tab, refresh_clicks):
        """Update top API keys by request count chart."""
        if active_tab != 'api-keys':
            return {}

        try:
            from database.connection import get_db_session
            from models.api_key import APIKey, APIUsage
            from sqlalchemy import func
            from datetime import timedelta
            import plotly.graph_objects as go

            with get_db_session() as session:
                user_id = 1  # TODO: Get from session
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)

                # Get top 10 API keys by request count
                top_keys = session.query(
                    APIKey.name,
                    APIKey.prefix,
                    func.count(APIUsage.id).label('count')
                )\
                    .join(APIUsage, APIUsage.api_key_id == APIKey.id)\
                    .filter(APIKey.user_id == user_id)\
                    .filter(APIUsage.timestamp >= thirty_days_ago)\
                    .group_by(APIKey.id, APIKey.name, APIKey.prefix)\
                    .order_by(func.count(APIUsage.id).desc())\
                    .limit(10)\
                    .all()

                if not top_keys:
                    return {
                        'data': [],
                        'layout': {
                            'title': 'No data available',
                            'height': 300,
                        }
                    }

                labels = [f"{key.name} ({key.prefix}...)" for key in top_keys]
                values = [key.count for key in top_keys]

                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        marker=dict(color='#198754')
                    )
                ])

                fig.update_layout(
                    xaxis_title='API Key',
                    yaxis_title='Request Count',
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=80),
                    xaxis={'tickangle': -45}
                )

                return fig

        except Exception as e:
            logger.error(f"Error loading top keys chart: {e}", exc_info=True)
            return {}

    @app.callback(
        Output('api-endpoints-chart', 'figure'),
        [
            Input('settings-tabs', 'active_tab'),
            Input('refresh-api-stats-btn', 'n_clicks'),
        ]
    )
    def update_endpoints_chart(active_tab, refresh_clicks):
        """Update requests by endpoint chart."""
        if active_tab != 'api-keys':
            return {}

        try:
            from database.connection import get_db_session
            from models.api_key import APIKey, APIUsage
            from sqlalchemy import func
            from datetime import timedelta
            import plotly.graph_objects as go

            with get_db_session() as session:
                user_id = 1  # TODO: Get from session
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)

                # Get top endpoints by request count
                top_endpoints = session.query(
                    APIUsage.endpoint,
                    func.count(APIUsage.id).label('count')
                )\
                    .join(APIKey, APIUsage.api_key_id == APIKey.id)\
                    .filter(APIKey.user_id == user_id)\
                    .filter(APIUsage.timestamp >= thirty_days_ago)\
                    .group_by(APIUsage.endpoint)\
                    .order_by(func.count(APIUsage.id).desc())\
                    .limit(10)\
                    .all()

                if not top_endpoints:
                    return {
                        'data': [],
                        'layout': {
                            'title': 'No data available',
                            'height': 300,
                        }
                    }

                labels = [endpoint.endpoint for endpoint in top_endpoints]
                values = [endpoint.count for endpoint in top_endpoints]

                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4
                    )
                ])

                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )

                return fig

        except Exception as e:
            logger.error(f"Error loading endpoints chart: {e}", exc_info=True)
            return {}

    @app.callback(
        Output('api-usage-detail-table', 'children'),
        [
            Input('settings-tabs', 'active_tab'),
            Input('refresh-api-stats-btn', 'n_clicks'),
        ]
    )
    def update_api_usage_detail_table(active_tab, refresh_clicks):
        """Update detailed API usage table."""
        if active_tab != 'api-keys':
            return html.Div()

        try:
            from database.connection import get_db_session
            from models.api_key import APIKey, APIUsage
            from sqlalchemy import func
            from datetime import timedelta

            with get_db_session() as session:
                user_id = 1  # TODO: Get from session
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)

                # Get detailed stats per API key
                key_stats = session.query(
                    APIKey.name,
                    APIKey.prefix,
                    APIKey.is_active,
                    func.count(APIUsage.id).label('total_requests'),
                    func.avg(APIUsage.response_time_ms).label('avg_response_time'),
                    func.max(APIUsage.timestamp).label('last_used')
                )\
                    .outerjoin(APIUsage,
                              (APIUsage.api_key_id == APIKey.id) &
                              (APIUsage.timestamp >= thirty_days_ago))\
                    .filter(APIKey.user_id == user_id)\
                    .group_by(APIKey.id, APIKey.name, APIKey.prefix, APIKey.is_active)\
                    .all()

                if not key_stats:
                    return html.P("No API keys found.", className="text-muted text-center py-4")

                # Build table
                table_header = html.Thead(html.Tr([
                    html.Th("API Key Name"),
                    html.Th("Prefix"),
                    html.Th("Status"),
                    html.Th("Total Requests"),
                    html.Th("Avg Response Time"),
                    html.Th("Last Used"),
                ]))

                rows = []
                for stat in key_stats:
                    status_badge = dbc.Badge(
                        "Active",
                        color="success"
                    ) if stat.is_active else dbc.Badge(
                        "Inactive",
                        color="secondary"
                    )

                    last_used_str = stat.last_used.strftime('%Y-%m-%d %H:%M') if stat.last_used else "Never"
                    avg_time_str = f"{int(stat.avg_response_time)}ms" if stat.avg_response_time else "N/A"

                    row = html.Tr([
                        html.Td(stat.name),
                        html.Td(html.Code(f"{stat.prefix}...")),
                        html.Td(status_badge),
                        html.Td(f"{stat.total_requests:,}"),
                        html.Td(avg_time_str),
                        html.Td(last_used_str),
                    ])
                    rows.append(row)

                table_body = html.Tbody(rows)
                return dbc.Table(
                    [table_header, table_body],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm"
                )

        except Exception as e:
            logger.error(f"Error loading API usage detail table: {e}", exc_info=True)
            return dbc.Alert(f"Error loading usage details: {str(e)}", color="danger")

    logger.info("API key management callbacks registered")
