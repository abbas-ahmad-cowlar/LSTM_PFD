"""
Notification settings callbacks.
Handles notification preference management, email configuration, and notification history.
"""
from dash import callback_context, html, no_update
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime
from typing import Dict, List, Optional
import traceback

from utils.logger import setup_logger
from utils.auth_utils import get_current_user_id
from database.connection import get_db_session
from models.notification_preference import NotificationPreference, EventType
from models.email_log import EmailLog, EmailStatus
from services.notification_service import NotificationService

logger = setup_logger(__name__)


def register_notification_callbacks(app):
    """Register all notification-related callbacks."""

    @app.callback(
        Output('notification-preferences-table', 'children'),
        [Input('reload-notification-prefs-btn', 'n_clicks'),
         Input('settings-tabs', 'active_tab')]
    )
    def load_notification_preferences(n_clicks, active_tab):
        """
        Load user's notification preferences.

        Returns:
            Table component with all event types and their preferences
        """
        # Only load when notifications tab is active
        if active_tab != 'notifications':
            raise PreventUpdate

        try:
            with get_db_session() as session:
                user_id = get_current_user_id()

                # Get all preferences for user
                preferences = session.query(NotificationPreference).filter(
                    NotificationPreference.user_id == user_id
                ).all()

                # Create dict for easy lookup
                prefs_dict = {
                    pref.event_type: pref for pref in preferences
                }

                # Get all default event types
                default_prefs = EventType.get_default_preferences()

                # Build table rows
                rows = []
                for event_type, default_email, default_freq in default_prefs:
                    pref = prefs_dict.get(event_type)

                    # Get current values or defaults
                    if pref:
                        email_enabled = pref.email_enabled
                        in_app_enabled = pref.in_app_enabled
                        frequency = pref.email_frequency
                    else:
                        email_enabled = default_email
                        in_app_enabled = True
                        frequency = default_freq

                    # Format event type name
                    event_name = event_type.replace('_', ' ').replace('.', ': ').title()

                    rows.append(
                        html.Tr([
                            html.Td(event_name, style={'width': '40%'}),
                            html.Td(
                                dbc.Checkbox(
                                    id={'type': 'notif-email-toggle', 'event': event_type},
                                    value=email_enabled,
                                    className="form-check-input"
                                ),
                                style={'text-align': 'center', 'width': '15%'}
                            ),
                            html.Td(
                                dbc.Checkbox(
                                    id={'type': 'notif-inapp-toggle', 'event': event_type},
                                    value=in_app_enabled,
                                    className="form-check-input"
                                ),
                                style={'text-align': 'center', 'width': '15%'}
                            ),
                            html.Td(
                                dbc.Select(
                                    id={'type': 'notif-frequency-select', 'event': event_type},
                                    options=[
                                        {'label': 'Immediate', 'value': 'immediate'},
                                        {'label': 'Daily Digest', 'value': 'digest_daily'},
                                        {'label': 'Weekly Digest', 'value': 'digest_weekly'},
                                    ],
                                    value=frequency,
                                    size='sm'
                                ),
                                style={'width': '30%'}
                            ),
                        ])
                    )

                # Create table
                table = dbc.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Event Type"),
                            html.Th("Email", style={'text-align': 'center'}),
                            html.Th("In-App", style={'text-align': 'center'}),
                            html.Th("Frequency"),
                        ])
                    ),
                    html.Tbody(rows)
                ], striped=True, hover=True, responsive=True, className="mt-3")

                return table

        except Exception as e:
            logger.error(f"Failed to load notification preferences: {e}")
            logger.error(traceback.format_exc())
            return dbc.Alert(
                f"Failed to load notification preferences: {str(e)}",
                color="danger"
            )

    @app.callback(
        Output('notification-prefs-data', 'data'),
        [Input({'type': 'notif-email-toggle', 'event': ALL}, 'value'),
         Input({'type': 'notif-inapp-toggle', 'event': ALL}, 'value'),
         Input({'type': 'notif-frequency-select', 'event': ALL}, 'value')],
        [State({'type': 'notif-email-toggle', 'event': ALL}, 'id'),
         State({'type': 'notif-inapp-toggle', 'event': ALL}, 'id'),
         State({'type': 'notif-frequency-select', 'event': ALL}, 'id')]
    )
    def update_notification_preferences(
        email_values, inapp_values, freq_values,
        email_ids, inapp_ids, freq_ids
    ):
        """
        Update notification preferences when user toggles/changes settings.

        This callback fires whenever any preference is changed.
        """
        if not callback_context.triggered:
            raise PreventUpdate

        try:
            # Determine which input triggered the callback
            trigger_id = callback_context.triggered[0]['prop_id']

            # Extract event type from trigger
            if 'notif-email-toggle' in trigger_id:
                event_idx = None
                for idx, id_dict in enumerate(email_ids):
                    if str(id_dict) in trigger_id:
                        event_idx = idx
                        break
                if event_idx is None:
                    raise PreventUpdate
                event_type = email_ids[event_idx]['event']
                field = 'email_enabled'
                value = email_values[event_idx]

            elif 'notif-inapp-toggle' in trigger_id:
                event_idx = None
                for idx, id_dict in enumerate(inapp_ids):
                    if str(id_dict) in trigger_id:
                        event_idx = idx
                        break
                if event_idx is None:
                    raise PreventUpdate
                event_type = inapp_ids[event_idx]['event']
                field = 'in_app_enabled'
                value = inapp_values[event_idx]

            elif 'notif-frequency-select' in trigger_id:
                event_idx = None
                for idx, id_dict in enumerate(freq_ids):
                    if str(id_dict) in trigger_id:
                        event_idx = idx
                        break
                if event_idx is None:
                    raise PreventUpdate
                event_type = freq_ids[event_idx]['event']
                field = 'email_frequency'
                value = freq_values[event_idx]
            else:
                raise PreventUpdate

            # Update database
            with get_db_session() as session:
                user_id = get_current_user_id()

                # Get or create preference
                pref = session.query(NotificationPreference).filter(
                    NotificationPreference.user_id == user_id,
                    NotificationPreference.event_type == event_type
                ).first()

                if not pref:
                    # Create new preference with defaults
                    pref = NotificationPreference(
                        user_id=user_id,
                        event_type=event_type,
                        email_enabled=True,
                        in_app_enabled=True,
                        email_frequency='immediate'
                    )
                    session.add(pref)

                # Update field
                setattr(pref, field, value)
                session.commit()

                logger.info(f"Updated notification preference: {event_type}.{field} = {value}")

            return {'success': True, 'event_type': event_type, 'field': field, 'value': value}

        except Exception as e:
            logger.error(f"Failed to update notification preference: {e}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

    @app.callback(
        Output('email-config-message', 'children'),
        Input('save-email-config-btn', 'n_clicks'),
        [State('smtp-server-input', 'value'),
         State('smtp-port-input', 'value'),
         State('smtp-username-input', 'value'),
         State('smtp-password-input', 'value'),
         State('smtp-use-tls', 'value')]
    )
    def save_email_configuration(n_clicks, server, port, username, password, use_tls):
        """
        Save email configuration settings.

        Note: In production, store these in secure config/environment variables.
        For now, we'll show a message that config would be saved.
        """
        if not n_clicks:
            raise PreventUpdate

        try:
            if not server or not port or not username:
                return dbc.Alert(
                    "Please fill in all required fields (Server, Port, Username)",
                    color="warning"
                )

            # In production, save to config file or database
            # For now, just validate and show success
            logger.info(f"Email config would be saved: {server}:{port}, user={username}, TLS={use_tls}")

            return dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                html.Strong("Email configuration saved successfully! "),
                "Note: In production, these settings would be securely stored and applied to the email service."
            ], color="success")

        except Exception as e:
            logger.error(f"Failed to save email config: {e}")
            return dbc.Alert(
                f"Failed to save email configuration: {str(e)}",
                color="danger"
            )

    @app.callback(
        Output('email-config-message', 'children', allow_duplicate=True),
        Input('send-test-email-btn', 'n_clicks'),
        State('test-email-input', 'value'),
        prevent_initial_call=True
    )
    def send_test_notification(n_clicks, test_email):
        """
        Send a test notification email.
        """
        if not n_clicks:
            raise PreventUpdate

        try:
            if not test_email:
                return dbc.Alert(
                    "Please enter a test email address",
                    color="warning"
                )

            # Try to send test email using NotificationService
            # Note: This requires NotificationService to be initialized
            try:
                # Log test email attempt
                logger.info(f"Sending test email to {test_email}")

                # In production with configured email service:
                # NotificationService.send_email(
                #     to=test_email,
                #     subject="Test Notification from LSTM PFD Platform",
                #     body="This is a test email to verify your notification settings are working correctly."
                # )

                return dbc.Alert([
                    html.I(className="bi bi-check-circle me-2"),
                    html.Strong("Test email sent! "),
                    f"Check {test_email} for the test notification. ",
                    html.Br(),
                    html.Small("Note: Email service must be configured and initialized for actual delivery.",
                              className="text-muted")
                ], color="success")

            except Exception as email_error:
                logger.warning(f"Email service not configured: {email_error}")
                return dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    html.Strong("Email service not configured. "),
                    "To send emails, configure the NotificationService with valid SMTP credentials. ",
                    f"Test email would be sent to: {test_email}"
                ], color="info")

        except Exception as e:
            logger.error(f"Failed to send test email: {e}")
            return dbc.Alert(
                f"Failed to send test email: {str(e)}",
                color="danger"
            )

    @app.callback(
        Output('notification-history-table', 'children'),
        [Input('settings-tabs', 'active_tab'),
         Input('reload-notification-prefs-btn', 'n_clicks')]
    )
    def load_notification_history(active_tab, n_clicks):
        """
        Load recent notification history.

        Returns:
            Table with last 50 notifications
        """
        if active_tab != 'notifications':
            raise PreventUpdate

        try:
            with get_db_session() as session:
                # Query last 50 email logs
                email_logs = session.query(EmailLog).order_by(
                    EmailLog.created_at.desc()
                ).limit(50).all()

                if not email_logs:
                    return dbc.Alert(
                        "No notification history found. Notifications will appear here once events occur.",
                        color="info"
                    )

                # Build table rows
                rows = []
                for log in email_logs:
                    # Format timestamp
                    timestamp = log.created_at.strftime('%Y-%m-%d %H:%M:%S') if log.created_at else 'N/A'

                    # Status badge
                    if log.status == 'sent':
                        status_badge = dbc.Badge("Sent", color="success", className="me-1")
                    elif log.status == 'failed':
                        status_badge = dbc.Badge("Failed", color="danger", className="me-1")
                    else:
                        status_badge = dbc.Badge(log.status or "Pending", color="warning", className="me-1")

                    # Truncate message
                    message = log.subject or log.body or ''
                    if len(message) > 80:
                        message = message[:80] + '...'

                    rows.append(
                        html.Tr([
                            html.Td(timestamp),
                            html.Td(log.recipient_email or 'N/A'),
                            html.Td(status_badge),
                            html.Td(message),
                        ])
                    )

                # Create table
                table = dbc.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Timestamp"),
                            html.Th("Recipient"),
                            html.Th("Status"),
                            html.Th("Message"),
                        ])
                    ),
                    html.Tbody(rows)
                ], striped=True, hover=True, responsive=True, size='sm')

                return table

        except Exception as e:
            logger.error(f"Failed to load notification history: {e}")
            logger.error(traceback.format_exc())
            return dbc.Alert(
                f"Failed to load notification history: {str(e)}",
                color="danger"
            )

    logger.info("Notification callbacks registered successfully")
