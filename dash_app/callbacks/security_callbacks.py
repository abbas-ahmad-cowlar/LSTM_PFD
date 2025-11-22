"""
Security Settings Callbacks (Phase 6, Feature 3).
Handles UI interactions for security settings (password, 2FA, sessions).
"""
from dash import Input, Output, State, html, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime
import re
import pyotp
import qrcode
import io
import base64

from database.connection import get_db_session
from models.user import User
from models.session_log import SessionLog
from models.login_history import LoginHistory
from utils.logger import setup_logger

logger = setup_logger(__name__)


def register_security_callbacks(app):
    """
    Register all security settings callbacks.

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output('password-strength-indicator', 'children'),
        Input('new-password-input', 'value'),
        prevent_initial_call=True
    )
    def check_password_strength(password):
        """Display password strength indicator."""
        if not password:
            return ""

        strength = 0
        feedback = []

        # Length check
        if len(password) >= 8:
            strength += 1
        else:
            feedback.append("At least 8 characters")

        # Uppercase check
        if re.search(r'[A-Z]', password):
            strength += 1
        else:
            feedback.append("One uppercase letter")

        # Lowercase check
        if re.search(r'[a-z]', password):
            strength += 1
        else:
            feedback.append("One lowercase letter")

        # Number check
        if re.search(r'\d', password):
            strength += 1
        else:
            feedback.append("One number")

        # Special character check
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            strength += 1

        # Determine strength level
        if strength == 0:
            color = "danger"
            label = "Very Weak"
        elif strength <= 2:
            color = "danger"
            label = "Weak"
        elif strength == 3:
            color = "warning"
            label = "Fair"
        elif strength == 4:
            color = "info"
            label = "Good"
        else:
            color = "success"
            label = "Strong"

        # Build indicator
        progress = dbc.Progress(
            value=(strength / 5) * 100,
            color=color,
            className="mb-2",
            style={'height': '8px'}
        )

        message = html.Div([
            html.Small([
                html.Strong(f"Password Strength: {label}"),
                html.Br() if feedback else "",
                " " + ", ".join(feedback) if feedback else ""
            ], className=f"text-{color}")
        ])

        return html.Div([progress, message])

    @app.callback(
        Output('password-change-message', 'children'),
        Input('change-password-btn', 'n_clicks'),
        [
            State('current-password-input', 'value'),
            State('new-password-input', 'value'),
            State('confirm-password-input', 'value'),
        ],
        prevent_initial_call=True
    )
    def change_password(n_clicks, current_password, new_password, confirm_password):
        """Handle password change."""
        if not n_clicks:
            return ""

        try:
            # Validate inputs
            if not current_password or not current_password.strip():
                return dbc.Alert("Current password is required", color="danger", dismissable=True)

            if not new_password or not new_password.strip():
                return dbc.Alert("New password is required", color="danger", dismissable=True)

            if not confirm_password or not confirm_password.strip():
                return dbc.Alert("Please confirm your new password", color="danger", dismissable=True)

            # Check if passwords match
            if new_password != confirm_password:
                return dbc.Alert("New passwords do not match", color="danger", dismissable=True)

            # Validate password strength
            if len(new_password) < 8:
                return dbc.Alert("Password must be at least 8 characters long", color="danger", dismissable=True)

            if not re.search(r'[A-Z]', new_password):
                return dbc.Alert("Password must contain at least one uppercase letter", color="danger", dismissable=True)

            if not re.search(r'[a-z]', new_password):
                return dbc.Alert("Password must contain at least one lowercase letter", color="danger", dismissable=True)

            if not re.search(r'\d', new_password):
                return dbc.Alert("Password must contain at least one number", color="danger", dismissable=True)

            # TODO: Verify current password against stored hash
            # TODO: Hash new password and update database
            # For now, just show success (placeholder implementation)

            user_id = 1  # TODO: Get from authenticated session

            logger.info(f"Password change requested for user {user_id}")

            return dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "Password change functionality will be implemented with full authentication system. ",
                "Your password strength requirements are valid!"
            ], color="info", dismissable=True)

        except Exception as e:
            logger.error(f"Error changing password: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)

    @app.callback(
        Output('active-sessions-table', 'children'),
        [
            Input('settings-tabs', 'active_tab'),
            Input('refresh-sessions-btn', 'n_clicks'),
        ],
        prevent_initial_call=False
    )
    def load_active_sessions(active_tab, refresh_clicks):
        """Load active sessions table."""
        if active_tab != 'security':
            return ""

        try:
            # TODO: Get user_id from authenticated session
            user_id = 1  # Placeholder until authentication is implemented

            with get_db_session() as session:
                # Query active sessions from database
                active_sessions = session.query(SessionLog)\
                    .filter_by(user_id=user_id, is_active=True)\
                    .order_by(SessionLog.last_active.desc())\
                    .all()

                if not active_sessions:
                    return dbc.Alert([
                        html.I(className="bi bi-info-circle me-2"),
                        "No active sessions found. Sessions will be tracked once full authentication is implemented."
                    ], color="info")

                table_rows = []
                for sess in active_sessions:
                    # Format device info
                    device_info = sess.device_type or "Unknown Device"
                    if sess.browser:
                        device_info = f"{sess.browser} on {sess.device_type or 'Unknown'}"

                    # Format last active time
                    if sess.last_active:
                        time_diff = datetime.utcnow() - sess.last_active
                        if time_diff.total_seconds() < 60:
                            last_active = "Just now"
                        elif time_diff.total_seconds() < 3600:
                            minutes = int(time_diff.total_seconds() / 60)
                            last_active = f"{minutes} min ago"
                        elif time_diff.total_seconds() < 86400:
                            hours = int(time_diff.total_seconds() / 3600)
                            last_active = f"{hours} hour{'s' if hours > 1 else ''} ago"
                        else:
                            last_active = sess.last_active.strftime('%Y-%m-%d %H:%M')
                    else:
                        last_active = "Unknown"

                    # Determine if this is the current session
                    # TODO: Compare with current session token when auth is implemented
                    is_current = False  # Placeholder

                    badge = dbc.Badge("Current Session", color="success", className="ms-2") if is_current else ""

                    table_rows.append(html.Tr([
                        html.Td([
                            html.Strong(device_info),
                            badge
                        ]),
                        html.Td(sess.ip_address or "N/A"),
                        html.Td(sess.location or "Unknown"),
                        html.Td(last_active),
                    ]))

                return dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Device"),
                        html.Th("IP Address"),
                        html.Th("Location"),
                        html.Th("Last Active"),
                    ])),
                    html.Tbody(table_rows)
                ], bordered=True, hover=True, size="sm", responsive=True)

        except Exception as e:
            logger.error(f"Error loading active sessions: {e}", exc_info=True)
            return dbc.Alert(f"Error loading sessions: {str(e)}", color="danger")

    @app.callback(
        Output('login-history-table', 'children'),
        Input('settings-tabs', 'active_tab'),
        prevent_initial_call=False
    )
    def load_login_history(active_tab):
        """Load login history table."""
        if active_tab != 'security':
            return ""

        try:
            # TODO: Get user_id from authenticated session
            user_id = 1  # Placeholder until authentication is implemented

            with get_db_session() as session:
                # Query login history from database
                history = session.query(LoginHistory)\
                    .filter_by(user_id=user_id)\
                    .order_by(LoginHistory.timestamp.desc())\
                    .limit(50)\
                    .all()

                if not history:
                    return dbc.Alert([
                        html.I(className="bi bi-info-circle me-2"),
                        "No login history found. Login attempts will be tracked once full authentication is implemented."
                    ], color="info")

                table_rows = []
                for entry in history:
                    # Format timestamp
                    timestamp = entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') if entry.timestamp else "Unknown"

                    # Create status badge
                    if entry.success:
                        status_badge = dbc.Badge([
                            html.I(className="bi bi-check-circle me-1"),
                            "Success"
                        ], color="success")
                    else:
                        status_badge = dbc.Badge([
                            html.I(className="bi bi-x-circle me-1"),
                            "Failed"
                        ], color="danger")

                    # Format login method
                    login_method = entry.login_method or "password"
                    if entry.login_method == "2fa":
                        method_display = dbc.Badge("2FA", color="info", className="me-1")
                    elif entry.login_method == "oauth":
                        method_display = dbc.Badge("OAuth", color="primary", className="me-1")
                    elif entry.login_method == "api_key":
                        method_display = dbc.Badge("API Key", color="secondary", className="me-1")
                    else:
                        method_display = dbc.Badge("Password", color="light", text_color="dark", className="me-1")

                    table_rows.append(html.Tr([
                        html.Td(timestamp),
                        html.Td(entry.ip_address or "N/A"),
                        html.Td(entry.location or "Unknown"),
                        html.Td(method_display),
                        html.Td([
                            status_badge,
                            html.Small(f" - {entry.failure_reason}", className="text-muted") if not entry.success and entry.failure_reason else ""
                        ]),
                    ]))

                return dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Timestamp"),
                        html.Th("IP Address"),
                        html.Th("Location"),
                        html.Th("Method"),
                        html.Th("Status"),
                    ])),
                    html.Tbody(table_rows)
                ], bordered=True, hover=True, size="sm", responsive=True, striped=True)

        except Exception as e:
            logger.error(f"Error loading login history: {e}", exc_info=True)
            return dbc.Alert(f"Error loading login history: {str(e)}", color="danger")

    @app.callback(
        [
            Output('2fa-setup-modal', 'is_open'),
            Output('2fa-qr-code', 'children'),
            Output('2fa-secret-key', 'children'),
        ],
        [
            Input('enable-2fa-btn', 'n_clicks'),
            Input('cancel-2fa-setup-btn', 'n_clicks'),
            Input('verify-2fa-btn', 'n_clicks'),
        ],
        prevent_initial_call=True
    )
    def manage_2fa_setup(enable_clicks, cancel_clicks, verify_clicks):
        """Handle 2FA setup modal."""
        ctx = callback_context
        if not ctx.triggered:
            return False, "", ""

        trigger_id = ctx.triggered[0]['prop_id']

        if 'enable-2fa-btn' in trigger_id:
            try:
                # TODO: Get user_id from authenticated session
                user_id = 1  # Placeholder until authentication is implemented

                with get_db_session() as session:
                    user = session.query(User).filter_by(id=user_id).first()

                    if not user:
                        logger.error(f"User {user_id} not found")
                        return False, "", ""

                    # Generate TOTP secret if not already exists
                    if not user.totp_secret:
                        secret = pyotp.random_base32()
                        user.totp_secret = secret
                        session.commit()
                    else:
                        secret = user.totp_secret

                    # Generate TOTP URI for QR code
                    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                        name=user.email,
                        issuer_name='LSTM Dashboard'
                    )

                    # Generate QR code
                    qr = qrcode.QRCode(
                        version=1,
                        error_correction=qrcode.constants.ERROR_CORRECT_L,
                        box_size=10,
                        border=4
                    )
                    qr.add_data(totp_uri)
                    qr.make(fit=True)

                    img = qr.make_image(fill_color="black", back_color="white")

                    # Convert to base64
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()

                    qr_code = html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{img_str}',
                            style={'width': '200px', 'height': '200px', 'margin': '0 auto', 'display': 'block'}
                        ),
                        html.P(
                            "Scan this QR code with your authenticator app (Google Authenticator, Authy, etc.)",
                            className="text-center mt-2 small text-muted"
                        )
                    ])

                    # Format secret key for display (groups of 4 characters)
                    formatted_secret = ' '.join([secret[i:i+4] for i in range(0, len(secret), 4)])

                    logger.info(f"Generated 2FA QR code for user {user_id}")
                    return True, qr_code, formatted_secret

            except Exception as e:
                logger.error(f"Error generating 2FA setup: {e}", exc_info=True)
                return False, dbc.Alert(f"Error: {str(e)}", color="danger"), ""

        # Close modal
        return False, "", ""

    @app.callback(
        Output('2fa-setup-message', 'children'),
        Input('verify-2fa-btn', 'n_clicks'),
        State('2fa-verify-code-input', 'value'),
        prevent_initial_call=True
    )
    def verify_2fa_code(n_clicks, code):
        """Verify 2FA code."""
        if not n_clicks:
            return ""

        if not code or len(code) != 6:
            return dbc.Alert("Please enter a 6-digit code", color="danger", dismissable=True)

        try:
            # TODO: Get user_id from authenticated session
            user_id = 1  # Placeholder until authentication is implemented

            with get_db_session() as session:
                user = session.query(User).filter_by(id=user_id).first()

                if not user:
                    logger.error(f"User {user_id} not found")
                    return dbc.Alert("User not found", color="danger", dismissable=True)

                if not user.totp_secret:
                    return dbc.Alert("2FA not set up. Please enable 2FA first.", color="danger", dismissable=True)

                # Verify TOTP code
                totp = pyotp.TOTP(user.totp_secret)
                if totp.verify(code, valid_window=1):  # Allow 30s time window
                    user.totp_enabled = True
                    session.commit()
                    logger.info(f"2FA enabled successfully for user {user_id}")
                    return dbc.Alert([
                        html.I(className="bi bi-check-circle me-2"),
                        "2FA enabled successfully! Your account is now protected with two-factor authentication."
                    ], color="success", dismissable=True)
                else:
                    logger.warning(f"Invalid 2FA code attempt for user {user_id}")
                    return dbc.Alert([
                        html.I(className="bi bi-x-circle me-2"),
                        "Invalid code. Please check your authenticator app and try again."
                    ], color="danger", dismissable=True)

        except Exception as e:
            logger.error(f"Error verifying 2FA code: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)
