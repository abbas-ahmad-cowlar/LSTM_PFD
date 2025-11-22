"""
Security Settings Callbacks (Phase 6, Feature 3).
Handles UI interactions for security settings (password, 2FA, sessions).
"""
from dash import Input, Output, State, html, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime
import re

from database.connection import get_db_session
from models.user import User
from utils.logger import setup_logger
from middleware.auth import verify_password, hash_password

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

            # Get user from session
            user_id = 1  # TODO: Get from authenticated session (will be fixed with full auth system)

            # Verify current password and update with new hashed password
            with get_db_session() as session:
                user = session.query(User).filter_by(id=user_id).first()

                if not user:
                    return dbc.Alert("User not found", color="danger", dismissable=True)

                # Verify current password
                if not verify_password(current_password, user.password_hash):
                    return dbc.Alert("Current password is incorrect", color="danger", dismissable=True)

                # Hash and update new password
                user.password_hash = hash_password(new_password)
                user.updated_at = datetime.utcnow()
                session.commit()

            logger.info(f"Password changed successfully for user {user_id}")

            return dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                "Password changed successfully!"
            ], color="success", dismissable=True)

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

        # Placeholder implementation
        # TODO: Implement session tracking in database
        sessions_data = [
            {
                'device': 'Chrome on Windows',
                'ip': '192.168.1.100',
                'location': 'New York, US',
                'last_active': 'Just now',
                'current': True
            },
        ]

        table_rows = []
        for session in sessions_data:
            badge = dbc.Badge("Current Session", color="success") if session['current'] else ""

            table_rows.append(html.Tr([
                html.Td([
                    html.Strong(session['device']),
                    html.Br(),
                    html.Small(badge, className="text-muted")
                ]),
                html.Td(session['ip']),
                html.Td(session['location']),
                html.Td(session['last_active']),
            ]))

        if not table_rows:
            return dbc.Alert("No active sessions found", color="info")

        return dbc.Table([
            html.Thead(html.Tr([
                html.Th("Device"),
                html.Th("IP Address"),
                html.Th("Location"),
                html.Th("Last Active"),
            ])),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, size="sm")

    @app.callback(
        Output('login-history-table', 'children'),
        Input('settings-tabs', 'active_tab'),
        prevent_initial_call=False
    )
    def load_login_history(active_tab):
        """Load login history table."""
        if active_tab != 'security':
            return ""

        # Placeholder implementation
        # TODO: Implement login history tracking in database
        history_data = [
            {
                'timestamp': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                'device': 'Chrome on Windows',
                'ip': '192.168.1.100',
                'location': 'New York, US',
                'status': 'success'
            },
        ]

        table_rows = []
        for login in history_data:
            status_badge = dbc.Badge(
                "✓ Success" if login['status'] == 'success' else "✗ Failed",
                color="success" if login['status'] == 'success' else "danger"
            )

            table_rows.append(html.Tr([
                html.Td(login['timestamp']),
                html.Td(login['device']),
                html.Td(login['ip']),
                html.Td(login['location']),
                html.Td(status_badge),
            ]))

        if not table_rows:
            return dbc.Alert("No login history found", color="info")

        return dbc.Table([
            html.Thead(html.Tr([
                html.Th("Timestamp"),
                html.Th("Device"),
                html.Th("IP Address"),
                html.Th("Location"),
                html.Th("Status"),
            ])),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, size="sm")

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
            # Generate placeholder QR code and secret
            # TODO: Generate actual TOTP secret and QR code
            qr_placeholder = html.Div([
                html.I(className="bi bi-qr-code", style={'fontSize': '120px', 'color': '#ccc'}),
                html.P("QR Code generation will be implemented with 2FA library", className="mt-2 small text-muted")
            ])

            secret_key = "ABCD EFGH IJKL MNOP"  # Placeholder

            return True, qr_placeholder, secret_key

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
            return dbc.Alert("Please enter a 6-digit code", color="danger")

        # TODO: Verify code against TOTP secret
        return dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "2FA verification will be implemented with TOTP library"
        ], color="info")
