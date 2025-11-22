"""
User Profile Management Callbacks (Phase 6, Feature 2).
Handles UI interactions for user profile management.
"""
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from datetime import datetime

from database.connection import get_db_session
from models.user import User
from utils.logger import setup_logger
from utils.auth_utils import get_current_user_id

logger = setup_logger(__name__)


def register_profile_callbacks(app):
    """
    Register all user profile callbacks.

    Args:
        app: Dash application instance
    """

    @app.callback(
        [
            Output('profile-username-display', 'value'),
            Output('profile-role-display', 'value'),
            Output('profile-email-input', 'value'),
            Output('profile-status-display', 'value'),
            Output('profile-created-at', 'children'),
            Output('profile-updated-at', 'children'),
            Output('profile-user-data', 'data'),
        ],
        Input('settings-tabs', 'active_tab'),
        prevent_initial_call=False
    )
    def load_profile_data(active_tab):
        """Load user profile data when tab is activated."""
        if active_tab != 'profile':
            return "", "", "", "", "", "", None

        try:
            user_id = get_current_user_id()

            with get_db_session() as session:
                user = session.query(User).filter_by(id=user_id).first()

                if not user:
                    return "N/A", "N/A", "", "Unknown", "N/A", "N/A", None

                # Format dates
                created_at = user.created_at.strftime("%Y-%m-%d %H:%M") if user.created_at else "N/A"
                updated_at = user.updated_at.strftime("%Y-%m-%d %H:%M") if user.updated_at else "N/A"

                # Status
                status = "Active" if user.is_active else "Inactive"

                return (
                    user.username or "N/A",
                    user.role or "user",
                    user.email or "",
                    status,
                    created_at,
                    updated_at,
                    {'id': user.id, 'username': user.username, 'email': user.email}
                )

        except Exception as e:
            logger.error(f"Error loading profile: {e}", exc_info=True)
            return "Error", "Error", "", "Error", "Error", "Error", None

    @app.callback(
        Output('profile-update-message', 'children'),
        Input('save-profile-btn', 'n_clicks'),
        [
            State('profile-email-input', 'value'),
            State('profile-user-data', 'data'),
        ],
        prevent_initial_call=True
    )
    def update_profile(n_clicks, email, user_data):
        """Update user profile information."""
        if not n_clicks or not user_data:
            return ""

        try:
            # Validate email
            if not email or not email.strip():
                return dbc.Alert("Email is required", color="danger", dismissable=True)

            # Simple email validation
            if '@' not in email or '.' not in email:
                return dbc.Alert("Please enter a valid email address", color="danger", dismissable=True)

            user_id = user_data['id']

            with get_db_session() as session:
                user = session.query(User).filter_by(id=user_id).first()

                if not user:
                    return dbc.Alert("User not found", color="danger", dismissable=True)

                # Update email
                user.email = email.strip()
                user.updated_at = datetime.utcnow()

                session.commit()

                logger.info(f"Updated profile for user {user_id}")

                return dbc.Alert([
                    html.I(className="bi bi-check-circle me-2"),
                    "Profile updated successfully!"
                ], color="success", dismissable=True)

        except Exception as e:
            logger.error(f"Error updating profile: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)
