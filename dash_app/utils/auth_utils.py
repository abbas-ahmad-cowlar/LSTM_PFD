"""
Authentication utilities for retrieving current user information.
Provides helper functions for accessing authenticated user data in Dash callbacks.
"""
from flask import has_request_context, request, session
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_current_user_id() -> int:
    """
    Get the current authenticated user's ID.

    This function attempts to retrieve the user ID from:
    1. Flask request context (if @require_auth decorator was used)
    2. Flask session (if user is logged in via session)
    3. Falls back to user_id=1 for development/testing

    Returns:
        int: User ID of the current authenticated user

    Usage in callbacks:
        >>> from utils.auth_utils import get_current_user_id
        >>> user_id = get_current_user_id()

    TODO: Implement proper session-based authentication for Dash callbacks
    TODO: Consider using dcc.Store for client-side user state management
    TODO: Integrate with JWT token validation for API requests
    """
    # Try to get user from Flask request context (if using @require_auth)
    if has_request_context():
        # Check if current_user was set by @require_auth decorator
        if hasattr(request, 'current_user') and request.current_user:
            user_id = request.current_user.get('user_id')
            if user_id:
                logger.debug(f"Retrieved user_id={user_id} from request context")
                return user_id

        # Check Flask session
        if 'user_id' in session:
            user_id = session['user_id']
            logger.debug(f"Retrieved user_id={user_id} from session")
            return user_id

    # Development fallback
    # TODO: Remove this fallback and require authentication in production
    logger.warning(
        "Could not retrieve user_id from request context or session. "
        "Using fallback user_id=1. This should be replaced with proper "
        "authentication in production."
    )
    return 1


def get_current_username() -> Optional[str]:
    """
    Get the current authenticated user's username.

    Returns:
        str: Username of the current user, or None if not available

    Usage:
        >>> from utils.auth_utils import get_current_username
        >>> username = get_current_username()
    """
    if has_request_context():
        # Check if current_user was set by @require_auth decorator
        if hasattr(request, 'current_user') and request.current_user:
            username = request.current_user.get('username')
            if username:
                return username

        # Check Flask session
        if 'username' in session:
            return session['username']

    return None


def set_current_user(user_id: int, username: str) -> None:
    """
    Set the current user in the Flask session.

    This should be called after successful login.

    Args:
        user_id: User ID
        username: Username

    Usage:
        >>> from utils.auth_utils import set_current_user
        >>> set_current_user(user_id=1, username='admin')
    """
    if has_request_context():
        session['user_id'] = user_id
        session['username'] = username
        logger.info(f"Set session for user_id={user_id}, username={username}")
    else:
        logger.warning("Cannot set user session - no request context available")


def clear_current_user() -> None:
    """
    Clear the current user from the Flask session.

    This should be called on logout.

    Usage:
        >>> from utils.auth_utils import clear_current_user
        >>> clear_current_user()
    """
    if has_request_context():
        session.pop('user_id', None)
        session.pop('username', None)
        logger.info("Cleared user session")
    else:
        logger.warning("Cannot clear user session - no request context available")
