"""
Session helper utilities for extracting user information from Dash callbacks.
Provides consistent authentication across all callbacks.
"""
from typing import Optional
from flask import request, has_request_context
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_current_user_id(session_store_data: Optional[dict] = None) -> Optional[int]:
    """
    Extract authenticated user_id from session or request context.

    This function provides a centralized way to get the current user's ID
    across all Dash callbacks. It tries multiple sources in order of preference:
    1. Dash session store (from dcc.Store component)
    2. Flask request context (from JWT middleware)
    3. API key authentication (from API key middleware)

    Args:
        session_store_data: Data from dcc.Store(id='session-store', storage_type='session')
                           Expected format: {'user_id': int, 'username': str, ...}

    Returns:
        int: User ID if found, None otherwise

    Example:
        >>> # In a Dash callback:
        >>> @app.callback(
        ...     Output('user-info', 'children'),
        ...     Input('some-input', 'value'),
        ...     State('session-store', 'data')
        ... )
        ... def show_user_info(input_val, session_data):
        ...     user_id = get_current_user_id(session_data)
        ...     if not user_id:
        ...         return "Please log in"
        ...     # Use user_id...

    Note:
        For development/testing without authentication, this will return None.
        Callbacks should handle None gracefully or redirect to login.
    """
    # Priority 1: Check Dash session store
    if session_store_data and isinstance(session_store_data, dict):
        user_id = session_store_data.get('user_id')
        if user_id:
            logger.debug(f"User ID from session store: {user_id}")
            return int(user_id)

    # Priority 2: Check Flask request context (if available)
    if has_request_context():
        # Check for JWT authentication
        if hasattr(request, 'current_user') and request.current_user:
            user_id = request.current_user.get('user_id')
            if user_id:
                logger.debug(f"User ID from JWT auth: {user_id}")
                return int(user_id)

        # Check for API key authentication
        if hasattr(request, 'user_id') and request.user_id:
            logger.debug(f"User ID from API key auth: {request.user_id}")
            return int(request.user_id)

    # No authentication found
    logger.warning("No authenticated user found in session or request context")
    return None


def get_current_username(session_store_data: Optional[dict] = None) -> Optional[str]:
    """
    Extract authenticated username from session or request context.

    Args:
        session_store_data: Data from dcc.Store(id='session-store')

    Returns:
        str: Username if found, None otherwise
    """
    # Check session store
    if session_store_data and isinstance(session_store_data, dict):
        username = session_store_data.get('username')
        if username:
            return str(username)

    # Check request context
    if has_request_context():
        if hasattr(request, 'current_user') and request.current_user:
            username = request.current_user.get('username')
            if username:
                return str(username)

    return None


def require_authentication(user_id: Optional[int]) -> bool:
    """
    Check if user is authenticated.

    Args:
        user_id: User ID from get_current_user_id()

    Returns:
        bool: True if authenticated, False otherwise

    Example:
        >>> user_id = get_current_user_id(session_data)
        >>> if not require_authentication(user_id):
        ...     return dcc.Location(pathname='/login', id='redirect')
    """
    if user_id is None:
        logger.warning("Authentication required but user_id is None")
        return False
    return True


def get_demo_user_id() -> int:
    """
    Get demo user ID for development/testing.

    WARNING: This should NEVER be used in production!
    Only use when authentication is not yet implemented or for testing.

    Returns:
        int: Demo user ID (always 1)
    """
    logger.warning("Using demo user ID - NOT FOR PRODUCTION!")
    return 1
