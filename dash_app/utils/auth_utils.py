"""
Authentication Utilities for Dash Callbacks.

Provides a robust, production-ready authentication system for retrieving
current user information in Dash callbacks with multiple fallback strategies.

Architecture:
    - Tier 1: Dash Store ('session-store') - Client-side session state
    - Tier 2: Flask Session - Server-side session management
    - Tier 3: Flask Request Context - For @require_auth decorated endpoints
    - Tier 4: Development Fallback - Default user for development/testing

Usage:
    from utils.auth_utils import get_current_user_id, require_user_auth

    # In callbacks:
    user_id = get_current_user_id()

    # With Dash Store:
    user_id = get_current_user_id(session_store=session_store_data)

    # Decorator for callbacks requiring auth:
    @require_user_auth
    def my_callback(...):
        user_id = get_current_user_id()
        ...
"""
from flask import has_request_context, request, session
from typing import Optional, Dict, Any
from functools import wraps
from dash.exceptions import PreventUpdate
import os

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Configuration
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "False").lower() == "true"
DEFAULT_USER_ID = int(os.getenv("DEFAULT_USER_ID", "1"))
DEFAULT_USERNAME = os.getenv("DEFAULT_USERNAME", "dev_user")

# Global flag to track if warning has been logged (prevents spam)
_fallback_warning_logged = False


def get_current_user_id(session_store: Optional[Dict[str, Any]] = None) -> int:
    """
    Get the current authenticated user's ID with multiple fallback strategies.

    This function implements a robust multi-tier authentication check:
    1. Dash Store ('session-store') - Preferred for Dash applications
    2. Flask Session - Server-side session (requires login)
    3. Flask Request Context - From @require_auth middleware
    4. Development Fallback - Default user (with rate-limited warning)

    Args:
        session_store: Optional dict from dcc.Store('session-store').
                      If provided, will check for user_id in this data first.

    Returns:
        int: User ID of the current authenticated user

    Examples:
        # Simple usage (without Dash Store):
        >>> user_id = get_current_user_id()

        # With Dash Store (recommended):
        >>> @app.callback(
        ...     Output('result', 'children'),
        ...     Input('btn', 'n_clicks'),
        ...     State('session-store', 'data')
        ... )
        >>> def my_callback(n_clicks, session_store):
        ...     user_id = get_current_user_id(session_store)
        ...     return f"User ID: {user_id}"

    Note:
        In production, set ENABLE_AUTH=true in environment variables
        and ensure proper login flow populates session or session-store.
    """
    global _fallback_warning_logged

    # Tier 1: Check Dash Store (session-store) - RECOMMENDED for Dash
    if session_store and isinstance(session_store, dict):
        user_id = session_store.get('user_id')
        if user_id is not None:
            logger.debug(f"Retrieved user_id={user_id} from Dash Store")
            return int(user_id)

    # Tier 2: Check Flask Session - Server-side session
    if has_request_context():
        # Try Flask session
        user_id = session.get('user_id')
        if user_id is not None:
            logger.debug(f"Retrieved user_id={user_id} from Flask session")
            return int(user_id)

        # Tier 3: Check Flask request context (from @require_auth middleware)
        if hasattr(request, 'current_user') and request.current_user:
            user_id = request.current_user.get('user_id')
            if user_id is not None:
                logger.debug(f"Retrieved user_id={user_id} from request context")
                return int(user_id)

    # Tier 4: Development Fallback
    # Only log warning once to avoid spam
    if not _fallback_warning_logged:
        if ENABLE_AUTH:
            logger.warning(
                "AUTHENTICATION ENABLED but no user session found! "
                f"Using fallback user_id={DEFAULT_USER_ID}. "
                "This indicates missing login flow or session configuration. "
                "Subsequent calls will not log this warning."
            )
        else:
            logger.info(
                f"Authentication disabled (ENABLE_AUTH=false). "
                f"Using default user_id={DEFAULT_USER_ID} for development."
            )
        _fallback_warning_logged = True

    return DEFAULT_USER_ID


def get_current_username(session_store: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get the current authenticated user's username.

    Args:
        session_store: Optional dict from dcc.Store('session-store')

    Returns:
        str: Username of the current user, or None if not available

    Examples:
        >>> username = get_current_username()
        >>> print(f"Logged in as: {username or 'Anonymous'}")
    """
    # Tier 1: Dash Store
    if session_store and isinstance(session_store, dict):
        username = session_store.get('username')
        if username:
            return str(username)

    # Tier 2: Flask Session
    if has_request_context():
        username = session.get('username')
        if username:
            return str(username)

        # Tier 3: Request context
        if hasattr(request, 'current_user') and request.current_user:
            username = request.current_user.get('username')
            if username:
                return str(username)

    # Tier 4: Fallback
    return DEFAULT_USERNAME if not ENABLE_AUTH else None


def get_current_user_info(session_store: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get comprehensive current user information.

    Returns:
        dict: User information including user_id, username, role, etc.

    Examples:
        >>> user_info = get_current_user_info(session_store)
        >>> print(f"User: {user_info['username']} (ID: {user_info['user_id']})")
    """
    return {
        'user_id': get_current_user_id(session_store),
        'username': get_current_username(session_store),
        'is_authenticated': is_authenticated(session_store),
    }


def is_authenticated(session_store: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if a user is currently authenticated.

    Returns:
        bool: True if user is authenticated, False otherwise
    """
    # Check Dash Store
    if session_store and isinstance(session_store, dict):
        return session_store.get('user_id') is not None

    # Check Flask Session or Request Context
    if has_request_context():
        if session.get('user_id') is not None:
            return True
        if hasattr(request, 'current_user') and request.current_user:
            return True

    # In development mode, consider always authenticated
    return not ENABLE_AUTH


def set_current_user(user_id: int, username: str, role: str = "user",
                    additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Set the current user in Flask session.

    This should be called after successful login in a login callback.

    Args:
        user_id: User ID
        username: Username
        role: User role (default: "user")
        additional_data: Any additional user data to store

    Returns:
        dict: User data to be stored in dcc.Store('session-store')

    Example:
        >>> # In login callback:
        >>> session_data = set_current_user(
        ...     user_id=user.id,
        ...     username=user.username,
        ...     role=user.role
        ... )
        >>> return session_data  # Return to session-store
    """
    user_data = {
        'user_id': user_id,
        'username': username,
        'role': role,
    }

    if additional_data:
        user_data.update(additional_data)

    # Set in Flask session (if request context available)
    if has_request_context():
        session['user_id'] = user_id
        session['username'] = username
        session['role'] = role
        if additional_data:
            for key, value in additional_data.items():
                session[key] = value
        logger.info(f"Set Flask session for user_id={user_id}, username={username}")

    return user_data


def clear_current_user() -> None:
    """
    Clear the current user from Flask session.

    This should be called on logout.

    Returns:
        dict: Empty dict to clear dcc.Store('session-store')

    Example:
        >>> # In logout callback:
        >>> clear_current_user()
        >>> return {}  # Return empty dict to session-store
    """
    if has_request_context():
        session.clear()
        logger.info("Cleared user session")

    return {}


def require_user_auth(func):
    """
    Decorator to require authentication for a callback.

    If user is not authenticated (and ENABLE_AUTH=true), prevents callback execution.

    Usage:
        >>> @app.callback(
        ...     Output('protected-content', 'children'),
        ...     Input('btn', 'n_clicks'),
        ...     State('session-store', 'data')
        ... )
        ... @require_user_auth
        ... def protected_callback(n_clicks, session_store):
        ...     user_id = get_current_user_id(session_store)
        ...     return f"Protected data for user {user_id}"

    Note:
        - Only enforces authentication if ENABLE_AUTH=true
        - Requires 'session-store' as a State in the callback signature
        - Raises PreventUpdate if not authenticated
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Only enforce if authentication is enabled
        if not ENABLE_AUTH:
            return func(*args, **kwargs)

        # Try to find session_store in kwargs or args
        session_store = None

        # Check kwargs for session_store
        if 'session_store' in kwargs:
            session_store = kwargs['session_store']

        # Check if last arg looks like session store
        elif args and isinstance(args[-1], dict) and 'user_id' in args[-1]:
            session_store = args[-1]

        # Check authentication
        if not is_authenticated(session_store):
            logger.warning(
                f"Unauthorized callback access to {func.__name__}. "
                "Preventing callback execution."
            )
            raise PreventUpdate

        return func(*args, **kwargs)

    return wrapper


# Compatibility alias for backward compatibility
def get_user_id_from_session() -> int:
    """
    DEPRECATED: Use get_current_user_id() instead.

    Backward compatibility function.
    """
    logger.warning("get_user_id_from_session() is deprecated. Use get_current_user_id() instead.")
    return get_current_user_id()
