"""
Authentication utilities for Dash callbacks.

This module provides session-based authentication helpers for Dash callbacks,
with proper error handling, logging, and production-ready features.
"""
import os
from functools import wraps
from typing import Optional, Dict, Any
from flask import session, has_request_context, g
import dash_bootstrap_components as dbc
from dash import html, no_update

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Cache user_id per request to avoid multiple session lookups
_REQUEST_USER_CACHE_KEY = '_cached_user_id'


def get_current_user_id() -> int:
    """
    Get authenticated user ID from Flask session.

    This function attempts to retrieve the user ID in the following order:
    1. From request-level cache (g object)
    2. From Flask session
    3. From development mode fallback (if ENV=development)

    Returns:
        int: Authenticated user ID

    Raises:
        RuntimeError: If called outside Flask request context
        ValueError: If user is not authenticated and not in development mode

    Example:
        >>> user_id = get_current_user_id()
        >>> experiments = get_user_experiments(user_id)
    """
    # Check if we're in a Flask request context
    if not has_request_context():
        error_msg = "get_current_user_id() called outside Flask request context"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Check request-level cache first (performance optimization)
    if hasattr(g, _REQUEST_USER_CACHE_KEY):
        return getattr(g, _REQUEST_USER_CACHE_KEY)

    # Check Flask session
    if 'user_id' in session:
        user_id = session['user_id']
        logger.debug(f"Retrieved user_id={user_id} from session")

        # Cache for this request
        setattr(g, _REQUEST_USER_CACHE_KEY, user_id)
        return user_id

    # Development mode fallback
    env = os.getenv('ENV', 'development')
    if env == 'development':
        dev_user_id = int(os.getenv('DEV_USER_ID', '1'))
        logger.debug(f"Using development fallback user_id={dev_user_id} (ENV={env})")

        # Cache for this request
        setattr(g, _REQUEST_USER_CACHE_KEY, dev_user_id)
        return dev_user_id

    # Production mode: no user authenticated
    logger.warning(f"Authentication required but no user in session (ENV={env})")
    raise ValueError("User not authenticated. Please log in to access this feature.")


def get_current_user_id_safe() -> Optional[int]:
    """
    Safely get current user ID without raising exceptions.

    This is useful for optional authentication scenarios or when you want
    to check if a user is authenticated without handling exceptions.

    Returns:
        Optional[int]: User ID if authenticated, None otherwise

    Example:
        >>> user_id = get_current_user_id_safe()
        >>> if user_id:
        >>>     show_personalized_content(user_id)
        >>> else:
        >>>     show_login_prompt()
    """
    try:
        return get_current_user_id()
    except (ValueError, RuntimeError) as e:
        logger.debug(f"get_current_user_id_safe() returned None: {e}")
        return None


def is_authenticated() -> bool:
    """
    Check if current request has an authenticated user.

    Returns:
        bool: True if user is authenticated, False otherwise

    Example:
        >>> if is_authenticated():
        >>>     return user_dashboard()
        >>> else:
        >>>     return login_page()
    """
    return get_current_user_id_safe() is not None


def get_user_info() -> Optional[Dict[str, Any]]:
    """
    Get full user information from session.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with user info or None if not authenticated

    Example:
        >>> user_info = get_user_info()
        >>> if user_info:
        >>>     print(f"Welcome {user_info['username']}")
    """
    if not has_request_context():
        return None

    if not is_authenticated():
        return None

    return {
        'user_id': session.get('user_id'),
        'username': session.get('username'),
        'email': session.get('email'),
        'role': session.get('role', 'user'),
    }


def require_authentication(f):
    """
    Decorator to require authentication for Dash callbacks.

    If user is not authenticated, returns no_update to prevent callback execution.
    This silently fails - use require_authentication_with_message for user feedback.

    Args:
        f: Callback function to wrap

    Returns:
        Wrapped function that checks authentication before executing

    Example:
        >>> @app.callback(Output('data', 'children'), Input('btn', 'n_clicks'))
        >>> @require_authentication
        >>> def load_user_data(n_clicks):
        >>>     user_id = get_current_user_id()
        >>>     return f"Data for user {user_id}"
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            get_current_user_id()  # Will raise if not authenticated
            return f(*args, **kwargs)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Callback {f.__name__} blocked: {e}")
            return no_update

    return decorated_function


def require_authentication_with_message(message: Optional[str] = None):
    """
    Decorator factory that requires authentication and shows error message.

    Unlike require_authentication, this shows a user-friendly alert when
    authentication fails.

    Args:
        message: Custom error message (optional)

    Returns:
        Decorator function

    Example:
        >>> @app.callback(Output('content', 'children'), Input('btn', 'n_clicks'))
        >>> @require_authentication_with_message("Please log in to view experiments")
        >>> def load_experiments(n_clicks):
        >>>     return get_experiments(get_current_user_id())
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                get_current_user_id()  # Will raise if not authenticated
                return f(*args, **kwargs)
            except ValueError as e:
                error_msg = message or str(e)
                logger.warning(f"Callback {f.__name__} blocked: {error_msg}")
                return dbc.Alert([
                    html.I(className="bi bi-lock me-2"),
                    html.Strong("Authentication Required: "),
                    error_msg
                ], color="warning", className="m-3")
            except RuntimeError as e:
                logger.error(f"Request context error in {f.__name__}: {e}")
                return dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    html.Strong("Error: "),
                    "An unexpected error occurred. Please refresh the page."
                ], color="danger", className="m-3")

        return decorated_function
    return decorator


def set_current_user(user_id: int, username: str = None,
                     email: str = None, role: str = 'user') -> None:
    """
    Set current user in session (for login implementations).

    This should be called after successful authentication to establish
    a user session.

    Args:
        user_id: User's database ID
        username: Username (optional)
        email: Email address (optional)
        role: User role (default: 'user')

    Raises:
        RuntimeError: If called outside Flask request context

    Example:
        >>> # After successful login
        >>> set_current_user(
        >>>     user_id=user.id,
        >>>     username=user.username,
        >>>     email=user.email,
        >>>     role=user.role
        >>> )
    """
    if not has_request_context():
        raise RuntimeError("set_current_user() called outside Flask request context")

    session['user_id'] = user_id
    if username:
        session['username'] = username
    if email:
        session['email'] = email
    session['role'] = role

    # Clear request cache
    if hasattr(g, _REQUEST_USER_CACHE_KEY):
        delattr(g, _REQUEST_USER_CACHE_KEY)

    logger.info(f"User session established: user_id={user_id}, username={username}, role={role}")


def clear_current_user() -> None:
    """
    Clear current user session (for logout implementations).

    Example:
        >>> @app.route('/logout')
        >>> def logout():
        >>>     clear_current_user()
        >>>     return redirect('/login')
    """
    if not has_request_context():
        return

    user_id = session.get('user_id')

    # Clear all user-related session data
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('email', None)
    session.pop('role', None)

    # Clear request cache
    if hasattr(g, _REQUEST_USER_CACHE_KEY):
        delattr(g, _REQUEST_USER_CACHE_KEY)

    if user_id:
        logger.info(f"User session cleared: user_id={user_id}")


# Backward compatibility - keep original function name
def get_current_user_id_or_none() -> Optional[int]:
    """Alias for get_current_user_id_safe() - kept for backward compatibility."""
    return get_current_user_id_safe()
