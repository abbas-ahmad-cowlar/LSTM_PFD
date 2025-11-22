"""Authentication utilities for Dash callbacks."""
from flask import session

def get_current_user_id() -> int:
    """
    Get authenticated user ID from session.

    Returns:
        int: User ID from session, or 1 if not authenticated (dev mode)

    Raises:
        ValueError: If no user in session and ENV=production
    """
    if 'user_id' in session:
        return session['user_id']

    # Development fallback
    import os
    if os.getenv('ENV', 'development') == 'development':
        return 1  # Default dev user

    raise ValueError("User not authenticated")

def require_authentication(f):
    """Decorator to require authentication for callback."""
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            get_current_user_id()  # Will raise if not authenticated
            return f(*args, **kwargs)
        except ValueError:
            from dash import no_update
            return no_update

    return decorated_function
