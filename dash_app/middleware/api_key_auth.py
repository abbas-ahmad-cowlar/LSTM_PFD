"""
API Key Authentication Middleware (Feature #1).
Handles API key validation for programmatic access.
"""
from functools import wraps
from flask import request, jsonify
from datetime import datetime

from services.api_key_service import APIKeyService
from utils.logger import setup_logger

logger = setup_logger(__name__)


class APIKeyAuth:
    """
    API Key Authentication Middleware.

    Supports multiple authentication methods:
        1. X-API-Key header (preferred)
        2. Authorization: Bearer <key> header
        3. Query parameter: api_key (discouraged, but supported)

    Usage:
        @app.route('/api/v1/predict')
        @APIKeyAuth.require_api_key
        @RateLimiter.rate_limit_decorator  # Apply after auth
        def predict():
            # Access authenticated user via request.api_key
            user_id = request.api_key.user_id
            return {"result": "success"}
    """

    @staticmethod
    def require_api_key(f):
        """
        Decorator requiring valid API key for endpoint access.

        Attaches the following to the request object:
            - request.api_key: APIKey model instance
            - request.user_id: User ID for convenience

        Returns 401 if:
            - No API key provided
            - Invalid API key
            - Expired API key
            - Inactive (revoked) API key

        Example Response (401):
            {
                "error": "invalid_api_key",
                "message": "Invalid or inactive API key."
            }
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract API key from request
            api_key = APIKeyAuth._extract_api_key_from_request()

            if not api_key:
                logger.warning(
                    f"API key missing for {request.method} {request.path} "
                    f"from {request.remote_addr}"
                )
                return jsonify({
                    'error': 'missing_api_key',
                    'message': (
                        'API key required. Provide via X-API-Key header, '
                        'Authorization: Bearer header, or api_key query parameter.'
                    )
                }), 401

            # Verify API key
            api_key_record = APIKeyService.verify_key(api_key)

            if not api_key_record:
                logger.warning(
                    f"Invalid API key attempt for {request.method} {request.path} "
                    f"from {request.remote_addr}: key_prefix={api_key[:20]}"
                )
                return jsonify({
                    'error': 'invalid_api_key',
                    'message': 'Invalid or inactive API key.'
                }), 401

            # Check expiration (extra safety check, should be caught in verify_key)
            if api_key_record.is_expired():
                logger.warning(
                    f"Expired API key attempt: id={api_key_record.id}, "
                    f"expired_at={api_key_record.expires_at}"
                )
                return jsonify({
                    'error': 'expired_api_key',
                    'message': f'API key expired at {api_key_record.expires_at.isoformat()}.'
                }), 401

            # Attach API key record to request for downstream use
            request.api_key = api_key_record
            request.user_id = api_key_record.user_id

            logger.info(
                f"API key authenticated: id={api_key_record.id}, "
                f"user_id={api_key_record.user_id}, "
                f"endpoint={request.path}"
            )

            # Execute endpoint
            return f(*args, **kwargs)

        return decorated_function

    @staticmethod
    def _extract_api_key_from_request() -> str:
        """
        Extract API key from request using multiple methods.

        Priority:
            1. X-API-Key header (preferred)
            2. Authorization: Bearer <key> header
            3. Query parameter: api_key (not recommended)

        Returns:
            API key string or None if not found

        Security Note:
            Query parameters are logged by proxies and load balancers,
            so headers are strongly preferred.
        """
        # Method 1: X-API-Key header (preferred)
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return api_key.strip()

        # Method 2: Authorization Bearer header
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
            return api_key.strip()

        # Method 3: Query parameter (discouraged)
        api_key = request.args.get('api_key')
        if api_key:
            logger.warning(
                "API key provided via query parameter. "
                "This is insecure - use X-API-Key header instead."
            )
            return api_key.strip()

        return None

    @staticmethod
    def require_scope(*required_scopes):
        """
        Decorator to require specific scopes.

        Usage:
            @app.route('/api/v1/data')
            @APIKeyAuth.require_api_key
            @APIKeyAuth.require_scope('read', 'write')
            def update_data():
                return {"result": "updated"}

        Args:
            *required_scopes: One or more required scopes

        Returns:
            403 if API key doesn't have required scopes
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                api_key_record = getattr(request, 'api_key', None)

                if not api_key_record:
                    return jsonify({
                        'error': 'unauthorized',
                        'message': 'API key required. Apply @require_api_key first.'
                    }), 401

                # Check if API key has all required scopes
                api_key_scopes = set(api_key_record.scopes or [])
                required_scopes_set = set(required_scopes)

                if not required_scopes_set.issubset(api_key_scopes):
                    missing_scopes = required_scopes_set - api_key_scopes
                    logger.warning(
                        f"API key {api_key_record.id} missing scopes: {missing_scopes}"
                    )
                    return jsonify({
                        'error': 'insufficient_permissions',
                        'message': f'API key missing required scopes: {list(missing_scopes)}',
                        'required_scopes': list(required_scopes),
                        'current_scopes': list(api_key_scopes)
                    }), 403

                # Execute endpoint
                return f(*args, **kwargs)

            return decorated_function
        return decorator

    @staticmethod
    def optional_api_key(f):
        """
        Decorator that allows but doesn't require API key.

        Use cases:
            - Public endpoints with higher rate limits for authenticated users
            - Endpoints that behave differently for authenticated vs anonymous

        Usage:
            @app.route('/api/v1/public-data')
            @APIKeyAuth.optional_api_key
            def get_public_data():
                if hasattr(request, 'api_key'):
                    # Authenticated user - return more data
                    return {"data": "full"}
                else:
                    # Anonymous user - return limited data
                    return {"data": "limited"}
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Try to extract API key
            api_key = APIKeyAuth._extract_api_key_from_request()

            if api_key:
                # Verify API key
                api_key_record = APIKeyService.verify_key(api_key)

                if api_key_record and not api_key_record.is_expired():
                    # Valid API key - attach to request
                    request.api_key = api_key_record
                    request.user_id = api_key_record.user_id
                    logger.info(f"Optional API key authenticated: id={api_key_record.id}")
                else:
                    logger.warning("Invalid optional API key - proceeding as anonymous")

            # Execute endpoint (with or without API key)
            return f(*args, **kwargs)

        return decorated_function
