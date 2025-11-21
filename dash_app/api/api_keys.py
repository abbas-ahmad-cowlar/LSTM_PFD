"""
API Key Management Endpoints (Feature #1).
Provides REST API for managing API keys.
"""
from flask import Blueprint, request, jsonify
from middleware.auth import AuthMiddleware
from services.api_key_service import APIKeyService
from middleware.rate_limiter import RateLimiter
from utils.logger import setup_logger

logger = setup_logger(__name__)

api_keys_bp = Blueprint('api_keys', __name__, url_prefix='/api/v1')


@api_keys_bp.route('/api-keys', methods=['GET'])
@AuthMiddleware.require_auth
def list_api_keys():
    """
    List all API keys for the current user.

    Authentication:
        Requires JWT token in Authorization header

    Response:
        {
            "api_keys": [
                {
                    "id": 1,
                    "name": "Production API",
                    "prefix": "sk_live_abc12345678",
                    "rate_limit": 1000,
                    "scopes": ["read", "write"],
                    "is_active": true,
                    "last_used_at": "2025-01-15T10:30:00",
                    "expires_at": null,
                    "created_at": "2025-01-01T00:00:00"
                },
                ...
            ],
            "total": 2
        }

    Status Codes:
        200: Success
        401: Unauthorized (no JWT token)
        500: Server error
    """
    try:
        user_id = request.current_user['user_id']

        # List user's API keys
        keys = APIKeyService.list_user_keys(user_id, include_inactive=True)

        return jsonify({
            'api_keys': [key.to_dict() for key in keys],
            'total': len(keys)
        }), 200

    except Exception as e:
        logger.error(f"Error listing API keys: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@api_keys_bp.route('/api-keys', methods=['POST'])
@AuthMiddleware.require_auth
def create_api_key():
    """
    Generate a new API key.

    Authentication:
        Requires JWT token in Authorization header

    Request Body:
        {
            "name": "Production API",      # Required
            "environment": "live",          # Optional: "live" or "test", default "live"
            "rate_limit": 1000,             # Optional: default 1000 req/hour
            "expires_in_days": 365,         # Optional: null = never expires
            "scopes": ["read", "write"]     # Optional: default ["read", "write"]
        }

    Response:
        {
            "api_key": "sk_live_a1b2c3d4e5f6...",  # SHOWN ONCE!
            "id": 1,
            "name": "Production API",
            "prefix": "sk_live_a1b2c3d4",
            "rate_limit": 1000,
            "expires_at": "2026-01-15T00:00:00",
            "message": "API key generated. Save it securely - you won't be able to see it again."
        }

    Status Codes:
        201: Created successfully
        400: Invalid request (missing name, invalid environment, etc.)
        401: Unauthorized
        500: Server error
    """
    try:
        user_id = request.current_user['user_id']
        data = request.get_json() or {}

        # Validate input
        name = data.get('name', '').strip()
        if not name:
            return jsonify({
                'error': 'validation_error',
                'message': 'API key name is required'
            }), 400

        # Extract optional parameters
        environment = data.get('environment', 'live')
        rate_limit = data.get('rate_limit', 1000)
        expires_in_days = data.get('expires_in_days')
        scopes = data.get('scopes')

        # Validate environment
        if environment not in ['live', 'test']:
            return jsonify({
                'error': 'validation_error',
                'message': 'Environment must be "live" or "test"'
            }), 400

        # Validate rate limit
        if rate_limit <= 0:
            return jsonify({
                'error': 'validation_error',
                'message': 'Rate limit must be a positive number'
            }), 400

        # Generate key
        result = APIKeyService.generate_key(
            user_id=user_id,
            name=name,
            environment=environment,
            rate_limit=rate_limit,
            expires_in_days=expires_in_days,
            scopes=scopes
        )

        record = result['record']

        logger.info(
            f"User {user_id} generated API key: "
            f"id={record.id}, name='{name}'"
        )

        return jsonify({
            'api_key': result['api_key'],  # Plain text, shown ONCE
            'id': record.id,
            'name': record.name,
            'prefix': record.prefix,
            'rate_limit': record.rate_limit,
            'scopes': record.scopes,
            'expires_at': record.expires_at.isoformat() if record.expires_at else None,
            'created_at': record.created_at.isoformat(),
            'message': "API key generated. Save it securely - you won't be able to see it again."
        }), 201

    except ValueError as e:
        return jsonify({
            'error': 'validation_error',
            'message': str(e)
        }), 400

    except Exception as e:
        logger.error(f"Error creating API key: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@api_keys_bp.route('/api-keys/<int:key_id>', methods=['DELETE'])
@AuthMiddleware.require_auth
def revoke_api_key(key_id):
    """
    Revoke an API key.

    Authentication:
        Requires JWT token in Authorization header

    Path Parameters:
        key_id: ID of the API key to revoke

    Response:
        {
            "message": "API key revoked successfully",
            "key_id": 1
        }

    Status Codes:
        200: Revoked successfully
        401: Unauthorized
        404: API key not found or unauthorized
        500: Server error
    """
    try:
        user_id = request.current_user['user_id']

        # Revoke key
        success = APIKeyService.revoke_key(key_id, user_id)

        if not success:
            return jsonify({
                'error': 'not_found',
                'message': 'API key not found or you do not have permission to revoke it'
            }), 404

        logger.info(f"User {user_id} revoked API key {key_id}")

        return jsonify({
            'message': 'API key revoked successfully',
            'key_id': key_id
        }), 200

    except Exception as e:
        logger.error(f"Error revoking API key: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@api_keys_bp.route('/api-keys/<int:key_id>/usage', methods=['GET'])
@AuthMiddleware.require_auth
def get_api_key_usage(key_id):
    """
    Get usage statistics for an API key.

    Authentication:
        Requires JWT token in Authorization header

    Path Parameters:
        key_id: ID of the API key

    Query Parameters:
        hours: Hours to look back (default 24, max 720 = 30 days)

    Response:
        {
            "key_id": 1,
            "total_requests": 450,
            "success_rate": 98.5,
            "avg_response_time_ms": 42.3,
            "requests_by_endpoint": {
                "/api/v1/predict": 300,
                "/api/v1/data": 150
            },
            "current_limit_usage": {
                "current_count": 45,
                "limit": 1000,
                "remaining": 955,
                "reset_time": 1705320000,
                "window_start": 1705316400
            }
        }

    Status Codes:
        200: Success
        401: Unauthorized
        404: API key not found
        500: Server error
    """
    try:
        user_id = request.current_user['user_id']
        hours = min(int(request.args.get('hours', 24)), 720)  # Max 30 days

        # Verify user owns this key
        keys = APIKeyService.list_user_keys(user_id, include_inactive=True)
        key_ids = [k.id for k in keys]

        if key_id not in key_ids:
            return jsonify({
                'error': 'not_found',
                'message': 'API key not found or you do not have permission to view it'
            }), 404

        # Get usage statistics
        stats = APIKeyService.get_key_usage_stats(key_id, hours=hours)

        # Get current rate limit usage
        current_usage = RateLimiter.get_current_usage(key_id)

        # Add rate limit info from database
        key = next(k for k in keys if k.id == key_id)
        current_usage['limit'] = key.rate_limit
        current_usage['remaining'] = max(
            0,
            key.rate_limit - current_usage.get('current_count', 0)
        )

        return jsonify({
            'key_id': key_id,
            **stats,
            'current_limit_usage': current_usage
        }), 200

    except Exception as e:
        logger.error(f"Error getting API key usage: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@api_keys_bp.route('/api-keys/<int:key_id>', methods=['PATCH'])
@AuthMiddleware.require_auth
def update_api_key(key_id):
    """
    Update an API key (name, rate_limit, scopes).

    Authentication:
        Requires JWT token in Authorization header

    Path Parameters:
        key_id: ID of the API key

    Request Body:
        {
            "name": "New Name",         # Optional
            "rate_limit": 2000,         # Optional
            "scopes": ["read"]          # Optional
        }

    Response:
        {
            "message": "API key updated successfully",
            "key": { ... }
        }

    Status Codes:
        200: Updated successfully
        401: Unauthorized
        404: API key not found
        500: Server error
    """
    try:
        user_id = request.current_user['user_id']
        data = request.get_json() or {}

        # Verify user owns this key
        keys = APIKeyService.list_user_keys(user_id, include_inactive=True)
        key = next((k for k in keys if k.id == key_id), None)

        if not key:
            return jsonify({
                'error': 'not_found',
                'message': 'API key not found or you do not have permission to update it'
            }), 404

        # Update fields
        from database.connection import get_db_session
        with get_db_session() as session:
            db_key = session.query(APIKeyService.__class__).filter_by(id=key_id).first()

            if 'name' in data:
                db_key.name = data['name'].strip()

            if 'rate_limit' in data:
                rate_limit = int(data['rate_limit'])
                if rate_limit > 0:
                    db_key.rate_limit = rate_limit

            if 'scopes' in data:
                db_key.scopes = data['scopes']

            session.flush()

            logger.info(f"User {user_id} updated API key {key_id}")

            return jsonify({
                'message': 'API key updated successfully',
                'key': db_key.to_dict()
            }), 200

    except Exception as e:
        logger.error(f"Error updating API key: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
