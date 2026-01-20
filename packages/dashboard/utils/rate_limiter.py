"""
Flask-Limiter Rate Limiting Configuration.

Reference: Master Roadmap Chapter 4.8.A

Provides distributed rate limiting using Redis backend.
Protects heavy endpoints from abuse:
- /predict: 10/minute (ML inference)
- /predict/batch: 5/minute (batch inference)
- /api/hpo/start: 3/hour (expensive HPO)

Usage:
    from packages.dashboard.utils.rate_limiter import limiter, init_limiter
    
    # Initialize in app.py
    init_limiter(app)
    
    # Protect routes
    @app.route('/api/heavy')
    @limiter.limit("10 per minute")
    def heavy_endpoint():
        pass
"""

import os
from flask import Flask, request, jsonify

# Try to import Flask-Limiter
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    FLASK_LIMITER_AVAILABLE = True
except ImportError:
    FLASK_LIMITER_AVAILABLE = False
    Limiter = None
    get_remote_address = None

# Import fallback rate limiter
from packages.dashboard.middleware.security import RateLimiter as FallbackRateLimiter


# Redis URL from environment
REDIS_URL = os.getenv('REDIS_URL', os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))

# Rate limit storage URI
# Use Redis in production, memory in development
RATELIMIT_STORAGE_URI = os.getenv('RATELIMIT_STORAGE_URI', REDIS_URL)

# Default limits
DEFAULT_LIMITS = [
    "200 per day",
    "50 per hour"
]


def get_client_ip():
    """
    Get client IP address, respecting proxy headers if configured.
    
    Returns:
        Client IP address string
    """
    # Check for forwarded header (behind proxy)
    if os.getenv('TRUST_PROXY_HEADERS', 'false').lower() == 'true':
        forwarded = request.headers.get('X-Forwarded-For', '')
        if forwarded:
            # X-Forwarded-For can have multiple IPs, take the first (client)
            return forwarded.split(',')[0].strip()
    
    return request.remote_addr or '127.0.0.1'


# Create limiter instance
if FLASK_LIMITER_AVAILABLE:
    limiter = Limiter(
        key_func=get_client_ip,
        default_limits=DEFAULT_LIMITS,
        storage_uri=RATELIMIT_STORAGE_URI,
        strategy="fixed-window-elastic-expiry",
        headers_enabled=True,  # Add X-RateLimit-* headers
    )
else:
    limiter = None


def init_limiter(app: Flask) -> None:
    """
    Initialize rate limiter on Flask app.
    
    Args:
        app: Flask application instance
    """
    if FLASK_LIMITER_AVAILABLE and limiter:
        limiter.init_app(app)
        
        # Register error handler for rate limit exceeded
        @app.errorhandler(429)
        def ratelimit_handler(e):
            return jsonify({
                "error": "rate_limit_exceeded",
                "message": str(e.description),
                "retry_after": e.retry_after if hasattr(e, 'retry_after') else 60
            }), 429
        
        app.logger.info(f"Flask-Limiter initialized with storage: {RATELIMIT_STORAGE_URI}")
    else:
        # Use fallback rate limiter
        fallback = FallbackRateLimiter(requests_per_minute=60)
        app.before_request(_create_fallback_rate_check(fallback))
        app.logger.warning("Flask-Limiter not available, using fallback in-memory rate limiter")


def _create_fallback_rate_check(rate_limiter):
    """Create a before_request handler for fallback rate limiting."""
    def check_rate_limit():
        from flask import request, jsonify
        
        # Skip rate limiting for health checks
        if request.path in ['/health', '/metrics', '/']:
            return None
        
        identifier = get_client_ip()
        if not rate_limiter.is_allowed(identifier):
            return jsonify({
                "error": "rate_limit_exceeded",
                "message": "Rate limit exceeded: 60 per minute",
                "retry_after": 60
            }), 429
        
        return None
    
    return check_rate_limit


# Endpoint-specific rate limit configurations
ENDPOINT_LIMITS = {
    # ML inference endpoints
    'predict': "10 per minute",
    'predict_batch': "5 per minute",
    
    # HPO (expensive)
    'hpo_start': "3 per hour",
    'hpo_create': "3 per hour",
    
    # Authentication
    'login': "10 per minute",
    'register': "5 per hour",
    'password_reset': "3 per hour",
    
    # File uploads
    'dataset_upload': "10 per hour",
    'model_upload': "5 per hour",
    
    # API key management
    'api_key_create': "10 per day",
}


def get_endpoint_limit(endpoint: str) -> str:
    """
    Get the rate limit for a specific endpoint.
    
    Args:
        endpoint: Endpoint name/key
        
    Returns:
        Rate limit string
    """
    return ENDPOINT_LIMITS.get(endpoint, "100 per minute")


# Decorator for endpoint-specific limits
def endpoint_limit(endpoint_key: str):
    """
    Decorator to apply endpoint-specific rate limit.
    
    Usage:
        @app.route('/predict')
        @endpoint_limit('predict')
        def predict():
            pass
    """
    if FLASK_LIMITER_AVAILABLE and limiter:
        return limiter.limit(get_endpoint_limit(endpoint_key))
    else:
        # Return no-op decorator if limiter not available
        def decorator(f):
            return f
        return decorator


# Exempt decorator
def rate_limit_exempt(f):
    """
    Exempt a route from rate limiting.
    
    Usage:
        @app.route('/health')
        @rate_limit_exempt
        def health():
            pass
    """
    if FLASK_LIMITER_AVAILABLE and limiter:
        return limiter.exempt(f)
    return f


if __name__ == '__main__':
    # Test rate limiter
    print("Rate Limiter Configuration:")
    print(f"  Flask-Limiter available: {FLASK_LIMITER_AVAILABLE}")
    print(f"  Storage URI: {RATELIMIT_STORAGE_URI}")
    print(f"  Default limits: {DEFAULT_LIMITS}")
    print("\nEndpoint Limits:")
    for endpoint, limit in ENDPOINT_LIMITS.items():
        print(f"  {endpoint}: {limit}")
