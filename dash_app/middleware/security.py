"""
Security middleware (Phase 11D).
Rate limiting, CORS, and security headers.
"""
from flask import request, jsonify
from functools import wraps
import time
from collections import defaultdict
import threading

from utils.logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter.
    For production, use Redis-based rate limiting (e.g., flask-limiter).
    """

    def __init__(self, requests_per_minute=60):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for identifier.

        Args:
            identifier: IP address or user ID

        Returns:
            True if allowed, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            minute_ago = now - 60

            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > minute_ago
            ]

            # Check limit
            if len(self.requests[identifier]) >= self.requests_per_minute:
                return False

            # Add current request
            self.requests[identifier].append(now)
            return True

    def limit(self, requests_per_minute=None):
        """
        Decorator for rate limiting Flask routes.

        Args:
            requests_per_minute: Override default limit

        Usage:
            @app.route('/api/endpoint')
            @rate_limiter.limit(requests_per_minute=30)
            def my_endpoint():
                return "Limited endpoint"
        """
        limit = requests_per_minute or self.requests_per_minute

        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Get identifier (IP address)
                identifier = request.remote_addr

                # Check if request is allowed
                if not self.is_allowed(identifier):
                    logger.warning(f"Rate limit exceeded for {identifier}")
                    return jsonify({
                        "error": "Rate limit exceeded",
                        "limit": limit,
                        "period": "1 minute"
                    }), 429

                return f(*args, **kwargs)

            return decorated_function

        return decorator


class SecurityMiddleware:
    """Security headers and CORS configuration."""

    @staticmethod
    def add_security_headers(response):
        """
        Add security headers to response.

        Headers added:
        - X-Content-Type-Options: nosniff
        - X-Frame-Options: DENY
        - X-XSS-Protection: 1; mode=block
        - Strict-Transport-Security: HTTPS only
        - Content-Security-Policy: XSS protection

        Args:
            response: Flask response object

        Returns:
            Response with security headers
        """
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'

        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'

        # XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # Force HTTPS (only in production)
        if not response.headers.get('Strict-Transport-Security'):
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.plot.ly https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "img-src 'self' data: https:; "
            "font-src 'self' data: https://cdnjs.cloudflare.com; "
        )

        return response

    @staticmethod
    def configure_cors(app, allowed_origins=None):
        """
        Configure CORS for Flask app.

        Args:
            app: Flask app instance
            allowed_origins: List of allowed origins (default: localhost only)
        """
        from flask_cors import CORS

        if allowed_origins is None:
            allowed_origins = [
                "http://localhost:8050",
                "http://127.0.0.1:8050",
            ]

        CORS(app, resources={
            r"/api/*": {
                "origins": allowed_origins,
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["Content-Type", "Authorization"],
                "max_age": 3600,
            }
        })

        logger.info(f"CORS configured for origins: {allowed_origins}")

    @staticmethod
    def sanitize_input(input_str: str, max_length=1000) -> str:
        """
        Sanitize user input to prevent injection attacks.

        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not input_str:
            return ""

        # Truncate to max length
        sanitized = input_str[:max_length]

        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '$', '`']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')

        return sanitized.strip()

    @staticmethod
    def validate_file_upload(filename: str, allowed_extensions=None) -> bool:
        """
        Validate uploaded file.

        Args:
            filename: Name of uploaded file
            allowed_extensions: Set of allowed file extensions

        Returns:
            True if file is safe to upload
        """
        if allowed_extensions is None:
            allowed_extensions = {'h5', 'hdf5', 'mat', 'csv', 'json'}

        if not filename:
            return False

        # Check extension
        if '.' not in filename:
            return False

        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in allowed_extensions:
            logger.warning(f"Rejected file upload: {filename} (invalid extension)")
            return False

        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            logger.warning(f"Rejected file upload: {filename} (path traversal attempt)")
            return False

        return True


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)
