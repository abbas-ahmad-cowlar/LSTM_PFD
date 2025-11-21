"""
Rate Limiting Middleware (Feature #1).
Implements sliding window rate limiting using Redis.
"""
import time
from functools import wraps
from flask import request, jsonify
from typing import Tuple

import redis

from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    API_KEY_RATE_LIMIT_WINDOW, API_KEY_EXPIRY_REDIS,
    RATE_LIMIT_FAIL_OPEN
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Initialize Redis connection
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_timeout=1,  # 1 second timeout
        socket_connect_timeout=1
    )
    # Test connection
    redis_client.ping()
    logger.info(f"Redis connection established: {REDIS_HOST}:{REDIS_PORT}")
except redis.RedisError as e:
    logger.error(f"Redis connection failed: {e}")
    if not RATE_LIMIT_FAIL_OPEN:
        raise
    redis_client = None


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.

    Algorithm:
        1. Round current timestamp to hour boundary (window_start)
        2. Create Redis key: rate_limit:{api_key_id}:{window_start}
        3. Increment counter atomically (INCR)
        4. Set expiry on first request (2 hours for cleanup)
        5. Check if count exceeds limit
        6. Return allowed/rejected with headers

    Features:
        - Atomic operations (thread-safe)
        - Sliding window (resets every hour)
        - Fail-open if Redis is down (configurable)
        - Rate limit headers in response
    """

    @staticmethod
    def check_rate_limit(
        api_key_id: int,
        rate_limit: int
    ) -> Tuple[bool, int, int, int]:
        """
        Check if request is within rate limit.

        Args:
            api_key_id: Database ID of the API key
            rate_limit: Requests per hour limit for this key

        Returns:
            Tuple of:
                - allowed (bool): True if request should be allowed
                - current_count (int): Current request count in window
                - limit (int): Rate limit value
                - reset_time (int): Unix timestamp when limit resets

        Example:
            >>> allowed, count, limit, reset = RateLimiter.check_rate_limit(
            ...     api_key_id=5,
            ...     rate_limit=1000
            ... )
            >>> if not allowed:
            ...     print(f"Rate limit exceeded: {count}/{limit}")
        """
        if redis_client is None:
            # Redis not available
            if RATE_LIMIT_FAIL_OPEN:
                logger.warning("Redis unavailable, failing open (allowing request)")
                current_time = int(time.time())
                return True, 0, rate_limit, current_time + API_KEY_RATE_LIMIT_WINDOW
            else:
                logger.error("Redis unavailable, failing closed (rejecting request)")
                current_time = int(time.time())
                return False, rate_limit + 1, rate_limit, current_time + API_KEY_RATE_LIMIT_WINDOW

        try:
            current_time = int(time.time())

            # Round to hour boundary (sliding window)
            window_start = current_time - (current_time % API_KEY_RATE_LIMIT_WINDOW)

            # Redis key format: rate_limit:key_{id}:{timestamp}
            redis_key = f"rate_limit:key_{api_key_id}:{window_start}"

            # Atomic increment
            count = redis_client.incr(redis_key)

            # Set expiry on first request in window
            if count == 1:
                redis_client.expire(redis_key, API_KEY_EXPIRY_REDIS)

            # Calculate reset time (next hour boundary)
            reset_time = window_start + API_KEY_RATE_LIMIT_WINDOW

            # Check limit
            allowed = count <= rate_limit

            if not allowed:
                logger.warning(
                    f"Rate limit exceeded for API key {api_key_id}: "
                    f"{count}/{rate_limit} in window starting at {window_start}"
                )

            return allowed, count, rate_limit, reset_time

        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}")

            # Fail open or closed based on configuration
            if RATE_LIMIT_FAIL_OPEN:
                logger.warning("Redis error, failing open (allowing request)")
                current_time = int(time.time())
                return True, 0, rate_limit, current_time + API_KEY_RATE_LIMIT_WINDOW
            else:
                logger.error("Redis error, failing closed (rejecting request)")
                current_time = int(time.time())
                return False, rate_limit + 1, rate_limit, current_time + API_KEY_RATE_LIMIT_WINDOW

    @staticmethod
    def rate_limit_decorator(f):
        """
        Decorator to apply rate limiting to Flask API endpoints.

        Usage:
            @app.route('/api/v1/predict')
            @AuthMiddleware.require_api_key  # Must be applied first
            @RateLimiter.rate_limit_decorator
            def predict():
                return {"result": "success"}

        Response Headers:
            X-RateLimit-Limit: Maximum requests per hour
            X-RateLimit-Remaining: Requests remaining in current window
            X-RateLimit-Reset: Unix timestamp when limit resets

        Error Response (429):
            {
                "error": "rate_limit_exceeded",
                "message": "Rate limit of 1000 requests per hour exceeded...",
                "current_usage": 1001,
                "limit": 1000,
                "reset_at": 1634567890
            }
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get API key from request (set by auth middleware)
            api_key_record = getattr(request, 'api_key', None)

            if not api_key_record:
                # No API key (should be caught by auth middleware)
                return jsonify({
                    'error': 'unauthorized',
                    'message': 'API key required. Apply @require_api_key decorator first.'
                }), 401

            # Check rate limit
            allowed, count, limit, reset_time = RateLimiter.check_rate_limit(
                api_key_id=api_key_record.id,
                rate_limit=api_key_record.rate_limit
            )

            # Prepare rate limit headers
            response_headers = {
                'X-RateLimit-Limit': str(limit),
                'X-RateLimit-Remaining': str(max(0, limit - count)),
                'X-RateLimit-Reset': str(reset_time)
            }

            if not allowed:
                # Rate limit exceeded - return 429
                reset_datetime = time.strftime(
                    '%Y-%m-%d %H:%M:%S UTC',
                    time.gmtime(reset_time)
                )

                return jsonify({
                    'error': 'rate_limit_exceeded',
                    'message': (
                        f'Rate limit of {limit} requests per hour exceeded. '
                        f'Limit resets at {reset_datetime}.'
                    ),
                    'current_usage': count,
                    'limit': limit,
                    'reset_at': reset_time
                }), 429, response_headers

            # Execute endpoint
            response = f(*args, **kwargs)

            # Add rate limit headers to successful response
            if isinstance(response, tuple):
                # Response is (data, status_code) or (data, status_code, headers)
                if len(response) == 3:
                    data, status_code, headers = response
                    headers.update(response_headers)
                    response = (data, status_code, headers)
                elif len(response) == 2:
                    data, status_code = response
                    response = (data, status_code, response_headers)
            else:
                # Response is just data
                response = (response, 200, response_headers)

            return response

        return decorated_function

    @staticmethod
    def get_current_usage(api_key_id: int) -> dict:
        """
        Get current usage statistics for an API key.

        Args:
            api_key_id: API key ID

        Returns:
            dict with:
                - current_count: Requests in current window
                - limit: Rate limit
                - remaining: Requests remaining
                - reset_time: When window resets
                - window_start: Start of current window

        Example:
            >>> stats = RateLimiter.get_current_usage(api_key_id=5)
            >>> print(f"Used: {stats['current_count']}/{stats['limit']}")
        """
        if redis_client is None:
            return {
                'current_count': 0,
                'limit': 0,
                'remaining': 0,
                'reset_time': 0,
                'window_start': 0,
                'error': 'Redis unavailable'
            }

        try:
            current_time = int(time.time())
            window_start = current_time - (current_time % API_KEY_RATE_LIMIT_WINDOW)
            redis_key = f"rate_limit:key_{api_key_id}:{window_start}"

            # Get current count (without incrementing)
            count = redis_client.get(redis_key)
            count = int(count) if count else 0

            # Note: We don't know the limit without querying the database
            # Caller should provide it or query separately
            reset_time = window_start + API_KEY_RATE_LIMIT_WINDOW

            return {
                'current_count': count,
                'reset_time': reset_time,
                'window_start': window_start
            }

        except redis.RedisError as e:
            logger.error(f"Redis error getting usage stats: {e}")
            return {
                'current_count': 0,
                'limit': 0,
                'remaining': 0,
                'reset_time': 0,
                'window_start': 0,
                'error': str(e)
            }

    @staticmethod
    def reset_limit(api_key_id: int) -> bool:
        """
        Reset rate limit for an API key (admin function).

        Args:
            api_key_id: API key ID

        Returns:
            True if reset successfully

        Example:
            >>> RateLimiter.reset_limit(api_key_id=5)
            True
        """
        if redis_client is None:
            logger.warning("Cannot reset limit: Redis unavailable")
            return False

        try:
            current_time = int(time.time())
            window_start = current_time - (current_time % API_KEY_RATE_LIMIT_WINDOW)
            redis_key = f"rate_limit:key_{api_key_id}:{window_start}"

            redis_client.delete(redis_key)
            logger.info(f"Reset rate limit for API key {api_key_id}")
            return True

        except redis.RedisError as e:
            logger.error(f"Redis error resetting limit: {e}")
            return False
