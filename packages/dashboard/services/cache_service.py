"""
Redis caching service.
"""
import redis
import json
from typing import Any, Optional
from config import REDIS_URL, CACHE_TTL_MEDIUM
from utils.logger import setup_logger
from utils.exceptions import CacheError
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)

# Initialize Redis client
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info(f"Connected to Redis at {REDIS_URL}")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}. Caching will be disabled.")
    redis_client = None


class CacheService:
    """Redis caching wrapper."""

    @staticmethod
    def get(key: str) -> Optional[Any]:
        """Get value from cache."""
        if redis_client is None:
            return None

        try:
            value = redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None

    @staticmethod
    def set(key: str, value: Any, ttl: int = CACHE_TTL_MEDIUM) -> bool:
        """Set value in cache with TTL."""
        if redis_client is None:
            return False

        try:
            redis_client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False

    @staticmethod
    def delete(key: str) -> bool:
        """Delete value from cache."""
        if redis_client is None:
            return False

        try:
            redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False

    @staticmethod
    def invalidate_pattern(pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        if redis_client is None:
            return 0

        try:
            keys = redis_client.keys(pattern)
            if keys:
                return redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate error for pattern '{pattern}': {e}")
            return 0

    @staticmethod
    def get_stats() -> dict:
        """Get cache statistics."""
        if redis_client is None:
            return {"status": "disconnected"}

        try:
            info = redis_client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                )
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
