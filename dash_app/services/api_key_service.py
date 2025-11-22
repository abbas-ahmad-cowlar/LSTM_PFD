"""
API Key Service (Feature #1).
Handles generation, verification, and management of API keys.
"""
import secrets
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

from models.api_key import APIKey, APIUsage
from models.user import User
from database.connection import get_db_session
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


class APIKeyService:
    """
    Service for managing API keys.

    Features:
        - Cryptographically secure key generation
        - bcrypt hashing (cost factor 12)
        - Fast verification using prefix index
        - Automatic expiration handling
        - Usage tracking
    """

    @staticmethod
    def generate_key(
        user_id: int,
        name: str,
        environment: str = 'live',
        rate_limit: int = 1000,
        expires_in_days: Optional[int] = None,
        scopes: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate a new API key for a user.

        Args:
            user_id: ID of the user
            name: Descriptive name (e.g., "CI/CD Pipeline")
            environment: 'live' or 'test'
            rate_limit: Requests per hour (default 1000)
            expires_in_days: Key expiry in days (None = never expires)
            scopes: List of permissions (default ['read', 'write'])

        Returns:
            dict with:
                - 'api_key': Plain text key (SHOW ONCE)
                - 'record': APIKey database object
                - 'created': Boolean success indicator

        Raises:
            ValueError: If user doesn't exist or name is empty

        Example:
            >>> result = APIKeyService.generate_key(
            ...     user_id=1,
            ...     name="Production API",
            ...     expires_in_days=365
            ... )
            >>> print(result['api_key'])
            'sk_live_a1b2c3d4e5f6...'
        """
        # Validate inputs
        if not name or len(name.strip()) == 0:
            raise ValueError("API key name cannot be empty")

        if environment not in ['live', 'test']:
            raise ValueError("Environment must be 'live' or 'test'")

        if rate_limit <= 0:
            raise ValueError("Rate limit must be positive")

        with get_db_session() as session:
            # Verify user exists
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                raise ValueError(f"User {user_id} not found")

            # Generate cryptographically secure random key
            # 32 bytes = 43 characters in base64url encoding
            random_bytes = secrets.token_urlsafe(32)
            api_key = f"sk_{environment}_{random_bytes}"

            # Hash key for storage (bcrypt cost factor 12)
            key_hash = bcrypt.hashpw(
                api_key.encode('utf-8'),
                bcrypt.gensalt(rounds=12)
            )

            # Extract prefix for display (first 20 chars)
            # Example: "sk_live_abc12345678"
            prefix = api_key[:20]

            # Calculate expiry
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

            # Set default scopes
            if scopes is None:
                scopes = ['read', 'write']

            # Create database record
            api_key_record = APIKey(
                user_id=user_id,
                key_hash=key_hash.decode('utf-8'),
                name=name.strip(),
                prefix=prefix,
                scopes=scopes,
                rate_limit=rate_limit,
                expires_at=expires_at
            )

            session.add(api_key_record)
            session.flush()  # Get ID before commit

            logger.info(
                f"Generated API key for user {user_id}: "
                f"id={api_key_record.id}, prefix={prefix}, "
                f"rate_limit={rate_limit}"
            )

            return {
                'api_key': api_key,  # Plain text, return to user ONCE
                'record': api_key_record,
                'created': True
            }

    @staticmethod
    def verify_key(api_key: str) -> Optional[APIKey]:
        """
        Verify API key and return corresponding database record.

        Uses prefix-based lookup for performance:
        1. Extract prefix (first 20 chars)
        2. Find candidates with matching prefix (indexed query)
        3. Check bcrypt hash for each candidate

        Args:
            api_key: Plain text API key from request header

        Returns:
            APIKey object if valid and active, None if invalid

        Example:
            >>> key_record = APIKeyService.verify_key('sk_live_abc...')
            >>> if key_record:
            ...     print(f"Authenticated as user {key_record.user_id}")
        """
        # Validate format
        if not api_key or len(api_key) < 20:
            logger.warning("Invalid API key format (too short)")
            return None

        if not api_key.startswith('sk_'):
            logger.warning("Invalid API key format (missing sk_ prefix)")
            return None

        # Extract prefix for fast lookup
        prefix = api_key[:20]

        with get_db_session() as session:
            # Find keys with matching prefix (fast indexed query)
            candidates = session.query(APIKey).filter(
                APIKey.prefix == prefix,
                APIKey.is_active == True
            ).all()

            # Check hash for each candidate
            for candidate in candidates:
                try:
                    if bcrypt.checkpw(
                        api_key.encode('utf-8'),
                        candidate.key_hash.encode('utf-8')
                    ):
                        # Check expiration
                        if candidate.is_expired():
                            logger.warning(
                                f"API key {candidate.id} is expired "
                                f"(expired at {candidate.expires_at})"
                            )
                            return None

                        # Update last used timestamp (async to avoid blocking)
                        candidate.last_used_at = datetime.utcnow()
                        session.flush()

                        logger.info(
                            f"API key verified: id={candidate.id}, "
                            f"user_id={candidate.user_id}"
                        )
                        return candidate

                except Exception as e:
                    logger.error(f"Error verifying API key hash: {e}")
                    continue

            logger.warning(f"No valid API key found for prefix {prefix}")
            return None

    @staticmethod
    def revoke_key(api_key_id: int, user_id: int) -> bool:
        """
        Revoke (deactivate) an API key.

        Args:
            api_key_id: ID of the key to revoke
            user_id: ID of the user (authorization check)

        Returns:
            True if revoked, False if not found or unauthorized

        Example:
            >>> success = APIKeyService.revoke_key(key_id=5, user_id=1)
            >>> if success:
            ...     print("Key revoked successfully")
        """
        with get_db_session() as session:
            key = session.query(APIKey).filter(
                APIKey.id == api_key_id,
                APIKey.user_id == user_id
            ).first()

            if not key:
                logger.warning(
                    f"Cannot revoke key {api_key_id}: "
                    f"not found or unauthorized for user {user_id}"
                )
                return False

            key.is_active = False
            session.flush()

            logger.info(
                f"Revoked API key: id={api_key_id}, "
                f"user_id={user_id}, prefix={key.prefix}"
            )
            return True

    @staticmethod
    def list_user_keys(user_id: int, include_inactive: bool = False) -> List[APIKey]:
        """
        Get all API keys for a user.

        Args:
            user_id: User ID
            include_inactive: If True, include revoked keys

        Returns:
            List of APIKey objects

        Example:
            >>> keys = APIKeyService.list_user_keys(user_id=1)
            >>> for key in keys:
            ...     print(f"{key.name}: {key.prefix}...")
        """
        with get_db_session() as session:
            query = session.query(APIKey).filter(APIKey.user_id == user_id)

            if not include_inactive:
                query = query.filter(APIKey.is_active == True)

            keys = query.order_by(APIKey.created_at.desc()).all()

            logger.info(f"Listed {len(keys)} API keys for user {user_id}")
            return keys

    @staticmethod
    def get_key_usage_stats(
        api_key_id: int,
        hours: int = 24
    ) -> Dict:
        """
        Get usage statistics for an API key.

        Args:
            api_key_id: API key ID
            hours: Hours to look back (default 24)

        Returns:
            dict with:
                - total_requests: Total request count
                - success_rate: Percentage of successful requests
                - avg_response_time_ms: Average response time
                - requests_by_endpoint: Dict of endpoint -> count

        Example:
            >>> stats = APIKeyService.get_key_usage_stats(key_id=5)
            >>> print(f"Total requests: {stats['total_requests']}")
        """
        with get_db_session() as session:
            # Get usage records from last N hours
            since = datetime.utcnow() - timedelta(hours=hours)

            records = session.query(APIUsage).filter(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= since
            ).all()

            if not records:
                return {
                    'total_requests': 0,
                    'success_rate': 0,
                    'avg_response_time_ms': 0,
                    'requests_by_endpoint': {}
                }

            # Calculate statistics
            total_requests = len(records)
            successful_requests = sum(
                1 for r in records if 200 <= r.status_code < 400
            )
            success_rate = (successful_requests / total_requests) * 100

            # Average response time
            response_times = [
                r.response_time_ms for r in records
                if r.response_time_ms is not None
            ]
            avg_response_time = (
                sum(response_times) / len(response_times)
                if response_times else 0
            )

            # Requests by endpoint
            endpoint_counts = {}
            for record in records:
                endpoint = record.endpoint
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1

            return {
                'total_requests': total_requests,
                'success_rate': round(success_rate, 2),
                'avg_response_time_ms': round(avg_response_time, 2),
                'requests_by_endpoint': endpoint_counts
            }

    @staticmethod
    def log_usage(
        api_key_id: int,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: Optional[int] = None
    ) -> bool:
        """
        Log API usage for analytics.

        This should be called asynchronously to avoid blocking requests.

        Args:
            api_key_id: API key ID
            endpoint: API endpoint path
            method: HTTP method
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds

        Returns:
            True if logged successfully

        Example:
            >>> APIKeyService.log_usage(
            ...     api_key_id=5,
            ...     endpoint='/api/v1/predict',
            ...     method='POST',
            ...     status_code=200,
            ...     response_time_ms=45
            ... )
        """
        try:
            with get_db_session() as session:
                usage = APIUsage(
                    api_key_id=api_key_id,
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    response_time_ms=response_time_ms
                )
                session.add(usage)
                session.flush()
                return True

        except Exception as e:
            logger.error(f"Failed to log API usage: {e}")
            return False
