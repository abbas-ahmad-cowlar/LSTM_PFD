"""
Authentication middleware (Phase 11D).
JWT-based authentication for production deployment.

Security Note:
- JWT_SECRET_KEY is validated at startup via config module
- Uses HS256 algorithm for token signing
- Tokens expire after 24 hours
"""
import jwt
import datetime
from functools import wraps
from flask import request, jsonify
import os
from typing import Optional, Tuple
import bcrypt

from utils.logger import setup_logger
from database.connection import get_db_session
from models.user import User

logger = setup_logger(__name__)

# Import config validator for JWT secret
try:
    from utils.config_validator import get_required_config
    SECRET_KEY = get_required_config("JWT_SECRET_KEY")
except ImportError:
    # Fallback for backwards compatibility
    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    if not SECRET_KEY:
        raise ValueError(
            "JWT_SECRET_KEY must be set in environment variables. "
            "Generate with: python -c 'import secrets; print(secrets.token_hex(32))'"
        )

ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

# Password hashing configuration
# Bcrypt rounds: higher = more secure but slower (10-14 recommended for 2024)
# Each increment doubles the computation time
BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))

# Password policy constants
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 72  # bcrypt's hard limit (silently truncates beyond this)


class AuthMiddleware:
    """Authentication middleware for securing endpoints."""

    @staticmethod
    def generate_token(user_id: int, username: str) -> str:
        """
        Generate JWT token for user.

        Args:
            user_id: User ID
            username: Username

        Returns:
            JWT token string
        """
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),
            "iat": datetime.datetime.utcnow()
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return token

    @staticmethod
    def verify_token(token: str) -> dict:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    @staticmethod
    def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Authenticate user with username and password using secure bcrypt verification.

        Args:
            username: Username
            password: Plain text password

        Returns:
            Tuple of (success: bool, token: str or None, error: str or None)

        Security notes:
            - Uses constant-time password comparison (bcrypt.checkpw)
            - Same error message for invalid username and password (prevents user enumeration)
            - Never reveals whether username exists
        """
        try:
            with get_db_session() as session:
                user = session.query(User).filter_by(username=username).first()

                if not user:
                    return False, None, "Invalid username or password"

                # Verify password using bcrypt
                if not verify_password(password, user.password_hash):
                    return False, None, "Invalid username or password"

                # Generate token
                token = AuthMiddleware.generate_token(user.id, user.username)
                logger.info(f"User {username} authenticated successfully")
                return True, token, None

        except Exception as e:
            logger.error(f"Authentication error: {e}", exc_info=True)
            return False, None, "Authentication service error"

    @staticmethod
    def require_auth(f):
        """
        Decorator to require authentication for Flask routes.

        Usage:
            @app.route('/protected')
            @require_auth
            def protected_route():
                return "Protected content"
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = None

            # Get token from Authorization header
            if 'Authorization' in request.headers:
                auth_header = request.headers['Authorization']
                try:
                    token = auth_header.split(" ")[1]  # Bearer <token>
                except IndexError:
                    return jsonify({"error": "Invalid Authorization header format"}), 401

            if not token:
                return jsonify({"error": "Authentication token is missing"}), 401

            # Verify token
            payload = AuthMiddleware.verify_token(token)
            if not payload:
                return jsonify({"error": "Invalid or expired token"}), 401

            # Add user info to request context
            request.current_user = payload

            return f(*args, **kwargs)

        return decorated_function

    @staticmethod
    def create_user(username: str, email: str, password: str, role: str = "user") -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Create a new user with secure password hashing.

        Args:
            username: Username
            email: Email address
            password: Plain text password (will be hashed with bcrypt)
            role: User role (user/admin)

        Returns:
            Tuple of (success: bool, user_id: int or None, error: str or None)

        Note:
            Password validation is performed by hash_password() which enforces:
            - Minimum 8 characters
            - Maximum 72 bytes (bcrypt limit)
            - Non-empty, non-None values
        """
        try:
            # Validate password before attempting database operations
            # This will raise ValueError if password doesn't meet requirements
            hashed_password = hash_password(password)

            with get_db_session() as session:
                # Check if user exists
                existing_user = session.query(User).filter_by(username=username).first()
                if existing_user:
                    return False, None, "Username already exists"

                # Create user with hashed password
                user = User(
                    username=username,
                    email=email,
                    password_hash=hashed_password,
                    role=role
                )
                session.add(user)
                session.flush()
                user_id = user.id

                logger.info(f"Created user: {username} with role: {role}")
                return True, user_id, None

        except ValueError as e:
            # Password validation errors (from hash_password)
            logger.warning(f"User creation failed - password validation: {e}")
            return False, None, str(e)

        except Exception as e:
            logger.error(f"User creation error: {e}", exc_info=True)
            return False, None, "User creation failed"


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt with comprehensive security checks.

    Args:
        password: Plain text password to hash

    Returns:
        Hashed password string

    Raises:
        ValueError: If password is None, empty, or exceeds bcrypt's 72-byte limit
        Exception: If bcrypt hashing fails

    Security notes:
        - Uses configurable bcrypt rounds (default: 12) via BCRYPT_ROUNDS env var
        - Enforces minimum password length (8 characters)
        - Warns if password exceeds 72 bytes (bcrypt's limit)
        - Each hash generates unique salt (same password = different hashes)
    """
    # Input validation
    if password is None:
        logger.error("Attempted to hash None password")
        raise ValueError("Password cannot be None")

    if not isinstance(password, str):
        logger.error(f"Attempted to hash non-string password: {type(password)}")
        raise ValueError("Password must be a string")

    if not password or not password.strip():
        logger.error("Attempted to hash empty password")
        raise ValueError("Password cannot be empty")

    if len(password) < MIN_PASSWORD_LENGTH:
        logger.warning(f"Password length {len(password)} below minimum {MIN_PASSWORD_LENGTH}")
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    # Check bcrypt's 72-byte limit
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > MAX_PASSWORD_LENGTH:
        logger.warning(
            f"Password exceeds {MAX_PASSWORD_LENGTH} bytes ({len(password_bytes)} bytes). "
            "bcrypt will truncate it, reducing security."
        )
        raise ValueError(
            f"Password too long ({len(password_bytes)} bytes). "
            f"Maximum is {MAX_PASSWORD_LENGTH} bytes due to bcrypt limitations."
        )

    try:
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(password_bytes, salt)
        hashed_str = hashed.decode('utf-8')

        logger.debug(f"Password hashed successfully (rounds={BCRYPT_ROUNDS})")
        return hashed_str

    except Exception as e:
        logger.error(f"Password hashing failed: {e}", exc_info=True)
        raise Exception(f"Password hashing failed: {str(e)}") from e


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against bcrypt hash with timing attack protection.

    Args:
        password: Plain text password to verify
        hashed: Bcrypt hash to check against

    Returns:
        bool: True if password matches hash, False otherwise

    Security notes:
        - Returns False on any error (prevents information leakage)
        - Uses constant-time comparison (bcrypt.checkpw is timing-safe)
        - Logs errors for debugging but doesn't expose them to caller
        - Never raises exceptions (graceful degradation)

    Note:
        This function never raises exceptions. It returns False for:
        - None or invalid inputs
        - Malformed hashes
        - Encryption errors
        This prevents attackers from distinguishing between different error conditions.
    """
    # Input validation - return False to prevent timing attacks
    if password is None or hashed is None:
        logger.warning("Password verification attempted with None value")
        return False

    if not isinstance(password, str) or not isinstance(hashed, str):
        logger.warning(
            f"Password verification attempted with invalid types: "
            f"password={type(password)}, hash={type(hashed)}"
        )
        return False

    if not password or not hashed:
        logger.warning("Password verification attempted with empty value")
        return False

    try:
        # Encode inputs
        password_bytes = password.encode('utf-8')
        hashed_bytes = hashed.encode('utf-8')

        # Perform constant-time comparison
        # bcrypt.checkpw is designed to resist timing attacks
        result = bcrypt.checkpw(password_bytes, hashed_bytes)

        if result:
            logger.debug("Password verification successful")
        else:
            logger.debug("Password verification failed: incorrect password")

        return result

    except ValueError as e:
        # Invalid salt or hash format
        logger.warning(f"Password verification failed: invalid hash format - {e}")
        return False

    except Exception as e:
        # Catch all other errors to prevent information leakage
        logger.error(f"Password verification error: {e}", exc_info=True)
        return False
