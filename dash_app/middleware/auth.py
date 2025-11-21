"""
Authentication middleware (Phase 11D).
JWT-based authentication for production deployment.
"""
import jwt
import datetime
from functools import wraps
from flask import request, jsonify
import os

from utils.logger import setup_logger
from database.connection import get_db_session
from models.user import User

logger = setup_logger(__name__)

# Secret key for JWT (should be in environment variable in production)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production-please")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24


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
    def authenticate_user(username: str, password: str) -> tuple:
        """
        Authenticate user with username and password.

        Args:
            username: Username
            password: Password

        Returns:
            Tuple of (success: bool, token: str or None, error: str or None)
        """
        try:
            with get_db_session() as session:
                user = session.query(User).filter_by(username=username).first()

                if not user:
                    return False, None, "Invalid username or password"

                # Verify password (implement proper password hashing in production)
                # For now, simple comparison (INSECURE - replace with bcrypt/argon2)
                if user.password_hash != password:  # TODO: Use proper password verification
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
    def create_user(username: str, email: str, password: str, role: str = "user") -> tuple:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            password: Plain text password (will be hashed)
            role: User role (user/admin)

        Returns:
            Tuple of (success: bool, user_id: int or None, error: str or None)
        """
        try:
            with get_db_session() as session:
                # Check if user exists
                existing_user = session.query(User).filter_by(username=username).first()
                if existing_user:
                    return False, None, "Username already exists"

                # Create user (TODO: Implement proper password hashing)
                user = User(
                    username=username,
                    email=email,
                    password_hash=password,  # TODO: Hash password with bcrypt/argon2
                    role=role
                )
                session.add(user)
                session.flush()
                user_id = user.id

                logger.info(f"Created user: {username}")
                return True, user_id, None

        except Exception as e:
            logger.error(f"User creation error: {e}", exc_info=True)
            return False, None, str(e)


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.
    TODO: Implement in production.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    import bcrypt
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against hash.
    TODO: Implement in production.

    Args:
        password: Plain text password
        hashed: Hashed password

    Returns:
        True if password matches
    """
    import bcrypt
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
