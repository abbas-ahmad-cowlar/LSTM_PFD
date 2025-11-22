"""
Authentication Service (Phase 6, Feature 3).

Centralized service for handling all authentication and security operations:
- TOTP/2FA management
- Backup code generation and verification
- Session tracking and management
- Login history recording
- Rate limiting for brute force protection

This service follows the separation of concerns principle and provides
a clean API for security operations.
"""
import secrets
import bcrypt
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

from database.connection import get_db_session
from models.user import User
from models.session_log import SessionLog
from models.login_history import LoginHistory
from models.backup_code import BackupCode
from utils.logger import setup_logger
from utils.validators import validate_totp_code, validate_backup_code, validate_ip_address
from config.security import (
    TOTP_WINDOW,
    TOTP_ISSUER_NAME,
    BACKUP_CODES_COUNT,
    BACKUP_CODE_LENGTH,
    BACKUP_CODE_CHARSET,
    BACKUP_CODE_BCRYPT_ROUNDS,
    SESSION_TOKEN_LENGTH,
    MAX_2FA_ATTEMPTS,
    LOCKOUT_DURATION_MINUTES,
)

logger = setup_logger(__name__)


# In-memory storage for rate limiting (use Redis in production)
_2fa_attempts: Dict[int, List[datetime]] = defaultdict(list)
_2fa_lockouts: Dict[int, datetime] = {}


class AuthenticationService:
    """
    Service for managing authentication and security operations.

    This service provides a clean, testable interface for all security-related
    operations including 2FA, sessions, and login tracking.
    """

    # ==========================================================================
    # TOTP/2FA Management
    # ==========================================================================

    @staticmethod
    def generate_totp_secret() -> str:
        """
        Generate a cryptographically secure TOTP secret.

        Returns:
            Base32-encoded secret string (32 characters)

        Example:
            >>> secret = AuthenticationService.generate_totp_secret()
            >>> len(secret)
            32
            >>> secret.isalnum() and secret.isupper()
            True
        """
        return pyotp.random_base32()

    @staticmethod
    def setup_2fa(user_id: int) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Set up 2FA for a user by generating TOTP secret and QR code.

        Args:
            user_id: ID of the user

        Returns:
            Tuple of (success, qr_code_base64, secret_formatted, error_message)

        Example:
            >>> success, qr, secret, error = AuthenticationService.setup_2fa(1)
            >>> if success:
            ...     print(f"QR code: {qr[:20]}...")
            ...     print(f"Secret: {secret}")
        """
        try:
            with get_db_session() as session:
                user = session.query(User).filter_by(id=user_id).first()

                if not user:
                    return False, None, None, "User not found"

                # Generate TOTP secret if not exists
                if not user.totp_secret:
                    secret = AuthenticationService.generate_totp_secret()
                    user.totp_secret = secret
                    session.commit()
                    logger.info(f"Generated new TOTP secret for user {user_id}")
                else:
                    secret = user.totp_secret

                # Generate TOTP URI for QR code
                totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                    name=user.email,
                    issuer_name=TOTP_ISSUER_NAME
                )

                # Generate QR code
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4
                )
                qr.add_data(totp_uri)
                qr.make(fit=True)

                img = qr.make_image(fill_color="black", back_color="white")

                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()

                # Format secret key for display (groups of 4 characters)
                formatted_secret = ' '.join([secret[i:i+4] for i in range(0, len(secret), 4)])

                return True, img_str, formatted_secret, None

        except Exception as e:
            logger.error(f"Error setting up 2FA for user {user_id}: {e}", exc_info=True)
            return False, None, None, str(e)

    @staticmethod
    def verify_totp(user_id: int, code: str, ip_address: Optional[str] = None) -> Tuple[bool, str]:
        """
        Verify TOTP code with rate limiting protection.

        Args:
            user_id: ID of the user
            code: 6-digit TOTP code to verify
            ip_address: IP address of the request (for logging)

        Returns:
            Tuple of (success, error_message)

        Example:
            >>> success, error = AuthenticationService.verify_totp(1, "123456")
            >>> if not success:
            ...     print(f"Error: {error}")
        """
        # Validate input
        is_valid, error_msg = validate_totp_code(code)
        if not is_valid:
            return False, error_msg

        # Check rate limiting
        is_locked, lockout_msg = AuthenticationService._check_2fa_rate_limit(user_id)
        if is_locked:
            return False, lockout_msg

        try:
            with get_db_session() as session:
                user = session.query(User).filter_by(id=user_id).first()

                if not user:
                    return False, "User not found"

                if not user.totp_secret:
                    return False, "2FA not set up"

                # Verify TOTP code
                totp = pyotp.TOTP(user.totp_secret)
                if totp.verify(code, valid_window=TOTP_WINDOW):
                    # Success - clear rate limit attempts
                    AuthenticationService._clear_2fa_attempts(user_id)

                    # Enable 2FA if not already enabled
                    if not user.totp_enabled:
                        user.totp_enabled = True
                        session.commit()
                        logger.info(f"2FA enabled for user {user_id}")

                    return True, ""
                else:
                    # Failed verification - record attempt
                    AuthenticationService._record_2fa_attempt(user_id)
                    logger.warning(f"Invalid 2FA code for user {user_id} from IP {ip_address}")
                    return False, "Invalid code. Please check your authenticator app and try again."

        except Exception as e:
            logger.error(f"Error verifying 2FA code for user {user_id}: {e}", exc_info=True)
            return False, "An error occurred during verification"

    @staticmethod
    def _check_2fa_rate_limit(user_id: int) -> Tuple[bool, str]:
        """
        Check if user is locked out due to too many failed 2FA attempts.

        Args:
            user_id: ID of the user

        Returns:
            Tuple of (is_locked, error_message)
        """
        # Check if currently locked out
        if user_id in _2fa_lockouts:
            lockout_until = _2fa_lockouts[user_id]
            if datetime.utcnow() < lockout_until:
                remaining = int((lockout_until - datetime.utcnow()).total_seconds() / 60)
                return True, f"Too many failed attempts. Try again in {remaining} minutes."
            else:
                # Lockout expired
                del _2fa_lockouts[user_id]
                AuthenticationService._clear_2fa_attempts(user_id)

        # Check if too many recent attempts
        if user_id in _2fa_attempts:
            recent_attempts = [
                attempt for attempt in _2fa_attempts[user_id]
                if datetime.utcnow() - attempt < timedelta(minutes=LOCKOUT_DURATION_MINUTES)
            ]

            if len(recent_attempts) >= MAX_2FA_ATTEMPTS:
                # Lock out the user
                lockout_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
                _2fa_lockouts[user_id] = lockout_until
                logger.warning(f"User {user_id} locked out due to {MAX_2FA_ATTEMPTS} failed 2FA attempts")
                return True, f"Too many failed attempts. Try again in {LOCKOUT_DURATION_MINUTES} minutes."

        return False, ""

    @staticmethod
    def _record_2fa_attempt(user_id: int):
        """Record a failed 2FA attempt."""
        _2fa_attempts[user_id].append(datetime.utcnow())

    @staticmethod
    def _clear_2fa_attempts(user_id: int):
        """Clear all 2FA attempts for a user."""
        if user_id in _2fa_attempts:
            del _2fa_attempts[user_id]
        if user_id in _2fa_lockouts:
            del _2fa_lockouts[user_id]

    # ==========================================================================
    # Backup Codes Management
    # ==========================================================================

    @staticmethod
    def generate_backup_codes(user_id: int) -> Tuple[bool, Optional[List[str]], Optional[str]]:
        """
        Generate new backup codes for a user.

        This will invalidate all previous backup codes.

        Args:
            user_id: ID of the user

        Returns:
            Tuple of (success, list_of_codes, error_message)

        Example:
            >>> success, codes, error = AuthenticationService.generate_backup_codes(1)
            >>> if success:
            ...     print(f"Generated {len(codes)} backup codes")
            ...     for code in codes:
            ...         print(code)
        """
        try:
            with get_db_session() as session:
                user = session.query(User).filter_by(id=user_id).first()

                if not user:
                    return False, None, "User not found"

                # Delete all existing backup codes for this user
                session.query(BackupCode).filter_by(user_id=user_id).delete()

                # Generate new codes
                codes = []
                for _ in range(BACKUP_CODES_COUNT):
                    # Generate random code
                    code = ''.join(
                        secrets.choice(BACKUP_CODE_CHARSET)
                        for _ in range(BACKUP_CODE_LENGTH)
                    )

                    # Format with hyphens for readability (XXXX-XXXX-XXXX)
                    formatted_code = '-'.join([
                        code[i:i+4] for i in range(0, len(code), 4)
                    ])
                    codes.append(formatted_code)

                    # Hash and store
                    code_hash = bcrypt.hashpw(
                        code.encode('utf-8'),
                        bcrypt.gensalt(rounds=BACKUP_CODE_BCRYPT_ROUNDS)
                    ).decode('utf-8')

                    backup_code = BackupCode(
                        user_id=user_id,
                        code_hash=code_hash,
                        is_used=False
                    )
                    session.add(backup_code)

                session.commit()
                logger.info(f"Generated {BACKUP_CODES_COUNT} backup codes for user {user_id}")

                return True, codes, None

        except Exception as e:
            logger.error(f"Error generating backup codes for user {user_id}: {e}", exc_info=True)
            return False, None, str(e)

    @staticmethod
    def verify_backup_code(user_id: int, code: str, ip_address: Optional[str] = None) -> Tuple[bool, str]:
        """
        Verify and consume a backup code.

        Args:
            user_id: ID of the user
            code: Backup code to verify
            ip_address: IP address (for audit trail)

        Returns:
            Tuple of (success, error_message)

        Example:
            >>> success, error = AuthenticationService.verify_backup_code(1, "ABCD-EFGH-IJKL")
            >>> if success:
            ...     print("Code verified and consumed")
        """
        # Validate input
        is_valid, error_msg = validate_backup_code(code)
        if not is_valid:
            return False, error_msg

        # Clean code (remove hyphens and whitespace)
        clean_code = code.replace("-", "").replace(" ", "").upper()

        try:
            with get_db_session() as session:
                # Get all unused backup codes for this user
                backup_codes = session.query(BackupCode).filter_by(
                    user_id=user_id,
                    is_used=False
                ).all()

                if not backup_codes:
                    logger.warning(f"No unused backup codes for user {user_id}")
                    return False, "No valid backup codes found"

                # Try to match against each code
                for backup_code in backup_codes:
                    if bcrypt.checkpw(clean_code.encode('utf-8'), backup_code.code_hash.encode('utf-8')):
                        # Code matches - mark as used
                        backup_code.mark_as_used(ip_address)
                        session.commit()
                        logger.info(f"Backup code used for user {user_id} from IP {ip_address}")
                        return True, ""

                # No match found
                logger.warning(f"Invalid backup code attempt for user {user_id} from IP {ip_address}")
                return False, "Invalid backup code"

        except Exception as e:
            logger.error(f"Error verifying backup code for user {user_id}: {e}", exc_info=True)
            return False, "An error occurred during verification"

    # ==========================================================================
    # Session Management
    # ==========================================================================

    @staticmethod
    def create_session(
        user_id: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_type: Optional[str] = None,
        browser: Optional[str] = None,
        location: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Create a new session for a user.

        Args:
            user_id: ID of the user
            ip_address: IP address of the session
            user_agent: Browser user agent string
            device_type: Type of device (desktop, mobile, tablet)
            browser: Browser name and version
            location: Geographic location (City, Country)

        Returns:
            Tuple of (success, session_token, error_message)

        Example:
            >>> success, token, error = AuthenticationService.create_session(
            ...     user_id=1,
            ...     ip_address="192.168.1.1",
            ...     device_type="desktop",
            ...     browser="Chrome 120"
            ... )
        """
        try:
            # Generate cryptographically secure session token
            session_token = secrets.token_urlsafe(SESSION_TOKEN_LENGTH)

            with get_db_session() as session:
                session_log = SessionLog(
                    user_id=user_id,
                    session_token=session_token,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    device_type=device_type,
                    browser=browser,
                    location=location,
                    is_active=True
                )
                session.add(session_log)
                session.commit()

                logger.info(f"Created session for user {user_id} from IP {ip_address}")
                return True, session_token, None

        except Exception as e:
            logger.error(f"Error creating session for user {user_id}: {e}", exc_info=True)
            return False, None, str(e)

    @staticmethod
    def revoke_session(session_id: int, user_id: int) -> Tuple[bool, str]:
        """
        Revoke (terminate) a session.

        Args:
            session_id: ID of the session to revoke
            user_id: ID of the user (for authorization check)

        Returns:
            Tuple of (success, error_message)

        Example:
            >>> success, error = AuthenticationService.revoke_session(123, 1)
            >>> if success:
            ...     print("Session revoked")
        """
        try:
            with get_db_session() as session:
                session_log = session.query(SessionLog).filter_by(
                    id=session_id,
                    user_id=user_id  # Ensure user can only revoke their own sessions
                ).first()

                if not session_log:
                    return False, "Session not found"

                if not session_log.is_active:
                    return False, "Session already inactive"

                # Mark as inactive
                session_log.is_active = False
                session_log.logged_out_at = datetime.utcnow()
                session.commit()

                logger.info(f"Revoked session {session_id} for user {user_id}")
                return True, ""

        except Exception as e:
            logger.error(f"Error revoking session {session_id}: {e}", exc_info=True)
            return False, str(e)

    # ==========================================================================
    # Login History
    # ==========================================================================

    @staticmethod
    def record_login_attempt(
        user_id: int,
        success: bool,
        login_method: str = "password",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        location: Optional[str] = None,
        failure_reason: Optional[str] = None
    ) -> bool:
        """
        Record a login attempt in the history.

        Args:
            user_id: ID of the user
            success: Whether login was successful
            login_method: Method used (password, 2fa, oauth, api_key)
            ip_address: IP address of the attempt
            user_agent: Browser user agent string
            location: Geographic location
            failure_reason: Reason for failure (if unsuccessful)

        Returns:
            True if recorded successfully, False otherwise

        Example:
            >>> AuthenticationService.record_login_attempt(
            ...     user_id=1,
            ...     success=True,
            ...     login_method="2fa",
            ...     ip_address="192.168.1.1"
            ... )
            True
        """
        try:
            with get_db_session() as session:
                login_history = LoginHistory(
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    location=location,
                    login_method=login_method,
                    success=success,
                    failure_reason=failure_reason,
                    timestamp=datetime.utcnow()
                )
                session.add(login_history)
                session.commit()

                status = "successful" if success else "failed"
                logger.info(f"Recorded {status} {login_method} login for user {user_id} from IP {ip_address}")
                return True

        except Exception as e:
            logger.error(f"Error recording login attempt for user {user_id}: {e}", exc_info=True)
            return False
