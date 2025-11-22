"""
Input validation utilities for security features (Phase 6, Feature 3).

This module provides validation functions for:
- TOTP codes
- IP addresses
- User agents
- Session tokens
- Backup codes
- Passwords
"""
import re
import ipaddress
from typing import Tuple, Optional
from config.security import (
    TOTP_CODE_LENGTH,
    BACKUP_CODE_LENGTH,
    PASSWORD_MIN_LENGTH,
    PASSWORD_MAX_LENGTH,
    PASSWORD_REQUIRE_UPPERCASE,
    PASSWORD_REQUIRE_LOWERCASE,
    PASSWORD_REQUIRE_NUMBER,
    PASSWORD_REQUIRE_SPECIAL,
    PASSWORD_SPECIAL_CHARS,
)


def validate_totp_code(code: Optional[str]) -> Tuple[bool, str]:
    """
    Validate TOTP code format.

    Args:
        code: The TOTP code to validate

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_totp_code("123456")
        (True, "")
        >>> validate_totp_code("12345")
        (False, "Code must be 6 digits")
        >>> validate_totp_code("12345a")
        (False, "Code must contain only digits")
    """
    if not code:
        return False, "TOTP code is required"

    # Remove whitespace
    code = code.strip()

    # Check length
    if len(code) != TOTP_CODE_LENGTH:
        return False, f"Code must be {TOTP_CODE_LENGTH} digits"

    # Check if all digits
    if not code.isdigit():
        return False, "Code must contain only digits"

    return True, ""


def validate_backup_code(code: Optional[str]) -> Tuple[bool, str]:
    """
    Validate backup code format.

    Args:
        code: The backup code to validate

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_backup_code("ABCD-EFGH-IJKL")
        (True, "")
        >>> validate_backup_code("123")
        (False, "Backup code must be 12 characters")
    """
    if not code:
        return False, "Backup code is required"

    # Remove hyphens and whitespace for validation
    clean_code = code.replace("-", "").replace(" ", "").upper()

    # Check length
    if len(clean_code) != BACKUP_CODE_LENGTH:
        return False, f"Backup code must be {BACKUP_CODE_LENGTH} characters"

    # Check if alphanumeric
    if not clean_code.isalnum():
        return False, "Backup code must contain only letters and numbers"

    return True, ""


def validate_ip_address(ip: Optional[str]) -> Tuple[bool, str]:
    """
    Validate IP address format (IPv4 or IPv6).

    Args:
        ip: The IP address to validate

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_ip_address("192.168.1.1")
        (True, "")
        >>> validate_ip_address("2001:0db8:85a3::8a2e:0370:7334")
        (True, "")
        >>> validate_ip_address("999.999.999.999")
        (False, "Invalid IP address format")
    """
    if not ip:
        return False, "IP address is required"

    try:
        # This validates both IPv4 and IPv6
        ipaddress.ip_address(ip.strip())
        return True, ""
    except ValueError:
        return False, "Invalid IP address format"


def sanitize_user_agent(user_agent: Optional[str], max_length: int = 500) -> str:
    """
    Sanitize and truncate user agent string.

    Args:
        user_agent: The user agent string to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized user agent string

    Examples:
        >>> sanitize_user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        >>> sanitize_user_agent("A" * 1000, max_length=10)
        "AAAAAAAAAA"
    """
    if not user_agent:
        return "Unknown"

    # Strip whitespace
    user_agent = user_agent.strip()

    # Remove control characters
    user_agent = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', user_agent)

    # Truncate to max length
    if len(user_agent) > max_length:
        user_agent = user_agent[:max_length]

    return user_agent if user_agent else "Unknown"


def validate_password(password: Optional[str]) -> Tuple[bool, list[str]]:
    """
    Validate password against security policy.

    Args:
        password: The password to validate

    Returns:
        Tuple of (is_valid, list_of_errors)

    Examples:
        >>> validate_password("StrongPass123!")
        (True, [])
        >>> validate_password("weak")
        (False, ["Password must be at least 8 characters", ...])
    """
    errors = []

    if not password:
        return False, ["Password is required"]

    # Length check
    if len(password) < PASSWORD_MIN_LENGTH:
        errors.append(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")

    if len(password) > PASSWORD_MAX_LENGTH:
        errors.append(f"Password must not exceed {PASSWORD_MAX_LENGTH} characters")

    # Uppercase check
    if PASSWORD_REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")

    # Lowercase check
    if PASSWORD_REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")

    # Number check
    if PASSWORD_REQUIRE_NUMBER and not re.search(r'\d', password):
        errors.append("Password must contain at least one number")

    # Special character check
    if PASSWORD_REQUIRE_SPECIAL:
        special_pattern = f"[{re.escape(PASSWORD_SPECIAL_CHARS)}]"
        if not re.search(special_pattern, password):
            errors.append(f"Password must contain at least one special character: {PASSWORD_SPECIAL_CHARS}")

    return (len(errors) == 0, errors)


def validate_session_token(token: Optional[str]) -> Tuple[bool, str]:
    """
    Validate session token format.

    Args:
        token: The session token to validate

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_session_token("abc123XYZ_-")
        (True, "")
        >>> validate_session_token("")
        (False, "Session token is required")
    """
    if not token:
        return False, "Session token is required"

    # Check if URL-safe (alphanumeric, -, _)
    if not re.match(r'^[A-Za-z0-9_-]+$', token):
        return False, "Session token contains invalid characters"

    # Check minimum length (should be cryptographically secure)
    if len(token) < 16:
        return False, "Session token too short"

    return True, ""


def sanitize_location(location: Optional[str], max_length: int = 200) -> str:
    """
    Sanitize location string.

    Args:
        location: The location string to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized location string

    Examples:
        >>> sanitize_location("New York, US")
        "New York, US"
        >>> sanitize_location("<script>alert('xss')</script>")
        "scriptalert('xss')/script"
    """
    if not location:
        return "Unknown"

    # Strip whitespace
    location = location.strip()

    # Remove HTML tags
    location = re.sub(r'<[^>]+>', '', location)

    # Remove control characters
    location = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', location)

    # Truncate to max length
    if len(location) > max_length:
        location = location[:max_length]

    return location if location else "Unknown"


def validate_email(email: Optional[str]) -> Tuple[bool, str]:
    """
    Validate email address format.

    Args:
        email: The email address to validate

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_email("user@example.com")
        (True, "")
        >>> validate_email("invalid-email")
        (False, "Invalid email format")
    """
    if not email:
        return False, "Email is required"

    # Basic email regex (RFC 5322 simplified)
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(email_pattern, email.strip()):
        return False, "Invalid email format"

    # Check length
    if len(email) > 255:
        return False, "Email address too long"

    return True, ""


def is_strong_password(password: str) -> bool:
    """
    Quick check if password meets all requirements.

    Args:
        password: The password to check

    Returns:
        True if password is strong, False otherwise
    """
    is_valid, _ = validate_password(password)
    return is_valid
