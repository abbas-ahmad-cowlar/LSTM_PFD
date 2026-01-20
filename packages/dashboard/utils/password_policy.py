"""
Password Policy Validation.

Reference: Master Roadmap Chapter 4.1.1

Provides password validation and hashing utilities using passlib/bcrypt.
Enforces strong password requirements:
- Minimum 12 characters (or 8 in development)
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character

Usage:
    from packages.dashboard.utils.password_policy import (
        validate_password, hash_password, verify_password
    )
    
    # Validate password strength
    is_valid, message = validate_password("MyP@ssw0rd123")
    
    # Hash for storage
    hashed = hash_password("MyP@ssw0rd123")
    
    # Verify on login
    if verify_password("MyP@ssw0rd123", hashed):
        # Password correct
"""

import re
import os
from typing import Tuple

# Try to import passlib, fall back to hashlib if not available
try:
    from passlib.context import CryptContext
    
    # bcrypt context with configurable rounds
    pwd_context = CryptContext(
        schemes=["bcrypt"],
        deprecated="auto",
        bcrypt__rounds=12  # Increased from default 10 for security
    )
    PASSLIB_AVAILABLE = True
except ImportError:
    import hashlib
    import secrets
    PASSLIB_AVAILABLE = False
    pwd_context = None


# Configuration from environment or defaults
PASSWORD_MIN_LENGTH = int(os.getenv('PASSWORD_MIN_LENGTH', '12'))
PASSWORD_MAX_LENGTH = int(os.getenv('PASSWORD_MAX_LENGTH', '128'))
REQUIRE_UPPERCASE = os.getenv('PASSWORD_REQUIRE_UPPERCASE', 'true').lower() == 'true'
REQUIRE_LOWERCASE = os.getenv('PASSWORD_REQUIRE_LOWERCASE', 'true').lower() == 'true'
REQUIRE_DIGIT = os.getenv('PASSWORD_REQUIRE_DIGIT', 'true').lower() == 'true'
REQUIRE_SPECIAL = os.getenv('PASSWORD_REQUIRE_SPECIAL', 'true').lower() == 'true'

# Special characters allowed in passwords
SPECIAL_CHARS = r'!@#$%^&*(),.?":{}|<>_\-=+\[\]\\;\'`~'


def validate_password(password: str, min_length: int = None) -> Tuple[bool, str]:
    """
    Validate password against security policy.
    
    Args:
        password: Password to validate
        min_length: Override minimum length (useful for testing)
        
    Returns:
        Tuple of (is_valid, message)
    """
    min_len = min_length or PASSWORD_MIN_LENGTH
    
    # Length check
    if len(password) < min_len:
        return False, f"Password must be at least {min_len} characters"
    
    if len(password) > PASSWORD_MAX_LENGTH:
        return False, f"Password must be at most {PASSWORD_MAX_LENGTH} characters"
    
    # Uppercase check
    if REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    # Lowercase check
    if REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    # Digit check
    if REQUIRE_DIGIT and not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    # Special character check
    if REQUIRE_SPECIAL and not re.search(f'[{re.escape(SPECIAL_CHARS)}]', password):
        return False, "Password must contain at least one special character (!@#$%^&* etc.)"
    
    # Common password check (basic)
    common_passwords = {
        'password', 'password123', '123456789', 'qwerty123',
        'letmein', 'welcome', 'admin123', 'root123'
    }
    if password.lower() in common_passwords:
        return False, "Password is too common. Please choose a stronger password."
    
    return True, "Password valid"


def hash_password(password: str) -> str:
    """
    Hash a password for secure storage.
    
    Uses bcrypt via passlib if available, otherwise falls back to
    PBKDF2-SHA256 (less secure but no external dependencies).
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    if PASSLIB_AVAILABLE and pwd_context:
        return pwd_context.hash(password)
    else:
        # Fallback to hashlib (not as secure as bcrypt)
        import hashlib
        import secrets
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        return f"pbkdf2:sha256:100000${salt}${hashed.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Password to check
        hashed_password: Stored hash
        
    Returns:
        True if password matches
    """
    if PASSLIB_AVAILABLE and pwd_context:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False
    else:
        # Fallback verification for pbkdf2 hashes
        try:
            parts = hashed_password.split('$')
            if len(parts) != 3 or not parts[0].startswith('pbkdf2'):
                return False
            
            salt = parts[1]
            stored_hash = parts[2]
            
            # Recreate hash
            import hashlib
            new_hash = hashlib.pbkdf2_hmac(
                'sha256',
                plain_password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            
            # Constant-time comparison
            import hmac
            return hmac.compare_digest(new_hash.hex(), stored_hash)
        except Exception:
            return False


def needs_rehash(hashed_password: str) -> bool:
    """
    Check if a password hash needs to be upgraded.
    
    This is useful when increasing bcrypt rounds or changing algorithms.
    
    Args:
        hashed_password: Stored hash
        
    Returns:
        True if password should be rehashed on next login
    """
    if PASSLIB_AVAILABLE and pwd_context:
        return pwd_context.needs_update(hashed_password)
    return False


def generate_temp_password(length: int = 16) -> str:
    """
    Generate a temporary password that meets policy requirements.
    
    Args:
        length: Password length (min 12)
        
    Returns:
        Random password string
    """
    import secrets
    import string
    
    length = max(length, 12)
    
    # Ensure all character types are included
    password = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
        secrets.choice('!@#$%^&*'),
    ]
    
    # Fill remaining length with random chars
    all_chars = string.ascii_letters + string.digits + '!@#$%^&*'
    password.extend(secrets.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle
    import random
    random.shuffle(password)
    
    return ''.join(password)


if __name__ == '__main__':
    # Test password validation
    print("Password Policy Tests:")
    
    test_passwords = [
        ("short", False),
        ("nouppercase123!", False),
        ("NOLOWERCASE123!", False),
        ("NoDigitsHere!!", False),
        ("NoSpecialChars123", False),
        ("ValidP@ssw0rd!", True),
        ("Another$ecure1", True),
    ]
    
    for pwd, expected in test_passwords:
        is_valid, msg = validate_password(pwd)
        status = "✓" if is_valid == expected else "✗"
        print(f"  {status} '{pwd}': {msg}")
    
    # Test hashing
    print("\nHash Test:")
    password = "TestP@ssw0rd123"
    hashed = hash_password(password)
    print(f"  Original: {password}")
    print(f"  Hashed: {hashed[:50]}...")
    print(f"  Verify correct: {verify_password(password, hashed)}")
    print(f"  Verify wrong: {verify_password('wrong', hashed)}")
    
    # Test temp password
    print(f"\nGenerated temp password: {generate_temp_password()}")
