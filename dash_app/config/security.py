"""
Security configuration constants (Phase 6, Feature 3).

This module contains all security-related configuration values for:
- TOTP/2FA settings
- Session management
- Rate limiting
- Backup codes
- Password policies
"""

# ==============================================================================
# TOTP/2FA Configuration
# ==============================================================================

# TOTP validity window (number of 30-second intervals to accept)
# Window of 1 = accept codes from current + previous/next interval (90s total)
TOTP_WINDOW = 1

# Length of TOTP codes
TOTP_CODE_LENGTH = 6

# TOTP issuer name (shown in authenticator apps)
TOTP_ISSUER_NAME = "LSTM Dashboard"

# TOTP secret length in characters (Base32 encoded)
# 32 chars = 160 bits of entropy
TOTP_SECRET_LENGTH = 32

# ==============================================================================
# Rate Limiting Configuration
# ==============================================================================

# Maximum failed 2FA verification attempts before lockout
MAX_2FA_ATTEMPTS = 5

# Lockout duration in minutes after max attempts reached
LOCKOUT_DURATION_MINUTES = 15

# Maximum login attempts before account lockout
MAX_LOGIN_ATTEMPTS = 10

# Login lockout duration in minutes
LOGIN_LOCKOUT_MINUTES = 30

# ==============================================================================
# Backup Codes Configuration
# ==============================================================================

# Number of backup codes to generate per user
BACKUP_CODES_COUNT = 10

# Length of each backup code (characters)
BACKUP_CODE_LENGTH = 12

# Characters allowed in backup codes (excludes ambiguous: 0, O, I, l, 1)
BACKUP_CODE_CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

# bcrypt cost factor for backup code hashing
BACKUP_CODE_BCRYPT_ROUNDS = 12

# ==============================================================================
# Session Management Configuration
# ==============================================================================

# Session token length (URL-safe base64 characters)
SESSION_TOKEN_LENGTH = 32

# Session inactivity timeout in hours
SESSION_TIMEOUT_HOURS = 24

# Maximum concurrent sessions per user (0 = unlimited)
MAX_CONCURRENT_SESSIONS = 5

# Remember me duration in days
REMEMBER_ME_DAYS = 30

# ==============================================================================
# Password Policy Configuration
# ==============================================================================

# Minimum password length
PASSWORD_MIN_LENGTH = 8

# Maximum password length
PASSWORD_MAX_LENGTH = 128

# Require uppercase letter
PASSWORD_REQUIRE_UPPERCASE = True

# Require lowercase letter
PASSWORD_REQUIRE_LOWERCASE = True

# Require number
PASSWORD_REQUIRE_NUMBER = True

# Require special character
PASSWORD_REQUIRE_SPECIAL = True

# Special characters allowed
PASSWORD_SPECIAL_CHARS = r'!@#$%^&*(),.?":{}|<>'

# bcrypt cost factor for password hashing
PASSWORD_BCRYPT_ROUNDS = 12

# ==============================================================================
# Login History Configuration
# ==============================================================================

# Number of login history records to display
LOGIN_HISTORY_DISPLAY_LIMIT = 50

# Days to retain login history (0 = forever)
LOGIN_HISTORY_RETENTION_DAYS = 365

# ==============================================================================
# Security Headers
# ==============================================================================

# Content Security Policy
CSP_POLICY = {
    'default-src': ["'self'"],
    'script-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
    'style-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
    'img-src': ["'self'", "data:", "https:"],
    'font-src': ["'self'", "cdn.jsdelivr.net"],
}

# ==============================================================================
# IP Address Configuration
# ==============================================================================

# Trust X-Forwarded-For header (enable if behind proxy/load balancer)
TRUST_PROXY_HEADERS = False

# Number of proxies to trust (if TRUST_PROXY_HEADERS=True)
NUM_PROXIES = 1

# ==============================================================================
# Geolocation Configuration
# ==============================================================================

# Enable IP geolocation lookup
ENABLE_GEOLOCATION = False

# Geolocation API endpoint (placeholder - requires service)
GEOLOCATION_API_URL = "https://ipapi.co/{ip}/json/"

# Geolocation API timeout in seconds
GEOLOCATION_TIMEOUT = 2

# ==============================================================================
# Security Event Logging
# ==============================================================================

# Log all 2FA setup attempts
LOG_2FA_SETUP = True

# Log all 2FA verification attempts (success and failure)
LOG_2FA_VERIFICATION = True

# Log all session creation
LOG_SESSION_CREATION = True

# Log all login attempts
LOG_LOGIN_ATTEMPTS = True

# Log suspicious activity (multiple failures, unusual locations, etc.)
LOG_SUSPICIOUS_ACTIVITY = True

# ==============================================================================
# Development/Testing Overrides
# ==============================================================================

# Allow weak passwords in development (NEVER in production)
ALLOW_WEAK_PASSWORDS_DEV = False

# Disable rate limiting in development
DISABLE_RATE_LIMITING_DEV = False

# Skip 2FA in development
SKIP_2FA_DEV = False
