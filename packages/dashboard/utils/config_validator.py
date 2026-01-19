"""
Professional configuration validator with security best practices.
Provides centralized validation for all sensitive configuration variables.
"""
import os
import re
import warnings
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    """
    Centralized configuration validator for security-sensitive variables.

    Features:
    - Validates required variables exist
    - Checks secret strength (length, entropy)
    - Environment-specific validation (dev vs prod)
    - Clear, actionable error messages
    - Warnings for weak but acceptable configs
    """

    # Minimum requirements
    MIN_SECRET_LENGTH = 32
    MIN_PASSWORD_LENGTH = 12

    # Required variables per environment
    REQUIRED_VARS = {
        'production': ['DATABASE_URL', 'SECRET_KEY', 'JWT_SECRET_KEY'],
        'development': ['DATABASE_URL', 'SECRET_KEY', 'JWT_SECRET_KEY'],
        'testing': []  # Tests can mock these
    }

    # Optional but recommended variables
    RECOMMENDED_VARS = {
        'production': ['REDIS_URL', 'CELERY_BROKER_URL'],
        'development': [],
        'testing': []
    }

    @classmethod
    def validate_environment(cls, env: str = None) -> ValidationResult:
        """
        Validate all configuration for the current environment.

        Args:
            env: Environment name (production/development/testing).
                 If None, reads from ENV variable, defaults to 'development'

        Returns:
            ValidationResult with errors and warnings
        """
        if env is None:
            env = os.getenv('ENV', 'development').lower()

        errors = []
        warnings_list = []

        # 1. Validate required variables
        required = cls.REQUIRED_VARS.get(env, cls.REQUIRED_VARS['development'])
        for var_name in required:
            value = os.getenv(var_name)
            if not value:
                errors.append(
                    f"❌ {var_name} is required but not set.\n"
                    f"   Set it in .env file or environment variables.\n"
                    f"   See .env.example for guidance."
                )
            else:
                # Validate the specific variable
                var_result = cls._validate_variable(var_name, value, env)
                errors.extend(var_result.errors)
                warnings_list.extend(var_result.warnings)

        # 2. Check recommended variables
        recommended = cls.RECOMMENDED_VARS.get(env, [])
        for var_name in recommended:
            if not os.getenv(var_name):
                warnings_list.append(
                    f"⚠️  {var_name} is not set (recommended for {env}).\n"
                    f"   The application will use defaults, which may not be optimal."
                )

        # 3. Special validation for production
        if env == 'production':
            prod_result = cls._validate_production_specific()
            errors.extend(prod_result.errors)
            warnings_list.extend(prod_result.warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings_list
        )

    @classmethod
    def _validate_variable(cls, var_name: str, value: str, env: str) -> ValidationResult:
        """Validate a specific configuration variable."""
        errors = []
        warnings_list = []

        # Database URL validation
        if var_name == 'DATABASE_URL':
            result = cls._validate_database_url(value, env)
            errors.extend(result.errors)
            warnings_list.extend(result.warnings)

        # Secret key validation
        elif var_name in ['SECRET_KEY', 'JWT_SECRET_KEY']:
            result = cls._validate_secret(var_name, value, env)
            errors.extend(result.errors)
            warnings_list.extend(result.warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings_list
        )

    @classmethod
    def _validate_database_url(cls, url: str, env: str) -> ValidationResult:
        """Validate DATABASE_URL format and security."""
        errors = []
        warnings_list = []

        # Check format - SQLite is allowed in development/testing, PostgreSQL required in production
        if env == 'production':
            if not url.startswith(('postgresql://', 'postgres://')):
                errors.append(
                    f"❌ DATABASE_URL must start with 'postgresql://' or 'postgres://' in production\n"
                    f"   Current value: {url[:20]}..."
                )
                return ValidationResult(False, errors, warnings_list)
        else:
            # Development/testing: allow SQLite or PostgreSQL
            if not url.startswith(('postgresql://', 'postgres://', 'sqlite://')):
                errors.append(
                    f"❌ DATABASE_URL must start with 'postgresql://', 'postgres://', or 'sqlite://'\n"
                    f"   Current value: {url[:20]}..."
                )
                return ValidationResult(False, errors, warnings_list)
            
            # If using SQLite, skip password validation
            if url.startswith('sqlite://'):
                return ValidationResult(True, [], [])

        # Extract password from URL for validation
        password_match = re.search(r':([^@]+)@', url)
        if password_match:
            password = password_match.group(1)

            # Check for default/weak passwords in production
            if env == 'production':
                weak_passwords = [
                    'password', 'admin', 'root', '123456', 'postgres',
                    'lstm_password', 'secure_password', 'changeme'
                ]
                if password.lower() in weak_passwords:
                    errors.append(
                        f"❌ DATABASE_URL contains a weak password in production!\n"
                        f"   Use a strong, unique password (minimum {cls.MIN_PASSWORD_LENGTH} characters)."
                    )
                elif len(password) < cls.MIN_PASSWORD_LENGTH:
                    warnings_list.append(
                        f"⚠️  Database password is shorter than recommended "
                        f"({len(password)} < {cls.MIN_PASSWORD_LENGTH} chars)"
                    )

            # Check for example passwords
            if 'example' in password.lower() or 'sample' in password.lower():
                errors.append(
                    f"❌ DATABASE_URL appears to contain an example password.\n"
                    f"   Replace it with a real, secure password."
                )

        # Warn about localhost in production
        if env == 'production' and 'localhost' in url:
            warnings_list.append(
                f"⚠️  DATABASE_URL uses 'localhost' in production.\n"
                f"   Ensure your database is properly configured."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings_list
        )

    @classmethod
    def _validate_secret(cls, var_name: str, value: str, env: str) -> ValidationResult:
        """Validate secret keys (SECRET_KEY, JWT_SECRET_KEY)."""
        errors = []
        warnings_list = []

        # Check length
        if len(value) < cls.MIN_SECRET_LENGTH:
            errors.append(
                f"❌ {var_name} is too short ({len(value)} < {cls.MIN_SECRET_LENGTH} characters).\n"
                f"   Generate a strong secret:\n"
                f"   python -c 'import secrets; print(secrets.token_hex(32))'"
            )

        # Check for common weak secrets
        weak_secrets = [
            'dev-secret', 'change-this', 'changeme', 'secret', 'password',
            'your-secret-key-here', 'example', 'sample', 'test', 'demo'
        ]
        for weak in weak_secrets:
            if weak in value.lower():
                errors.append(
                    f"❌ {var_name} contains weak or example text: '{weak}'\n"
                    f"   Replace with a cryptographically random secret."
                )
                break

        # Check entropy (basic check - should have varied characters)
        if env == 'production':
            unique_chars = len(set(value))
            if unique_chars < 16:
                warnings_list.append(
                    f"⚠️  {var_name} has low character diversity ({unique_chars} unique chars).\n"
                    f"   Consider regenerating with: python -c 'import secrets; print(secrets.token_hex(32))'"
                )

            # Check if it's all lowercase or all numbers
            if value.islower() or value.isdigit():
                warnings_list.append(
                    f"⚠️  {var_name} lacks character variety (no uppercase/special chars).\n"
                    f"   This reduces entropy. Regenerate for better security."
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings_list
        )

    @classmethod
    def _validate_production_specific(cls) -> ValidationResult:
        """Additional validation specific to production environment."""
        errors = []
        warnings_list = []

        # Check DEBUG is disabled
        debug = os.getenv('DEBUG', 'False').lower()
        if debug == 'true':
            errors.append(
                f"❌ DEBUG=True in production is a security risk!\n"
                f"   Set DEBUG=False or remove it (defaults to False)."
            )

        # Warn about rate limiting
        if not os.getenv('API_KEY_RATE_LIMIT_DEFAULT'):
            warnings_list.append(
                f"⚠️  API_KEY_RATE_LIMIT_DEFAULT not set.\n"
                f"   Using default (1000 req/hour). Consider tuning for production."
            )

        # Check HTTPS/SSL settings (if configured)
        secure_ssl = os.getenv('SECURE_SSL_REDIRECT', 'False').lower()
        if secure_ssl != 'true':
            warnings_list.append(
                f"⚠️  SECURE_SSL_REDIRECT not enabled in production.\n"
                f"   Consider enabling for HTTPS enforcement."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings_list
        )

    @classmethod
    def validate_or_exit(cls, env: str = None) -> None:
        """
        Validate configuration and exit if critical errors found.
        Prints warnings but doesn't exit for non-critical issues.

        This is the main entry point for application startup validation.
        """
        result = cls.validate_environment(env)

        # Print warnings
        if result.warnings:
            print("\n⚠️  Configuration Warnings:")
            print("=" * 80)
            for warning in result.warnings:
                print(warning)
            print("=" * 80)
            print()

        # Handle errors
        if not result.is_valid:
            print("\n❌ Configuration Validation Failed!")
            print("=" * 80)
            for error in result.errors:
                print(error)
            print("=" * 80)
            print("\n💡 Quick Fix:")
            print("   1. Copy .env.example to .env")
            print("   2. Run: python -c 'import secrets; print(secrets.token_hex(32))'")
            print("   3. Set the generated values for SECRET_KEY and JWT_SECRET_KEY")
            print("   4. Configure DATABASE_URL with your actual credentials")
            print("\nSee .env.example for complete configuration template.\n")
            raise SystemExit(1)

        # Success message (only in development with verbose mode)
        if os.getenv('CONFIG_VERBOSE', 'False').lower() == 'true':
            print("✅ Configuration validation passed!\n")


def get_required_config(var_name: str, default: Optional[str] = None) -> str:
    """
    Get a required configuration variable with lazy validation.

    This is useful for configs that may not be needed at startup
    but should be validated when accessed.

    Args:
        var_name: Environment variable name
        default: Default value (use with caution!)

    Returns:
        Configuration value

    Raises:
        ValueError: If variable not set and no default provided
    """
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(
            f"{var_name} is required but not set in environment variables.\n"
            f"Set it in .env file. See .env.example for guidance."
        )
    return value
