"""
Unit tests for configuration validator.
Tests validation logic for security-sensitive configuration variables.
"""
import os
import pytest
from unittest.mock import patch
from utils.config_validator import ConfigValidator, ValidationResult, get_required_config


class TestConfigValidator:
    """Test suite for ConfigValidator."""

    def test_validate_strong_secret(self):
        """Test that strong secrets pass validation."""
        # Generate a proper secret
        import secrets
        strong_secret = secrets.token_hex(32)

        with patch.dict(os.environ, {'SECRET_KEY': strong_secret, 'ENV': 'production'}):
            result = ConfigValidator._validate_secret('SECRET_KEY', strong_secret, 'production')
            assert result.is_valid
            assert len(result.errors) == 0

    def test_validate_weak_secret(self):
        """Test that weak secrets fail validation."""
        weak_secrets = [
            'short',  # Too short
            'dev-secret-key-change-in-production',  # Contains weak text
            'your-secret-key-here-minimum-32-chars-change-this',  # Example text
            '12345678901234567890123456789012',  # All numbers
        ]

        for weak in weak_secrets:
            result = ConfigValidator._validate_secret('SECRET_KEY', weak, 'production')
            assert not result.is_valid or len(result.warnings) > 0, f"Failed to detect weak secret: {weak}"

    def test_validate_database_url_format(self):
        """Test DATABASE_URL format validation."""
        # Valid URLs
        valid_urls = [
            'postgresql://user:pass@localhost:5432/db',
            'postgres://user:pass@host.com:5432/db',
        ]
        for url in valid_urls:
            result = ConfigValidator._validate_database_url(url, 'development')
            assert result.is_valid, f"Valid URL rejected: {url}"

        # Invalid URLs
        invalid_urls = [
            'mysql://user:pass@localhost/db',  # Wrong protocol
            'http://localhost:5432/db',  # Not a database URL
        ]
        for url in invalid_urls:
            result = ConfigValidator._validate_database_url(url, 'development')
            assert not result.is_valid, f"Invalid URL accepted: {url}"

    def test_validate_weak_database_password(self):
        """Test detection of weak database passwords."""
        weak_passwords = [
            'postgresql://user:password@localhost:5432/db',
            'postgresql://user:admin@localhost:5432/db',
            'postgresql://user:123456@localhost:5432/db',
        ]

        for url in weak_passwords:
            result = ConfigValidator._validate_database_url(url, 'production')
            assert not result.is_valid, f"Weak password not detected in: {url}"

    def test_required_variables_development(self):
        """Test that required variables are checked in development."""
        with patch.dict(os.environ, {}, clear=True):
            result = ConfigValidator.validate_environment('development')
            assert not result.is_valid
            assert any('DATABASE_URL' in err for err in result.errors)
            assert any('SECRET_KEY' in err for err in result.errors)

    def test_required_variables_testing(self):
        """Test that testing environment has relaxed requirements."""
        with patch.dict(os.environ, {}, clear=True):
            result = ConfigValidator.validate_environment('testing')
            # Testing should not require all variables
            assert result.is_valid or len(result.errors) == 0

    def test_production_debug_check(self):
        """Test that DEBUG=True is flagged in production."""
        import secrets
        with patch.dict(os.environ, {
            'ENV': 'production',
            'DEBUG': 'True',
            'DATABASE_URL': 'postgresql://user:securepass123@localhost:5432/db',
            'SECRET_KEY': secrets.token_hex(32),
            'JWT_SECRET_KEY': secrets.token_hex(32),
        }):
            result = ConfigValidator.validate_environment('production')
            assert not result.is_valid
            assert any('DEBUG' in err for err in result.errors)

    def test_get_required_config_missing(self):
        """Test get_required_config raises error when variable missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MISSING_VAR"):
                get_required_config('MISSING_VAR')

    def test_get_required_config_with_default(self):
        """Test get_required_config with default value."""
        with patch.dict(os.environ, {}, clear=True):
            value = get_required_config('MISSING_VAR', 'default_value')
            assert value == 'default_value'

    def test_get_required_config_exists(self):
        """Test get_required_config returns value when set."""
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            value = get_required_config('TEST_VAR')
            assert value == 'test_value'

    def test_localhost_warning_in_production(self):
        """Test warning for localhost in production DATABASE_URL."""
        import secrets
        with patch.dict(os.environ, {
            'ENV': 'production',
            'DATABASE_URL': 'postgresql://user:securepass123@localhost:5432/db',
            'SECRET_KEY': secrets.token_hex(32),
            'JWT_SECRET_KEY': secrets.token_hex(32),
        }):
            result = ConfigValidator.validate_environment('production')
            assert any('localhost' in warn for warn in result.warnings)

    def test_complete_valid_configuration(self):
        """Test a complete, valid configuration."""
        import secrets
        with patch.dict(os.environ, {
            'ENV': 'production',
            'DEBUG': 'False',
            'DATABASE_URL': 'postgresql://user:' + secrets.token_urlsafe(16) + '@prod.db.com:5432/db',
            'SECRET_KEY': secrets.token_hex(32),
            'JWT_SECRET_KEY': secrets.token_hex(32),
            'REDIS_URL': 'redis://redis.prod.com:6379/0',
        }):
            result = ConfigValidator.validate_environment('production')
            assert result.is_valid
            assert len(result.errors) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_variables(self):
        """Test that empty strings are treated as missing."""
        with patch.dict(os.environ, {
            'DATABASE_URL': '',
            'SECRET_KEY': '',
        }):
            result = ConfigValidator.validate_environment('development')
            assert not result.is_valid

    def test_whitespace_only_secret(self):
        """Test that whitespace-only secrets fail validation."""
        with patch.dict(os.environ, {
            'SECRET_KEY': '   ' * 20,  # 60 spaces
        }):
            result = ConfigValidator._validate_secret('SECRET_KEY', '   ' * 20, 'production')
            # Should fail due to low entropy
            assert not result.is_valid or len(result.warnings) > 0

    def test_sql_injection_in_database_url(self):
        """Test that potential SQL injection attempts are handled."""
        # These should still be valid URLs (format-wise) but might have warnings
        suspicious_urls = [
            "postgresql://user:pass';DROP TABLE users;--@localhost:5432/db",
        ]
        for url in suspicious_urls:
            # Should not crash, even with suspicious content
            result = ConfigValidator._validate_database_url(url, 'development')
            # Should at least process without exception
            assert isinstance(result, ValidationResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
