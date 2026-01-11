"""
Base configuration classes for Milestone 1.

Simple base classes for configuration management.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BaseConfig:
    """Base configuration class."""

    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for validation."""
        return {}

    def validate(self) -> bool:
        """Validate configuration."""
        return True


class ConfigValidator:
    """Simple configuration validator."""

    @staticmethod
    def validate(config: BaseConfig) -> bool:
        """Validate a configuration object."""
        return config.validate()
