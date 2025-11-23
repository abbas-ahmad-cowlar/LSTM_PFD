"""
Base configuration class with YAML serialization and validation.

Purpose:
    Abstract base class for all configuration objects with:
    - YAML load/save capabilities
    - JSON schema validation
    - Configuration merging
    - Type-safe dataclass-based configs

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
import yaml
import json
from jsonschema import validate, ValidationError

T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig(ABC):
    """
    Abstract base class for all configuration objects.

    All config classes should inherit from this and use @dataclass decorator.
    Provides automatic YAML serialization, validation, and merging.

    Example:
        @dataclass
        class MyConfig(BaseConfig):
            learning_rate: float = 0.001
            batch_size: int = 32

            def get_schema(self) -> Dict:
                return {
                    "type": "object",
                    "properties": {
                        "learning_rate": {"type": "number", "minimum": 0},
                        "batch_size": {"type": "integer", "minimum": 1}
                    }
                }
    """

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for validation.

        Returns:
            Dictionary containing JSON schema specification
        """
        pass

    def validate(self) -> bool:
        """
        Validate configuration against schema.

        Returns:
            True if valid

        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            validate(instance=asdict(self), schema=self.get_schema())
            return True
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}") from e

    @classmethod
    def from_yaml(cls: Type[T], path: Path) -> T:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Configuration object

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Create instance from loaded data
        instance = cls(**data)
        instance.validate()
        return instance

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        self.validate()  # Validate before saving

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return asdict(self)

    @classmethod
    def merge_configs(cls: Type[T], *configs: T) -> T:
        """
        Merge multiple configurations (later configs override earlier).

        Args:
            *configs: Variable number of config objects to merge

        Returns:
            Merged configuration

        Example:
            base_cfg = MyConfig(lr=0.001, batch_size=32)
            override_cfg = MyConfig(lr=0.01, batch_size=32)
            merged = MyConfig.merge_configs(base_cfg, override_cfg)
            # merged.lr == 0.01 (overridden)
        """
        if not configs:
            raise ValueError("At least one config required for merging")

        # Start with first config's dict
        merged_dict = asdict(configs[0])

        # Merge subsequent configs
        for config in configs[1:]:
            config_dict = asdict(config)
            merged_dict.update(config_dict)

        # Create new instance
        merged = cls(**merged_dict)
        merged.validate()
        return merged

    def __repr__(self) -> str:
        """Pretty print configuration."""
        field_strs = []
        for field in fields(self):
            value = getattr(self, field.name)
            field_strs.append(f"  {field.name}={value}")

        return f"{self.__class__.__name__}(\n" + ",\n".join(field_strs) + "\n)"


class ConfigValidator:
    """
    Utility class for JSON schema validation.

    Provides helper methods for common validation patterns.
    """

    @staticmethod
    def validate_positive(value: float, name: str) -> None:
        """Validate that value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
        """Validate that value is in range [min_val, max_val]."""
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")

    @staticmethod
    def validate_probability(value: float, name: str) -> None:
        """Validate that value is a valid probability [0, 1]."""
        ConfigValidator.validate_range(value, 0.0, 1.0, name)
