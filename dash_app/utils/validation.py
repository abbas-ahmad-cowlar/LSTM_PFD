"""Input validation utilities."""
import re
from typing import Optional, Any


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_string_length(value: str, max_length: int, field_name: str = "Field") -> str:
    """
    Validate string doesn't exceed max length.

    Args:
        value: String to validate
        max_length: Maximum allowed length
        field_name: Name of the field for error messages

    Returns:
        Trimmed string value

    Raises:
        ValidationError: If string exceeds max length
    """
    if value and len(value) > max_length:
        raise ValidationError(f"{field_name} must be at most {max_length} characters")
    return value.strip() if value else ""


def validate_alphanumeric(value: str, allow_spaces: bool = False, field_name: str = "Field") -> str:
    """
    Validate string contains only alphanumeric characters.

    Args:
        value: String to validate
        allow_spaces: Whether to allow spaces in the string
        field_name: Name of the field for error messages

    Returns:
        Trimmed string value

    Raises:
        ValidationError: If string contains invalid characters
    """
    if not value:
        return ""

    pattern = r'^[a-zA-Z0-9\s_-]+$' if allow_spaces else r'^[a-zA-Z0-9_-]+$'
    if not re.match(pattern, value):
        raise ValidationError(f"{field_name} contains invalid characters")
    return value.strip()


def validate_email(email: str) -> str:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        Lowercase trimmed email address

    Raises:
        ValidationError: If email format is invalid
    """
    if not email:
        raise ValidationError("Email is required")

    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError("Invalid email format")
    return email.strip().lower()


def validate_positive_integer(value: Any, field_name: str = "Value") -> int:
    """
    Validate integer is positive.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Returns:
        The validated integer value

    Raises:
        ValidationError: If value is not a positive integer
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValidationError(f"{field_name} must be a positive integer")
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid integer")


def validate_non_negative_integer(value: Any, field_name: str = "Value") -> int:
    """
    Validate integer is non-negative (>= 0).

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Returns:
        The validated integer value

    Raises:
        ValidationError: If value is not a non-negative integer
    """
    try:
        int_value = int(value)
        if int_value < 0:
            raise ValidationError(f"{field_name} must be a non-negative integer")
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid integer")


def validate_positive_number(value: Any, field_name: str = "Value") -> float:
    """
    Validate number is positive.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Returns:
        The validated float value

    Raises:
        ValidationError: If value is not a positive number
    """
    try:
        float_value = float(value)
        if float_value <= 0:
            raise ValidationError(f"{field_name} must be a positive number")
        return float_value
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid number")


def validate_range(value: Any, min_val: float, max_val: float, field_name: str = "Value") -> float:
    """
    Validate number is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of the field for error messages

    Returns:
        The validated float value

    Raises:
        ValidationError: If value is outside the allowed range
    """
    try:
        float_value = float(value)
        if not (min_val <= float_value <= max_val):
            raise ValidationError(f"{field_name} must be between {min_val} and {max_val}")
        return float_value
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid number")


def validate_integer_range(value: Any, min_val: int, max_val: int, field_name: str = "Value") -> int:
    """
    Validate integer is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of the field for error messages

    Returns:
        The validated integer value

    Raises:
        ValidationError: If value is outside the allowed range
    """
    try:
        int_value = int(value)
        if not (min_val <= int_value <= max_val):
            raise ValidationError(f"{field_name} must be between {min_val} and {max_val}")
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid integer")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for file system operations
    """
    if not filename:
        return "unnamed_file"

    # Remove any path separators
    filename = filename.replace('/', '').replace('\\', '').replace('..', '')
    # Keep only safe characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # Ensure it's not empty after sanitization
    if not filename or filename in ['.', '..']:
        return "unnamed_file"
    return filename


def validate_required(value: Any, field_name: str = "Field") -> Any:
    """
    Validate field is not None or empty.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Returns:
        The value if valid

    Raises:
        ValidationError: If value is None or empty
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(f"{field_name} is required")
    return value


def validate_url(url: str, field_name: str = "URL") -> str:
    """
    Validate URL format.

    Args:
        url: URL to validate
        field_name: Name of the field for error messages

    Returns:
        Trimmed URL

    Raises:
        ValidationError: If URL format is invalid
    """
    if not url:
        raise ValidationError(f"{field_name} is required")

    # Basic URL pattern - allows http, https, and common URL patterns
    pattern = r'^https?://[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(/.*)?$'
    if not re.match(pattern, url.strip()):
        raise ValidationError(f"{field_name} must be a valid URL (http:// or https://)")
    return url.strip()


def validate_choice(value: Any, choices: list, field_name: str = "Value") -> Any:
    """
    Validate value is one of the allowed choices.

    Args:
        value: Value to validate
        choices: List of allowed values
        field_name: Name of the field for error messages

    Returns:
        The value if valid

    Raises:
        ValidationError: If value is not in allowed choices
    """
    if value not in choices:
        raise ValidationError(f"{field_name} must be one of: {', '.join(map(str, choices))}")
    return value


def validate_list_not_empty(value: list, field_name: str = "List") -> list:
    """
    Validate list is not empty.

    Args:
        value: List to validate
        field_name: Name of the field for error messages

    Returns:
        The list if valid

    Raises:
        ValidationError: If list is None or empty
    """
    if not value or len(value) == 0:
        raise ValidationError(f"{field_name} must contain at least one item")
    return value


def validate_json_serializable(value: Any, field_name: str = "Value") -> Any:
    """
    Validate value is JSON serializable.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Returns:
        The value if valid

    Raises:
        ValidationError: If value cannot be serialized to JSON
    """
    import json
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{field_name} must be JSON serializable: {str(e)}")
