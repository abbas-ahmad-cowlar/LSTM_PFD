"""
Base notification provider interface and message format.
Provides abstraction for multiple notification providers (Slack, Teams, custom webhooks).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


@dataclass
class NotificationMessage:
    """
    Standardized message format across all providers.
    Providers translate this to their specific format (Slack Block Kit, Teams MessageCard, etc.).
    """
    title: str
    body: str
    event_type: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    data: Dict[str, Any]  # Event-specific data
    actions: Optional[List[Dict[str, str]]] = None  # Buttons/links: [{'label': '...', 'url': '...', 'style': '...'}]
    color: Optional[str] = None  # Hex color for sidebar/accent

    # Provider-specific overrides (optional)
    slack_override: Optional[Dict] = None
    teams_override: Optional[Dict] = None

    def __post_init__(self):
        """Validate message fields."""
        if self.priority not in ['low', 'medium', 'high', 'critical']:
            raise ValueError(f"Invalid priority: {self.priority}. Must be one of: low, medium, high, critical")

        # Set default color based on priority if not provided
        if not self.color:
            color_map = {
                'low': '#0078d4',      # Blue
                'medium': '#00ff00',   # Green
                'high': '#ffcc00',     # Yellow/Warning
                'critical': '#ff0000'  # Red/Error
            }
            self.color = color_map.get(self.priority, '#0078d4')


class NotificationProvider(ABC):
    """
    Abstract base class for all notification providers.

    New providers (Discord, Mattermost) simply implement this interface.
    """

    @abstractmethod
    def send(self, webhook_url: str, message: NotificationMessage) -> bool:
        """
        Send notification via provider.

        Args:
            webhook_url: Provider-specific webhook URL
            message: Standardized message object

        Returns:
            True if sent successfully, False if failed

        Raises:
            ProviderError: If provider-specific error occurs
        """
        pass

    @abstractmethod
    def validate_webhook_url(self, webhook_url: str) -> bool:
        """
        Validate that webhook URL is correctly formatted for this provider.

        Args:
            webhook_url: URL to validate

        Returns:
            True if valid format, False otherwise
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (e.g., 'slack', 'teams')"""
        pass

    @abstractmethod
    def supports_rich_formatting(self) -> bool:
        """Return True if provider supports rich cards/buttons"""
        pass


class ProviderError(Exception):
    """Base exception for provider-specific errors."""
    pass


class RateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded."""
    def __init__(self, reset_time: Optional[str] = None):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Reset time: {reset_time}")
