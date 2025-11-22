"""
Factory for creating notification provider instances.
Centralizes provider instantiation logic and feature flag checking.
"""
from typing import Dict, List
from services.notification_providers.base import NotificationProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class NotificationProviderFactory:
    """
    Factory for creating notification provider instances.

    Centralizes provider instantiation logic.
    Makes it easy to add new providers.
    """

    _providers: Dict[str, NotificationProvider] = {}  # Cache provider instances

    @staticmethod
    def get_provider(provider_type: str, config: Dict = None) -> NotificationProvider:
        """
        Get provider instance by type.

        Args:
            provider_type: 'slack', 'teams', 'webhook', etc.
            config: Optional provider-specific configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider type unknown or disabled
        """
        from config import (
            NOTIFICATIONS_SLACK_ENABLED,
            NOTIFICATIONS_TEAMS_ENABLED,
            NOTIFICATIONS_WEBHOOK_ENABLED
        )

        # Check if provider globally enabled
        if provider_type == 'slack':
            if not NOTIFICATIONS_SLACK_ENABLED:
                raise ValueError("Slack notifications are disabled globally")
        elif provider_type == 'teams':
            if not NOTIFICATIONS_TEAMS_ENABLED:
                raise ValueError("Teams notifications are disabled globally")
        elif provider_type == 'webhook':
            if not NOTIFICATIONS_WEBHOOK_ENABLED:
                raise ValueError("Custom webhooks are disabled globally")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

        # Return cached instance or create new
        cache_key = f"{provider_type}_{id(config) if config else 'default'}"
        if cache_key not in NotificationProviderFactory._providers:
            if provider_type == 'slack':
                from services.notification_providers.slack_notifier import SlackNotifier
                NotificationProviderFactory._providers[cache_key] = SlackNotifier(config or {})
            elif provider_type == 'teams':
                from services.notification_providers.teams_notifier import TeamsNotifier
                NotificationProviderFactory._providers[cache_key] = TeamsNotifier(config or {})
            elif provider_type == 'webhook':
                from services.notification_providers.custom_webhook_notifier import CustomWebhookNotifier
                NotificationProviderFactory._providers[cache_key] = CustomWebhookNotifier(config or {})

        return NotificationProviderFactory._providers[cache_key]

    @staticmethod
    def get_enabled_providers() -> List[str]:
        """
        Get list of globally enabled provider types.

        Returns:
            List of enabled provider strings (e.g., ['email', 'slack', 'teams'])
        """
        from config import (
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
            EMAIL_ENABLED,
            NOTIFICATIONS_SLACK_ENABLED,
            NOTIFICATIONS_TEAMS_ENABLED,
            NOTIFICATIONS_WEBHOOK_ENABLED
        )

        enabled = []

        if EMAIL_ENABLED:
            enabled.append('email')
        if NOTIFICATIONS_SLACK_ENABLED:
            enabled.append('slack')
        if NOTIFICATIONS_TEAMS_ENABLED:
            enabled.append('teams')
        if NOTIFICATIONS_WEBHOOK_ENABLED:
            enabled.append('webhook')

        return enabled

    @staticmethod
    def clear_cache():
        """Clear provider cache (useful for testing)."""
        NotificationProviderFactory._providers.clear()
        logger.info("Provider cache cleared")
