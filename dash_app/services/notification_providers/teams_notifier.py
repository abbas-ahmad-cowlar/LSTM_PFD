"""
Microsoft Teams notification provider with MessageCard formatting.
Implements Teams Incoming Webhooks integration for team notifications.
"""
import re
import time
import requests
from typing import Dict, Any
from services.notification_providers.base import NotificationProvider, NotificationMessage, ProviderError, RateLimitError
from services.notification_providers.slack_notifier import TokenBucket
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TeamsNotifier(NotificationProvider):
    """
    Microsoft Teams notification provider.

    Features:
    - Rich formatting with MessageCard (Adaptive Cards v1)
    - Color themes (themeColor)
    - Action buttons (potentialAction)
    - Rate limiting (2 msg/sec - more generous than Slack)
    - Retry logic with exponential backoff
    - Webhook URL validation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Teams notifier.

        Args:
            config: Configuration dict with rate limits, timeouts, etc.
        """
        self.config = config
        self.rate_limiter = TokenBucket(
            capacity=2,  # Teams allows 2 msg/sec
            refill_rate=config.get('rate_limit_per_webhook', 2.0)
        )
        self.timeout = config.get('timeout_seconds', 10)
        self.retry_attempts = config.get('retry_attempts', 3)

    def send(self, webhook_url: str, message: NotificationMessage) -> bool:
        """
        Send notification to Microsoft Teams.

        Args:
            webhook_url: Teams webhook URL
            message: Notification message

        Returns:
            True if sent successfully

        Raises:
            ValueError: If webhook URL invalid
            ProviderError: If send fails after retries
        """
        # Validate webhook URL
        if not self.validate_webhook_url(webhook_url):
            raise ValueError(f"Invalid Teams webhook URL: {webhook_url}")

        # Rate limit
        self.rate_limiter.consume(1)

        # Build payload
        payload = self._build_teams_payload(message)

        # Send with retries
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    webhook_url,
                    json=payload,
                    timeout=self.timeout
                )

                # Handle response
                if response.status_code == 200:
                    logger.info(f"Teams notification sent successfully: {message.event_type}")
                    return True
                elif response.status_code == 429:
                    # Rate limited by Teams
                    logger.warning("Teams rate limit hit")
                    if attempt < self.retry_attempts - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RateLimitError()
                else:
                    error_msg = f"Teams API error: HTTP {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    if attempt < self.retry_attempts - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ProviderError(error_msg)

            except requests.exceptions.Timeout:
                logger.error(f"Teams webhook timeout (attempt {attempt + 1}/{self.retry_attempts})")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ProviderError("Teams webhook timeout")

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Cannot connect to Teams (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ProviderError(f"Cannot connect to Teams: {e}")

        return False

    def _build_teams_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """
        Convert NotificationMessage to Teams MessageCard format.

        Args:
            message: Standard notification message

        Returns:
            Teams-formatted payload dict (MessageCard format)
        """
        # Check if user provided Teams-specific override
        if message.teams_override:
            return message.teams_override

        # Map priority to color
        color_map = {
            'low': '0078d4',      # Blue
            'medium': '00ff00',   # Green (success)
            'high': 'ffcc00',     # Yellow (warning)
            'critical': 'ff0000'  # Red (error)
        }
        theme_color = message.color.lstrip('#') if message.color else color_map.get(message.priority, '0078d4')

        # Build facts array from message.data
        facts = []
        for key, value in message.data.items():
            # Skip URLs
            if key in ['dashboard_url', 'settings_url', 'help_url', 'unsubscribe_url']:
                continue

            # Format key (convert snake_case to Title Case)
            formatted_key = key.replace('_', ' ').title()

            facts.append({
                "name": f"{formatted_key}:",
                "value": str(value)
            })

        # Build sections
        sections = []
        section = {
            "activityTitle": message.title,
            "facts": facts,
            "markdown": True
        }

        # Add body as subtitle if provided
        if message.body:
            section["activitySubtitle"] = message.body

        sections.append(section)

        # Build potential actions (buttons)
        potential_actions = []
        if message.actions:
            for action in message.actions:
                potential_actions.append({
                    "@type": "OpenUri",
                    "name": action.get('label', 'View'),
                    "targets": [
                        {
                            "os": "default",
                            "uri": action['url']
                        }
                    ]
                })

        # Build final payload (MessageCard format)
        payload = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": message.title,
            "themeColor": theme_color,
            "sections": sections
        }

        if potential_actions:
            payload["potentialAction"] = potential_actions

        return payload

    def validate_webhook_url(self, webhook_url: str) -> bool:
        """
        Validate Microsoft Teams webhook URL format.

        Format: https://{region}.office.com/webhook/{tenant}/IncomingWebhook/{channel}/{secret}

        Args:
            webhook_url: URL to validate

        Returns:
            True if valid format
        """
        # Teams webhook URLs can have various formats
        pattern = r'^https://[a-z0-9]+\.office\.com/webhook/[a-zA-Z0-9-]+(@[a-zA-Z0-9-]+)?/IncomingWebhook/[a-zA-Z0-9-]+/[a-zA-Z0-9-]+$'
        return bool(re.match(pattern, webhook_url))

    def get_provider_name(self) -> str:
        """Return provider name."""
        return 'teams'

    def supports_rich_formatting(self) -> bool:
        """Teams supports rich formatting via MessageCard."""
        return True
