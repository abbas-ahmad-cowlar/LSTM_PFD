"""
Custom webhook notifier for generic HTTP webhooks.
Sends simple JSON payloads to user-defined webhook endpoints.
"""
import re
import time
import requests
from typing import Dict, Any
from services.notification_providers.base import NotificationProvider, NotificationMessage, ProviderError
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


class CustomWebhookNotifier(NotificationProvider):
    """
    Generic webhook notification provider.

    Features:
    - Simple JSON payload (no provider-specific formatting)
    - Basic retry logic
    - Flexible URL validation (allows any HTTPS URL)
    - Timeout handling
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize custom webhook notifier.

        Args:
            config: Configuration dict with timeouts, retries, etc.
        """
        self.config = config
        self.timeout = config.get('timeout_seconds', 5)
        self.retry_attempts = config.get('retry_attempts', 2)

    def send(self, webhook_url: str, message: NotificationMessage) -> bool:
        """
        Send notification to custom webhook.

        Args:
            webhook_url: Custom webhook URL
            message: Notification message

        Returns:
            True if sent successfully

        Raises:
            ValueError: If webhook URL invalid
            ProviderError: If send fails after retries
        """
        # Validate webhook URL
        if not self.validate_webhook_url(webhook_url):
            raise ValueError(f"Invalid custom webhook URL: {webhook_url}")

        # Build simple JSON payload
        payload = self._build_payload(message)

        # Send with retries
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    webhook_url,
                    json=payload,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )

                # Accept any 2xx status as success
                if 200 <= response.status_code < 300:
                    logger.info(f"Custom webhook notification sent successfully: {message.event_type}")
                    return True
                else:
                    error_msg = f"Custom webhook error: HTTP {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    if attempt < self.retry_attempts - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ProviderError(error_msg)

            except requests.exceptions.Timeout:
                logger.error(f"Custom webhook timeout (attempt {attempt + 1}/{self.retry_attempts})")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ProviderError("Custom webhook timeout")

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Cannot connect to custom webhook (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ProviderError(f"Cannot connect to custom webhook: {e}")

        return False

    def _build_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """
        Build simple JSON payload for custom webhook.

        Args:
            message: Standard notification message

        Returns:
            Simple JSON dict
        """
        return {
            'event_type': message.event_type,
            'title': message.title,
            'body': message.body,
            'priority': message.priority,
            'data': message.data,
            'timestamp': message.data.get('timestamp', ''),
            'actions': message.actions or []
        }

    def validate_webhook_url(self, webhook_url: str) -> bool:
        """
        Validate custom webhook URL.

        Accepts any HTTPS URL (enforces HTTPS for security).

        Args:
            webhook_url: URL to validate

        Returns:
            True if valid HTTPS URL
        """
        # Require HTTPS for security
        pattern = r'^https://[a-zA-Z0-9.-]+(:[0-9]+)?(/.*)?$'
        return bool(re.match(pattern, webhook_url))

    def get_provider_name(self) -> str:
        """Return provider name."""
        return 'webhook'

    def supports_rich_formatting(self) -> bool:
        """Custom webhooks use simple JSON (no rich formatting)."""
        return False
