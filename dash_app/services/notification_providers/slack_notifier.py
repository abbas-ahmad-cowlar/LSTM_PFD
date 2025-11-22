"""
Slack notification provider with rich formatting (Block Kit) and rate limiting.
Implements Slack Incoming Webhooks integration for team notifications.
"""
import re
import time
import requests
from typing import Dict, Any
from threading import Lock
from services.notification_providers.base import NotificationProvider, NotificationMessage, ProviderError, RateLimitError
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.
    Ensures we don't exceed Slack's 1 message/second limit.
    """
    def __init__(self, capacity: int = 1, refill_rate: float = 1.0):
        """
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens. Blocks if not enough tokens available.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed successfully
        """
        with self.lock:
            # Refill tokens based on time passed
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                # Wait until we have enough tokens
                wait_time = (tokens - self.tokens) / self.refill_rate
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for tokens")
                time.sleep(wait_time)
                self.tokens = 0
                self.last_refill = time.time()
                return True


class SlackNotifier(NotificationProvider):
    """
    Slack notification provider.

    Features:
    - Rich formatting with Block Kit
    - Color-coded messages (attachments)
    - Action buttons (links to dashboard)
    - Rate limiting (1 msg/sec per webhook)
    - Retry logic with exponential backoff
    - Webhook URL validation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Slack notifier.

        Args:
            config: Configuration dict with rate limits, timeouts, etc.
        """
        self.config = config
        self.rate_limiter = TokenBucket(
            capacity=1,
            refill_rate=config.get('rate_limit_per_webhook', 1.0)
        )
        self.timeout = config.get('timeout_seconds', 10)
        self.retry_attempts = config.get('retry_attempts', 3)

    def send(self, webhook_url: str, message: NotificationMessage) -> bool:
        """
        Send notification to Slack.

        Args:
            webhook_url: Slack webhook URL
            message: Notification message

        Returns:
            True if sent successfully

        Raises:
            ValueError: If webhook URL invalid
            ProviderError: If send fails after retries
        """
        # Validate webhook URL
        if not self.validate_webhook_url(webhook_url):
            raise ValueError(f"Invalid Slack webhook URL: {webhook_url}")

        # Rate limit
        self.rate_limiter.consume(1)

        # Build payload
        payload = self._build_slack_payload(message)

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
                    logger.info(f"Slack notification sent successfully: {message.event_type}")
                    return True
                elif response.status_code == 429:
                    # Rate limited by Slack
                    reset_time = response.headers.get('X-Rate-Limit-Reset')
                    logger.warning(f"Slack rate limit hit. Reset time: {reset_time}")
                    if attempt < self.retry_attempts - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RateLimitError(reset_time)
                else:
                    error_msg = f"Slack API error: HTTP {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    if attempt < self.retry_attempts - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ProviderError(error_msg)

            except requests.exceptions.Timeout:
                logger.error(f"Slack webhook timeout (attempt {attempt + 1}/{self.retry_attempts})")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ProviderError("Slack webhook timeout")

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Cannot connect to Slack (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ProviderError(f"Cannot connect to Slack: {e}")

        return False

    def _build_slack_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """
        Convert NotificationMessage to Slack Block Kit format.

        Args:
            message: Standard notification message

        Returns:
            Slack-formatted payload dict
        """
        # Check if user provided Slack-specific override
        if message.slack_override:
            return message.slack_override

        blocks = []

        # 1. Header block with emoji
        emoji_map = {
            'training.complete': '🎉',
            'training.failed': '⚠️',
            'training.started': '🚀',
            'hpo.campaign_complete': '🏆',
            'hpo.campaign_failed': '⚠️',
        }
        emoji = emoji_map.get(message.event_type, '📢')

        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {message.title}",
                "emoji": True
            }
        })

        # 2. Body section with fields from message.data
        fields = []
        for key, value in message.data.items():
            # Skip dashboard_url and other internal fields
            if key in ['dashboard_url', 'settings_url', 'help_url', 'unsubscribe_url']:
                continue

            # Format key (convert snake_case to Title Case)
            formatted_key = key.replace('_', ' ').title()

            # Format value
            formatted_value = str(value)

            fields.append({
                "type": "mrkdwn",
                "text": f"*{formatted_key}:*\n{formatted_value}"
            })

        if fields:
            blocks.append({
                "type": "section",
                "fields": fields
            })

        # 3. Body text (if provided)
        if message.body:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message.body
                }
            })

        # 4. Action buttons
        if message.actions:
            elements = []
            for action in message.actions:
                button = {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action.get('label', 'View')
                    },
                    "url": action['url']
                }

                # Set button style (primary, danger)
                style = action.get('style', 'default')
                if style == 'primary':
                    button['style'] = 'primary'
                elif style == 'danger':
                    button['style'] = 'danger'

                elements.append(button)

            if elements:
                blocks.append({
                    "type": "actions",
                    "elements": elements
                })

        # 5. Context footer
        context_elements = []
        if 'experiment_id' in message.data:
            context_elements.append({
                "type": "mrkdwn",
                "text": f"Experiment #{message.data['experiment_id']}"
            })
        if 'started_by' in message.data:
            context_elements.append({
                "type": "mrkdwn",
                "text": f"Started by {message.data['started_by']}"
            })

        if context_elements:
            blocks.append({
                "type": "context",
                "elements": context_elements
            })

        # Build final payload
        payload = {
            "text": message.title,  # Fallback text for notifications
            "blocks": blocks
        }

        # Add color via attachments (for visual sidebar)
        if message.color:
            payload["attachments"] = [{
                "color": message.color
            }]

        return payload

    def validate_webhook_url(self, webhook_url: str) -> bool:
        """
        Validate Slack webhook URL format.

        Format: https://hooks.slack.com/services/T{TEAM_ID}/B{CHANNEL_ID}/{SECRET_TOKEN}

        Args:
            webhook_url: URL to validate

        Returns:
            True if valid format
        """
        pattern = r'^https://hooks\.slack\.com/services/T[A-Z0-9]{8,10}/B[A-Z0-9]{8,10}/[a-zA-Z0-9]{24}$'
        return bool(re.match(pattern, webhook_url))

    def get_provider_name(self) -> str:
        """Return provider name."""
        return 'slack'

    def supports_rich_formatting(self) -> bool:
        """Slack supports rich formatting via Block Kit."""
        return True
