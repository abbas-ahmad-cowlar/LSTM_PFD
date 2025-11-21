"""
Email provider abstraction layer for sending emails.
Supports multiple providers (SendGrid, AWS SES, SMTP) with easy switching.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmailProvider(ABC):
    """Abstract base class for email providers."""

    @abstractmethod
    def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send email via provider.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text email body (fallback)
            from_email: Sender email address
            from_name: Sender name

        Returns:
            Dict with 'success', 'message_id', and optional 'error' keys
        """
        pass

    @abstractmethod
    def get_message_status(self, message_id: str) -> str:
        """
        Check delivery status of a message.

        Args:
            message_id: Provider's message ID

        Returns:
            Status string ('sent', 'delivered', 'bounced', etc.)
        """
        pass


class SendGridProvider(EmailProvider):
    """SendGrid email provider implementation."""

    def __init__(self, api_key: str, default_from_email: str, default_from_name: str = "LSTM Dashboard"):
        """
        Initialize SendGrid provider.

        Args:
            api_key: SendGrid API key
            default_from_email: Default sender email
            default_from_name: Default sender name
        """
        self.api_key = api_key
        self.default_from_email = default_from_email
        self.default_from_name = default_from_name
        self.client = SendGridAPIClient(api_key)

    def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send email via SendGrid API.

        Returns:
            Dict with 'success': bool, 'message_id': str, 'error': str (if failed)
        """
        try:
            from_email = from_email or self.default_from_email
            from_name = from_name or self.default_from_name

            message = Mail(
                from_email=Email(from_email, from_name),
                to_emails=To(to_email),
                subject=subject,
                plain_text_content=Content("text/plain", text_body),
                html_content=Content("text/html", html_body)
            )

            response = self.client.send(message)

            if response.status_code in [200, 201, 202]:
                # Extract message ID from headers
                message_id = response.headers.get('X-Message-Id', '')
                logger.info(f"Email sent successfully to {to_email} via SendGrid. Message ID: {message_id}")
                return {
                    'success': True,
                    'message_id': message_id,
                    'status_code': response.status_code
                }
            else:
                logger.error(f"SendGrid returned non-success status: {response.status_code}")
                return {
                    'success': False,
                    'error': f"SendGrid returned status {response.status_code}",
                    'status_code': response.status_code
                }

        except Exception as e:
            logger.error(f"Failed to send email via SendGrid: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def get_message_status(self, message_id: str) -> str:
        """
        Check delivery status via SendGrid API.
        Note: Requires Event Webhook setup for real-time status updates.
        """
        # This would require SendGrid Event Webhook integration
        # For now, return 'sent' as default
        return 'sent'


class SMTPProvider(EmailProvider):
    """SMTP email provider (fallback)."""

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str,
                 default_from_email: str, default_from_name: str = "LSTM Dashboard"):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.default_from_email = default_from_email
        self.default_from_name = default_from_name

    def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send email via SMTP."""
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        try:
            from_email = from_email or self.default_from_email
            from_name = from_name or self.default_from_name

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{from_name} <{from_email}>"
            msg['To'] = to_email

            part1 = MIMEText(text_body, 'plain')
            part2 = MIMEText(html_body, 'html')

            msg.attach(part1)
            msg.attach(part2)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(from_email, to_email, msg.as_string())

            logger.info(f"Email sent successfully to {to_email} via SMTP")
            return {
                'success': True,
                'message_id': f"smtp-{int(time.time())}"
            }

        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def get_message_status(self, message_id: str) -> str:
        """SMTP doesn't provide delivery tracking."""
        return 'sent'


class EmailProviderFactory:
    """Factory for creating email provider instances."""

    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> EmailProvider:
        """
        Create email provider instance.

        Args:
            provider_type: 'sendgrid', 'smtp', etc.
            config: Provider-specific configuration

        Returns:
            EmailProvider instance

        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type == 'sendgrid':
            return SendGridProvider(
                api_key=config['api_key'],
                default_from_email=config['from_email'],
                default_from_name=config.get('from_name', 'LSTM Dashboard')
            )
        elif provider_type == 'smtp':
            return SMTPProvider(
                smtp_host=config['smtp_host'],
                smtp_port=config['smtp_port'],
                username=config['username'],
                password=config['password'],
                default_from_email=config['from_email'],
                default_from_name=config.get('from_name', 'LSTM Dashboard')
            )
        else:
            raise ValueError(f"Unsupported email provider: {provider_type}")


class EmailRateLimiter:
    """
    Rate limiter for email sending (token bucket algorithm).
    Prevents hitting provider rate limits.
    """

    def __init__(self, redis_client, max_emails_per_minute: int = 100):
        """
        Initialize rate limiter.

        Args:
            redis_client: Redis client for distributed rate limiting
            max_emails_per_minute: Maximum emails per minute
        """
        self.redis = redis_client
        self.max_emails_per_minute = max_emails_per_minute
        self.key = "email_rate_limit"

    def can_send(self) -> bool:
        """
        Check if we can send an email without hitting rate limit.

        Returns:
            True if email can be sent, False otherwise
        """
        try:
            current = self.redis.get(self.key)
            if current is None:
                # First email in this minute
                self.redis.setex(self.key, 60, 1)
                return True

            count = int(current)
            if count < self.max_emails_per_minute:
                self.redis.incr(self.key)
                return True

            logger.warning(f"Rate limit exceeded: {count}/{self.max_emails_per_minute} emails this minute")
            return False

        except Exception as e:
            logger.error(f"Rate limiter error (failing open): {e}")
            return True  # Fail open to avoid blocking emails

    def reset(self):
        """Reset rate limit counter (for testing)."""
        try:
            self.redis.delete(self.key)
        except Exception as e:
            logger.error(f"Failed to reset rate limiter: {e}")
