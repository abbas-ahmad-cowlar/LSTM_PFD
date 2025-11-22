"""
Enhanced notification service with email and webhook support (Features #3 and #4).
Handles multi-channel notifications: in-app toasts, emails, Slack, Teams, and custom webhooks.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import traceback

from utils.logger import setup_logger
from database.connection import get_db_session
from models.notification_preference import NotificationPreference, EventType
from models.email_log import EmailLog, EmailStatus
from models.email_digest_queue import EmailDigestQueue
from models.webhook_configuration import WebhookConfiguration
from models.webhook_log import WebhookLog, WebhookStatus
from models.user import User
from services.email_provider import EmailProviderFactory, EmailRateLimiter
from services.notification_providers import NotificationProviderFactory, NotificationMessage
from config import DASH_APP_DIR
import redis

logger = setup_logger(__name__)


class NotificationService:
    """Service for handling multi-channel user notifications."""

    # Class-level email provider and rate limiter (initialized once)
    _email_provider = None
    _rate_limiter = None
    _jinja_env = None

    @classmethod
    def initialize(cls, email_config: Dict[str, Any], redis_client):
        """
        Initialize email provider and rate limiter.
        Should be called once at application startup.

        Args:
            email_config: Email provider configuration
            redis_client: Redis client for rate limiting
        """
        try:
            # Initialize email provider
            cls._email_provider = EmailProviderFactory.create_provider(
                provider_type=email_config.get('provider', 'sendgrid'),
                config=email_config
            )
            logger.info(f"Email provider initialized: {email_config.get('provider')}")

            # Initialize rate limiter
            cls._rate_limiter = EmailRateLimiter(
                redis_client=redis_client,
                max_emails_per_minute=email_config.get('rate_limit', 100)
            )
            logger.info("Email rate limiter initialized")

            # Initialize Jinja2 template environment
            template_dir = DASH_APP_DIR / 'templates'
            cls._jinja_env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(['html', 'xml'])
            )
            logger.info(f"Jinja2 environment initialized with template dir: {template_dir}")

        except Exception as e:
            logger.error(f"Failed to initialize NotificationService: {e}", exc_info=True)
            raise

    @staticmethod
    def emit_event(event_type: str, user_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for emitting notification events.
        Routes to appropriate channels based on user preferences.

        Args:
            event_type: Event type (e.g., 'training.complete')
            user_id: User ID to notify
            data: Event data (experiment details, metrics, etc.)

        Returns:
            Dict with notification results
        """
        try:
            logger.info(f"Emitting event '{event_type}' for user {user_id}")

            results = {
                'event_type': event_type,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'channels': {}
            }

            # Get user preferences
            with get_db_session() as session:
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    logger.warning(f"User {user_id} not found")
                    return {'error': 'User not found'}

                preference = session.query(NotificationPreference).filter_by(
                    user_id=user_id,
                    event_type=event_type
                ).first()

                # If no preference exists, check if it's a high-priority event
                if not preference:
                    # High-priority events default to enabled
                    high_priority_events = [
                        EventType.TRAINING_COMPLETE,
                        EventType.TRAINING_FAILED,
                        EventType.HPO_CAMPAIGN_COMPLETE,
                        EventType.HPO_CAMPAIGN_FAILED
                    ]
                    if event_type not in high_priority_events:
                        logger.info(f"No preference for event '{event_type}', skipping notification")
                        return results

                    # Create default preference on-the-fly
                    email_enabled = True
                    email_frequency = 'immediate'
                else:
                    email_enabled = preference.email_enabled
                    email_frequency = preference.email_frequency

                # Send email if enabled
                if email_enabled and NotificationService._email_provider:
                    if email_frequency == 'immediate':
                        email_result = NotificationService._send_email_immediate(
                            user=user,
                            event_type=event_type,
                            data=data,
                            session=session
                        )
                        results['channels']['email'] = email_result
                    elif email_frequency in ['digest_daily', 'digest_weekly']:
                        # Add to digest queue
                        digest_result = NotificationService._queue_for_digest(
                            user_id=user_id,
                            event_type=event_type,
                            data=data,
                            frequency=email_frequency,
                            session=session
                        )
                        results['channels']['email_digest'] = digest_result

                # Send webhooks if configured (Feature #4)
                webhook_results = NotificationService._send_webhook_notifications(
                    user_id=user_id,
                    event_type=event_type,
                    data=data,
                    session=session
                )
                if webhook_results:
                    results['channels']['webhooks'] = webhook_results

            return results

        except Exception as e:
            logger.error(f"Failed to emit event '{event_type}': {e}", exc_info=True)
            return {'error': str(e)}

    @staticmethod
    def _send_email_immediate(user: User, event_type: str, data: Dict[str, Any], session) -> Dict[str, Any]:
        """
        Send immediate email notification.

        Args:
            user: User object
            event_type: Event type
            data: Event data
            session: Database session

        Returns:
            Result dict with success status
        """
        try:
            # Check rate limit
            if not NotificationService._rate_limiter.can_send():
                logger.warning(f"Rate limit exceeded, queueing email for user {user.id}")
                return {'status': 'rate_limited', 'queued': True}

            # Select template based on event type
            template_map = {
                EventType.TRAINING_COMPLETE: 'email_templates/training_complete.html',
                EventType.TRAINING_FAILED: 'email_templates/training_failed.html',
                EventType.HPO_CAMPAIGN_COMPLETE: 'email_templates/hpo_campaign_complete.html',
                EventType.HPO_CAMPAIGN_FAILED: 'email_templates/training_failed.html',  # Reuse template
            }

            template_name = template_map.get(event_type)
            if not template_name:
                logger.warning(f"No email template for event type '{event_type}'")
                return {'status': 'no_template'}

            # Render email
            html_body, text_body, subject = NotificationService._render_email_template(
                template_name=template_name,
                event_type=event_type,
                data=data,
                user=user
            )

            # Send via provider
            send_result = NotificationService._email_provider.send(
                to_email=user.email,
                subject=subject,
                html_body=html_body,
                text_body=text_body
            )

            # Log to database
            email_log = EmailLog(
                user_id=user.id,
                recipient_email=user.email,
                event_type=event_type,
                subject=subject,
                template_name=template_name,
                event_data=data,
                provider='sendgrid',  # TODO: Make dynamic
                message_id=send_result.get('message_id', ''),
                status=EmailStatus.SENT if send_result['success'] else EmailStatus.FAILED,
                error_message=send_result.get('error'),
                sent_at=datetime.now() if send_result['success'] else None
            )
            session.add(email_log)
            session.commit()

            logger.info(f"Email sent to {user.email} for event '{event_type}'")
            return {'status': 'sent', 'success': send_result['success']}

        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    @staticmethod
    def _render_email_template(template_name: str, event_type: str, data: Dict[str, Any], user: User) -> tuple:
        """
        Render email template with context data.

        Args:
            template_name: Template file name
            event_type: Event type
            data: Event data
            user: User object

        Returns:
            Tuple of (html_body, text_body, subject)
        """
        try:
            # Prepare template context
            context = {
                'user_first_name': user.username.split()[0] if user.username else '',
                'dashboard_url': data.get('dashboard_url', 'http://localhost:8050'),
                'settings_url': f"{data.get('dashboard_url', 'http://localhost:8050')}/settings",
                'help_url': f"{data.get('dashboard_url', 'http://localhost:8050')}/help",
                'unsubscribe_url': f"{data.get('dashboard_url', 'http://localhost:8050')}/settings/notifications?unsubscribe={event_type}",
                'current_year': datetime.now().year,
                **data  # Merge event data
            }

            # Format metrics as percentages if needed
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'best_accuracy']:
                if key in context and isinstance(context[key], (int, float)):
                    if context[key] <= 1.0:  # If it's a decimal (0.96), convert to percentage
                        context[key] = f"{context[key] * 100:.1f}"
                    else:
                        context[key] = f"{context[key]:.1f}"

            # Render HTML
            template = NotificationService._jinja_env.get_template(template_name)
            html_body = template.render(context)

            # Generate plain text version (strip HTML tags)
            import re
            text_body = re.sub('<[^<]+?>', '', html_body)
            text_body = re.sub(r'\n\s*\n', '\n\n', text_body)

            # Generate subject
            subject_map = {
                EventType.TRAINING_COMPLETE: f"✅ Training Complete - {data.get('experiment_name', 'Your Model')}",
                EventType.TRAINING_FAILED: f"⚠️ Training Failed - {data.get('experiment_name', 'Your Model')}",
                EventType.HPO_CAMPAIGN_COMPLETE: f"🏆 HPO Campaign Complete - {data.get('campaign_name', 'Your Campaign')}",
                EventType.HPO_CAMPAIGN_FAILED: f"⚠️ HPO Campaign Failed - {data.get('campaign_name', 'Your Campaign')}",
            }
            subject = subject_map.get(event_type, "LSTM Dashboard Notification")

            return html_body, text_body, subject

        except Exception as e:
            logger.error(f"Failed to render email template: {e}", exc_info=True)
            raise

    @staticmethod
    def _queue_for_digest(user_id: int, event_type: str, data: Dict[str, Any], frequency: str, session) -> Dict[str, Any]:
        """
        Queue notification for digest email.

        Args:
            user_id: User ID
            event_type: Event type
            data: Event data
            frequency: Digest frequency ('digest_daily' or 'digest_weekly')
            session: Database session

        Returns:
            Result dict
        """
        try:
            # Calculate scheduled time
            now = datetime.now()
            if frequency == 'digest_daily':
                scheduled_for = now.replace(hour=9, minute=0, second=0, microsecond=0)
                if scheduled_for <= now:
                    scheduled_for += timedelta(days=1)
            else:  # digest_weekly
                days_until_monday = (7 - now.weekday()) % 7
                if days_until_monday == 0:
                    days_until_monday = 7
                scheduled_for = (now + timedelta(days=days_until_monday)).replace(hour=9, minute=0, second=0, microsecond=0)

            # Add to queue
            queue_item = EmailDigestQueue(
                user_id=user_id,
                event_type=event_type,
                event_data=data,
                scheduled_for=scheduled_for
            )
            session.add(queue_item)
            session.commit()

            logger.info(f"Event queued for {frequency} digest, scheduled for {scheduled_for}")
            return {'status': 'queued', 'scheduled_for': scheduled_for.isoformat()}

        except Exception as e:
            logger.error(f"Failed to queue digest: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    @staticmethod
    def _send_webhook_notifications(user_id: int, event_type: str, data: Dict[str, Any], session) -> List[Dict[str, Any]]:
        """
        Send webhook notifications to all configured webhooks for this user and event type.

        Args:
            user_id: User ID
            event_type: Event type
            data: Event data
            session: Database session

        Returns:
            List of results for each webhook
        """
        try:
            from config import (
                NOTIFICATIONS_SLACK_ENABLED,
                NOTIFICATIONS_TEAMS_ENABLED,
                NOTIFICATIONS_WEBHOOK_ENABLED,
                SLACK_RATE_LIMIT_PER_WEBHOOK,
                SLACK_RETRY_ATTEMPTS,
                SLACK_TIMEOUT_SECONDS,
                TEAMS_RATE_LIMIT_PER_WEBHOOK,
                TEAMS_RETRY_ATTEMPTS,
                TEAMS_TIMEOUT_SECONDS,
                WEBHOOK_CUSTOM_TIMEOUT_SECONDS,
                WEBHOOK_CUSTOM_RETRY_ATTEMPTS
            )

            # Check if any webhook providers are enabled globally
            if not any([NOTIFICATIONS_SLACK_ENABLED, NOTIFICATIONS_TEAMS_ENABLED, NOTIFICATIONS_WEBHOOK_ENABLED]):
                logger.debug("All webhook providers disabled globally")
                return []

            # Query active webhooks for this user
            webhooks = session.query(WebhookConfiguration).filter(
                WebhookConfiguration.user_id == user_id,
                WebhookConfiguration.is_active == True
            ).all()

            if not webhooks:
                logger.debug(f"No active webhooks configured for user {user_id}")
                return []

            results = []

            for webhook_config in webhooks:
                # Check if this event is enabled for this webhook
                if event_type not in webhook_config.enabled_events:
                    logger.debug(f"Event '{event_type}' not enabled for webhook {webhook_config.id}")
                    continue

                # Check if provider is globally enabled
                if webhook_config.provider_type == 'slack' and not NOTIFICATIONS_SLACK_ENABLED:
                    logger.debug(f"Slack disabled globally, skipping webhook {webhook_config.id}")
                    continue
                elif webhook_config.provider_type == 'teams' and not NOTIFICATIONS_TEAMS_ENABLED:
                    logger.debug(f"Teams disabled globally, skipping webhook {webhook_config.id}")
                    continue
                elif webhook_config.provider_type == 'webhook' and not NOTIFICATIONS_WEBHOOK_ENABLED:
                    logger.debug(f"Custom webhooks disabled globally, skipping webhook {webhook_config.id}")
                    continue

                # Send webhook
                result = NotificationService._send_single_webhook(
                    webhook_config=webhook_config,
                    event_type=event_type,
                    data=data,
                    session=session
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to send webhook notifications: {e}", exc_info=True)
            return [{'error': str(e)}]

    @staticmethod
    def _send_single_webhook(webhook_config: WebhookConfiguration, event_type: str, data: Dict[str, Any], session) -> Dict[str, Any]:
        """
        Send notification to a single webhook.

        Args:
            webhook_config: Webhook configuration
            event_type: Event type
            data: Event data
            session: Database session

        Returns:
            Result dict with status
        """
        from config import (
            SLACK_RATE_LIMIT_PER_WEBHOOK,
            SLACK_RETRY_ATTEMPTS,
            SLACK_TIMEOUT_SECONDS,
            TEAMS_RATE_LIMIT_PER_WEBHOOK,
            TEAMS_RETRY_ATTEMPTS,
            TEAMS_TIMEOUT_SECONDS,
            WEBHOOK_CUSTOM_TIMEOUT_SECONDS,
            WEBHOOK_CUSTOM_RETRY_ATTEMPTS
        )

        try:
            # Get provider config
            provider_config = {}
            if webhook_config.provider_type == 'slack':
                provider_config = {
                    'rate_limit_per_webhook': SLACK_RATE_LIMIT_PER_WEBHOOK,
                    'retry_attempts': SLACK_RETRY_ATTEMPTS,
                    'timeout_seconds': SLACK_TIMEOUT_SECONDS
                }
            elif webhook_config.provider_type == 'teams':
                provider_config = {
                    'rate_limit_per_webhook': TEAMS_RATE_LIMIT_PER_WEBHOOK,
                    'retry_attempts': TEAMS_RETRY_ATTEMPTS,
                    'timeout_seconds': TEAMS_TIMEOUT_SECONDS
                }
            elif webhook_config.provider_type == 'webhook':
                provider_config = {
                    'timeout_seconds': WEBHOOK_CUSTOM_TIMEOUT_SECONDS,
                    'retry_attempts': WEBHOOK_CUSTOM_RETRY_ATTEMPTS
                }

            # Get provider instance
            provider = NotificationProviderFactory.get_provider(
                provider_type=webhook_config.provider_type,
                config=provider_config
            )

            # Build notification message
            message = NotificationService._build_notification_message(
                event_type=event_type,
                data=data,
                settings=webhook_config.settings
            )

            # Send notification
            success = provider.send(webhook_config.webhook_url, message)

            # Log result
            webhook_log = WebhookLog(
                webhook_config_id=webhook_config.id,
                user_id=webhook_config.user_id,
                event_type=event_type,
                provider_type=webhook_config.provider_type,
                webhook_url=webhook_config.webhook_url,
                payload=message.__dict__,
                status=WebhookStatus.SENT if success else WebhookStatus.FAILED,
                http_status_code=200 if success else None,
                sent_at=datetime.now() if success else None
            )
            session.add(webhook_log)

            # Update webhook config status
            if success:
                webhook_config.last_used_at = datetime.now()
                webhook_config.consecutive_failures = 0
                webhook_config.last_error = None
            else:
                webhook_config.consecutive_failures += 1
                webhook_config.last_error = "Send failed"

                # Auto-disable after 10 consecutive failures
                if webhook_config.consecutive_failures >= 10:
                    webhook_config.is_active = False
                    logger.warning(f"Webhook {webhook_config.id} auto-disabled after 10 failures")

            session.commit()

            logger.info(f"Webhook notification sent to {webhook_config.provider_type}: {success}")
            return {
                'webhook_id': webhook_config.id,
                'provider': webhook_config.provider_type,
                'success': success
            }

        except Exception as e:
            logger.error(f"Failed to send webhook {webhook_config.id}: {e}", exc_info=True)

            # Log error
            webhook_log = WebhookLog(
                webhook_config_id=webhook_config.id,
                user_id=webhook_config.user_id,
                event_type=event_type,
                provider_type=webhook_config.provider_type,
                webhook_url=webhook_config.webhook_url,
                status=WebhookStatus.FAILED,
                error_message=str(e)
            )
            session.add(webhook_log)

            # Update webhook config
            webhook_config.consecutive_failures += 1
            webhook_config.last_error = str(e)
            if webhook_config.consecutive_failures >= 10:
                webhook_config.is_active = False

            session.commit()

            return {
                'webhook_id': webhook_config.id,
                'provider': webhook_config.provider_type,
                'success': False,
                'error': str(e)
            }

    @staticmethod
    def _build_notification_message(event_type: str, data: Dict[str, Any], settings: Dict[str, Any]) -> NotificationMessage:
        """
        Build NotificationMessage from event data.

        Args:
            event_type: Event type
            data: Event data
            settings: Webhook-specific settings

        Returns:
            NotificationMessage instance
        """
        # Map event types to titles and priorities
        event_config = {
            EventType.TRAINING_COMPLETE: {
                'title': f"Training Complete - {data.get('experiment_name', 'Experiment')}",
                'priority': 'medium',
                'emoji': '🎉'
            },
            EventType.TRAINING_FAILED: {
                'title': f"Training Failed - {data.get('experiment_name', 'Experiment')}",
                'priority': 'high',
                'emoji': '⚠️'
            },
            EventType.TRAINING_STARTED: {
                'title': f"Training Started - {data.get('experiment_name', 'Experiment')}",
                'priority': 'low',
                'emoji': '🚀'
            },
            EventType.HPO_CAMPAIGN_COMPLETE: {
                'title': f"HPO Campaign Complete - {data.get('campaign_name', 'Campaign')}",
                'priority': 'high',
                'emoji': '🏆'
            },
            EventType.HPO_CAMPAIGN_FAILED: {
                'title': f"HPO Campaign Failed - {data.get('campaign_name', 'Campaign')}",
                'priority': 'high',
                'emoji': '⚠️'
            }
        }

        config = event_config.get(event_type, {
            'title': 'Notification',
            'priority': 'medium',
            'emoji': '📢'
        })

        # Build actions (buttons)
        actions = []
        if 'dashboard_url' in data:
            actions.append({
                'label': 'View Results',
                'url': data['dashboard_url'],
                'style': 'primary'
            })

        # Build message
        message = NotificationMessage(
            title=config['title'],
            body=data.get('message', ''),
            event_type=event_type,
            priority=config['priority'],
            data=data,
            actions=actions if actions else None
        )

        return message

    # Legacy toast notification methods (backward compatibility)
    @staticmethod
    def create_toast(
        message: str,
        notification_type: str = "info",
        duration: int = 5000,
        link: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create toast notification (legacy method)."""
        return {
            "message": message,
            "type": notification_type,
            "duration": duration,
            "link": link,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def success(message: str, link: Optional[str] = None):
        """Create success notification."""
        return NotificationService.create_toast(message, "success", link=link)

    @staticmethod
    def error(message: str):
        """Create error notification."""
        return NotificationService.create_toast(message, "error", duration=10000)

    @staticmethod
    def warning(message: str):
        """Create warning notification."""
        return NotificationService.create_toast(message, "warning", duration=7000)

    @staticmethod
    def info(message: str, link: Optional[str] = None):
        """Create info notification."""
        return NotificationService.create_toast(message, "info", link=link)


def create_default_notification_preferences(user_id: int):
    """
    Create default notification preferences for a new user.
    Should be called during user registration.

    Args:
        user_id: User ID
    """
    try:
        with get_db_session() as session:
            defaults = EventType.get_default_preferences()

            for event_type, email_enabled, frequency in defaults:
                preference = NotificationPreference(
                    user_id=user_id,
                    event_type=event_type,
                    email_enabled=email_enabled,
                    email_frequency=frequency
                )
                session.add(preference)

            session.commit()
            logger.info(f"Created default notification preferences for user {user_id}")

    except Exception as e:
        logger.error(f"Failed to create default preferences: {e}", exc_info=True)
        raise


def get_error_suggestion(error_message: str) -> str:
    """
    Provide actionable suggestion based on error type.

    Args:
        error_message: Error message string

    Returns:
        Suggestion string
    """
    error_lower = error_message.lower()

    if 'cuda' in error_lower and 'out of memory' in error_lower:
        return "Reduce batch size (try 16 or 8) or use a smaller model variant"
    elif 'nan' in error_lower or 'inf' in error_lower:
        return "Learning rate may be too high. Try reducing to 1e-4 or enable gradient clipping"
    elif 'file' in error_lower and ('not found' in error_lower or 'no such file' in error_lower):
        return "Required file is missing. Check dataset availability and file paths"
    elif 'timeout' in error_lower:
        return "Training exceeded time limit. Reduce epochs or enable early stopping"
    elif 'connection' in error_lower:
        return "Database or network connection issue. Check connectivity and retry"
    else:
        return "Review error logs for detailed stack traces. Contact support if issue persists"
