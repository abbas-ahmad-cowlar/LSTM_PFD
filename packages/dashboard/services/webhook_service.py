"""
Webhook Configuration Service (Feature #4).
Handles CRUD operations for webhook configurations.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests
from sqlalchemy import and_

from database.connection import get_db_session
from models.webhook_configuration import WebhookConfiguration
from models.webhook_log import WebhookLog, WebhookStatus
from services.notification_providers import NotificationProviderFactory, NotificationMessage
from utils.logger import setup_logger

logger = setup_logger(__name__)


class WebhookService:
    """Service for managing webhook configurations."""

    @staticmethod
    def list_user_webhooks(user_id: int, include_inactive: bool = False) -> List[WebhookConfiguration]:
        """
        List all webhooks for a user.

        Args:
            user_id: User ID
            include_inactive: Include inactive webhooks

        Returns:
            List of webhook configurations
        """
        with get_db_session() as session:
            query = session.query(WebhookConfiguration).filter(
                WebhookConfiguration.user_id == user_id
            )

            if not include_inactive:
                query = query.filter(WebhookConfiguration.is_active == True)

            # Apply pagination to prevent loading too many webhooks
            from utils.query_utils import paginate_with_default_limit
            webhooks = paginate_with_default_limit(
                query.order_by(WebhookConfiguration.created_at.desc()),
                limit=100
            )

            # Detach from session
            session.expunge_all()
            return webhooks

    @staticmethod
    def get_webhook(webhook_id: int, user_id: int) -> Optional[WebhookConfiguration]:
        """
        Get a specific webhook by ID (with user ownership check).

        Args:
            webhook_id: Webhook ID
            user_id: User ID for ownership verification

        Returns:
            Webhook configuration or None
        """
        with get_db_session() as session:
            webhook = session.query(WebhookConfiguration).filter(
                and_(
                    WebhookConfiguration.id == webhook_id,
                    WebhookConfiguration.user_id == user_id
                )
            ).first()

            if webhook:
                session.expunge(webhook)
            return webhook

    @staticmethod
    def create_webhook(
        user_id: int,
        provider_type: str,
        webhook_url: str,
        name: str,
        enabled_events: List[str],
        description: Optional[str] = None,
        is_active: bool = True,
        settings: Optional[Dict[str, Any]] = None
    ) -> WebhookConfiguration:
        """
        Create a new webhook configuration.

        Args:
            user_id: User ID
            provider_type: 'slack', 'teams', or 'webhook'
            webhook_url: Webhook URL
            name: Webhook name
            enabled_events: List of event types
            description: Optional description
            is_active: Whether webhook is active
            settings: Provider-specific settings

        Returns:
            Created webhook configuration

        Raises:
            ValueError: If validation fails
        """
        # Validate provider type
        valid_providers = ['slack', 'teams', 'webhook']
        if provider_type not in valid_providers:
            raise ValueError(f"Invalid provider type. Must be one of: {valid_providers}")

        # Validate URL
        if not webhook_url or not webhook_url.startswith('http'):
            raise ValueError("Invalid webhook URL")

        # Validate events
        if not enabled_events or not isinstance(enabled_events, list):
            raise ValueError("At least one event must be selected")

        with get_db_session() as session:
            # Check for duplicate URL
            existing = session.query(WebhookConfiguration).filter(
                and_(
                    WebhookConfiguration.user_id == user_id,
                    WebhookConfiguration.webhook_url == webhook_url
                )
            ).first()

            if existing:
                raise ValueError("A webhook with this URL already exists")

            webhook = WebhookConfiguration(
                user_id=user_id,
                provider_type=provider_type,
                webhook_url=webhook_url,
                name=name,
                description=description,
                is_active=is_active,
                enabled_events=enabled_events,
                settings=settings or {},
                consecutive_failures=0
            )

            session.add(webhook)
            session.commit()
            session.refresh(webhook)

            webhook_id = webhook.id
            session.expunge(webhook)

            logger.info(f"Created webhook {webhook_id} for user {user_id}")
            return webhook

    @staticmethod
    def update_webhook(
        webhook_id: int,
        user_id: int,
        **updates
    ) -> Optional[WebhookConfiguration]:
        """
        Update a webhook configuration.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for ownership verification
            **updates: Fields to update

        Returns:
            Updated webhook or None if not found

        Raises:
            ValueError: If validation fails
        """
        with get_db_session() as session:
            webhook = session.query(WebhookConfiguration).filter(
                and_(
                    WebhookConfiguration.id == webhook_id,
                    WebhookConfiguration.user_id == user_id
                )
            ).first()

            if not webhook:
                return None

            # Validate and apply updates
            allowed_fields = [
                'name', 'description', 'webhook_url', 'provider_type',
                'enabled_events', 'is_active', 'settings'
            ]

            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(webhook, field, value)

            session.commit()
            session.refresh(webhook)
            session.expunge(webhook)

            logger.info(f"Updated webhook {webhook_id}")
            return webhook

    @staticmethod
    def delete_webhook(webhook_id: int, user_id: int) -> bool:
        """
        Delete a webhook configuration.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for ownership verification

        Returns:
            True if deleted, False if not found
        """
        with get_db_session() as session:
            webhook = session.query(WebhookConfiguration).filter(
                and_(
                    WebhookConfiguration.id == webhook_id,
                    WebhookConfiguration.user_id == user_id
                )
            ).first()

            if not webhook:
                return False

            session.delete(webhook)
            session.commit()

            logger.info(f"Deleted webhook {webhook_id}")
            return True

    @staticmethod
    def toggle_webhook(webhook_id: int, user_id: int, is_active: bool) -> bool:
        """
        Enable or disable a webhook.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for ownership verification
            is_active: New active status

        Returns:
            True if updated, False if not found
        """
        with get_db_session() as session:
            webhook = session.query(WebhookConfiguration).filter(
                and_(
                    WebhookConfiguration.id == webhook_id,
                    WebhookConfiguration.user_id == user_id
                )
            ).first()

            if not webhook:
                return False

            webhook.is_active = is_active
            session.commit()

            logger.info(f"Toggled webhook {webhook_id} to {'active' if is_active else 'inactive'}")
            return True

    @staticmethod
    def test_webhook(webhook_id: int, user_id: int) -> Dict[str, Any]:
        """
        Send a test notification to a webhook.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for ownership verification

        Returns:
            Dict with success status and message
        """
        webhook = WebhookService.get_webhook(webhook_id, user_id)
        if not webhook:
            return {
                'success': False,
                'message': 'Webhook not found'
            }

        try:
            # Create test notification message
            test_message = NotificationMessage(
                title="🧪 Test Notification",
                body="This is a test notification from LSTM_PFD. Your webhook is configured correctly!",
                color="info",
                fields={
                    "Event Type": "test.webhook",
                    "Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "Webhook ID": str(webhook_id)
                }
            )

            # Get appropriate notifier
            notifier = NotificationProviderFactory.create_notifier(
                provider_type=webhook.provider_type,
                webhook_url=webhook.webhook_url,
                settings=webhook.settings or {}
            )

            # Send test notification
            notifier.send(test_message)

            # Log success
            with get_db_session() as session:
                log = WebhookLog(
                    webhook_config_id=webhook_id,
                    user_id=user_id,
                    event_type='test.webhook',
                    provider_type=webhook.provider_type,
                    webhook_url=webhook.webhook_url,
                    status=WebhookStatus.SENT,
                    http_status_code=200,
                    sent_at=datetime.utcnow()
                )
                session.add(log)

                # Update webhook last_used_at
                webhook_obj = session.query(WebhookConfiguration).filter(
                    WebhookConfiguration.id == webhook_id
                ).first()
                if webhook_obj:
                    webhook_obj.last_used_at = datetime.utcnow()
                    webhook_obj.consecutive_failures = 0

                session.commit()

            logger.info(f"Test webhook sent successfully for webhook {webhook_id}")
            return {
                'success': True,
                'message': 'Test notification sent successfully! Check your channel.'
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to send test webhook {webhook_id}: {error_msg}", exc_info=True)

            # Log failure
            with get_db_session() as session:
                log = WebhookLog(
                    webhook_config_id=webhook_id,
                    user_id=user_id,
                    event_type='test.webhook',
                    provider_type=webhook.provider_type,
                    webhook_url=webhook.webhook_url,
                    status=WebhookStatus.FAILED,
                    error_message=error_msg
                )
                session.add(log)

                # Update consecutive failures
                webhook_obj = session.query(WebhookConfiguration).filter(
                    WebhookConfiguration.id == webhook_id
                ).first()
                if webhook_obj:
                    webhook_obj.consecutive_failures += 1
                    webhook_obj.last_error = error_msg

                session.commit()

            return {
                'success': False,
                'message': f'Test failed: {error_msg}'
            }

    @staticmethod
    def get_webhook_logs(webhook_id: int, user_id: int, limit: int = 10) -> List[WebhookLog]:
        """
        Get recent delivery logs for a webhook.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for ownership verification
            limit: Maximum number of logs to return

        Returns:
            List of webhook logs
        """
        # Verify ownership
        webhook = WebhookService.get_webhook(webhook_id, user_id)
        if not webhook:
            return []

        with get_db_session() as session:
            logs = session.query(WebhookLog).filter(
                WebhookLog.webhook_config_id == webhook_id
            ).order_by(
                WebhookLog.created_at.desc()
            ).limit(limit).all()

            session.expunge_all()
            return logs

    @staticmethod
    def get_webhook_stats(webhook_id: int, user_id: int) -> Dict[str, Any]:
        """
        Get statistics for a webhook.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for ownership verification

        Returns:
            Dict with webhook statistics
        """
        webhook = WebhookService.get_webhook(webhook_id, user_id)
        if not webhook:
            return {}

        with get_db_session() as session:
            # Total deliveries
            total = session.query(WebhookLog).filter(
                WebhookLog.webhook_config_id == webhook_id
            ).count()

            # Successful deliveries
            successful = session.query(WebhookLog).filter(
                and_(
                    WebhookLog.webhook_config_id == webhook_id,
                    WebhookLog.status == WebhookStatus.SENT
                )
            ).count()

            # Failed deliveries
            failed = session.query(WebhookLog).filter(
                and_(
                    WebhookLog.webhook_config_id == webhook_id,
                    WebhookLog.status == WebhookStatus.FAILED
                )
            ).count()

            # Last 24 hours
            day_ago = datetime.utcnow() - timedelta(days=1)
            recent = session.query(WebhookLog).filter(
                and_(
                    WebhookLog.webhook_config_id == webhook_id,
                    WebhookLog.created_at >= day_ago
                )
            ).count()

            return {
                'total_deliveries': total,
                'successful': successful,
                'failed': failed,
                'success_rate': round((successful / total * 100) if total > 0 else 0, 1),
                'last_24h': recent,
                'consecutive_failures': webhook.consecutive_failures
            }
