"""Notification provider abstraction layer for webhooks (Feature #4)."""
from services.notification_providers.base import NotificationProvider, NotificationMessage
from services.notification_providers.factory import NotificationProviderFactory

__all__ = ['NotificationProvider', 'NotificationMessage', 'NotificationProviderFactory']
