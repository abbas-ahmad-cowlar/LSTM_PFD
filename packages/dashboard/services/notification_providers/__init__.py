"""Notification provider abstraction layer for webhooks (Feature #4)."""
from services.notification_providers.base import NotificationProvider, NotificationMessage
from services.notification_providers.factory import NotificationProviderFactory
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

__all__ = ['NotificationProvider', 'NotificationMessage', 'NotificationProviderFactory']
