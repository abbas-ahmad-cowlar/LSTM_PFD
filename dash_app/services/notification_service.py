"""
Notification service for user notifications.
"""
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class NotificationService:
    """Service for creating user notifications."""

    @staticmethod
    def create_toast(
        message: str,
        notification_type: str = "info",
        duration: int = 5000,
        link: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create toast notification.

        Args:
            message: Notification message
            notification_type: Type (info, success, warning, error)
            duration: Display duration in milliseconds
            link: Optional link to navigate to

        Returns:
            Toast notification dict
        """
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


from datetime import datetime
from typing import Dict, Any
