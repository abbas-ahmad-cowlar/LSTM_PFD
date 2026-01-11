"""
Email Digest Service.
Handles business logic for email digest queue management.
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import and_, or_, func
from database.connection import get_db_session
from models.email_digest_queue import EmailDigestQueue
from models.user import User
from utils.logger import setup_logger
from dashboard_config import EMAIL_DIGEST_PAGE_SIZE

logger = setup_logger(__name__)


class EmailDigestService:
    """Service for managing email digest queue operations."""

    @staticmethod
    def get_queue_stats() -> Dict[str, int]:
        """
        Get summary statistics for the digest queue.

        Returns:
            Dict with keys: pending_count, included_count, today_count

        Example:
            >>> stats = EmailDigestService.get_queue_stats()
            >>> print(f"Pending: {stats['pending_count']}")
        """
        try:
            with get_db_session() as session:
                # Count pending items (not included in digest yet)
                pending_count = session.query(EmailDigestQueue)\
                    .filter_by(included_in_digest=False)\
                    .count()

                # Count items already included in digests
                included_count = session.query(EmailDigestQueue)\
                    .filter_by(included_in_digest=True)\
                    .count()

                # Count items scheduled for today
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_end = today_start + timedelta(days=1)

                today_count = session.query(EmailDigestQueue)\
                    .filter(
                        and_(
                            EmailDigestQueue.scheduled_for >= today_start,
                            EmailDigestQueue.scheduled_for < today_end
                        )
                    )\
                    .count()

                return {
                    'pending_count': pending_count,
                    'included_count': included_count,
                    'today_count': today_count
                }

        except Exception as e:
            logger.error(f"Error getting digest queue stats: {e}", exc_info=True)
            return {
                'pending_count': 0,
                'included_count': 0,
                'today_count': 0
            }

    @staticmethod
    def get_pending_digests(
        event_type_filter: str = 'all',
        user_id_filter: Optional[int] = None,
        time_filter: str = 'all',
        page: int = 1,
        page_size: Optional[int] = None
    ) -> Tuple[List[Tuple[EmailDigestQueue, User]], int]:
        """
        Get pending digest items with filtering and pagination.

        Args:
            event_type_filter: Event type to filter by ('all' or specific type)
            user_id_filter: User ID to filter by (None for all users)
            time_filter: Time range filter ('all', 'past_due', 'next_hour', 'next_24h', 'this_week')
            page: Page number (1-indexed)
            page_size: Items per page (defaults to EMAIL_DIGEST_PAGE_SIZE)

        Returns:
            Tuple of (list of (EmailDigestQueue, User) tuples, total_count)

        Raises:
            ValueError: If page or page_size is invalid
        """
        if page < 1:
            raise ValueError("Page must be >= 1")

        page_size = page_size or EMAIL_DIGEST_PAGE_SIZE

        try:
            with get_db_session() as session:
                # Base query - pending items with user info
                query = session.query(EmailDigestQueue, User)\
                    .join(User, EmailDigestQueue.user_id == User.id)\
                    .filter(EmailDigestQueue.included_in_digest == False)

                # Apply event type filter
                if event_type_filter and event_type_filter != 'all':
                    query = query.filter(EmailDigestQueue.event_type == event_type_filter)

                # Apply user filter
                if user_id_filter:
                    query = query.filter(EmailDigestQueue.user_id == user_id_filter)

                # Apply time filter
                now = datetime.utcnow()
                if time_filter == 'past_due':
                    query = query.filter(EmailDigestQueue.scheduled_for < now)
                elif time_filter == 'next_hour':
                    query = query.filter(
                        and_(
                            EmailDigestQueue.scheduled_for >= now,
                            EmailDigestQueue.scheduled_for < now + timedelta(hours=1)
                        )
                    )
                elif time_filter == 'next_24h':
                    query = query.filter(
                        and_(
                            EmailDigestQueue.scheduled_for >= now,
                            EmailDigestQueue.scheduled_for < now + timedelta(days=1)
                        )
                    )
                elif time_filter == 'this_week':
                    week_end = now + timedelta(days=7)
                    query = query.filter(
                        and_(
                            EmailDigestQueue.scheduled_for >= now,
                            EmailDigestQueue.scheduled_for < week_end
                        )
                    )

                # Order by scheduled time (earliest first)
                query = query.order_by(EmailDigestQueue.scheduled_for.asc())

                # Get total count
                total_count = query.count()

                # Pagination
                offset = (page - 1) * page_size
                items = query.limit(page_size).offset(offset).all()

                return items, total_count

        except Exception as e:
            logger.error(f"Error getting pending digests: {e}", exc_info=True)
            raise

    @staticmethod
    def get_digest_history(limit: int = 50) -> List[Tuple[EmailDigestQueue, User]]:
        """
        Get recently processed digest items.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of (EmailDigestQueue, User) tuples
        """
        try:
            with get_db_session() as session:
                history_items = session.query(EmailDigestQueue, User)\
                    .join(User, EmailDigestQueue.user_id == User.id)\
                    .filter(EmailDigestQueue.included_in_digest == True)\
                    .order_by(EmailDigestQueue.updated_at.desc())\
                    .limit(limit)\
                    .all()

                return history_items

        except Exception as e:
            logger.error(f"Error getting digest history: {e}", exc_info=True)
            return []

    @staticmethod
    def get_users_with_digests() -> List[User]:
        """
        Get list of users who have items in the digest queue.

        Returns:
            List of User objects
        """
        try:
            with get_db_session() as session:
                users = session.query(User)\
                    .join(EmailDigestQueue, EmailDigestQueue.user_id == User.id)\
                    .distinct()\
                    .all()

                return users

        except Exception as e:
            logger.error(f"Error getting users with digests: {e}", exc_info=True)
            return []

    @staticmethod
    def trigger_digest_processing():
        """
        Trigger immediate processing of pending digests.

        This attempts to start a Celery task for processing.
        If Celery is not available, logs a warning.

        Returns:
            bool: True if task was triggered, False otherwise
        """
        try:
            from tasks.notification_tasks import process_email_digests
            process_email_digests.delay()
            logger.info("Email digest processing task triggered successfully")
            return True

        except ImportError:
            logger.warning("Celery task not available - digest processing not triggered")
            return False

        except Exception as e:
            logger.error(f"Error triggering digest processing: {e}", exc_info=True)
            return False
