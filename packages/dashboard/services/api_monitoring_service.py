"""
API Monitoring Service.
Business logic for API request tracking and analytics.
"""
from typing import Dict, List, Optional, Any
from database.connection import get_db_session
from models.api_request_log import APIRequestLog, APIMetricsSummary
from models.api_key import APIKey
from utils.logger import setup_logger
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import func, and_

logger = setup_logger(__name__)


class APIMonitoringService:
    """Service for API monitoring and analytics."""

    @staticmethod
    def log_request(
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        ip_address: str,
        user_agent: str = None,
        api_key_id: int = None,
        request_size: int = 0,
        response_size: int = 0,
        error_message: str = None,
        request_sample: dict = None,
        response_sample: dict = None
    ) -> bool:
        """
        Log an API request.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            ip_address: Client IP address
            user_agent: User agent string
            api_key_id: API key ID if authenticated
            request_size: Request size in bytes
            response_size: Response size in bytes
            error_message: Error message if failed
            request_sample: Sample of request payload
            response_sample: Sample of response payload

        Returns:
            True if logged successfully
        """
        try:
            with get_db_session() as session:
                log = APIRequestLog(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    api_key_id=api_key_id,
                    request_size_bytes=request_size,
                    response_size_bytes=response_size,
                    error_message=error_message,
                    request_payload_sample=request_sample,
                    response_payload_sample=response_sample
                )
                session.add(log)
                session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to log API request: {e}", exc_info=True)
            return False

    @staticmethod
    def get_recent_requests(limit: int = 100, hours: int = 24) -> List[Dict]:
        """
        Get recent API requests.

        Args:
            limit: Maximum number of requests to return
            hours: Number of hours to look back

        Returns:
            List of request dictionaries
        """
        try:
            with get_db_session() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                requests = session.query(APIRequestLog).filter(
                    APIRequestLog.request_time >= cutoff
                ).order_by(
                    APIRequestLog.request_time.desc()
                ).limit(limit).all()

                return [
                    {
                        'id': req.id,
                        'endpoint': req.endpoint,
                        'method': req.method,
                        'status_code': req.status_code,
                        'response_time_ms': req.response_time_ms,
                        'ip_address': req.ip_address,
                        'request_time': req.request_time.isoformat(),
                        'error_message': req.error_message
                    }
                    for req in requests
                ]

        except Exception as e:
            logger.error(f"Failed to get recent requests: {e}", exc_info=True)
            return []

    @staticmethod
    def get_request_stats(hours: int = 24) -> Dict[str, Any]:
        """
        Get aggregate request statistics.

        Args:
            hours: Number of hours to analyze

        Returns:
            Statistics dictionary
        """
        try:
            with get_db_session() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                # Total requests
                total = session.query(func.count(APIRequestLog.id)).filter(
                    APIRequestLog.request_time >= cutoff
                ).scalar() or 0

                # Successful requests (2xx status codes)
                successful = session.query(func.count(APIRequestLog.id)).filter(
                    and_(
                        APIRequestLog.request_time >= cutoff,
                        APIRequestLog.status_code >= 200,
                        APIRequestLog.status_code < 300
                    )
                ).scalar() or 0

                # Failed requests (4xx, 5xx status codes)
                failed = session.query(func.count(APIRequestLog.id)).filter(
                    and_(
                        APIRequestLog.request_time >= cutoff,
                        APIRequestLog.status_code >= 400
                    )
                ).scalar() or 0

                # Average response time
                avg_time = session.query(func.avg(APIRequestLog.response_time_ms)).filter(
                    APIRequestLog.request_time >= cutoff
                ).scalar() or 0

                # Active API keys
                active_keys = session.query(func.count(func.distinct(APIRequestLog.api_key_id))).filter(
                    and_(
                        APIRequestLog.request_time >= cutoff,
                        APIRequestLog.api_key_id.isnot(None)
                    )
                ).scalar() or 0

                # Error rate
                error_rate = (failed / total * 100) if total > 0 else 0

                return {
                    'total_requests': total,
                    'successful_requests': successful,
                    'failed_requests': failed,
                    'avg_response_time_ms': round(avg_time, 2),
                    'active_api_keys': active_keys,
                    'error_rate': round(error_rate, 2),
                    'period_hours': hours
                }

        except Exception as e:
            logger.error(f"Failed to get request stats: {e}", exc_info=True)
            return {}

    @staticmethod
    def get_endpoint_metrics(hours: int = 24) -> List[Dict]:
        """
        Get per-endpoint metrics.

        Args:
            hours: Number of hours to analyze

        Returns:
            List of endpoint metrics
        """
        try:
            with get_db_session() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                # Query grouped by endpoint
                results = session.query(
                    APIRequestLog.endpoint,
                    func.count(APIRequestLog.id).label('total_requests'),
                    func.avg(APIRequestLog.response_time_ms).label('avg_time'),
                    func.count(APIRequestLog.id).filter(APIRequestLog.status_code >= 400).label('errors')
                ).filter(
                    APIRequestLog.request_time >= cutoff
                ).group_by(
                    APIRequestLog.endpoint
                ).all()

                metrics = []
                for result in results:
                    error_rate = (result.errors / result.total_requests * 100) if result.total_requests > 0 else 0
                    metrics.append({
                        'endpoint': result.endpoint,
                        'total_requests': result.total_requests,
                        'avg_response_time_ms': round(result.avg_time, 2) if result.avg_time else 0,
                        'errors': result.errors,
                        'error_rate': round(error_rate, 2)
                    })

                return sorted(metrics, key=lambda x: -x['total_requests'])

        except Exception as e:
            logger.error(f"Failed to get endpoint metrics: {e}", exc_info=True)
            return []

    @staticmethod
    def get_latency_percentiles(hours: int = 24) -> Dict[str, float]:
        """
        Get latency percentiles (P50, P95, P99).

        Args:
            hours: Number of hours to analyze

        Returns:
            Percentiles dictionary
        """
        try:
            with get_db_session() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                # Get all response times
                times = session.query(APIRequestLog.response_time_ms).filter(
                    APIRequestLog.request_time >= cutoff
                ).all()

                if not times:
                    return {'p50': 0, 'p95': 0, 'p99': 0}

                times_array = np.array([t[0] for t in times])

                return {
                    'p50': round(float(np.percentile(times_array, 50)), 2),
                    'p95': round(float(np.percentile(times_array, 95)), 2),
                    'p99': round(float(np.percentile(times_array, 99)), 2)
                }

        except Exception as e:
            logger.error(f"Failed to get latency percentiles: {e}", exc_info=True)
            return {'p50': 0, 'p95': 0, 'p99': 0}

    @staticmethod
    def get_request_timeline(hours: int = 24, interval_minutes: int = 5) -> List[Dict]:
        """
        Get request timeline (requests per interval).

        Args:
            hours: Number of hours to analyze
            interval_minutes: Time interval in minutes

        Returns:
            List of timeline points
        """
        try:
            with get_db_session() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                # Generate time buckets
                timeline = []
                current = cutoff
                end = datetime.utcnow()

                while current < end:
                    next_time = current + timedelta(minutes=interval_minutes)

                    count = session.query(func.count(APIRequestLog.id)).filter(
                        and_(
                            APIRequestLog.request_time >= current,
                            APIRequestLog.request_time < next_time
                        )
                    ).scalar() or 0

                    timeline.append({
                        'timestamp': current.isoformat(),
                        'requests': count
                    })

                    current = next_time

                return timeline

        except Exception as e:
            logger.error(f"Failed to get request timeline: {e}", exc_info=True)
            return []

    @staticmethod
    def get_top_api_keys(limit: int = 10, hours: int = 24) -> List[Dict]:
        """
        Get most active API keys.

        Args:
            limit: Number of keys to return
            hours: Number of hours to analyze

        Returns:
            List of API key usage statistics
        """
        try:
            with get_db_session() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                results = session.query(
                    APIRequestLog.api_key_id,
                    APIKey.name,
                    APIKey.prefix,
                    func.count(APIRequestLog.id).label('request_count'),
                    func.avg(APIRequestLog.response_time_ms).label('avg_time')
                ).join(
                    APIKey, APIRequestLog.api_key_id == APIKey.id
                ).filter(
                    APIRequestLog.request_time >= cutoff
                ).group_by(
                    APIRequestLog.api_key_id,
                    APIKey.name,
                    APIKey.prefix
                ).order_by(
                    func.count(APIRequestLog.id).desc()
                ).limit(limit).all()

                return [
                    {
                        'api_key_id': r.api_key_id,
                        'name': r.name,
                        'prefix': r.prefix,
                        'request_count': r.request_count,
                        'avg_response_time_ms': round(r.avg_time, 2) if r.avg_time else 0
                    }
                    for r in results
                ]

        except Exception as e:
            logger.error(f"Failed to get top API keys: {e}", exc_info=True)
            return []

    @staticmethod
    def get_error_logs(limit: int = 50, hours: int = 24) -> List[Dict]:
        """
        Get recent error logs.

        Args:
            limit: Number of errors to return
            hours: Number of hours to look back

        Returns:
            List of error log dictionaries
        """
        try:
            with get_db_session() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)

                errors = session.query(APIRequestLog).filter(
                    and_(
                        APIRequestLog.request_time >= cutoff,
                        APIRequestLog.status_code >= 400
                    )
                ).order_by(
                    APIRequestLog.request_time.desc()
                ).limit(limit).all()

                return [
                    {
                        'id': err.id,
                        'endpoint': err.endpoint,
                        'method': err.method,
                        'status_code': err.status_code,
                        'error_message': err.error_message,
                        'request_time': err.request_time.isoformat(),
                        'ip_address': err.ip_address
                    }
                    for err in errors
                ]

        except Exception as e:
            logger.error(f"Failed to get error logs: {e}", exc_info=True)
            return []
