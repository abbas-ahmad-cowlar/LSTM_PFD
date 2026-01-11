"""
Production monitoring service (Phase 11D).
Application health monitoring, metrics collection, and alerting.
"""
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List
import threading

from database.connection import get_db_session
from models.system_log import SystemLog
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MonitoringService:
    """Service for monitoring application health and performance."""

    def __init__(self):
        """Initialize monitoring service."""
        self.metrics = {}
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None

    def start_monitoring(self, interval_seconds=60):
        """
        Start background monitoring thread.

        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Monitoring started (interval: {interval_seconds}s)")

    def stop_monitoring(self):
        """Stop monitoring thread."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring stopped")

    def _monitoring_loop(self, interval):
        """Internal monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                self._check_alerts()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)

    def _collect_metrics(self):
        """Collect system and application metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            self.metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024 ** 3),
                    "memory_total_gb": memory.total / (1024 ** 3),
                    "disk_percent": disk.percent,
                    "disk_used_gb": disk.used / (1024 ** 3),
                    "disk_total_gb": disk.total / (1024 ** 3),
                },
                "application": self._get_application_metrics(),
            }

            # Log to database
            self._log_metrics()

        except Exception as e:
            logger.error(f"Metrics collection failed: {e}", exc_info=True)

    def _get_application_metrics(self) -> Dict:
        """Get application-specific metrics."""
        try:
            with get_db_session() as session:
                from models.experiment import Experiment, ExperimentStatus

                # Count experiments by status
                total_experiments = session.query(Experiment).count()
                running_experiments = session.query(Experiment).filter_by(
                    status=ExperimentStatus.RUNNING
                ).count()
                completed_experiments = session.query(Experiment).filter_by(
                    status=ExperimentStatus.COMPLETED
                ).count()
                failed_experiments = session.query(Experiment).filter_by(
                    status=ExperimentStatus.FAILED
                ).count()

                return {
                    "total_experiments": total_experiments,
                    "running_experiments": running_experiments,
                    "completed_experiments": completed_experiments,
                    "failed_experiments": failed_experiments,
                }

        except Exception as e:
            logger.error(f"Application metrics error: {e}")
            return {}

    def _log_metrics(self):
        """Log metrics to database."""
        try:
            with get_db_session() as session:
                system_log = SystemLog(
                    level="INFO",
                    message="System metrics",
                    details=self.metrics
                )
                session.add(system_log)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def _check_alerts(self):
        """Check for alert conditions."""
        if not self.metrics:
            return

        system = self.metrics.get("system", {})

        # CPU alert
        if system.get("cpu_percent", 0) > 90:
            self._create_alert("HIGH_CPU", f"CPU usage at {system['cpu_percent']:.1f}%")

        # Memory alert
        if system.get("memory_percent", 0) > 85:
            self._create_alert("HIGH_MEMORY", f"Memory usage at {system['memory_percent']:.1f}%")

        # Disk alert
        if system.get("disk_percent", 0) > 90:
            self._create_alert("HIGH_DISK", f"Disk usage at {system['disk_percent']:.1f}%")

        # Failed experiments alert
        app = self.metrics.get("application", {})
        failed_count = app.get("failed_experiments", 0)
        if failed_count > 10:
            self._create_alert("HIGH_FAILURES", f"{failed_count} failed experiments")

    def _create_alert(self, alert_type: str, message: str):
        """Create an alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow(),
            "severity": "WARNING"
        }

        # Check if alert already exists recently (prevent spam)
        recent_alerts = [
            a for a in self.alerts
            if a["type"] == alert_type and
            (datetime.utcnow() - a["timestamp"]).total_seconds() < 300  # 5 minutes
        ]

        if not recent_alerts:
            self.alerts.append(alert)
            logger.warning(f"ALERT: {alert_type} - {message}")

            # Log to database
            try:
                with get_db_session() as session:
                    system_log = SystemLog(
                        level="WARNING",
                        message=f"Alert: {alert_type}",
                        details=alert
                    )
                    session.add(system_log)
            except:
                pass

    def get_current_metrics(self) -> Dict:
        """Get current metrics."""
        return self.metrics

    def get_recent_alerts(self, hours=24) -> List[Dict]:
        """
        Get recent alerts.

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent alerts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if alert["timestamp"] > cutoff
        ]

    def get_health_status(self) -> Dict:
        """
        Get overall health status.

        Returns:
            Health status dictionary with status and details
        """
        if not self.metrics:
            return {
                "status": "unknown",
                "message": "No metrics available"
            }

        system = self.metrics.get("system", {})

        # Determine health status
        if (system.get("cpu_percent", 0) > 90 or
                system.get("memory_percent", 0) > 90 or
                system.get("disk_percent", 0) > 95):
            status = "unhealthy"
            message = "Critical resource usage"
        elif (system.get("cpu_percent", 0) > 75 or
              system.get("memory_percent", 0) > 75):
            status = "degraded"
            message = "High resource usage"
        else:
            status = "healthy"
            message = "All systems operational"

        return {
            "status": status,
            "message": message,
            "metrics": self.metrics,
            "alerts_count": len(self.get_recent_alerts(hours=1))
        }


# Global monitoring service instance
monitoring_service = MonitoringService()
