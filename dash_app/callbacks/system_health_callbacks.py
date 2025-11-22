"""
System Health callbacks (Phase 11D).
Real-time system monitoring callbacks.
"""
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
from layouts.system_health import (
    create_gauge_chart,
    get_gauge_color,
    create_health_banner,
    create_alert_card,
    create_application_metrics_table
)
from services.monitoring_service import monitoring_service
from utils.logger import setup_logger
from dash import html
import plotly.graph_objs as go
from datetime import datetime, timedelta
from database.connection import get_db_session
from models.system_log import SystemLog

logger = setup_logger(__name__)


def register_system_health_callbacks(app):
    """Register system health monitoring callbacks."""

    @app.callback(
        [
            Output('health-status-banner', 'children'),
            Output('cpu-usage-value', 'children'),
            Output('cpu-usage-gauge', 'figure'),
            Output('memory-usage-value', 'children'),
            Output('memory-usage-gauge', 'figure'),
            Output('disk-usage-value', 'children'),
            Output('disk-usage-gauge', 'figure'),
            Output('application-metrics', 'children'),
            Output('recent-alerts', 'children'),
            Output('metrics-history-chart', 'figure'),
        ],
        Input('system-health-interval', 'n_intervals')
    )
    def update_system_health(n_intervals):
        """
        Update all system health metrics in real-time.

        Args:
            n_intervals: Interval counter

        Returns:
            Tuple of updated components
        """
        try:
            # Get current health status
            health_status = monitoring_service.get_health_status()

            # Extract metrics
            metrics = health_status.get('metrics', {})
            system_metrics = metrics.get('system', {})
            app_metrics = metrics.get('application', {})

            # CPU metrics
            cpu_percent = system_metrics.get('cpu_percent', 0)
            cpu_value = f"{cpu_percent:.1f}%"
            cpu_gauge = create_gauge_chart(
                cpu_percent,
                "CPU",
                get_gauge_color(cpu_percent)
            )

            # Memory metrics
            memory_percent = system_metrics.get('memory_percent', 0)
            memory_value = f"{memory_percent:.1f}%"
            memory_gauge = create_gauge_chart(
                memory_percent,
                "Memory",
                get_gauge_color(memory_percent)
            )

            # Disk metrics
            disk_percent = system_metrics.get('disk_percent', 0)
            disk_value = f"{disk_percent:.1f}%"
            disk_gauge = create_gauge_chart(
                disk_percent,
                "Disk",
                get_gauge_color(disk_percent)
            )

            # Health status banner
            status = health_status.get('status', 'unknown')
            message = health_status.get('message', 'No status available')
            alerts_count = health_status.get('alerts_count', 0)
            health_banner = create_health_banner(status, message, alerts_count)

            # Application metrics table
            app_metrics_table = create_application_metrics_table(app_metrics)

            # Recent alerts
            recent_alerts = monitoring_service.get_recent_alerts(hours=24)
            if recent_alerts:
                alerts_components = [create_alert_card(alert) for alert in recent_alerts[:10]]
            else:
                alerts_components = html.P(
                    "No alerts in the last 24 hours",
                    className="text-muted text-center"
                )

            # Historical metrics chart
            history_chart = create_metrics_history_chart()

            return (
                health_banner,
                cpu_value,
                cpu_gauge,
                memory_value,
                memory_gauge,
                disk_value,
                disk_gauge,
                app_metrics_table,
                alerts_components,
                history_chart
            )

        except Exception as e:
            logger.error(f"Failed to update system health: {e}", exc_info=True)
            raise PreventUpdate


def create_metrics_history_chart():
    """
    Create historical metrics chart from database logs.

    Returns:
        Plotly figure with metrics history
    """
    try:
        # Query last 24 hours of system logs
        with get_db_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            logs = session.query(SystemLog).filter(
                SystemLog.created_at >= cutoff,
                SystemLog.message == "System metrics"
            ).order_by(SystemLog.created_at.asc()).limit(288).all()  # Max 288 points (5min intervals for 24h)

            if not logs:
                # Return empty chart
                return create_empty_metrics_chart()

            # Extract data
            timestamps = []
            cpu_values = []
            memory_values = []
            disk_values = []

            for log in logs:
                details = log.details or {}
                system_metrics = details.get('system', {})

                timestamps.append(log.created_at)
                cpu_values.append(system_metrics.get('cpu_percent', 0))
                memory_values.append(system_metrics.get('memory_percent', 0))
                disk_values.append(system_metrics.get('disk_percent', 0))

            # Create figure
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=cpu_values,
                mode='lines',
                name='CPU %',
                line=dict(color='#1f77b4', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=memory_values,
                mode='lines',
                name='Memory %',
                line=dict(color='#ff7f0e', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=disk_values,
                mode='lines',
                name='Disk %',
                line=dict(color='#2ca02c', width=2)
            ))

            # Add threshold lines
            fig.add_hline(y=90, line_dash="dash", line_color="red",
                          annotation_text="Critical (90%)",
                          annotation_position="right")
            fig.add_hline(y=75, line_dash="dash", line_color="orange",
                          annotation_text="Warning (75%)",
                          annotation_position="right")

            fig.update_layout(
                title="System Metrics (Last 24 Hours)",
                xaxis_title="Time",
                yaxis_title="Usage (%)",
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                height=400
            )

            return fig

    except Exception as e:
        logger.error(f"Failed to create metrics history chart: {e}", exc_info=True)
        return create_empty_metrics_chart()


def create_empty_metrics_chart():
    """
    Create empty metrics chart when no data available.

    Returns:
        Empty Plotly figure
    """
    fig = go.Figure()

    fig.update_layout(
        title="System Metrics (Last 24 Hours)",
        xaxis_title="Time",
        yaxis_title="Usage (%)",
        annotations=[
            dict(
                text="No data available yet. Metrics will appear as monitoring collects data.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
        ],
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )

    return fig
