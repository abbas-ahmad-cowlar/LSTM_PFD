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
        [
            Input('system-health-interval', 'n_intervals'),
            Input('url', 'pathname')
        ]
    )
    def update_system_health(n_intervals, pathname):
        """
        Update all system health metrics in real-time.

        Args:
            n_intervals: Interval counter
            pathname: Current URL path

        Returns:
            Tuple of updated components
        """
        if pathname != '/system-health':
            raise PreventUpdate

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


    @app.callback(
        [
            Output('system-log-table', 'children'),
            Output('log-count', 'children'),
        ],
        [
            Input('url', 'pathname'),
            Input('system-health-interval', 'n_intervals'),
            Input('refresh-system-logs-btn', 'n_clicks'),
            Input('log-search-input', 'value'),
            Input('log-status-filter', 'value'),
            Input('log-time-filter', 'value'),
            Input('log-page-number', 'data'),
        ],
        [
            State('log-items-per-page', 'data'),
        ]
    )
    def load_system_logs(
        pathname, n_intervals, refresh_clicks,
        search_query, status_filter, time_filter,
        page_number, items_per_page
    ):
        """Load and display system logs with filtering."""
        if pathname != '/system-health':
            return html.Div(), "Showing 0 logs"

        try:
            import dash_bootstrap_components as dbc
            from sqlalchemy import or_, and_

            with get_db_session() as session:
                # Base query
                query = session.query(SystemLog)

                # Apply search filter
                if search_query and search_query.strip():
                    search_term = f"%{search_query.strip()}%"
                    query = query.filter(
                        or_(
                            SystemLog.action.ilike(search_term),
                            SystemLog.details.cast(str).ilike(search_term),
                            SystemLog.error_message.ilike(search_term)
                        )
                    )

                # Apply status filter
                if status_filter and status_filter != 'all':
                    query = query.filter(SystemLog.status == status_filter)

                # Apply time filter
                now = datetime.utcnow()
                if time_filter == 'hour':
                    time_threshold = now - timedelta(hours=1)
                    query = query.filter(SystemLog.created_at >= time_threshold)
                elif time_filter == 'day':
                    time_threshold = now - timedelta(days=1)
                    query = query.filter(SystemLog.created_at >= time_threshold)
                elif time_filter == 'week':
                    time_threshold = now - timedelta(days=7)
                    query = query.filter(SystemLog.created_at >= time_threshold)
                elif time_filter == 'month':
                    time_threshold = now - timedelta(days=30)
                    query = query.filter(SystemLog.created_at >= time_threshold)

                # Order by most recent first
                query = query.order_by(SystemLog.created_at.desc())

                # Get total count
                total_logs = query.count()

                # Pagination
                page_number = page_number or 1
                items_per_page = items_per_page or 50
                offset = (page_number - 1) * items_per_page

                # Get page of logs
                logs = query.limit(items_per_page).offset(offset).all()

                # Build table
                if not logs:
                    return html.Div([
                        html.P("No system logs found.", className="text-muted text-center py-4")
                    ]), "Showing 0 logs"

                # Table header
                table_header = html.Thead(html.Tr([
                    html.Th("Timestamp", style={'width': '15%'}),
                    html.Th("Action", style={'width': '20%'}),
                    html.Th("Status", style={'width': '10%'}),
                    html.Th("Details", style={'width': '40%'}),
                    html.Th("Error", style={'width': '15%'}),
                ]))

                # Table rows
                rows = []
                for log in logs:
                    # Status badge with color
                    if log.status == 'success':
                        status_badge = dbc.Badge("Success", color="success")
                    elif log.status == 'error':
                        status_badge = dbc.Badge("Error", color="danger")
                    elif log.status == 'warning':
                        status_badge = dbc.Badge("Warning", color="warning")
                    else:
                        status_badge = dbc.Badge(log.status, color="secondary")

                    # Format details (JSON or dict)
                    details_str = ""
                    if log.details:
                        if isinstance(log.details, dict):
                            # Show key details in a compact format
                            detail_items = []
                            for key, value in list(log.details.items())[:3]:  # Limit to 3 items
                                detail_items.append(f"{key}: {value}")
                            details_str = ", ".join(detail_items)
                            if len(log.details) > 3:
                                details_str += "..."
                        else:
                            details_str = str(log.details)[:100]  # Truncate long strings
                    else:
                        details_str = "-"

                    row = html.Tr([
                        html.Td([
                            html.Div(log.created_at.strftime('%Y-%m-%d')),
                            html.Small(log.created_at.strftime('%H:%M:%S'), className="text-muted")
                        ]),
                        html.Td(log.action),
                        html.Td(status_badge),
                        html.Td(html.Small(details_str)),
                        html.Td(html.Small(log.error_message if log.error_message else "-", className="text-danger")),
                    ])
                    rows.append(row)

                table_body = html.Tbody(rows)
                table = dbc.Table(
                    [table_header, table_body],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                    className="mb-0"
                )

                # Count message
                start_idx = offset + 1
                end_idx = min(offset + len(logs), total_logs)
                count_msg = f"Showing {start_idx}-{end_idx} of {total_logs} logs"

                return table, count_msg

        except Exception as e:
            logger.error(f"Error loading system logs: {e}", exc_info=True)
            import dash_bootstrap_components as dbc
            return dbc.Alert(f"Error loading logs: {str(e)}", color="danger"), "Error"

    @app.callback(
        Output('export-logs-btn', 'n_clicks'),
        Input('export-logs-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def export_system_logs(n_clicks):
        """Export system logs to CSV."""
        if not n_clicks:
            raise PreventUpdate

        try:
            # TODO: Implement CSV export functionality
            # This would typically use dcc.Download component
            logger.info("System logs export requested")
            return None

        except Exception as e:
            logger.error(f"Error exporting logs: {e}", exc_info=True)
            return None


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
