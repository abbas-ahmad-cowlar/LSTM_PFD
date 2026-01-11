"""
API Monitoring callbacks.
Real-time API metrics and analytics.
"""
from dash import Input, Output, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from services.api_monitoring_service import APIMonitoringService
from utils.logger import setup_logger
import plotly.graph_objs as go

logger = setup_logger(__name__)


def register_api_monitoring_callbacks(app):
    """Register API monitoring callbacks."""

    @app.callback(
        [
            Output('api-total-requests', 'children'),
            Output('api-avg-latency', 'children'),
            Output('api-error-rate', 'children'),
            Output('api-active-keys', 'children'),
            Output('api-timeline-chart', 'figure'),
            Output('api-endpoint-table', 'children'),
            Output('api-latency-chart', 'figure'),
            Output('api-error-logs', 'children'),
        ],
        Input('api-monitoring-interval', 'n_intervals')
    )
    def update_api_metrics(n_intervals):
        """
        Update all API monitoring metrics.

        Args:
            n_intervals: Interval counter

        Returns:
            Tuple of updated components
        """
        try:
            # Get statistics
            stats = APIMonitoringService.get_request_stats(hours=24)

            # Overview cards
            total_requests = f"{stats.get('total_requests', 0):,}"
            avg_latency = f"{stats.get('avg_response_time_ms', 0):.1f} ms"
            error_rate = f"{stats.get('error_rate', 0):.1f}%"
            active_keys = str(stats.get('active_api_keys', 0))

            # Timeline chart
            timeline_data = APIMonitoringService.get_request_timeline(hours=24, interval_minutes=30)
            timeline_fig = go.Figure()
            if timeline_data:
                timestamps = [d['timestamp'] for d in timeline_data]
                requests = [d['requests'] for d in timeline_data]

                timeline_fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=requests,
                    mode='lines+markers',
                    name='Requests',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=5)
                ))

            timeline_fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Requests",
                hovermode='x unified',
                margin=dict(l=50, r=50, t=20, b=50),
                height=300
            )

            # Endpoint table
            endpoint_metrics = APIMonitoringService.get_endpoint_metrics(hours=24)
            if endpoint_metrics:
                rows = [
                    html.Tr([
                        html.Td(m['endpoint']),
                        html.Td(f"{m['total_requests']:,}"),
                        html.Td(f"{m['avg_response_time_ms']:.1f} ms"),
                        html.Td(m['errors']),
                        html.Td(f"{m['error_rate']:.1f}%", style={'color': 'red' if m['error_rate'] > 5 else 'inherit'})
                    ])
                    for m in endpoint_metrics[:10]
                ]
                endpoint_table = dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Endpoint"),
                        html.Th("Requests"),
                        html.Th("Avg Time"),
                        html.Th("Errors"),
                        html.Th("Error Rate")
                    ])),
                    html.Tbody(rows)
                ], bordered=True, hover=True, responsive=True, size="sm")
            else:
                endpoint_table = html.P("No endpoint data available", className="text-muted text-center")

            # Latency chart
            percentiles = APIMonitoringService.get_latency_percentiles(hours=24)
            latency_fig = go.Figure()
            latency_fig.add_trace(go.Bar(
                x=['P50', 'P95', 'P99'],
                y=[percentiles['p50'], percentiles['p95'], percentiles['p99']],
                marker_color=['#2ecc71', '#f39c12', '#e74c3c']
            ))
            latency_fig.update_layout(
                xaxis_title="Percentile",
                yaxis_title="Latency (ms)",
                margin=dict(l=50, r=50, t=20, b=50),
                height=300
            )

            # Error logs
            error_logs = APIMonitoringService.get_error_logs(limit=20, hours=24)
            if error_logs:
                error_rows = [
                    dbc.Alert([
                        html.Strong(f"{err['status_code']} - {err['endpoint']}"),
                        html.Br(),
                        html.Small(f"{err['error_message'] or 'No error message'} | {err['request_time'][:19]}")
                    ], color="danger", className="mb-2 py-2")
                    for err in error_logs[:10]
                ]
                error_component = html.Div(error_rows)
            else:
                error_component = dbc.Alert("No errors in the last 24 hours", color="success")

            return (
                total_requests,
                avg_latency,
                error_rate,
                active_keys,
                timeline_fig,
                endpoint_table,
                latency_fig,
                error_component
            )

        except Exception as e:
            logger.error(f"Failed to update API metrics: {e}", exc_info=True)
            raise PreventUpdate
