"""
System Health Dashboard (Phase 11D).
Real-time system monitoring with metrics visualization and alerts.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objs as go


def create_system_health_layout():
    """Create system health monitoring layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-heartbeat me-3"),
                    "System Health Monitoring"
                ], className="mb-1"),
                html.P("Real-time system metrics, alerts, and health status", className="text-muted mb-4")
            ])
        ]),

        # Health Status Banner
        dbc.Row([
            dbc.Col([
                html.Div(id="health-status-banner")
            ])
        ], className="mb-4"),

        # Real-time Metrics Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-microchip me-2"),
                        "CPU Usage"
                    ]),
                    dbc.CardBody([
                        html.H2(id="cpu-usage-value", className="text-center mb-3"),
                        dcc.Graph(
                            id="cpu-usage-gauge",
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ])
                ], className="shadow-sm")
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-memory me-2"),
                        "Memory Usage"
                    ]),
                    dbc.CardBody([
                        html.H2(id="memory-usage-value", className="text-center mb-3"),
                        dcc.Graph(
                            id="memory-usage-gauge",
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ])
                ], className="shadow-sm")
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-hdd me-2"),
                        "Disk Usage"
                    ]),
                    dbc.CardBody([
                        html.H2(id="disk-usage-value", className="text-center mb-3"),
                        dcc.Graph(
                            id="disk-usage-gauge",
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ])
                ], className="shadow-sm")
            ], width=4),
        ], className="mb-4"),

        # Application Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Application Metrics"),
                    dbc.CardBody([
                        html.Div(id="application-metrics")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Recent Alerts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Recent Alerts (Last 24 Hours)"
                    ]),
                    dbc.CardBody([
                        html.Div(id="recent-alerts")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Historical Metrics Chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Metrics History"),
                    dbc.CardBody([
                        dcc.Graph(id="metrics-history-chart")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # System Audit Logs Viewer
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-file-alt me-2"),
                            "System Audit Logs"
                        ], className="mb-0 d-inline"),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh"],
                            id='refresh-system-logs-btn',
                            color="secondary",
                            size="sm",
                            className="float-end"
                        ),
                    ]),
                    dbc.CardBody([
                        html.P("Search and filter system audit logs", className="text-muted mb-3"),

                        # Filter Controls
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Search", className="small"),
                                dbc.Input(
                                    id='log-search-input',
                                    placeholder="Search actions, details, or error messages...",
                                    type="text",
                                    debounce=True,
                                    className="mb-3"
                                ),
                            ], md=6),
                            dbc.Col([
                                dbc.Label("Filter by Status", className="small"),
                                dcc.Dropdown(
                                    id='log-status-filter',
                                    options=[
                                        {'label': 'All Status', 'value': 'all'},
                                        {'label': 'Success', 'value': 'success'},
                                        {'label': 'Error', 'value': 'error'},
                                        {'label': 'Warning', 'value': 'warning'},
                                    ],
                                    value='all',
                                    clearable=False,
                                    className="mb-3"
                                ),
                            ], md=3),
                            dbc.Col([
                                dbc.Label("Time Range", className="small"),
                                dcc.Dropdown(
                                    id='log-time-filter',
                                    options=[
                                        {'label': 'Last Hour', 'value': 'hour'},
                                        {'label': 'Last 24 Hours', 'value': 'day'},
                                        {'label': 'Last 7 Days', 'value': 'week'},
                                        {'label': 'Last 30 Days', 'value': 'month'},
                                        {'label': 'All Time', 'value': 'all'},
                                    ],
                                    value='day',
                                    clearable=False,
                                    className="mb-3"
                                ),
                            ], md=3),
                        ]),

                        dcc.Loading(
                            id="loading-system-logs",
                            children=[html.Div(id='system-log-table')],
                            type="default"
                        ),

                        # Pagination and Export
                        html.Div([
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    html.P(id='log-count', children="Showing 0 logs", className="text-muted small mb-0")
                                ], md=6),
                                dbc.Col([
                                    dbc.ButtonGroup([
                                        dbc.Button(
                                            [html.I(className="bi bi-download me-2"), "Export"],
                                            id='export-logs-btn',
                                            color="secondary",
                                            size="sm",
                                            outline=True,
                                            className="me-2"
                                        ),
                                        dbc.Button("Previous", id='log-prev-page-btn', size="sm", outline=True, disabled=True),
                                        dbc.Button("Next", id='log-next-page-btn', size="sm", outline=True, disabled=True),
                                    ], className="float-end")
                                ], md=6),
                            ])
                        ]),
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Auto-refresh interval
        # dcc.Interval(id='system-health-interval', interval=5000, n_intervals=0), # Moved to app.py

        # Store for pagination
        dcc.Store(id='log-page-number', data=1),
        dcc.Store(id='log-items-per-page', data=50),

    ], fluid=True)


def create_gauge_chart(value, title, color):
    """
    Create a gauge chart for system metrics.

    Args:
        value: Current metric value (0-100)
        title: Chart title
        color: Color based on threshold

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "lightyellow"},
                {'range': [75, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=200
    )

    return fig


def get_gauge_color(value):
    """
    Get gauge color based on value.

    Args:
        value: Metric value (0-100)

    Returns:
        Color string
    """
    if value < 50:
        return "green"
    elif value < 75:
        return "yellow"
    elif value < 90:
        return "orange"
    else:
        return "red"


def create_health_banner(status, message, alerts_count):
    """
    Create health status banner.

    Args:
        status: Health status (healthy, degraded, unhealthy, unknown)
        message: Status message
        alerts_count: Number of active alerts

    Returns:
        Dash component
    """
    # Determine color and icon
    if status == "healthy":
        color = "success"
        icon = "fa-check-circle"
    elif status == "degraded":
        color = "warning"
        icon = "fa-exclamation-triangle"
    elif status == "unhealthy":
        color = "danger"
        icon = "fa-times-circle"
    else:  # unknown
        color = "secondary"
        icon = "fa-question-circle"

    return dbc.Alert([
        html.H4([
            html.I(className=f"fas {icon} me-2"),
            f"System Status: {status.upper()}"
        ], className="alert-heading"),
        html.P(message, className="mb-2"),
        html.Hr(),
        html.P([
            html.Strong(f"{alerts_count} active alert(s) "),
            "in the last 24 hours"
        ], className="mb-0")
    ], color=color, className="shadow-sm")


def create_alert_card(alert):
    """
    Create alert card component.

    Args:
        alert: Alert dictionary with type, message, timestamp, severity

    Returns:
        Dash component
    """
    # Determine color based on severity
    severity_colors = {
        "WARNING": "warning",
        "ERROR": "danger",
        "CRITICAL": "danger",
        "INFO": "info"
    }

    color = severity_colors.get(alert.get("severity", "INFO"), "secondary")

    return dbc.Alert([
        html.Div([
            html.Strong(f"{alert.get('type', 'ALERT')}: ", className="me-2"),
            alert.get('message', 'No message'),
        ]),
        html.Small(
            alert.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S')
            if hasattr(alert.get('timestamp', ''), 'strftime')
            else str(alert.get('timestamp', '')),
            className="text-muted"
        )
    ], color=color, className="mb-2")


def create_application_metrics_table(metrics):
    """
    Create application metrics table.

    Args:
        metrics: Application metrics dictionary

    Returns:
        Dash component
    """
    rows = []

    metric_labels = {
        'total_experiments': 'Total Experiments',
        'running_experiments': 'Running Experiments',
        'completed_experiments': 'Completed Experiments',
        'failed_experiments': 'Failed Experiments'
    }

    for key, label in metric_labels.items():
        value = metrics.get(key, 0)
        rows.append(
            html.Tr([
                html.Td(label, style={'font-weight': 'bold'}),
                html.Td(value, className="text-end")
            ])
        )

    return dbc.Table([
        html.Tbody(rows)
    ], bordered=True, hover=True, responsive=True)
