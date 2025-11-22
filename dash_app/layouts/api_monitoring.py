"""
API Monitoring Dashboard Layout.
Real-time API monitoring, analytics, and key management.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_api_monitoring_layout():
    """Create API monitoring dashboard layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-chart-line me-3"),
                    "API Monitoring & Analytics"
                ], className="mb-1"),
                html.P("Real-time API metrics, request logs, and key management", className="text-muted mb-4")
            ])
        ]),

        # Overview Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Total Requests", className="text-muted mb-2"),
                        html.H3(id="api-total-requests", className="mb-0")
                    ])
                ], className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Avg Latency", className="text-muted mb-2"),
                        html.H3(id="api-avg-latency", className="mb-0")
                    ])
                ], className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Error Rate", className="text-muted mb-2"),
                        html.H3(id="api-error-rate", className="mb-0")
                    ])
                ], className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Active API Keys", className="text-muted mb-2"),
                        html.H3(id="api-active-keys", className="mb-0")
                    ])
                ], className="shadow-sm")
            ], width=3),
        ], className="mb-4"),

        # Timeline Chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Request Timeline (Last 24 Hours)"),
                    dbc.CardBody([
                        dcc.Graph(id="api-timeline-chart")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Endpoint Metrics Table & Latency Distribution
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Endpoint Metrics"),
                    dbc.CardBody([
                        html.Div(id="api-endpoint-table")
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Latency Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="api-latency-chart")
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Error Logs
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Recent Errors"
                    ]),
                    dbc.CardBody([
                        html.Div(id="api-error-logs")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Auto-refresh interval
        dcc.Interval(
            id='api-monitoring-interval',
            interval=10*1000,  # Update every 10 seconds
            n_intervals=0
        ),

    ], fluid=True)
