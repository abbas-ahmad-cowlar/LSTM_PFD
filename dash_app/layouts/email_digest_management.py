"""
Email Digest Queue Management layout.
Provides UI for managing digest email queue and monitoring digest delivery.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_email_digest_tab():
    """Create email digest queue management tab."""
    return dbc.Container([
        # Section Header
        html.H4("Email Digest Queue Management", className="mt-3 mb-3"),
        html.P([
            "Monitor and manage email digest queue. ",
            "View pending digests, delivery status, and manually trigger digest processing."
        ], className="text-muted mb-4"),

        # Queue Status Card
        dbc.Card([
            dbc.CardHeader(html.H5("Queue Status", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3(id='digest-pending-count', children="0", className="text-warning mb-0"),
                            html.Small("Pending Items", className="text-muted")
                        ], className="text-center")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.H3(id='digest-included-count', children="0", className="text-success mb-0"),
                            html.Small("Sent in Digests", className="text-muted")
                        ], className="text-center")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.H3(id='digest-today-count', children="0", className="text-info mb-0"),
                            html.Small("Scheduled Today", className="text-muted")
                        ], className="text-center")
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            dbc.Button(
                                [html.I(className="bi bi-send me-2"), "Trigger Processing"],
                                id='trigger-digests-btn',
                                color="primary",
                                size="sm",
                                className="w-100"
                            ),
                            html.Small(id='last-processed-time', children="Never", className="text-muted d-block mt-2")
                        ], className="text-center")
                    ], md=3),
                ])
            ])
        ], className="mb-4"),

        # Status Message Area
        html.Div(id='digest-trigger-message', className="mb-3"),

        # Pending Digest Queue Table
        dbc.Card([
            dbc.CardHeader([
                html.H5("Pending Digest Items", className="mb-0 d-inline"),
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh"],
                    id='refresh-digest-queue-btn',
                    color="secondary",
                    size="sm",
                    className="float-end"
                ),
            ]),
            dbc.CardBody([
                html.P("Events waiting to be included in digest emails", className="text-muted mb-3"),

                # Filter Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Filter by Event Type", className="small"),
                        dcc.Dropdown(
                            id='digest-event-type-filter',
                            options=[
                                {'label': 'All Event Types', 'value': 'all'},
                                {'label': 'Training Completed', 'value': 'training.complete'},
                                {'label': 'HPO Completed', 'value': 'hpo.complete'},
                                {'label': 'Prediction Completed', 'value': 'prediction.complete'},
                                {'label': 'Data Generated', 'value': 'data.generated'},
                                {'label': 'System Alert', 'value': 'system.alert'},
                            ],
                            value='all',
                            clearable=False,
                            className="mb-3"
                        ),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Filter by User", className="small"),
                        dcc.Dropdown(
                            id='digest-user-filter',
                            options=[{'label': 'All Users', 'value': 'all'}],
                            value='all',
                            clearable=False,
                            className="mb-3"
                        ),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Filter by Scheduled Time", className="small"),
                        dcc.Dropdown(
                            id='digest-time-filter',
                            options=[
                                {'label': 'All Time', 'value': 'all'},
                                {'label': 'Past Due', 'value': 'past_due'},
                                {'label': 'Next Hour', 'value': 'next_hour'},
                                {'label': 'Next 24 Hours', 'value': 'next_24h'},
                                {'label': 'This Week', 'value': 'this_week'},
                            ],
                            value='all',
                            clearable=False,
                            className="mb-3"
                        ),
                    ], md=4),
                ], className="mb-3"),

                dcc.Loading(
                    id="loading-digest-queue",
                    children=[html.Div(id='digest-queue-table')],
                    type="default"
                ),

                # Pagination
                html.Div([
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P(id='digest-queue-count', children="Showing 0 items", className="text-muted small mb-0")
                        ], md=6),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Previous", id='digest-prev-page-btn', size="sm", outline=True, disabled=True),
                                dbc.Button("Next", id='digest-next-page-btn', size="sm", outline=True, disabled=True),
                            ], className="float-end")
                        ], md=6),
                    ])
                ], id='digest-pagination'),
            ])
        ], className="mb-4"),

        # Recent Digest History
        dbc.Card([
            dbc.CardHeader(html.H5("Recent Digest Activity", className="mb-0")),
            dbc.CardBody([
                html.P("Recently processed digest items (last 50)", className="text-muted mb-3"),
                dcc.Loading(
                    id="loading-digest-history",
                    children=[html.Div(id='digest-history-table')],
                    type="default"
                )
            ])
        ], className="mb-4"),

        # Auto-refresh interval
        dcc.Interval(id='digest-refresh-interval', interval=30000, n_intervals=0),  # Refresh every 30 seconds

        # Store for pagination state
        dcc.Store(id='digest-page-number', data=1),
        dcc.Store(id='digest-items-per-page', data=50),

    ], className="py-4")
