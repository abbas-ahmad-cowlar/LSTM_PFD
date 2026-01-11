"""
Email Digest Queue callbacks.
Handles UI interactions for email digest queue management.
"""
from dash import Input, Output, State, html, callback_context, MATCH
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from database.connection import get_db_session
from models.email_digest_queue import EmailDigestQueue
from models.user import User
from sqlalchemy import func, and_


def register_email_digest_callbacks(app):
    """Register all email digest queue management callbacks."""

    @app.callback(
        [
            Output('digest-pending-count', 'children'),
            Output('digest-included-count', 'children'),
            Output('digest-today-count', 'children'),
        ],
        [
            Input('url', 'pathname'),
            Input('digest-refresh-interval', 'n_intervals'),
            Input('refresh-digest-queue-btn', 'n_clicks'),
        ]
    )
    def update_digest_stats(pathname, n_intervals, refresh_clicks):
        """Update digest queue statistics."""
        if pathname != '/settings':
            return "0", "0", "0"

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

                return str(pending_count), str(included_count), str(today_count)

        except Exception as e:
            print(f"Error updating digest stats: {e}")
            return "Error", "Error", "Error"

    @app.callback(
        [
            Output('digest-queue-table', 'children'),
            Output('digest-queue-count', 'children'),
        ],
        [
            Input('url', 'pathname'),
            Input('digest-refresh-interval', 'n_intervals'),
            Input('refresh-digest-queue-btn', 'n_clicks'),
            Input('digest-event-type-filter', 'value'),
            Input('digest-user-filter', 'value'),
            Input('digest-time-filter', 'value'),
            Input('digest-page-number', 'data'),
        ],
        [
            State('digest-items-per-page', 'data'),
        ]
    )
    def load_digest_queue_table(
        pathname, n_intervals, refresh_clicks,
        event_type_filter, user_filter, time_filter,
        page_number, items_per_page
    ):
        """Load and display digest queue items with filtering."""
        if pathname != '/settings':
            return html.Div(), "Showing 0 items"

        try:
            with get_db_session() as session:
                # Base query - pending items
                query = session.query(EmailDigestQueue, User)\
                    .join(User, EmailDigestQueue.user_id == User.id)\
                    .filter(EmailDigestQueue.included_in_digest == False)

                # Apply event type filter
                if event_type_filter and event_type_filter != 'all':
                    query = query.filter(EmailDigestQueue.event_type == event_type_filter)

                # Apply user filter
                if user_filter and user_filter != 'all':
                    query = query.filter(EmailDigestQueue.user_id == int(user_filter))

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
                total_items = query.count()

                # Pagination
                page_number = page_number or 1
                items_per_page = items_per_page or 50
                offset = (page_number - 1) * items_per_page

                # Get page of items
                queue_items = query.limit(items_per_page).offset(offset).all()

                # Build table
                if not queue_items:
                    return html.Div([
                        html.P("No pending digest items found.", className="text-muted text-center py-4")
                    ]), "Showing 0 items"

                # Table header
                table_header = html.Thead(html.Tr([
                    html.Th("User", style={'width': '15%'}),
                    html.Th("Event Type", style={'width': '20%'}),
                    html.Th("Event Details", style={'width': '30%'}),
                    html.Th("Scheduled For", style={'width': '15%'}),
                    html.Th("Created", style={'width': '15%'}),
                    html.Th("Status", style={'width': '5%'}),
                ]))

                # Table rows
                rows = []
                for digest_item, user in queue_items:
                    # Determine if past due
                    is_past_due = digest_item.scheduled_for < now
                    status_badge = dbc.Badge(
                        "Past Due",
                        color="danger",
                        className="ms-2"
                    ) if is_past_due else dbc.Badge(
                        "Pending",
                        color="warning",
                        className="ms-2"
                    )

                    # Format event data for display
                    event_summary = _format_event_data(digest_item.event_data)

                    row = html.Tr([
                        html.Td([
                            html.Div(user.username if user else f"User {digest_item.user_id}"),
                            html.Small(user.email if user else "", className="text-muted")
                        ]),
                        html.Td(digest_item.event_type),
                        html.Td(html.Small(event_summary)),
                        html.Td([
                            html.Div(digest_item.scheduled_for.strftime('%Y-%m-%d')),
                            html.Small(digest_item.scheduled_for.strftime('%H:%M UTC'), className="text-muted")
                        ]),
                        html.Td([
                            html.Small(digest_item.created_at.strftime('%Y-%m-%d %H:%M'), className="text-muted")
                        ]),
                        html.Td(status_badge),
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
                end_idx = min(offset + len(queue_items), total_items)
                count_msg = f"Showing {start_idx}-{end_idx} of {total_items} items"

                return table, count_msg

        except Exception as e:
            print(f"Error loading digest queue: {e}")
            import traceback
            traceback.print_exc()
            return html.Div([
                dbc.Alert(f"Error loading digest queue: {str(e)}", color="danger")
            ]), "Error loading items"

    @app.callback(
        Output('digest-history-table', 'children'),
        [
            Input('url', 'pathname'),
            Input('digest-refresh-interval', 'n_intervals'),
            Input('refresh-digest-queue-btn', 'n_clicks'),
        ]
    )
    def load_digest_history(pathname, n_intervals, refresh_clicks):
        """Load recently processed digest items."""
        if pathname != '/settings':
            return html.Div()

        try:
            with get_db_session() as session:
                # Get last 50 items that were included in digests
                history_items = session.query(EmailDigestQueue, User)\
                    .join(User, EmailDigestQueue.user_id == User.id)\
                    .filter(EmailDigestQueue.included_in_digest == True)\
                    .order_by(EmailDigestQueue.updated_at.desc())\
                    .limit(50)\
                    .all()

                if not history_items:
                    return html.P("No digest history found.", className="text-muted text-center py-4")

                # Build table
                table_header = html.Thead(html.Tr([
                    html.Th("User"),
                    html.Th("Event Type"),
                    html.Th("Scheduled For"),
                    html.Th("Processed At"),
                ]))

                rows = []
                for digest_item, user in history_items:
                    row = html.Tr([
                        html.Td(user.username if user else f"User {digest_item.user_id}"),
                        html.Td(digest_item.event_type),
                        html.Td(digest_item.scheduled_for.strftime('%Y-%m-%d %H:%M') if digest_item.scheduled_for else "N/A"),
                        html.Td(digest_item.updated_at.strftime('%Y-%m-%d %H:%M')),
                    ])
                    rows.append(row)

                table_body = html.Tbody(rows)
                return dbc.Table(
                    [table_header, table_body],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm"
                )

        except Exception as e:
            print(f"Error loading digest history: {e}")
            return dbc.Alert(f"Error loading digest history: {str(e)}", color="danger")

    @app.callback(
        Output('digest-trigger-message', 'children'),
        Input('trigger-digests-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def trigger_digest_processing(n_clicks):
        """Manually trigger digest processing."""
        if not n_clicks:
            return html.Div()

        try:
            # Import and trigger the Celery task
            # Note: This assumes a Celery task exists for processing digests
            try:
                from tasks.notification_tasks import process_email_digests
                process_email_digests.delay()
                return dbc.Alert(
                    [
                        html.I(className="bi bi-check-circle me-2"),
                        "Digest processing triggered successfully! Processing will begin shortly."
                    ],
                    color="success",
                    dismissable=True,
                    duration=5000
                )
            except ImportError:
                # Fallback: process synchronously if Celery is not available
                return dbc.Alert(
                    [
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        "Celery task not found. Background processing may not be configured."
                    ],
                    color="warning",
                    dismissable=True,
                    duration=5000
                )

        except Exception as e:
            print(f"Error triggering digest processing: {e}")
            return dbc.Alert(
                [
                    html.I(className="bi bi-x-circle me-2"),
                    f"Error triggering digest processing: {str(e)}"
                ],
                color="danger",
                dismissable=True,
                duration=5000
            )

    @app.callback(
        Output('digest-user-filter', 'options'),
        Input('url', 'pathname')
    )
    def populate_user_filter(pathname):
        """Populate user filter dropdown with available users."""
        if pathname != '/settings':
            return [{'label': 'All Users', 'value': 'all'}]

        try:
            with get_db_session() as session:
                # Get unique users who have digest queue items
                users = session.query(User)\
                    .join(EmailDigestQueue, EmailDigestQueue.user_id == User.id)\
                    .distinct()\
                    .all()

                options = [{'label': 'All Users', 'value': 'all'}]
                options.extend([
                    {'label': f"{u.username} ({u.email})", 'value': str(u.id)}
                    for u in users
                ])

                return options

        except Exception as e:
            print(f"Error populating user filter: {e}")
            return [{'label': 'All Users', 'value': 'all'}]


def _format_event_data(event_data):
    """Format event data JSON for display."""
    if not event_data:
        return "No details"

    try:
        # Extract key information based on common event data patterns
        if isinstance(event_data, dict):
            # Try to get meaningful summary
            if 'model_name' in event_data:
                return f"Model: {event_data['model_name']}"
            elif 'experiment_id' in event_data:
                return f"Experiment: {event_data['experiment_id']}"
            elif 'message' in event_data:
                return event_data['message']
            elif 'title' in event_data:
                return event_data['title']
            else:
                # Return first few key-value pairs
                items = list(event_data.items())[:3]
                return ", ".join([f"{k}: {v}" for k, v in items])
        else:
            return str(event_data)[:100]  # Truncate long strings

    except Exception:
        return "Invalid event data"
