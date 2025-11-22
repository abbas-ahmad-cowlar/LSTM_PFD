"""
Tag Management Callbacks.
Handles UI interactions for experiment tagging and organization.
"""
from dash import Input, Output, State, html, callback_context, ALL
import dash_bootstrap_components as dbc
from typing import List

from database.connection import get_db_session
from services.tag_service import TagService
from models.tag import Tag
from utils.logger import setup_logger

logger = setup_logger(__name__)


def register_tag_callbacks(app):
    """
    Register all tag management callbacks.

    Args:
        app: Dash application instance
    """

    @app.callback(
        [
            Output('experiment-tag-filter', 'options'),
            Output('tag-autocomplete', 'options'),
        ],
        Input('url', 'pathname'),
        prevent_initial_call=False
    )
    def load_tag_options(pathname):
        """Load tag options for filter and autocomplete."""
        if pathname != '/experiments':
            return [], []

        try:
            with get_db_session() as session:
                # Get all tags sorted by usage
                popular_tags = TagService.get_popular_tags(session, limit=50, min_usage=1)

                # Format for dropdown
                tag_options = [
                    {
                        'label': f"{tag.name} ({tag.usage_count})",
                        'value': tag.id
                    }
                    for tag in popular_tags
                ]

                # Autocomplete options (for creating new tags too)
                autocomplete_options = [
                    {
                        'label': tag.name,
                        'value': tag.name
                    }
                    for tag in popular_tags
                ]

                return tag_options, autocomplete_options

        except Exception as e:
            logger.error(f"Error loading tag options: {e}", exc_info=True)
            return [], []

    @app.callback(
        [
            Output('manage-tags-btn', 'disabled'),
            Output('compare-experiments-btn', 'disabled', allow_duplicate=True),
        ],
        Input('experiments-table', 'selected_rows'),
        prevent_initial_call=True
    )
    def update_action_buttons(selected_rows):
        """Enable/disable action buttons based on selection."""
        has_selection = selected_rows and len(selected_rows) > 0

        return (
            not has_selection,  # Manage tags enabled if selection
            not (has_selection and len(selected_rows) >= 2),  # Compare enabled if 2+ selected
        )

    @app.callback(
        [
            Output('tag-management-modal', 'is_open'),
            Output('selected-experiments-count', 'children'),
        ],
        [
            Input('manage-tags-btn', 'n_clicks'),
            Input('close-tag-modal-btn', 'n_clicks'),
        ],
        [
            State('experiments-table', 'selected_rows'),
            State('experiments-table', 'data'),
        ],
        prevent_initial_call=True
    )
    def toggle_tag_modal(manage_clicks, close_clicks, selected_rows, table_data):
        """Open/close tag management modal."""
        ctx = callback_context
        if not ctx.triggered:
            return False, ""

        trigger_id = ctx.triggered[0]['prop_id']

        if 'manage-tags-btn' in trigger_id:
            if selected_rows and table_data:
                count = len(selected_rows)
                exp_names = [table_data[i]['name'] for i in selected_rows if i < len(table_data)]
                message = f"Managing tags for {count} experiment{'s' if count != 1 else ''}: {', '.join(exp_names[:3])}"
                if count > 3:
                    message += f" and {count - 3} more"
                return True, message
            return False, ""

        # Close button
        return False, ""

    @app.callback(
        Output('popular-tags-chips', 'children'),
        Input('tag-management-modal', 'is_open'),
        prevent_initial_call=True
    )
    def load_popular_tags(is_open):
        """Load popular tags as clickable chips."""
        if not is_open:
            return []

        try:
            with get_db_session() as session:
                popular_tags = TagService.get_popular_tags(session, limit=15, min_usage=1)

                if not popular_tags:
                    return html.P("No tags yet. Create one above!", className="text-muted small")

                # Create badge chips for each tag
                chips = []
                for tag in popular_tags:
                    badge_style = {}
                    if tag.color:
                        badge_style = {'backgroundColor': tag.color, 'color': 'white'}

                    chip = dbc.Badge(
                        [
                            tag.name,
                            html.Span(f" ({tag.usage_count})", className="ms-1 opacity-75 small")
                        ],
                        id={'type': 'popular-tag-chip', 'index': tag.name},
                        color="secondary" if not tag.color else None,
                        style=badge_style,
                        className="me-2 mb-2 cursor-pointer",
                        pill=True
                    )
                    chips.append(chip)

                return chips

        except Exception as e:
            logger.error(f"Error loading popular tags: {e}", exc_info=True)
            return html.P("Error loading tags", className="text-danger")

    @app.callback(
        Output('current-tags-display', 'children'),
        [
            Input('tag-management-modal', 'is_open'),
            Input('tag-operation-status', 'children'),  # Refresh after operations
        ],
        [
            State('experiments-table', 'selected_rows'),
            State('experiments-table', 'data'),
        ],
        prevent_initial_call=True
    )
    def display_current_tags(is_open, status_update, selected_rows, table_data):
        """Display current tags on selected experiments."""
        if not is_open or not selected_rows or not table_data:
            return html.P("No experiments selected", className="text-muted small")

        try:
            # Get experiment IDs from selected rows
            experiment_ids = [table_data[i]['id'] for i in selected_rows if i < len(table_data)]

            with get_db_session() as session:
                # Get tags for all selected experiments in a single query (no N+1)
                # Use eager loading to fetch all tags at once
                from models.tag import ExperimentTag
                from sqlalchemy.orm import joinedload

                experiment_tag_mappings = session.query(ExperimentTag).options(
                    joinedload(ExperimentTag.tag)  # Eager load the Tag relationship!
                ).filter(
                    ExperimentTag.experiment_id.in_(experiment_ids)
                ).all()

                # Build tag count mapping
                all_tags = {}
                for exp_tag in experiment_tag_mappings:
                    if exp_tag.tag:  # Ensure tag exists
                        if exp_tag.tag.id not in all_tags:
                            all_tags[exp_tag.tag.id] = {
                                'tag': exp_tag.tag,
                                'count': 0
                            }
                        all_tags[exp_tag.tag.id]['count'] += 1

                if not all_tags:
                    return html.P("No tags on selected experiments", className="text-muted small")

                # Create removable badges
                tag_badges = []
                for tag_info in all_tags.values():
                    tag = tag_info['tag']
                    count = tag_info['count']
                    total = len(experiment_ids)

                    badge_style = {}
                    if tag.color:
                        badge_style = {'backgroundColor': tag.color, 'color': 'white'}

                    badge_content = [
                        tag.name,
                        html.Span(f" ({count}/{total})", className="ms-1 opacity-75 small"),
                        html.I(
                            className="bi bi-x-circle ms-2",
                            id={'type': 'remove-tag-btn', 'index': tag.id},
                            style={'cursor': 'pointer'}
                        )
                    ]

                    badge = dbc.Badge(
                        badge_content,
                        color="primary" if not tag.color else None,
                        style=badge_style,
                        className="me-2 mb-2",
                        pill=True
                    )
                    tag_badges.append(badge)

                return tag_badges

        except Exception as e:
            logger.error(f"Error displaying current tags: {e}", exc_info=True)
            return html.P("Error loading tags", className="text-danger small")

    @app.callback(
        [
            Output('tag-operation-status', 'children'),
            Output('tag-autocomplete', 'value'),
        ],
        [
            Input('add-tags-btn', 'n_clicks'),
            Input({'type': 'popular-tag-chip', 'index': ALL}, 'n_clicks'),
            Input({'type': 'remove-tag-btn', 'index': ALL}, 'n_clicks'),
        ],
        [
            State('tag-autocomplete', 'value'),
            State('experiments-table', 'selected_rows'),
            State('experiments-table', 'data'),
        ],
        prevent_initial_call=True
    )
    def handle_tag_operations(add_clicks, chip_clicks, remove_clicks,
                             selected_tag_names, selected_rows, table_data):
        """Handle adding and removing tags."""
        ctx = callback_context
        if not ctx.triggered or not selected_rows or not table_data:
            return "", []

        trigger = ctx.triggered[0]['prop_id']

        try:
            # Get experiment IDs
            experiment_ids = [table_data[i]['id'] for i in selected_rows if i < len(table_data)]
            user_id = 1  # TODO: Get from session

            # Handle add tags button
            if 'add-tags-btn' in trigger and selected_tag_names:
                with get_db_session() as session:
                    result = TagService.bulk_add_tags(
                        session,
                        experiment_ids=experiment_ids,
                        tag_names=selected_tag_names,
                        user_id=user_id
                    )

                    if result['success']:
                        message = f"Added {result['added']} tag(s)"
                        if result['skipped'] > 0:
                            message += f", skipped {result['skipped']} (already present)"

                        return dbc.Alert([
                            html.I(className="bi bi-check-circle me-2"),
                            message
                        ], color="success", dismissable=True), []
                    else:
                        return dbc.Alert("Failed to add tags", color="danger", dismissable=True), []

            # Handle popular tag chip click
            elif 'popular-tag-chip' in trigger:
                import json
                tag_id = json.loads(trigger.split('.')[0])
                tag_name = tag_id['index']

                with get_db_session() as session:
                    result = TagService.bulk_add_tags(
                        session,
                        experiment_ids=experiment_ids,
                        tag_names=[tag_name],
                        user_id=user_id
                    )

                    if result['success'] and result['added'] > 0:
                        return dbc.Alert([
                            html.I(className="bi bi-check-circle me-2"),
                            f"Added tag '{tag_name}' to {result['added']} experiment(s)"
                        ], color="success", dismissable=True), []
                    else:
                        return dbc.Alert(f"Tag '{tag_name}' already on selected experiments",
                                       color="info", dismissable=True), []

            # Handle remove tag button
            elif 'remove-tag-btn' in trigger:
                import json
                tag_id_obj = json.loads(trigger.split('.')[0])
                tag_id = tag_id_obj['index']

                with get_db_session() as session:
                    result = TagService.bulk_remove_tags(
                        session,
                        experiment_ids=experiment_ids,
                        tag_ids=[tag_id]
                    )

                    if result['success']:
                        return dbc.Alert([
                            html.I(className="bi bi-check-circle me-2"),
                            f"Removed tag from {result['removed']} experiment(s)"
                        ], color="success", dismissable=True), []
                    else:
                        return dbc.Alert("Failed to remove tag", color="danger", dismissable=True), []

        except Exception as e:
            logger.error(f"Error in tag operation: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True), []

        return "", []

    @app.callback(
        Output('tag-autocomplete', 'options', allow_duplicate=True),
        Input('tag-autocomplete', 'search_value'),
        prevent_initial_call=True
    )
    def autocomplete_tags(search_value):
        """Provide tag autocomplete suggestions."""
        if not search_value or len(search_value) < 2:
            # Return popular tags if no search
            try:
                with get_db_session() as session:
                    popular = TagService.get_popular_tags(session, limit=10)
                    return [{'label': tag.name, 'value': tag.name} for tag in popular]
            except:
                return []

        try:
            with get_db_session() as session:
                suggestions = TagService.suggest_tags(session, search_value, limit=10)

                options = [{'label': tag.name, 'value': tag.name} for tag in suggestions]

                # Add "Create new tag: <search_value>" option if not exact match
                if not any(opt['value'].lower() == search_value.lower() for opt in options):
                    options.insert(0, {
                        'label': f"➕ Create new tag: {search_value}",
                        'value': search_value
                    })

                return options

        except Exception as e:
            logger.error(f"Error in tag autocomplete: {e}", exc_info=True)
            return []
