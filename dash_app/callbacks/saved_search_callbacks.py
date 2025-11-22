"""
Saved Search Callbacks (Phase 6, Feature 1).
Handles UI interactions for saving and managing search queries.
"""
from dash import Input, Output, State, html, callback_context
import dash_bootstrap_components as dbc

from database.connection import get_db_session
from services.search_service import SearchService
from utils.logger import setup_logger
from utils.auth_utils import get_current_user_id

logger = setup_logger(__name__)


def register_saved_search_callbacks(app):
    """
    Register all saved search callbacks.

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output('saved-searches-dropdown', 'options'),
        Input('url', 'pathname'),
        prevent_initial_call=False
    )
    def load_saved_searches(pathname):
        """Load saved searches for dropdown."""
        if pathname != '/experiments':
            return []

        try:
            user_id = get_current_user_id()

            with get_db_session() as session:
                saved_searches = SearchService.get_saved_searches(session, user_id)

                if not saved_searches:
                    return []

                # Format for dropdown (pinned first, then alphabetically)
                options = []
                for search in saved_searches:
                    icon = "📌 " if search.is_pinned else ""
                    usage_info = f" ({search.usage_count} uses)" if search.usage_count > 0 else ""
                    label = f"{icon}{search.name}{usage_info}"

                    options.append({
                        'label': label,
                        'value': search.id
                    })

                return options

        except Exception as e:
            logger.error(f"Error loading saved searches: {e}", exc_info=True)
            return []

    @app.callback(
        [
            Output('experiment-search', 'value', allow_duplicate=True),
            Output('experiment-tag-filter', 'value', allow_duplicate=True),
            Output('experiment-model-filter', 'value', allow_duplicate=True),
            Output('experiment-status-filter', 'value', allow_duplicate=True),
        ],
        Input('saved-searches-dropdown', 'value'),
        prevent_initial_call=True
    )
    def apply_saved_search(saved_search_id):
        """Apply a saved search by populating filters."""
        if not saved_search_id:
            return "", [], [], []

        try:
            user_id = get_current_user_id()

            with get_db_session() as session:
                # Get saved search
                saved_search = session.query(SearchService.__bases__[0]).filter_by(
                    id=saved_search_id,
                    user_id=user_id
                ).first()

                if not saved_search:
                    return "", [], [], []

                # Update usage stats
                from models.saved_search import SavedSearch
                search_obj = session.query(SavedSearch).filter_by(id=saved_search_id).first()
                if search_obj:
                    from datetime import datetime
                    search_obj.usage_count += 1
                    search_obj.last_used_at = datetime.utcnow()
                    session.commit()

                # Get the query
                query = search_obj.query if search_obj else ""

                # Parse the query to extract filters
                # For now, we'll just put the whole query in the search box
                # Advanced parsing could populate individual filters
                return query, [], [], []

        except Exception as e:
            logger.error(f"Error applying saved search: {e}", exc_info=True)
            return "", [], [], []

    @app.callback(
        [
            Output('save-search-modal', 'is_open'),
            Output('current-search-query-display', 'children'),
            Output('save-search-name-input', 'value'),
            Output('pin-search-checkbox', 'value'),
        ],
        [
            Input('save-search-btn', 'n_clicks'),
            Input('cancel-save-search-btn', 'n_clicks'),
            Input('confirm-save-search-btn', 'n_clicks'),
        ],
        [
            State('experiment-search', 'value'),
            State('experiment-tag-filter', 'value'),
            State('experiment-model-filter', 'value'),
            State('experiment-status-filter', 'value'),
        ],
        prevent_initial_call=True
    )
    def manage_save_search_modal(save_clicks, cancel_clicks, confirm_clicks,
                                 search_value, tag_filter, model_filter, status_filter):
        """Open/close save search modal."""
        ctx = callback_context
        if not ctx.triggered:
            return False, "", "", False

        trigger_id = ctx.triggered[0]['prop_id']

        # Open modal and show current search
        if 'save-search-btn' in trigger_id:
            # Build query string from current filters
            query_parts = []

            if search_value and search_value.strip():
                query_parts.append(search_value.strip())

            if tag_filter and len(tag_filter) > 0:
                # Get tag names from IDs
                try:
                    with get_db_session() as session:
                        from models.tag import Tag
                        tags = session.query(Tag).filter(Tag.id.in_(tag_filter)).all()
                        tag_names = [t.name for t in tags]
                        if tag_names:
                            query_parts.append(f"tag:{','.join(tag_names)}")
                except Exception as e:
                    logger.error(f"Error loading tags for search query in preview: {e}", exc_info=True)

            if model_filter and len(model_filter) > 0:
                for model in model_filter:
                    query_parts.append(f"model:{model}")

            if status_filter and len(status_filter) > 0:
                for status in status_filter:
                    query_parts.append(f"status:{status}")

            query_string = " ".join(query_parts) if query_parts else "(no filters applied)"

            return True, query_string, "", False

        # Close modal (cancel or confirm)
        return False, "", "", False

    @app.callback(
        [
            Output('save-search-status', 'children'),
            Output('saved-searches-dropdown', 'options', allow_duplicate=True),
        ],
        Input('confirm-save-search-btn', 'n_clicks'),
        [
            State('save-search-name-input', 'value'),
            State('pin-search-checkbox', 'value'),
            State('experiment-search', 'value'),
            State('experiment-tag-filter', 'value'),
            State('experiment-model-filter', 'value'),
            State('experiment-status-filter', 'value'),
        ],
        prevent_initial_call=True
    )
    def save_search_handler(n_clicks, name, is_pinned, search_value, tag_filter, model_filter, status_filter):
        """Save the search query."""
        if not n_clicks:
            return "", []

        try:
            # Validate name
            if not name or not name.strip():
                return dbc.Alert("Please provide a name for this search", color="danger"), []

            # Build query string
            query_parts = []

            if search_value and search_value.strip():
                query_parts.append(search_value.strip())

            if tag_filter and len(tag_filter) > 0:
                try:
                    with get_db_session() as session:
                        from models.tag import Tag
                        tags = session.query(Tag).filter(Tag.id.in_(tag_filter)).all()
                        tag_names = [t.name for t in tags]
                        if tag_names:
                            query_parts.append(f"tag:{','.join(tag_names)}")
                except Exception as e:
                    logger.error(f"Error loading tags for saved search: {e}", exc_info=True)

            if model_filter and len(model_filter) > 0:
                for model in model_filter:
                    query_parts.append(f"model:{model}")

            if status_filter and len(status_filter) > 0:
                for status in status_filter:
                    query_parts.append(f"status:{status}")

            query_string = " ".join(query_parts)

            if not query_string or not query_string.strip():
                return dbc.Alert("No search query to save. Apply some filters first.", color="warning"), []

            # Save the search
            user_id = get_current_user_id()

            with get_db_session() as session:
                saved_search = SearchService.save_search(
                    session=session,
                    user_id=user_id,
                    name=name.strip(),
                    query=query_string,
                    is_pinned=is_pinned or False
                )

                logger.info(f"Saved search '{name}' for user {user_id}")

                # Reload saved searches
                saved_searches = SearchService.get_saved_searches(session, user_id)
                options = []
                for search in saved_searches:
                    icon = "📌 " if search.is_pinned else ""
                    usage_info = f" ({search.usage_count} uses)" if search.usage_count > 0 else ""
                    label = f"{icon}{search.name}{usage_info}"
                    options.append({'label': label, 'value': search.id})

                return dbc.Alert([
                    html.I(className="bi bi-check-circle me-2"),
                    f"Search '{name}' saved successfully!"
                ], color="success"), options

        except ValueError as e:
            return dbc.Alert(str(e), color="danger"), []
        except Exception as e:
            logger.error(f"Error saving search: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger"), []

    @app.callback(
        [
            Output('delete-search-modal', 'is_open'),
            Output('delete-search-info', 'children'),
            Output('selected-saved-search-id', 'data'),
        ],
        [
            Input('saved-searches-dropdown', 'value'),
            Input('cancel-delete-search-btn', 'n_clicks'),
            Input('confirm-delete-search-btn', 'n_clicks'),
        ],
        [
            State('selected-saved-search-id', 'data'),
        ],
        prevent_initial_call=True
    )
    def manage_delete_search_modal(selected_id, cancel_clicks, confirm_clicks, stored_id):
        """Handle delete saved search modal."""
        ctx = callback_context
        if not ctx.triggered:
            return False, "", None

        trigger_id = ctx.triggered[0]['prop_id']

        # Context menu to delete (right-click functionality would need separate implementation)
        # For now, we won't implement delete from this callback
        # User can manage saved searches from a dedicated section if needed

        return False, "", None
