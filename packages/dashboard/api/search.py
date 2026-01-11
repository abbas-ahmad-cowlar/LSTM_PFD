"""
REST API endpoints for experiment search (Feature #5).

Provides endpoints for:
- Full-text search with advanced filters
- Saved searches (create, list, execute, delete)
- Search suggestions
"""
from flask import Blueprint, request, jsonify
from database.connection import get_db_session
from services.search_service import SearchService

from middleware.api_key_auth import APIKeyAuth
require_api_key = APIKeyAuth.require_api_key
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Create Blueprint
search_bp = Blueprint('search', __name__, url_prefix='/api/search')


@search_bp.route('/', methods=['GET'])
@require_api_key
def search_experiments():
    """
    Search experiments with advanced filtering.

    Query params:
        - q (str): Search query
        - limit (int): Max results (default: 100, max: 500)

    Query syntax examples:
        - "tag:baseline accuracy:>0.95 resnet"
        - "model:cnn status:completed created:>2025-01-01"
        - "tag:baseline,production" (OR logic for tags)

    Returns:
        200: Search results
        400: Invalid request
        500: Server error
    """
    try:
        query = request.args.get('q', '').strip()
        limit = min(int(request.args.get('limit', 100)), 500)
        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        with get_db_session() as session:
            results = SearchService.search(session, query, user_id, limit)

            return jsonify({
                'success': True,
                **results
            }), 200

    except Exception as e:
        logger.error(f"Error searching experiments: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_bp.route('/saved', methods=['GET'])
@require_api_key
def get_saved_searches():
    """
    Get all saved searches for the current user.

    Returns:
        200: List of saved searches
        500: Server error
    """
    try:
        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        if not user_id:
            return jsonify({'success': False, 'error': 'User authentication required'}), 401

        with get_db_session() as session:
            saved_searches = SearchService.get_saved_searches(session, user_id)

            return jsonify({
                'success': True,
                'saved_searches': [ss.to_dict() for ss in saved_searches],
                'total': len(saved_searches)
            }), 200

    except Exception as e:
        logger.error(f"Error fetching saved searches: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_bp.route('/saved', methods=['POST'])
@require_api_key
def save_search():
    """
    Save a search query for later use.

    Request body:
        {
            "name": "Best baseline models",
            "query": "tag:baseline accuracy:>0.95",
            "is_pinned": false (optional)
        }

    Returns:
        201: Saved search created
        400: Invalid request
        409: Duplicate name
        500: Server error
    """
    try:
        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        if not user_id:
            return jsonify({'success': False, 'error': 'User authentication required'}), 401

        data = request.get_json()
        name = data.get('name', '').strip()
        query = data.get('query', '').strip()
        is_pinned = data.get('is_pinned', False)

        if not name or not query:
            return jsonify({
                'success': False,
                'error': 'Name and query required'
            }), 400

        with get_db_session() as session:
            saved_search = SearchService.save_search(
                session,
                user_id,
                name,
                query,
                is_pinned
            )

            return jsonify({
                'success': True,
                'saved_search': saved_search.to_dict(),
                'message': 'Search saved successfully'
            }), 201

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 409
    except Exception as e:
        logger.error(f"Error saving search: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_bp.route('/saved/<int:saved_search_id>', methods=['GET'])
@require_api_key
def execute_saved_search(saved_search_id):
    """
    Execute a saved search.

    Returns:
        200: Search results
        404: Saved search not found
        500: Server error
    """
    try:
        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        if not user_id:
            return jsonify({'success': False, 'error': 'User authentication required'}), 401

        with get_db_session() as session:
            results = SearchService.use_saved_search(session, saved_search_id, user_id)

            return jsonify({
                'success': True,
                **results
            }), 200

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error executing saved search: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_bp.route('/saved/<int:saved_search_id>', methods=['DELETE'])
@require_api_key
def delete_saved_search(saved_search_id):
    """
    Delete a saved search.

    Returns:
        200: Saved search deleted
        404: Saved search not found
        500: Server error
    """
    try:
        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        if not user_id:
            return jsonify({'success': False, 'error': 'User authentication required'}), 401

        with get_db_session() as session:
            SearchService.delete_saved_search(session, saved_search_id, user_id)

            return jsonify({
                'success': True,
                'message': 'Saved search deleted successfully'
            }), 200

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error deleting saved search: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_bp.route('/saved/<int:saved_search_id>/pin', methods=['PUT'])
@require_api_key
def toggle_pin_saved_search(saved_search_id):
    """
    Toggle pin status of a saved search.

    Request body:
        {
            "is_pinned": true
        }

    Returns:
        200: Pin status updated
        404: Saved search not found
        500: Server error
    """
    try:
        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        if not user_id:
            return jsonify({'success': False, 'error': 'User authentication required'}), 401

        data = request.get_json()
        is_pinned = data.get('is_pinned', False)

        with get_db_session() as session:
            from models.saved_search import SavedSearch

            saved_search = session.query(SavedSearch).filter_by(
                id=saved_search_id,
                user_id=user_id
            ).first()

            if not saved_search:
                return jsonify({
                    'success': False,
                    'error': 'Saved search not found'
                }), 404

            saved_search.is_pinned = is_pinned
            session.commit()

            return jsonify({
                'success': True,
                'saved_search': saved_search.to_dict(),
                'message': f'Saved search {"pinned" if is_pinned else "unpinned"}'
            }), 200

    except Exception as e:
        logger.error(f"Error toggling pin status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@search_bp.route('/help', methods=['GET'])
def search_help():
    """
    Get search syntax help.

    Returns:
        200: Search syntax documentation
    """
    help_text = {
        'success': True,
        'syntax': {
            'tags': {
                'single': 'tag:baseline',
                'multiple_or': 'tag:baseline,production',
                'multiple_and': 'tag:baseline tag:production',
                'description': 'Filter by experiment tags'
            },
            'accuracy': {
                'greater_than': 'accuracy:>0.95',
                'less_than': 'accuracy:<0.90',
                'equal': 'accuracy:=0.968',
                'description': 'Filter by test accuracy'
            },
            'date': {
                'after': 'created:>2025-01-01',
                'before': 'created:<2025-03-01',
                'description': 'Filter by creation date (YYYY-MM-DD)'
            },
            'model': {
                'example': 'model:resnet',
                'description': 'Filter by model type (partial match)'
            },
            'status': {
                'example': 'status:completed',
                'options': ['pending', 'running', 'paused', 'completed', 'failed', 'cancelled'],
                'description': 'Filter by experiment status'
            },
            'keywords': {
                'example': 'resnet batch_size learning_rate',
                'description': 'Full-text search in name, notes, and model type'
            }
        },
        'examples': [
            {
                'query': 'tag:baseline accuracy:>0.95',
                'description': 'Baseline experiments with >95% accuracy'
            },
            {
                'query': 'model:cnn status:completed created:>2025-01-01',
                'description': 'Completed CNN experiments created after Jan 1, 2025'
            },
            {
                'query': 'tag:baseline,production accuracy:>0.96',
                'description': 'Baseline OR production experiments with >96% accuracy'
            },
            {
                'query': 'resnet learning_rate',
                'description': 'Search for "resnet" and "learning_rate" in experiment details'
            }
        ]
    }

    return jsonify(help_text), 200
