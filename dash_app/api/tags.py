"""
REST API endpoints for tag management (Feature #5).

Provides endpoints for:
- Creating/getting tags
- Adding/removing tags from experiments
- Getting popular tags
- Tag autocomplete/suggestions
- Bulk tag operations
"""
from flask import Blueprint, request, jsonify
from database.connection import get_db_session
from services.tag_service import TagService
from middleware.rate_limiter import rate_limit_check
from middleware.api_auth import require_api_key
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Create Blueprint
tags_bp = Blueprint('tags', __name__, url_prefix='/api/tags')


@tags_bp.route('/popular', methods=['GET'])
@require_api_key
def get_popular_tags():
    """
    Get most popular tags.

    Query params:
        - limit (int): Max tags to return (default: 20)
        - min_usage (int): Minimum usage count (default: 1)

    Returns:
        200: List of tags
        500: Server error
    """
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        min_usage = int(request.args.get('min_usage', 1))

        with get_db_session() as session:
            tags = TagService.get_popular_tags(session, limit=limit, min_usage=min_usage)

            return jsonify({
                'success': True,
                'tags': [tag.to_dict() for tag in tags],
                'total': len(tags)
            }), 200

    except Exception as e:
        logger.error(f"Error fetching popular tags: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@tags_bp.route('/suggest', methods=['GET'])
@require_api_key
def suggest_tags():
    """
    Get tag suggestions (autocomplete).

    Query params:
        - q (str): Query string
        - limit (int): Max suggestions (default: 10)

    Returns:
        200: List of suggested tags
        400: Missing query parameter
        500: Server error
    """
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'success': False, 'error': 'Query parameter "q" required'}), 400

        limit = min(int(request.args.get('limit', 10)), 50)

        with get_db_session() as session:
            tags = TagService.suggest_tags(session, query, limit=limit)

            return jsonify({
                'success': True,
                'suggestions': [tag.to_dict() for tag in tags],
                'query': query
            }), 200

    except Exception as e:
        logger.error(f"Error suggesting tags: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@tags_bp.route('/create', methods=['POST'])
@require_api_key
def create_tag():
    """
    Create a new tag.

    Request body:
        {
            "name": "baseline",
            "color": "#3498db" (optional)
        }

    Returns:
        201: Tag created
        400: Invalid request
        500: Server error
    """
    try:
        data = request.get_json()
        name = data.get('name', '').strip()

        if not name:
            return jsonify({'success': False, 'error': 'Tag name required'}), 400

        color = data.get('color')
        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        with get_db_session() as session:
            tag = TagService.create_or_get_tag(session, name, color, user_id)

            return jsonify({
                'success': True,
                'tag': tag.to_dict(),
                'message': 'Tag created successfully'
            }), 201

    except Exception as e:
        logger.error(f"Error creating tag: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@tags_bp.route('/experiment/<int:experiment_id>', methods=['GET'])
@require_api_key
def get_experiment_tags(experiment_id):
    """
    Get all tags for an experiment.

    Returns:
        200: List of tags
        500: Server error
    """
    try:
        with get_db_session() as session:
            tags = TagService.get_experiment_tags(session, experiment_id)

            return jsonify({
                'success': True,
                'experiment_id': experiment_id,
                'tags': [tag.to_dict() for tag in tags],
                'total': len(tags)
            }), 200

    except Exception as e:
        logger.error(f"Error fetching experiment tags: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@tags_bp.route('/experiment/<int:experiment_id>/add', methods=['POST'])
@require_api_key
def add_tag_to_experiment(experiment_id):
    """
    Add a tag to an experiment.

    Request body:
        {
            "tag_name": "baseline",
            "color": "#3498db" (optional, for new tags)
        }

    Returns:
        200: Tag added
        400: Invalid request
        404: Experiment not found
        500: Server error
    """
    try:
        data = request.get_json()
        tag_name = data.get('tag_name', '').strip()

        if not tag_name:
            return jsonify({'success': False, 'error': 'Tag name required'}), 400

        color = data.get('color')
        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        with get_db_session() as session:
            result = TagService.add_tag_to_experiment(
                session,
                experiment_id,
                tag_name,
                user_id,
                color
            )

            status_code = 200 if result['success'] else 400
            return jsonify(result), status_code

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error adding tag to experiment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@tags_bp.route('/experiment/<int:experiment_id>/remove/<int:tag_id>', methods=['DELETE'])
@require_api_key
def remove_tag_from_experiment(experiment_id, tag_id):
    """
    Remove a tag from an experiment.

    Returns:
        200: Tag removed
        404: Tag not found on experiment
        500: Server error
    """
    try:
        with get_db_session() as session:
            result = TagService.remove_tag_from_experiment(session, experiment_id, tag_id)

            status_code = 200 if result['success'] else 404
            return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Error removing tag from experiment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@tags_bp.route('/bulk/add', methods=['POST'])
@require_api_key
def bulk_add_tags():
    """
    Add tags to multiple experiments.

    Request body:
        {
            "experiment_ids": [1, 2, 3],
            "tag_names": ["baseline", "production"]
        }

    Returns:
        200: Bulk operation complete
        400: Invalid request
        500: Server error
    """
    try:
        data = request.get_json()
        experiment_ids = data.get('experiment_ids', [])
        tag_names = data.get('tag_names', [])

        if not experiment_ids or not tag_names:
            return jsonify({
                'success': False,
                'error': 'experiment_ids and tag_names required'
            }), 400

        user_id = request.api_key.user_id if hasattr(request, 'api_key') else None

        with get_db_session() as session:
            result = TagService.bulk_add_tags(
                session,
                experiment_ids,
                tag_names,
                user_id
            )

            return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in bulk add tags: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@tags_bp.route('/bulk/remove', methods=['POST'])
@require_api_key
def bulk_remove_tags():
    """
    Remove tags from multiple experiments.

    Request body:
        {
            "experiment_ids": [1, 2, 3],
            "tag_ids": [1, 2]
        }

    Returns:
        200: Bulk operation complete
        400: Invalid request
        500: Server error
    """
    try:
        data = request.get_json()
        experiment_ids = data.get('experiment_ids', [])
        tag_ids = data.get('tag_ids', [])

        if not experiment_ids or not tag_ids:
            return jsonify({
                'success': False,
                'error': 'experiment_ids and tag_ids required'
            }), 400

        with get_db_session() as session:
            result = TagService.bulk_remove_tags(session, experiment_ids, tag_ids)

            return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in bulk remove tags: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@tags_bp.route('/statistics', methods=['GET'])
@require_api_key
def get_tag_statistics():
    """
    Get statistics about tag usage.

    Returns:
        200: Tag statistics
        500: Server error
    """
    try:
        with get_db_session() as session:
            stats = TagService.get_tag_statistics(session)

            return jsonify({
                'success': True,
                'statistics': stats
            }), 200

    except Exception as e:
        logger.error(f"Error fetching tag statistics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
