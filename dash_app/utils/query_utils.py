"""Database query utilities for pagination and optimization."""
from typing import Tuple, List, Any, Dict
from sqlalchemy.orm import Query


def paginate(query: Query, page: int = 1, per_page: int = 50) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Paginate a SQLAlchemy query.

    Args:
        query: SQLAlchemy query object
        page: Page number (1-indexed)
        per_page: Items per page

    Returns:
        Tuple of (items, pagination_info)

    Example:
        >>> from models.experiment import Experiment
        >>> query = session.query(Experiment).filter_by(status='completed')
        >>> experiments, pagination = paginate(query, page=1, per_page=50)
        >>> print(f"Total: {pagination['total']}, Page: {pagination['page']}/{pagination['total_pages']}")
    """
    # Validate inputs
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 50

    # Get total count
    total = query.count()

    # Calculate pagination
    offset = (page - 1) * per_page
    items = query.limit(per_page).offset(offset).all()

    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    pagination_info = {
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'has_next': page * per_page < total,
        'has_prev': page > 1,
        'offset': offset,
        'showing_from': offset + 1 if total > 0 else 0,
        'showing_to': min(offset + per_page, total)
    }

    return items, pagination_info


def paginate_with_default_limit(query: Query, limit: int = 500) -> List[Any]:
    """
    Execute query with a default limit to prevent loading too many records.

    This is a safety wrapper for queries that currently use .all() but might
    return a large number of results. It limits the results while maintaining
    backwards compatibility.

    Args:
        query: SQLAlchemy query object
        limit: Maximum number of records to return (default: 500)

    Returns:
        List of results (limited to `limit` records)

    Example:
        >>> # Instead of: experiments = query.all()
        >>> experiments = paginate_with_default_limit(query, limit=500)
    """
    return query.limit(limit).all()
