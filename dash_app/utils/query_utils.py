"""Database query utilities for pagination and optimization."""
from typing import Tuple, List, Any, Dict, Optional
from sqlalchemy.orm import Query
from utils.logger import setup_logger

logger = setup_logger(__name__)


def paginate(
    query: Query,
    page: int = 1,
    per_page: int = 50,
    count: Optional[int] = None,
    max_count_threshold: int = 10000
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Paginate a SQLAlchemy query with optimized counting.

    Args:
        query: SQLAlchemy query object
        page: Page number (1-indexed)
        per_page: Items per page
        count: Pre-calculated total count (pass this to avoid COUNT query)
        max_count_threshold: Skip exact count if estimated to be > this value

    Returns:
        Tuple of (items, pagination_info)

    Performance Notes:
        - If you already know the count, pass it via `count` parameter to skip COUNT query
        - For very large tables, consider using approximate counts or skip counting entirely
        - COUNT(*) can be slow on large tables with complex filters

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

    # Get total count (skip if pre-provided)
    if count is None:
        try:
            # Use LIMIT 1 to avoid fetching data for count
            total = query.count()
        except Exception as e:
            logger.warning(f"Error getting count: {e}. Proceeding without total count.")
            total = None
    else:
        total = count

    # Calculate pagination
    offset = (page - 1) * per_page

    # Fetch one extra item to check if there's a next page (efficient!)
    items = query.limit(per_page + 1).offset(offset).all()

    # Check if we have more items than requested
    has_next = len(items) > per_page
    if has_next:
        items = items[:per_page]  # Remove the extra item

    # Calculate total pages (if we have total)
    total_pages = (total + per_page - 1) // per_page if total and total > 0 else None

    pagination_info = {
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'has_next': has_next,
        'has_prev': page > 1,
        'offset': offset,
        'showing_from': offset + 1 if len(items) > 0 else 0,
        'showing_to': offset + len(items)
    }

    return items, pagination_info


def paginate_with_default_limit(query: Query, limit: int = 500, warn_if_truncated: bool = True) -> List[Any]:
    """
    Execute query with a default limit to prevent loading too many records.

    This is a safety wrapper for queries that currently use .all() but might
    return a large number of results. It limits the results while maintaining
    backwards compatibility.

    Args:
        query: SQLAlchemy query object
        limit: Maximum number of records to return (default: 500)
        warn_if_truncated: Log warning if results were truncated (default: True)

    Returns:
        List of results (limited to `limit` records)

    Performance Notes:
        - This function fetches LIMIT+1 records to detect truncation
        - If truncation is detected, a warning is logged
        - Consider upgrading to full pagination UI for better UX

    Example:
        >>> # Instead of: experiments = query.all()
        >>> experiments = paginate_with_default_limit(query, limit=500)
    """
    # Fetch one extra to detect if we're truncating
    items = query.limit(limit + 1).all()

    # Check if results were truncated
    if len(items) > limit:
        items = items[:limit]  # Remove the extra item
        if warn_if_truncated:
            # Extract table name from query for better logging
            try:
                table_name = query.column_descriptions[0]['entity'].__tablename__
            except:
                table_name = "unknown"

            logger.warning(
                f"Query results truncated: showing {limit} of {limit}+ records from {table_name}. "
                f"Consider implementing proper pagination for better UX."
            )

    return items


def get_fast_count_estimate(query: Query, threshold: int = 10000) -> Optional[int]:
    """
    Get fast approximate count for large tables.

    For tables with > threshold rows, this returns None instead of exact count
    to avoid slow COUNT(*) queries.

    Args:
        query: SQLAlchemy query object
        threshold: Return None if count exceeds this (default: 10000)

    Returns:
        Exact count if < threshold, None if >= threshold or on error

    Example:
        >>> count = get_fast_count_estimate(query, threshold=5000)
        >>> if count is None:
        ...     print("More than 5000 results")
        ... else:
        ...     print(f"Exactly {count} results")
    """
    try:
        # Use LIMIT threshold+1 to quickly check if we exceed threshold
        # This is much faster than full COUNT(*) on large tables
        sample = query.limit(threshold + 1).all()
        count = len(sample)

        if count > threshold:
            return None  # Don't bother with exact count
        else:
            # For small result sets, return exact count
            return count
    except Exception as e:
        logger.error(f"Error estimating count: {e}")
        return None
