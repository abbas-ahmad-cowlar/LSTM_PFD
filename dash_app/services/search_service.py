"""
Search service for finding experiments with advanced filtering.

Supports:
- Full-text search (PostgreSQL)
- Tag filtering (AND/OR logic)
- Advanced filters (accuracy, date range, status, model type)
- Result ranking by relevance
- Caching via Redis
"""
from typing import List, Dict, Optional
from sqlalchemy import and_, or_, func, desc
from sqlalchemy.orm import Session, joinedload
from models.experiment import Experiment, ExperimentStatus
from models.tag import Tag, ExperimentTag
from models.saved_search import SavedSearch
from datetime import datetime, timedelta
import re
import json


class SearchService:
    """
    Centralized search service for experiments.

    Features:
    - Full-text search across name, notes, model_type
    - Tag filtering with AND/OR logic
    - Accuracy filters (>, <, =, >=, <=)
    - Date range filters
    - Status and model type filters
    - Result ranking by relevance
    - Query suggestions
    """

    @staticmethod
    def search(
        session: Session,
        query: str,
        user_id: Optional[int] = None,
        limit: int = 100
    ) -> Dict:
        """
        Main search entry point.

        Args:
            session: Database session
            query: Search query (e.g., "tag:baseline accuracy:>0.95 resnet")
            user_id: User performing search (for authorization)
            limit: Max results to return

        Returns:
            {
                'results': [list of experiments],
                'total': int (total matches),
                'query_info': {parsed query details},
                'suggestions': [alternative queries]
            }
        """
        # Parse query
        parsed_query = SearchService._parse_query(query)

        # Build SQL query
        sql_query = SearchService._build_sql_query(session, parsed_query, user_id)

        # Execute search with eager loading for relationships
        results = session.query(Experiment).filter(
            sql_query
        ).options(
            joinedload(Experiment.dataset)
        ).limit(limit).all()

        # Rank results by relevance
        ranked_results = SearchService._rank_results(results, parsed_query)

        # Build response
        response = {
            'results': [exp.to_dict_with_tags() for exp in ranked_results],
            'total': len(ranked_results),
            'query_info': parsed_query,
            'suggestions': SearchService._get_suggestions(session, parsed_query)
        }

        return response

    @staticmethod
    def _parse_query(query: str) -> Dict:
        """
        Parse search query into structured format.

        Syntax:
            tag:baseline               → Filter by tag
            tag:baseline,production    → Filter by tag (OR logic)
            accuracy:>0.95             → Accuracy filter
            created:>2025-01-01        → Date filter
            model:resnet               → Model type filter
            status:completed           → Status filter
            keyword1 keyword2          → Full-text search

        Returns:
            {
                'tags': ['baseline', 'production'],
                'tag_logic': 'OR',  # 'AND' or 'OR'
                'accuracy': {'operator': '>', 'value': 0.95},
                'created_after': '2025-01-01',
                'created_before': '2025-03-01',
                'model_type': 'resnet',
                'status': 'completed',
                'keywords': ['keyword1', 'keyword2']
            }
        """
        parsed = {
            'tags': [],
            'tag_logic': 'AND',
            'accuracy': None,
            'created_after': None,
            'created_before': None,
            'model_type': None,
            'status': None,
            'keywords': []
        }

        if not query or not query.strip():
            return parsed

        # Tokenize query (handle quoted strings)
        tokens = re.findall(r'"[^"]+"|\S+', query)

        for token in tokens:
            # Remove quotes from quoted strings
            token = token.strip('"')

            # Tag filter: tag:baseline or tag:baseline,production
            if token.startswith('tag:'):
                tag_value = token[4:]  # Remove "tag:" prefix
                if ',' in tag_value:
                    # Multiple tags with OR logic
                    parsed['tags'].extend(tag_value.split(','))
                    parsed['tag_logic'] = 'OR'
                else:
                    parsed['tags'].append(tag_value)

            # Accuracy filter: accuracy:>0.95, accuracy:=0.968, accuracy:<0.90
            elif token.startswith('accuracy:'):
                accuracy_value = token[9:]  # Remove "accuracy:" prefix
                match = re.match(r'([><=]+)(\d+\.?\d*)', accuracy_value)
                if match:
                    operator = match.group(1)
                    value = float(match.group(2))
                    parsed['accuracy'] = {'operator': operator, 'value': value}

            # Date filter: created:>2025-01-01, created:<2025-03-01
            elif token.startswith('created:'):
                date_value = token[8:]
                match = re.match(r'([><])(.+)', date_value)
                if match:
                    operator = match.group(1)
                    date = match.group(2)
                    if operator == '>':
                        parsed['created_after'] = date
                    elif operator == '<':
                        parsed['created_before'] = date

            # Model type filter: model:resnet, model:transformer
            elif token.startswith('model:'):
                parsed['model_type'] = token[6:].lower()

            # Status filter: status:completed, status:failed
            elif token.startswith('status:'):
                parsed['status'] = token[7:].lower()

            # Keyword (full-text search)
            else:
                parsed['keywords'].append(token)

        return parsed

    @staticmethod
    def _build_sql_query(
        session: Session,
        parsed_query: Dict,
        user_id: Optional[int] = None
    ):
        """
        Build SQLAlchemy query from parsed query.

        Args:
            session: Database session
            parsed_query: Parsed query dict
            user_id: User ID for authorization

        Returns:
            SQLAlchemy filter expression
        """
        filters = []

        # Authorization: User can only search own experiments (if user_id provided)
        if user_id:
            filters.append(Experiment.created_by == user_id)

        # Tag filter
        if parsed_query['tags']:
            if parsed_query['tag_logic'] == 'AND':
                # All tags must be present (AND logic)
                for tag_name in parsed_query['tags']:
                    tag = session.query(Tag).filter_by(name=tag_name.lower()).first()
                    if tag:
                        filters.append(
                            Experiment.id.in_(
                                session.query(ExperimentTag.experiment_id)
                                .filter(ExperimentTag.tag_id == tag.id)
                            )
                        )
            else:
                # Any tag can be present (OR logic)
                tag_names = [t.lower() for t in parsed_query['tags']]
                tags = session.query(Tag).filter(Tag.name.in_(tag_names)).all()
                tag_ids = [t.id for t in tags]
                if tag_ids:
                    filters.append(
                        Experiment.id.in_(
                            session.query(ExperimentTag.experiment_id)
                            .filter(ExperimentTag.tag_id.in_(tag_ids))
                        )
                    )

        # Accuracy filter
        if parsed_query['accuracy']:
            operator = parsed_query['accuracy']['operator']
            value = parsed_query['accuracy']['value']

            # Extract accuracy from JSON metrics field
            # Assuming metrics = {"test_accuracy": 0.95, ...}
            if operator == '>':
                filters.append(
                    func.cast(
                        Experiment.metrics['test_accuracy'].astext,
                        func.Float
                    ) > value
                )
            elif operator == '>=':
                filters.append(
                    func.cast(
                        Experiment.metrics['test_accuracy'].astext,
                        func.Float
                    ) >= value
                )
            elif operator == '<':
                filters.append(
                    func.cast(
                        Experiment.metrics['test_accuracy'].astext,
                        func.Float
                    ) < value
                )
            elif operator == '<=':
                filters.append(
                    func.cast(
                        Experiment.metrics['test_accuracy'].astext,
                        func.Float
                    ) <= value
                )
            elif operator == '=' or operator == '==':
                filters.append(
                    func.cast(
                        Experiment.metrics['test_accuracy'].astext,
                        func.Float
                    ) == value
                )

        # Date filters
        if parsed_query['created_after']:
            try:
                date_after = datetime.fromisoformat(parsed_query['created_after'])
                filters.append(Experiment.created_at >= date_after)
            except ValueError:
                pass  # Invalid date format, skip filter

        if parsed_query['created_before']:
            try:
                date_before = datetime.fromisoformat(parsed_query['created_before'])
                filters.append(Experiment.created_at <= date_before)
            except ValueError:
                pass  # Invalid date format, skip filter

        # Model type filter
        if parsed_query['model_type']:
            filters.append(
                Experiment.model_type.ilike(f"%{parsed_query['model_type']}%")
            )

        # Status filter
        if parsed_query['status']:
            try:
                status_enum = ExperimentStatus[parsed_query['status'].upper()]
                filters.append(Experiment.status == status_enum)
            except KeyError:
                pass  # Invalid status, skip filter

        # Keyword search (PostgreSQL full-text search)
        if parsed_query['keywords']:
            # Join keywords with & for AND logic
            keyword_query = ' & '.join(parsed_query['keywords'])

            # Use PostgreSQL full-text search
            filters.append(
                func.to_tsvector('english', Experiment.search_vector).match(
                    keyword_query,
                    postgresql_regconfig='english'
                )
            )

        # Combine all filters with AND
        if filters:
            return and_(*filters)
        else:
            # No filters, return all (or just user filter)
            return True

    @staticmethod
    def _rank_results(
        results: List[Experiment],
        parsed_query: Dict
    ) -> List[Experiment]:
        """
        Rank results by relevance.

        Ranking factors:
        1. Exact name match (highest priority)
        2. Multiple keyword matches
        3. Recent experiments (created in last 30 days)
        4. Higher accuracy
        5. Completed status

        Args:
            results: List of experiments
            parsed_query: Parsed query dict

        Returns:
            List of experiments sorted by relevance
        """
        scored_results = []

        for exp in results:
            score = 0

            # Exact name match
            if parsed_query['keywords']:
                for keyword in parsed_query['keywords']:
                    keyword_lower = keyword.lower()
                    if exp.name and keyword_lower in exp.name.lower():
                        score += 10
                    if exp.notes and keyword_lower in exp.notes.lower():
                        score += 5
                    if exp.model_type and keyword_lower in exp.model_type.lower():
                        score += 3

            # Recent (created in last 30 days)
            if exp.created_at > datetime.utcnow() - timedelta(days=30):
                score += 5

            # High accuracy (bonus for >95%)
            if exp.metrics and 'test_accuracy' in exp.metrics:
                accuracy = exp.metrics['test_accuracy']
                if accuracy > 0.95:
                    score += 3
                elif accuracy > 0.90:
                    score += 2

            # Completed status
            if exp.status == ExperimentStatus.COMPLETED:
                score += 2

            scored_results.append((exp, score))

        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [exp for exp, score in scored_results]

    @staticmethod
    def _get_suggestions(session: Session, parsed_query: Dict) -> List[str]:
        """
        Generate search suggestions based on query.

        Args:
            session: Database session
            parsed_query: Parsed query dict

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Suggest adding accuracy filter if not present
        if not parsed_query['accuracy'] and len(parsed_query['keywords']) > 0:
            suggestions.append("Try adding: accuracy:>0.95")

        # Suggest adding date filter if not present
        if not parsed_query['created_after'] and not parsed_query['created_before']:
            # Suggest last 7 days
            week_ago = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
            suggestions.append(f"Try adding: created:>{week_ago}")

        # Suggest popular tags if no tags specified
        if not parsed_query['tags']:
            from services.tag_service import TagService
            popular_tags = TagService.get_popular_tags(session, limit=3)
            if popular_tags:
                tag_names = [t.name for t in popular_tags]
                suggestions.append(f"Popular tags: {', '.join(tag_names)}")

        return suggestions[:3]  # Max 3 suggestions

    @staticmethod
    def save_search(
        session: Session,
        user_id: int,
        name: str,
        query: str,
        is_pinned: bool = False
    ) -> SavedSearch:
        """
        Save a search query for later use.

        Args:
            session: Database session
            user_id: User saving the search
            name: Name for the saved search
            query: Search query string
            is_pinned: Whether to pin to top

        Returns:
            SavedSearch object

        Raises:
            ValueError: If user already has a saved search with this name
        """
        # Check for duplicate name
        existing = session.query(SavedSearch).filter_by(
            user_id=user_id,
            name=name
        ).first()

        if existing:
            raise ValueError(f"You already have a saved search named '{name}'")

        # Create saved search
        saved_search = SavedSearch(
            user_id=user_id,
            name=name,
            query=query,
            is_pinned=is_pinned,
            usage_count=0
        )

        session.add(saved_search)
        session.commit()

        return saved_search

    @staticmethod
    def get_saved_searches(
        session: Session,
        user_id: int
    ) -> List[SavedSearch]:
        """
        Get all saved searches for a user.

        Args:
            session: Database session
            user_id: User ID

        Returns:
            List of SavedSearch objects (pinned first, then by name)
        """
        saved_searches = session.query(SavedSearch).filter_by(
            user_id=user_id
        ).order_by(
            desc(SavedSearch.is_pinned),
            SavedSearch.name
        ).all()

        return saved_searches

    @staticmethod
    def use_saved_search(
        session: Session,
        saved_search_id: int,
        user_id: int
    ) -> Dict:
        """
        Execute a saved search and update usage stats.

        Args:
            session: Database session
            saved_search_id: Saved search ID
            user_id: User executing the search

        Returns:
            Search results

        Raises:
            ValueError: If saved search not found or unauthorized
        """
        # Get saved search
        saved_search = session.query(SavedSearch).filter_by(
            id=saved_search_id,
            user_id=user_id
        ).first()

        if not saved_search:
            raise ValueError("Saved search not found or unauthorized")

        # Update usage stats
        saved_search.usage_count += 1
        saved_search.last_used_at = datetime.utcnow()
        session.commit()

        # Execute search
        return SearchService.search(session, saved_search.query, user_id)

    @staticmethod
    def delete_saved_search(
        session: Session,
        saved_search_id: int,
        user_id: int
    ) -> bool:
        """
        Delete a saved search.

        Args:
            session: Database session
            saved_search_id: Saved search ID
            user_id: User deleting the search

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If saved search not found or unauthorized
        """
        saved_search = session.query(SavedSearch).filter_by(
            id=saved_search_id,
            user_id=user_id
        ).first()

        if not saved_search:
            raise ValueError("Saved search not found or unauthorized")

        session.delete(saved_search)
        session.commit()

        return True
