"""
Tag management service for organizing experiments.

Provides functionality for creating, managing, and suggesting tags.
"""
from typing import List, Dict, Optional
from sqlalchemy import func, desc
from sqlalchemy.orm import Session
from models.tag import Tag, ExperimentTag
from models.experiment import Experiment
from datetime import datetime
import re


class TagService:
    """
    Service for managing experiment tags.

    Features:
    - Create or get existing tags
    - Add/remove tags from experiments
    - Get popular tags (sorted by usage)
    - Tag autocomplete/suggestions
    - Bulk tag operations
    """

    @staticmethod
    def slugify(text: str) -> str:
        """
        Convert tag name to URL-safe slug.

        Args:
            text: Original tag name

        Returns:
            Slugified version (lowercase, hyphenated)

        Example:
            "My Tag" -> "my-tag"
            "High_Accuracy" -> "high-accuracy"
        """
        # Convert to lowercase
        slug = text.lower()
        # Replace spaces and underscores with hyphens
        slug = re.sub(r'[\s_]+', '-', slug)
        # Remove non-alphanumeric characters (except hyphens)
        slug = re.sub(r'[^\w-]', '', slug)
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        return slug

    @staticmethod
    def create_or_get_tag(
        session: Session,
        name: str,
        color: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Tag:
        """
        Create a new tag or return existing one.

        Args:
            session: Database session
            name: Tag name (will be normalized to lowercase)
            color: Optional hex color code (e.g., "#3498db")
            user_id: User creating the tag

        Returns:
            Tag object (new or existing)
        """
        # Normalize name
        name_normalized = name.lower().strip()
        slug = TagService.slugify(name_normalized)

        # Check if tag already exists
        existing_tag = session.query(Tag).filter_by(name=name_normalized).first()
        if existing_tag:
            return existing_tag

        # Create new tag
        new_tag = Tag(
            name=name_normalized,
            slug=slug,
            color=color,
            created_by=user_id,
            usage_count=0
        )
        session.add(new_tag)
        session.commit()
        return new_tag

    @staticmethod
    def add_tag_to_experiment(
        session: Session,
        experiment_id: int,
        tag_name: str,
        user_id: Optional[int] = None,
        color: Optional[str] = None
    ) -> Dict:
        """
        Add a tag to an experiment.

        Args:
            session: Database session
            experiment_id: Experiment ID
            tag_name: Name of the tag to add
            user_id: User adding the tag
            color: Optional color for new tags

        Returns:
            Dict with status and message

        Raises:
            ValueError: If experiment doesn't exist
        """
        # Verify experiment exists
        experiment = session.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Create or get tag
        tag = TagService.create_or_get_tag(session, tag_name, color, user_id)

        # Check if tag already added to experiment
        existing = session.query(ExperimentTag).filter_by(
            experiment_id=experiment_id,
            tag_id=tag.id
        ).first()

        if existing:
            return {
                'success': False,
                'message': f'Tag "{tag_name}" already added to experiment',
                'tag': tag.to_dict()
            }

        # Create experiment-tag relationship
        exp_tag = ExperimentTag(
            experiment_id=experiment_id,
            tag_id=tag.id,
            added_by=user_id
        )
        session.add(exp_tag)

        # Increment tag usage count
        tag.usage_count += 1

        session.commit()

        return {
            'success': True,
            'message': f'Tag "{tag_name}" added successfully',
            'tag': tag.to_dict(),
            'experiment_tag': exp_tag.to_dict()
        }

    @staticmethod
    def remove_tag_from_experiment(
        session: Session,
        experiment_id: int,
        tag_id: int
    ) -> Dict:
        """
        Remove a tag from an experiment.

        Args:
            session: Database session
            experiment_id: Experiment ID
            tag_id: Tag ID

        Returns:
            Dict with status and message
        """
        # Find experiment-tag relationship
        exp_tag = session.query(ExperimentTag).filter_by(
            experiment_id=experiment_id,
            tag_id=tag_id
        ).first()

        if not exp_tag:
            return {
                'success': False,
                'message': 'Tag not found on experiment'
            }

        # Get tag to decrement usage count
        tag = session.query(Tag).filter_by(id=tag_id).first()
        if tag and tag.usage_count > 0:
            tag.usage_count -= 1

        # Remove relationship
        session.delete(exp_tag)
        session.commit()

        return {
            'success': True,
            'message': 'Tag removed successfully'
        }

    @staticmethod
    def get_experiment_tags(session: Session, experiment_id: int) -> List[Tag]:
        """
        Get all tags for an experiment.

        Args:
            session: Database session
            experiment_id: Experiment ID

        Returns:
            List of Tag objects
        """
        tags = session.query(Tag).join(ExperimentTag).filter(
            ExperimentTag.experiment_id == experiment_id
        ).all()
        return tags

    @staticmethod
    def get_popular_tags(
        session: Session,
        limit: int = 20,
        min_usage: int = 1
    ) -> List[Tag]:
        """
        Get most popular tags (sorted by usage count).

        Args:
            session: Database session
            limit: Maximum number of tags to return
            min_usage: Minimum usage count to include

        Returns:
            List of Tag objects sorted by usage_count (descending)
        """
        tags = session.query(Tag).filter(
            Tag.usage_count >= min_usage
        ).order_by(
            desc(Tag.usage_count)
        ).limit(limit).all()

        return tags

    @staticmethod
    def suggest_tags(
        session: Session,
        query: str,
        limit: int = 10
    ) -> List[Tag]:
        """
        Suggest tags based on partial query (autocomplete).

        Args:
            session: Database session
            query: Partial tag name
            limit: Maximum suggestions to return

        Returns:
            List of matching Tag objects

        Example:
            query="base" -> returns ["baseline", "database", "base-model"]
        """
        if not query or len(query) < 2:
            # Return popular tags if query too short
            return TagService.get_popular_tags(session, limit=limit)

        # Search tags that start with query (case-insensitive)
        query_lower = query.lower()
        tags = session.query(Tag).filter(
            Tag.name.like(f'{query_lower}%')
        ).order_by(
            desc(Tag.usage_count)
        ).limit(limit).all()

        # If no exact prefix matches, do fuzzy search (contains query)
        if not tags:
            tags = session.query(Tag).filter(
                Tag.name.like(f'%{query_lower}%')
            ).order_by(
                desc(Tag.usage_count)
            ).limit(limit).all()

        return tags

    @staticmethod
    def bulk_add_tags(
        session: Session,
        experiment_ids: List[int],
        tag_names: List[str],
        user_id: Optional[int] = None
    ) -> Dict:
        """
        Add multiple tags to multiple experiments.

        Args:
            session: Database session
            experiment_ids: List of experiment IDs
            tag_names: List of tag names to add
            user_id: User performing the operation

        Returns:
            Dict with statistics about the operation
        """
        added_count = 0
        skipped_count = 0
        errors = []

        for exp_id in experiment_ids:
            for tag_name in tag_names:
                try:
                    result = TagService.add_tag_to_experiment(
                        session, exp_id, tag_name, user_id
                    )
                    if result['success']:
                        added_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    errors.append(f"Experiment {exp_id}, Tag {tag_name}: {str(e)}")

        return {
            'success': True,
            'added': added_count,
            'skipped': skipped_count,
            'errors': errors
        }

    @staticmethod
    def bulk_remove_tags(
        session: Session,
        experiment_ids: List[int],
        tag_ids: List[int]
    ) -> Dict:
        """
        Remove multiple tags from multiple experiments.

        Args:
            session: Database session
            experiment_ids: List of experiment IDs
            tag_ids: List of tag IDs to remove

        Returns:
            Dict with statistics about the operation
        """
        removed_count = 0
        errors = []

        for exp_id in experiment_ids:
            for tag_id in tag_ids:
                try:
                    result = TagService.remove_tag_from_experiment(
                        session, exp_id, tag_id
                    )
                    if result['success']:
                        removed_count += 1
                except Exception as e:
                    errors.append(f"Experiment {exp_id}, Tag {tag_id}: {str(e)}")

        return {
            'success': True,
            'removed': removed_count,
            'errors': errors
        }

    @staticmethod
    def get_tag_statistics(session: Session) -> Dict:
        """
        Get statistics about tag usage.

        Args:
            session: Database session

        Returns:
            Dict with tag statistics
        """
        total_tags = session.query(func.count(Tag.id)).scalar()
        total_tag_uses = session.query(func.count(ExperimentTag.id)).scalar()
        most_used_tag = session.query(Tag).order_by(
            desc(Tag.usage_count)
        ).first()

        return {
            'total_tags': total_tags,
            'total_tag_uses': total_tag_uses,
            'most_used_tag': most_used_tag.to_dict() if most_used_tag else None,
            'average_tags_per_experiment': round(
                total_tag_uses / session.query(func.count(Experiment.id)).scalar()
                if session.query(func.count(Experiment.id)).scalar() > 0 else 0,
                2
            )
        }
