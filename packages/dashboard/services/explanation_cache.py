"""
Explanation Caching Service.
Manages caching and retrieval of XAI explanations from database.
"""
from typing import Optional, List, Dict, Any
from database.connection import get_db_session
from models.explanation import Explanation
from utils.logger import setup_logger
from datetime import datetime, timedelta

logger = setup_logger(__name__)


class ExplanationCache:
    """Service for managing explanation cache."""

    @staticmethod
    def get_explanation(
        experiment_id: int,
        signal_id: int,
        method: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached explanation from database.

        Args:
            experiment_id: Experiment ID
            signal_id: Signal ID
            method: XAI method name

        Returns:
            Explanation data dictionary or None if not found
        """
        try:
            with get_db_session() as session:
                explanation = session.query(Explanation).filter_by(
                    experiment_id=experiment_id,
                    signal_id=str(signal_id),
                    method=method
                ).first()

                if explanation:
                    logger.info(f"Cache HIT: {method} explanation for exp={experiment_id}, signal={signal_id}")
                    return explanation.explanation_data
                else:
                    logger.info(f"Cache MISS: {method} explanation for exp={experiment_id}, signal={signal_id}")
                    return None

        except Exception as e:
            logger.error(f"Failed to get cached explanation: {e}", exc_info=True)
            return None

    @staticmethod
    def cache_explanation(
        experiment_id: int,
        signal_id: int,
        method: str,
        explanation_data: Dict[str, Any]
    ) -> bool:
        """
        Cache explanation to database.

        Args:
            experiment_id: Experiment ID
            signal_id: Signal ID
            method: XAI method name
            explanation_data: Explanation results dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_session() as session:
                # Check if exists
                existing = session.query(Explanation).filter_by(
                    experiment_id=experiment_id,
                    signal_id=str(signal_id),
                    method=method
                ).first()

                if existing:
                    # Update existing
                    existing.explanation_data = explanation_data
                    existing.updated_at = datetime.utcnow()
                    session.commit()
                    logger.info(f"Updated cached explanation ID {existing.id}")
                else:
                    # Create new
                    explanation = Explanation(
                        experiment_id=experiment_id,
                        signal_id=str(signal_id),
                        method=method,
                        explanation_data=explanation_data
                    )
                    session.add(explanation)
                    session.commit()
                    logger.info(f"Created new cached explanation ID {explanation.id}")

                return True

        except Exception as e:
            logger.error(f"Failed to cache explanation: {e}", exc_info=True)
            return False

    @staticmethod
    def get_recent_explanations(
        experiment_id: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get list of recent cached explanations.

        Args:
            experiment_id: Filter by experiment ID (None = all)
            limit: Maximum number of results

        Returns:
            List of explanation summaries
        """
        try:
            with get_db_session() as session:
                query = session.query(Explanation)

                if experiment_id:
                    query = query.filter_by(experiment_id=experiment_id)

                explanations = query.order_by(Explanation.created_at.desc()).limit(limit).all()

                results = []
                for exp in explanations:
                    results.append({
                        'id': exp.id,
                        'experiment_id': exp.experiment_id,
                        'signal_id': exp.signal_id,
                        'method': exp.method,
                        'created_at': exp.created_at.isoformat() if exp.created_at else None,
                        'updated_at': exp.updated_at.isoformat() if exp.updated_at else None,
                        'predicted_class': exp.explanation_data.get('predicted_class') if exp.explanation_data else None,
                        'confidence': exp.explanation_data.get('confidence') if exp.explanation_data else None,
                    })

                logger.info(f"Retrieved {len(results)} recent explanations")
                return results

        except Exception as e:
            logger.error(f"Failed to get recent explanations: {e}", exc_info=True)
            return []

    @staticmethod
    def delete_explanation(explanation_id: int) -> bool:
        """
        Delete cached explanation.

        Args:
            explanation_id: Explanation database ID

        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_session() as session:
                explanation = session.query(Explanation).filter_by(id=explanation_id).first()

                if explanation:
                    session.delete(explanation)
                    session.commit()
                    logger.info(f"Deleted explanation ID {explanation_id}")
                    return True
                else:
                    logger.warning(f"Explanation ID {explanation_id} not found")
                    return False

        except Exception as e:
            logger.error(f"Failed to delete explanation: {e}", exc_info=True)
            return False

    @staticmethod
    def clear_old_explanations(days: int = 30) -> int:
        """
        Clear explanations older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of explanations deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            with get_db_session() as session:
                old_explanations = session.query(Explanation).filter(
                    Explanation.created_at < cutoff_date
                ).all()

                count = len(old_explanations)

                for exp in old_explanations:
                    session.delete(exp)

                session.commit()

                logger.info(f"Cleared {count} explanations older than {days} days")
                return count

        except Exception as e:
            logger.error(f"Failed to clear old explanations: {e}", exc_info=True)
            return 0

    @staticmethod
    def get_cache_statistics() -> Dict[str, Any]:
        """
        Get cache usage statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            with get_db_session() as session:
                total_count = session.query(Explanation).count()

                # Count by method
                method_counts = {}
                for method in ['shap', 'lime', 'integrated_gradients', 'gradcam']:
                    count = session.query(Explanation).filter_by(method=method).count()
                    method_counts[method] = count

                # Count by experiment
                from sqlalchemy import func
                experiment_counts = session.query(
                    Explanation.experiment_id,
                    func.count(Explanation.id).label('count')
                ).group_by(Explanation.experiment_id).all()

                # Recent activity (last 7 days)
                week_ago = datetime.utcnow() - timedelta(days=7)
                recent_count = session.query(Explanation).filter(
                    Explanation.created_at >= week_ago
                ).count()

                return {
                    'total_explanations': total_count,
                    'by_method': method_counts,
                    'by_experiment': {exp_id: count for exp_id, count in experiment_counts},
                    'recent_7_days': recent_count,
                    'unique_experiments': len(experiment_counts),
                }

        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}", exc_info=True)
            return {
                'total_explanations': 0,
                'by_method': {},
                'by_experiment': {},
                'recent_7_days': 0,
                'unique_experiments': 0,
            }

    @staticmethod
    def explanation_exists(
        experiment_id: int,
        signal_id: int,
        method: str
    ) -> bool:
        """
        Check if explanation exists in cache.

        Args:
            experiment_id: Experiment ID
            signal_id: Signal ID
            method: XAI method

        Returns:
            True if exists, False otherwise
        """
        try:
            with get_db_session() as session:
                exists = session.query(Explanation).filter_by(
                    experiment_id=experiment_id,
                    signal_id=str(signal_id),
                    method=method
                ).first() is not None

                return exists

        except Exception as e:
            logger.error(f"Failed to check explanation existence: {e}")
            return False
