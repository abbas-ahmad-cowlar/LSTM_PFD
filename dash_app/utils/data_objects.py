"""Data object classes for callbacks and utilities."""
from datetime import datetime
from typing import Any, Dict, Optional


class CampaignObject:
    """
    Campaign object wrapper for HPO campaign data.

    Converts campaign dictionary data into an object-like structure
    for easier access in templates and callbacks.
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize campaign object from dictionary.

        Args:
            data: Dictionary containing campaign data
        """
        self.id = data.get('id')
        self.name = data.get('name')
        self.status = data.get('status')
        self.model_type = data.get('base_model_type')
        self.optimization_method = data.get('method')
        self.completed_trials = data.get('trials_completed', 0)
        self.total_trials = data.get('trials_total', 0)
        self.best_score = data.get('best_accuracy') or 0.0

        # Parse created_at timestamp
        created_at_str = data.get('created_at')
        if created_at_str:
            try:
                self.created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                self.created_at = datetime.now()
        else:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object back to dictionary.

        Returns:
            Dictionary representation of the campaign object
        """
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'base_model_type': self.model_type,
            'method': self.optimization_method,
            'trials_completed': self.completed_trials,
            'trials_total': self.total_trials,
            'best_accuracy': self.best_score,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
