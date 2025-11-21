"""
Data service for dataset operations.
"""
import h5py
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from database.connection import get_db_session
from models.dataset import Dataset
from models.signal import Signal
from utils.logger import setup_logger
from utils.exceptions import DatasetNotFoundError
from services.cache_service import CacheService

logger = setup_logger(__name__)


class DataService:
    """Service for dataset operations."""

    @staticmethod
    def list_datasets() -> List[Dict[str, Any]]:
        """List all datasets."""
        with get_db_session() as session:
            datasets = session.query(Dataset).all()
            return [d.to_dict() for d in datasets]

    @staticmethod
    def get_dataset(dataset_id: int) -> Dict[str, Any]:
        """Get dataset by ID."""
        with get_db_session() as session:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise DatasetNotFoundError(f"Dataset {dataset_id} not found")
            return dataset.to_dict()

    @staticmethod
    def get_dataset_stats(dataset_id: int) -> Dict[str, Any]:
        """Get dataset statistics (cached)."""
        cache_key = f"dataset_stats:{dataset_id}"
        cached = CacheService.get(cache_key)
        if cached:
            return cached

        with get_db_session() as session:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise DatasetNotFoundError(f"Dataset {dataset_id} not found")

            # Calculate stats
            stats = {
                "dataset_id": dataset_id,
                "name": dataset.name,
                "num_signals": dataset.num_signals,
                "fault_types": dataset.fault_types,
                "class_distribution": {}
            }

            # Get class distribution from signals
            for fault in dataset.fault_types:
                count = session.query(Signal).filter_by(
                    dataset_id=dataset_id, fault_class=fault
                ).count()
                stats["class_distribution"][fault] = count

            CacheService.set(cache_key, stats, ttl=600)  # Cache for 10 minutes
            return stats

    @staticmethod
    def load_signals(
        dataset_id: int,
        fault_filter: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load signals from dataset with optional filtering."""
        with get_db_session() as session:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise DatasetNotFoundError(f"Dataset {dataset_id} not found")

            # Query signals
            query = session.query(Signal).filter_by(dataset_id=dataset_id)

            if fault_filter:
                query = query.filter(Signal.fault_class.in_(fault_filter))

            if limit:
                query = query.limit(limit)

            signals = query.all()
            return [s.to_dict() for s in signals]

    @staticmethod
    def load_signal_data(dataset_id: int, signal_id: str) -> np.ndarray:
        """Load actual signal data from HDF5 file."""
        with get_db_session() as session:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise DatasetNotFoundError(f"Dataset {dataset_id} not found")

            # Load from HDF5
            try:
                with h5py.File(dataset.file_path, 'r') as f:
                    if signal_id in f:
                        signal_data = f[signal_id][:]
                        return signal_data
                    else:
                        raise ValueError(f"Signal {signal_id} not found in dataset")
            except Exception as e:
                logger.error(f"Error loading signal {signal_id}: {e}")
                raise

    @staticmethod
    def create_dataset(
        name: str,
        description: str,
        fault_types: List[str],
        severity_levels: List[str],
        file_path: str,
        num_signals: int,
        created_by: int
    ) -> int:
        """Create new dataset entry."""
        with get_db_session() as session:
            dataset = Dataset(
                name=name,
                description=description,
                fault_types=fault_types,
                severity_levels=severity_levels,
                file_path=file_path,
                num_signals=num_signals,
                created_by=created_by
            )
            session.add(dataset)
            session.commit()
            return dataset.id

    @staticmethod
    def delete_dataset(dataset_id: int) -> bool:
        """Delete dataset and associated signals."""
        with get_db_session() as session:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                return False

            # Delete file if exists
            file_path = Path(dataset.file_path)
            if file_path.exists():
                file_path.unlink()

            session.delete(dataset)
            session.commit()

            # Invalidate cache
            CacheService.invalidate_pattern(f"dataset_stats:{dataset_id}*")
            return True
