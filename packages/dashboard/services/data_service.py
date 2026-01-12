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
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

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

            # Try to get class distribution from HDF5 file first
            try:
                file_path = Path(dataset.file_path)
                if file_path.exists():
                    with h5py.File(file_path, 'r') as f:
                        # Check for metadata group
                        if 'metadata' in f and 'fault_classes' in f['metadata']:
                            fault_classes = f['metadata']['fault_classes'][:]
                            if hasattr(fault_classes[0], 'decode'):
                                fault_classes = [fc.decode() for fc in fault_classes]
                            # Count occurrences
                            from collections import Counter
                            dist = Counter(fault_classes)
                            stats["class_distribution"] = dict(dist)
                        elif 'train' in f and 'labels' in f['train']:
                            # Fall back to counting labels from train split
                            labels = f['train']['labels'][:]
                            from collections import Counter
                            label_counts = Counter(labels.tolist())
                            # Map label indices to fault names if possible
                            fault_types = dataset.fault_types or []
                            for label_idx, count in label_counts.items():
                                if label_idx < len(fault_types):
                                    stats["class_distribution"][fault_types[label_idx]] = count
                                else:
                                    stats["class_distribution"][f"class_{label_idx}"] = count
                            # Update num_signals from actual data
                            total = sum(label_counts.values())
                            if 'val' in f and 'labels' in f['val']:
                                total += len(f['val']['labels'][:])
                            if 'test' in f and 'labels' in f['test']:
                                total += len(f['test']['labels'][:])
                            stats["num_signals"] = total
            except Exception as e:
                logger.warning(f"Could not read HDF5 stats for dataset {dataset_id}: {e}")
                # Fall back to Signal table query if HDF5 fails
                for fault in (dataset.fault_types or []):
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
    def load_signal_data(dataset_id: int, signal_id) -> np.ndarray:
        """Load actual signal data from HDF5 file.
        
        Args:
            dataset_id: Database ID of the dataset
            signal_id: Signal index (integer) within the dataset
            
        Returns:
            Signal data as numpy array
        """
        with get_db_session() as session:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise DatasetNotFoundError(f"Dataset {dataset_id} not found")

            # Ensure signal_id is an integer index
            signal_idx = int(signal_id)
            
            # Load from HDF5
            try:
                with h5py.File(dataset.file_path, 'r') as f:
                    # Try different HDF5 structures
                    
                    # Structure 1: Flat signals array
                    if 'signals' in f:
                        signals = f['signals']
                        if signal_idx < len(signals):
                            return signals[signal_idx][:]
                        else:
                            raise ValueError(f"Signal index {signal_idx} out of range (max {len(signals)-1})")
                    
                    # Structure 2: Train/Val/Test splits
                    elif 'train' in f and 'signals' in f['train']:
                        # Combine all splits or use just train for now
                        train_signals = f['train']['signals']
                        if signal_idx < len(train_signals):
                            return train_signals[signal_idx][:]
                        else:
                            raise ValueError(f"Signal index {signal_idx} out of range (max {len(train_signals)-1})")
                    
                    # Structure 3: Signal by string ID (legacy)
                    elif str(signal_idx) in f:
                        return f[str(signal_idx)][:]
                    
                    else:
                        available_keys = list(f.keys())[:5]
                        raise ValueError(f"Cannot find signals in HDF5. Available keys: {available_keys}")
                        
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
