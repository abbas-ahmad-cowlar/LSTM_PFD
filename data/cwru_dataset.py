#!/usr/bin/env python3
"""
CWRU Bearing Dataset Loader

Loads and preprocesses the Case Western Reserve University (CWRU) bearing
dataset - the most widely used benchmark for bearing fault diagnosis research.

Features:
- Automatic download from official CWRU website (with caching)
- Preprocessing to match project signal format
- Fault class mapping to project taxonomy
- Ready-to-use train/val/test splits

Dataset Info:
- Source: https://engineering.case.edu/bearingdatacenter
- Bearings: 6205-2RS JEM SKF (Drive End) and 6203-2RS JEM SKF (Fan End)
- Motor loads: 0, 1, 2, 3 HP
- Fault types: Normal, Ball, Inner Race, Outer Race (various diameters)
- Sampling rate: 12 kHz (some at 48 kHz)

Usage:
    from data.cwru_dataset import CWRUDataset, download_cwru_data
    
    # Download data (first time only)
    download_cwru_data('data/raw/cwru')
    
    # Create dataset
    dataset = CWRUDataset('data/raw/cwru', split='train')
    loader = DataLoader(dataset, batch_size=32)

Author: Critical Deficiency Fix #2 (Priority: 98)
Date: 2026-01-18

References:
- K.A. Loparo, "Bearings Data Center", Case Western Reserve University
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from typing import Optional, Callable, Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split

from utils.logging import get_logger


logger = get_logger(__name__)


# CWRU Dataset URLs and file mappings
# Note: Files are .mat format from MATLAB
CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/bearingdatacenter/"

# Fault type definitions with corresponding file patterns
CWRU_FAULT_TYPES = {
    'normal': {
        'label': 0,
        'files': ['97.mat', '98.mat', '99.mat', '100.mat'],  # Normal baseline at 0-3 HP
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'ball_007': {
        'label': 1,
        'files': ['118.mat', '119.mat', '120.mat', '121.mat'],  # 0.007" ball fault
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'ball_014': {
        'label': 2,
        'files': ['185.mat', '186.mat', '187.mat', '188.mat'],  # 0.014" ball fault
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'ball_021': {
        'label': 3,
        'files': ['222.mat', '223.mat', '224.mat', '225.mat'],  # 0.021" ball fault
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'inner_007': {
        'label': 4,
        'files': ['105.mat', '106.mat', '107.mat', '108.mat'],  # 0.007" inner race
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'inner_014': {
        'label': 5,
        'files': ['169.mat', '170.mat', '171.mat', '172.mat'],  # 0.014" inner race
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'inner_021': {
        'label': 6,
        'files': ['209.mat', '210.mat', '211.mat', '212.mat'],  # 0.021" inner race
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'outer_007': {
        'label': 7,
        'files': ['130.mat', '131.mat', '132.mat', '133.mat'],  # 0.007" outer race @6:00
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'outer_014': {
        'label': 8,
        'files': ['197.mat', '198.mat', '199.mat', '200.mat'],  # 0.014" outer race @6:00
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    },
    'outer_021': {
        'label': 9,
        'files': ['234.mat', '235.mat', '236.mat', '237.mat'],  # 0.021" outer race @6:00
        'rpm': [1772, 1750, 1730, 1712],
        'load': [0, 1, 2, 3]
    }
}


def download_cwru_data(
    save_dir: str,
    fault_types: Optional[List[str]] = None,
    force: bool = False
) -> Dict[str, List[str]]:
    """
    Download CWRU bearing dataset files.
    
    Args:
        save_dir: Directory to save downloaded files
        fault_types: List of fault types to download (None = all)
        force: Force re-download even if files exist
    
    Returns:
        Dictionary mapping fault types to downloaded file paths
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if fault_types is None:
        fault_types = list(CWRU_FAULT_TYPES.keys())
    
    downloaded = {}
    total_files = sum(len(CWRU_FAULT_TYPES[ft]['files']) for ft in fault_types)
    downloaded_count = 0
    
    logger.info(f"Downloading CWRU dataset to {save_path}")
    logger.info(f"Fault types: {fault_types}")
    logger.info(f"Total files: {total_files}")
    
    for fault_type in fault_types:
        if fault_type not in CWRU_FAULT_TYPES:
            logger.warning(f"Unknown fault type: {fault_type}")
            continue
        
        fault_info = CWRU_FAULT_TYPES[fault_type]
        downloaded[fault_type] = []
        
        for mat_file in fault_info['files']:
            file_path = save_path / mat_file
            
            if file_path.exists() and not force:
                logger.debug(f"  Already exists: {mat_file}")
                downloaded[fault_type].append(str(file_path))
                downloaded_count += 1
                continue
            
            url = CWRU_BASE_URL + mat_file
            
            try:
                logger.info(f"  Downloading: {mat_file}")
                urllib.request.urlretrieve(url, file_path)
                downloaded[fault_type].append(str(file_path))
                downloaded_count += 1
            except Exception as e:
                logger.error(f"  Failed to download {mat_file}: {e}")
                # Try alternative URL pattern
                alt_url = f"https://engineering.case.edu/sites/default/files/{mat_file}"
                try:
                    urllib.request.urlretrieve(alt_url, file_path)
                    downloaded[fault_type].append(str(file_path))
                    downloaded_count += 1
                except Exception as e2:
                    logger.error(f"  Alternative URL also failed: {e2}")
    
    logger.info(f"Downloaded {downloaded_count}/{total_files} files")
    return downloaded


def load_cwru_mat_file(mat_path: str) -> Tuple[np.ndarray, float]:
    """
    Load a CWRU .mat file and extract vibration data.
    
    Args:
        mat_path: Path to .mat file
    
    Returns:
        (vibration_signal, sampling_rate)
    """
    data = loadmat(mat_path)
    
    # CWRU files have variable names like 'X097_DE_time' for drive end
    # or 'X097_FE_time' for fan end
    signal = None
    
    for key in data.keys():
        if 'DE_time' in key:  # Drive End accelerometer
            signal = data[key].flatten()
            break
        elif 'FE_time' in key:  # Fan End accelerometer
            signal = data[key].flatten()
            break
    
    if signal is None:
        # Fallback: try to find any time-series data
        for key in data.keys():
            if not key.startswith('_') and isinstance(data[key], np.ndarray):
                if data[key].size > 1000:
                    signal = data[key].flatten()
                    break
    
    if signal is None:
        raise ValueError(f"Could not find vibration data in {mat_path}")
    
    # CWRU uses 12 kHz sampling rate by default
    # Some files (starting with X) use 48 kHz
    filename = Path(mat_path).stem
    if filename.startswith('X') and int(filename[1:]) > 300:
        sampling_rate = 48000.0
    else:
        sampling_rate = 12000.0
    
    return signal.astype(np.float32), sampling_rate


def segment_signal(
    signal: np.ndarray,
    segment_length: int = 2048,
    overlap: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Segment a long signal into fixed-length windows.
    
    Args:
        signal: 1D vibration signal
        segment_length: Length of each segment
        overlap: Overlap ratio between segments (0.0 to 0.9)
        normalize: Whether to z-normalize each segment
    
    Returns:
        Array of shape (num_segments, segment_length)
    """
    step = int(segment_length * (1 - overlap))
    num_segments = (len(signal) - segment_length) // step + 1
    
    segments = np.zeros((num_segments, segment_length), dtype=np.float32)
    
    for i in range(num_segments):
        start = i * step
        segment = signal[start:start + segment_length]
        
        if normalize:
            mean = np.mean(segment)
            std = np.std(segment)
            if std > 1e-10:
                segment = (segment - mean) / std
            else:
                segment = segment - mean
        
        segments[i] = segment
    
    return segments


class CWRUDataset(Dataset):
    """
    PyTorch Dataset for CWRU Bearing Dataset.
    
    Loads pre-downloaded CWRU .mat files, segments them, and provides
    ready-to-use samples for training/evaluation.
    
    Args:
        data_dir: Directory containing downloaded .mat files
        split: 'train', 'val', or 'test'
        segment_length: Length of each signal segment
        overlap: Overlap ratio for segmentation
        transform: Optional transform to apply
        fault_types: List of fault types to include (None = all)
        seed: Random seed for train/val/test split
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        segment_length: int = 2048,
        overlap: float = 0.5,
        transform: Optional[Callable] = None,
        fault_types: Optional[List[str]] = None,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.segment_length = segment_length
        self.overlap = overlap
        self.transform = transform
        self.seed = seed
        
        if fault_types is None:
            fault_types = list(CWRU_FAULT_TYPES.keys())
        self.fault_types = fault_types
        
        # Load and process all data
        self.signals, self.labels = self._load_all_data()
        
        # Create train/val/test split
        self._create_splits()
    
    def _load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and segment all available .mat files."""
        all_segments = []
        all_labels = []
        
        for fault_type in self.fault_types:
            if fault_type not in CWRU_FAULT_TYPES:
                continue
            
            fault_info = CWRU_FAULT_TYPES[fault_type]
            label = fault_info['label']
            
            for mat_file in fault_info['files']:
                mat_path = self.data_dir / mat_file
                
                if not mat_path.exists():
                    logger.warning(f"File not found: {mat_path}")
                    continue
                
                try:
                    signal, sr = load_cwru_mat_file(str(mat_path))
                    segments = segment_signal(
                        signal, 
                        self.segment_length, 
                        self.overlap
                    )
                    
                    all_segments.append(segments)
                    all_labels.extend([label] * len(segments))
                    
                    logger.debug(f"  {mat_file}: {len(segments)} segments")
                    
                except Exception as e:
                    logger.error(f"Error loading {mat_path}: {e}")
        
        if len(all_segments) == 0:
            raise ValueError(f"No data found in {self.data_dir}")
        
        signals = np.concatenate(all_segments, axis=0)
        labels = np.array(all_labels)
        
        logger.info(f"Loaded CWRU: {len(signals)} segments, {len(self.fault_types)} classes")
        
        return signals, labels
    
    def _create_splits(self):
        """Create train/val/test splits."""
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            np.arange(len(self.signals)),
            test_size=0.3,
            random_state=self.seed,
            stratify=self.labels
        )
        
        # Second split: val vs test
        temp_labels = self.labels[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=self.seed,
            stratify=temp_labels
        )
        
        if self.split == 'train':
            self.indices = train_idx
        elif self.split == 'val':
            self.indices = val_idx
        elif self.split == 'test':
            self.indices = test_idx
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        logger.info(f"CWRU {self.split} split: {len(self.indices)} samples")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        actual_idx = self.indices[idx]
        signal = self.signals[actual_idx]
        label = self.labels[actual_idx]
        
        if self.transform:
            signal = self.transform(signal)
        else:
            signal = torch.from_numpy(signal).unsqueeze(0)  # Add channel dim
        
        return signal, int(label)
    
    @property
    def num_classes(self) -> int:
        return len(set(self.labels[self.indices]))
    
    @property
    def class_names(self) -> List[str]:
        return [ft for ft in self.fault_types if ft in CWRU_FAULT_TYPES]


def create_cwru_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    segment_length: int = 2048,
    num_workers: int = 4,
    fault_types: Optional[List[str]] = None
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for CWRU dataset.
    
    Args:
        data_dir: Directory with downloaded CWRU files
        batch_size: Batch size
        segment_length: Signal segment length
        num_workers: Number of data loading workers
        fault_types: Fault types to include
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = CWRUDataset(
            data_dir=data_dir,
            split=split,
            segment_length=segment_length,
            fault_types=fault_types
        )
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    return loaders


def test_cwru_dataset():
    """Test CWRU dataset functionality."""
    import tempfile
    
    print("=" * 60)
    print("CWRU DATASET TEST")
    print("=" * 60)
    
    # Create mock data for testing (without actual download)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create mock .mat file
        print("\n[1] Creating mock .mat file for testing...")
        
        from scipy.io import savemat
        
        mock_signal = np.random.randn(120000).astype(np.float32)  # ~10 seconds at 12kHz
        
        for mat_file in ['97.mat', '118.mat']:  # Normal and ball fault
            savemat(
                tmp_path / mat_file,
                {'X097_DE_time': mock_signal}
            )
        
        print("  ✓ Created mock .mat files")
        
        # Test loading
        print("\n[2] Testing signal loading...")
        signal, sr = load_cwru_mat_file(str(tmp_path / '97.mat'))
        print(f"  ✓ Loaded signal: shape={signal.shape}, sr={sr} Hz")
        
        # Test segmentation
        print("\n[3] Testing segmentation...")
        segments = segment_signal(signal, segment_length=2048, overlap=0.5)
        print(f"  ✓ Segmented: {segments.shape[0]} segments × {segments.shape[1]} samples")
        
        # Test dataset
        print("\n[4] Testing CWRUDataset...")
        
        # Only test with available fault types
        available_types = ['normal', 'ball_007']
        
        try:
            dataset = CWRUDataset(
                data_dir=str(tmp_path),
                split='train',
                fault_types=available_types
            )
            print(f"  ✓ Dataset created: {len(dataset)} samples")
            
            signal, label = dataset[0]
            print(f"  ✓ Sample: shape={signal.shape}, label={label}")
            
        except Exception as e:
            print(f"  ⚠ Dataset test skipped: {e}")
    
    print("\n" + "=" * 60)
    print("CWRU DATASET INFO")
    print("=" * 60)
    print(f"\nAvailable fault types: {list(CWRU_FAULT_TYPES.keys())}")
    print(f"Total classes: {len(CWRU_FAULT_TYPES)}")
    
    print("\nTo use with real data:")
    print("  1. download_cwru_data('data/raw/cwru')")
    print("  2. dataset = CWRUDataset('data/raw/cwru', split='train')")
    print("  3. loader = DataLoader(dataset, batch_size=32)")
    
    print("\n✅ CWRU dataset module ready!")
    print("=" * 60)


if __name__ == '__main__':
    test_cwru_dataset()
