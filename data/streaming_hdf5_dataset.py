#!/usr/bin/env python3
"""
Streaming HDF5 Dataset for Memory-Efficient Training

This module provides a memory-efficient dataset class that reads signals
on-demand from HDF5 files instead of loading the entire dataset into RAM.

Benefits:
- Handles TB-scale datasets without memory issues
- Uses lazy indexing via h5py
- Optional chunked prefetching for performance

Usage:
    from data.streaming_hdf5_dataset import StreamingHDF5Dataset
    
    dataset = StreamingHDF5Dataset('data/processed/dataset.h5', split='train')
    loader = DataLoader(dataset, batch_size=32, num_workers=4)

Author: Critical Deficiency Fix #3 (Priority: 95)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple, Dict, Any, List
import threading
from queue import Queue


class StreamingHDF5Dataset(Dataset):
    """
    Memory-efficient HDF5 dataset that reads signals on-demand.
    
    Instead of loading the entire dataset into RAM, this class opens
    the HDF5 file and reads individual samples as needed.
    
    Thread-safe for use with DataLoader's num_workers > 0.
    
    Args:
        hdf5_path: Path to HDF5 file
        split: Which split to use ('train', 'val', or 'test')
        transform: Optional transform to apply to signals
        cache_size: Number of samples to cache in memory (0 = no caching)
    """
    
    def __init__(
        self,
        hdf5_path: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        cache_size: int = 0
    ):
        self.hdf5_path = str(hdf5_path)
        self.split = split
        self.transform = transform
        self.cache_size = cache_size
        
        # We don't keep the file open since it's not thread-safe
        # Instead, each worker will open its own handle
        self._local = threading.local()
        
        # Read metadata (this is fast and doesn't load data)
        with h5py.File(self.hdf5_path, 'r') as f:
            if split not in f:
                available = list(f.keys())
                raise ValueError(f"Split '{split}' not found. Available: {available}")
            
            self.num_samples = f[split]['signals'].shape[0]
            self.signal_length = f[split]['signals'].shape[1]
            self.sampling_rate = f.attrs.get('sampling_rate', 20480)
            self.num_classes = f.attrs.get('num_classes', 11)
            
            # Store labels in memory (small compared to signals)
            self.labels = f[split]['labels'][:]
        
        # LRU cache for frequently accessed samples
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_order: List[int] = []
    
    def _get_file_handle(self) -> h5py.File:
        """Get thread-local file handle."""
        if not hasattr(self._local, 'file') or self._local.file is None:
            self._local.file = h5py.File(self.hdf5_path, 'r', swmr=True)
        return self._local.file
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample by index."""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        # Check cache first
        if idx in self._cache:
            signal = self._cache[idx]
        else:
            # Read from HDF5
            f = self._get_file_handle()
            signal = f[self.split]['signals'][idx]
            
            # Update cache
            if self.cache_size > 0:
                self._cache[idx] = signal
                self._cache_order.append(idx)
                
                # Evict oldest if cache full
                if len(self._cache) > self.cache_size:
                    oldest = self._cache_order.pop(0)
                    if oldest in self._cache:
                        del self._cache[oldest]
        
        label = self.labels[idx]
        
        # Apply transform
        if self.transform:
            signal = self.transform(signal)
        else:
            # Default: convert to tensor with channel dimension
            signal = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)
        
        return signal, int(label)
    
    def __del__(self):
        """Close file handle on deletion."""
        if hasattr(self._local, 'file') and self._local.file is not None:
            try:
                self._local.file.close()
            except:
                pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        return {
            'path': self.hdf5_path,
            'split': self.split,
            'num_samples': self.num_samples,
            'signal_length': self.signal_length,
            'sampling_rate': self.sampling_rate,
            'num_classes': self.num_classes,
            'cache_size': self.cache_size
        }


class ChunkedStreamingDataset(StreamingHDF5Dataset):
    """
    Streaming dataset with chunked prefetching for better I/O performance.
    
    Reads signals in chunks to reduce random I/O overhead, which is
    especially beneficial for HDDs and network storage.
    
    Args:
        hdf5_path: Path to HDF5 file
        split: Which split to use
        chunk_size: Number of samples to prefetch at once
        transform: Optional transform to apply to signals
    """
    
    def __init__(
        self,
        hdf5_path: str,
        split: str = 'train',
        chunk_size: int = 256,
        transform: Optional[Callable] = None
    ):
        super().__init__(hdf5_path, split, transform, cache_size=0)
        self.chunk_size = chunk_size
        self._current_chunk_start = -1
        self._current_chunk_data = None
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample, using chunked prefetching."""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        # Calculate chunk boundaries
        chunk_start = (idx // self.chunk_size) * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, self.num_samples)
        
        # Load chunk if not already loaded
        if self._current_chunk_start != chunk_start:
            f = self._get_file_handle()
            self._current_chunk_data = f[self.split]['signals'][chunk_start:chunk_end]
            self._current_chunk_start = chunk_start
        
        # Get signal from chunk
        chunk_idx = idx - chunk_start
        signal = self._current_chunk_data[chunk_idx]
        label = self.labels[idx]
        
        # Apply transform  
        if self.transform:
            signal = self.transform(signal)
        else:
            signal = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)
        
        return signal, int(label)


def create_streaming_dataloaders(
    hdf5_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    use_chunked: bool = False,
    chunk_size: int = 256
) -> Dict[str, DataLoader]:
    """
    Create memory-efficient streaming dataloaders for train/val/test.
    
    Args:
        hdf5_path: Path to HDF5 dataset
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        train_transform: Transform for training data
        test_transform: Transform for validation/test data
        use_chunked: Use chunked prefetching (better for HDDs)
        chunk_size: Chunk size for prefetching
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    DatasetClass = ChunkedStreamingDataset if use_chunked else StreamingHDF5Dataset
    
    loaders = {}
    
    # Check which splits exist
    with h5py.File(hdf5_path, 'r') as f:
        available_splits = list(f.keys())
    
    for split_name, is_train in [('train', True), ('val', False), ('test', False)]:
        if split_name not in available_splits:
            continue
        
        transform = train_transform if is_train else test_transform
        
        if use_chunked:
            dataset = ChunkedStreamingDataset(
                hdf5_path, split=split_name, 
                chunk_size=chunk_size, transform=transform
            )
        else:
            dataset = StreamingHDF5Dataset(
                hdf5_path, split=split_name, transform=transform
            )
        
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=is_train
        )
    
    return loaders


def test_streaming_dataset():
    """Test streaming dataset functionality."""
    import tempfile
    import time
    
    print("=" * 60)
    print("STREAMING HDF5 DATASET TEST")
    print("=" * 60)
    
    # Create temporary test dataset
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    num_samples = 1000
    signal_length = 1024
    
    print(f"\nCreating test dataset: {num_samples} samples × {signal_length} length")
    
    with h5py.File(tmp_path, 'w') as f:
        for split in ['train', 'val', 'test']:
            n = num_samples if split == 'train' else num_samples // 4
            signals = np.random.randn(n, signal_length).astype(np.float32)
            labels = np.random.randint(0, 11, n)
            
            grp = f.create_group(split)
            grp.create_dataset('signals', data=signals)
            grp.create_dataset('labels', data=labels)
        
        f.attrs['sampling_rate'] = 20480
        f.attrs['num_classes'] = 11
    
    # Test basic streaming dataset
    print("\n[1] Testing StreamingHDF5Dataset...")
    dataset = StreamingHDF5Dataset(tmp_path, split='train')
    print(f"  ✓ Loaded dataset: {len(dataset)} samples")
    print(f"  ✓ Metadata: {dataset.get_metadata()}")
    
    # Test indexing
    signal, label = dataset[0]
    print(f"  ✓ Sample shape: {signal.shape}, label: {label}")
    
    # Test DataLoader
    print("\n[2] Testing DataLoader integration...")
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    start = time.time()
    batch_count = 0
    for batch_signals, batch_labels in loader:
        batch_count += 1
    elapsed = time.time() - start
    
    print(f"  ✓ Processed {batch_count} batches in {elapsed:.2f}s")
    
    # Test chunked dataset
    print("\n[3] Testing ChunkedStreamingDataset...")
    chunked_dataset = ChunkedStreamingDataset(tmp_path, split='train', chunk_size=128)
    
    start = time.time()
    for i in range(100):
        signal, _ = chunked_dataset[i]
    elapsed = time.time() - start
    print(f"  ✓ Read 100 samples in {elapsed*1000:.2f}ms (chunked)")
    
    # Test streaming dataloaders
    print("\n[4] Testing create_streaming_dataloaders...")
    loaders = create_streaming_dataloaders(tmp_path, batch_size=32, num_workers=0)
    print(f"  ✓ Created loaders: {list(loaders.keys())}")
    
    # Memory comparison (conceptual)
    print("\n" + "=" * 60)
    print("MEMORY COMPARISON (conceptual)")
    print("=" * 60)
    
    full_load_mb = (num_samples * signal_length * 4) / (1024 * 1024)
    streaming_mb = (32 * signal_length * 4) / (1024 * 1024)  # Just one batch
    
    print(f"  Full dataset in RAM:     {full_load_mb:.2f} MB")
    print(f"  Streaming (1 batch):     {streaming_mb:.2f} MB")
    print(f"  Memory reduction factor: {full_load_mb / streaming_mb:.1f}x")
    
    # Close dataset file handles before cleanup
    del dataset
    del chunked_dataset
    for loader in loaders.values():
        if hasattr(loader.dataset, '_local'):
            if hasattr(loader.dataset._local, 'file') and loader.dataset._local.file:
                try:
                    loader.dataset._local.file.close()
                except:
                    pass
    del loaders
    
    # Cleanup
    import os
    import gc
    gc.collect()  # Force garbage collection
    
    try:
        os.unlink(tmp_path)
    except PermissionError:
        print(f"  ⚠ Could not delete temp file (Windows lock): {tmp_path}")
    
    print("\n✅ All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_streaming_dataset()

