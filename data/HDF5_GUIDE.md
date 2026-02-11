# HDF5 Guide

> Schema reference, reading/writing patterns, and migration instructions for HDF5 datasets in LSTM_PFD.

## Overview

The LSTM_PFD project uses HDF5 (`.h5`) as its primary binary storage format for cached signal datasets. HDF5 provides efficient random access, built-in compression, and a hierarchical structure that naturally maps to the train/val/test split paradigm used throughout the training pipeline.

All HDF5 I/O goes through two pathways:

1. **`CacheManager`** (`data/cache_manager.py`) — Writes datasets to HDF5 with optional compression and train/val/test splitting.
2. **`StreamingHDF5Dataset`** (`data/streaming_hdf5_dataset.py`) — Reads HDF5 files on-demand as a PyTorch `Dataset`, avoiding loading the entire dataset into RAM.

## HDF5 Schema

### Flat Cache Schema

Created by `CacheManager.cache_dataset()`. Used when you want a simple dump of signals and labels without splits.

```
{cache_name}.h5
│
├── signals          Dataset (N, signal_length)  dtype=float32
│                    compression='gzip', compression_opts=4
│
├── labels           Dataset (N,)                dtype=int
│
├── metadata         Dataset (N,)                dtype=string (UTF-8)
│                    (optional — JSON-serialized dicts)
│
└── Attributes
    ├── cache_name      string    — Name identifier
    ├── num_signals     int       — Number of signals stored
    ├── signal_length   int       — Length of each signal
    ├── cached_at       string    — ISO 8601 timestamp
    └── compression     string    — 'gzip', 'lzf', or 'none'
```

### Split Cache Schema

Created by `CacheManager.cache_dataset_with_splits()`. This is the structure expected by `StreamingHDF5Dataset`, the dashboard's `deep_learning_adapter.py`, and the standard training loops.

```
{cache_name}.h5
│
├── train/
│   ├── signals      Dataset (N_train, signal_length)  gzip
│   ├── labels       Dataset (N_train,)
│   └── Attributes
│       └── num_samples  int
│
├── val/
│   ├── signals      Dataset (N_val, signal_length)    gzip
│   ├── labels       Dataset (N_val,)
│   └── Attributes
│       └── num_samples  int
│
├── test/
│   ├── signals      Dataset (N_test, signal_length)   gzip
│   ├── labels       Dataset (N_test,)
│   └── Attributes
│       └── num_samples  int
│
├── metadata         Dataset (optional, JSON strings)
│
└── Attributes
    ├── cache_name      string
    ├── num_signals     int       — Total signals (all splits)
    ├── signal_length   int
    ├── cached_at       string    — ISO 8601 timestamp
    ├── split_ratios    tuple     — e.g. (0.7, 0.15, 0.15)
    ├── stratified      bool      — Whether splits are stratified
    └── random_seed     int       — Seed for reproducibility
```

### Signal Generator Schema

Created by `SignalGenerator.save_dataset(..., format='hdf5')`. Extends the split schema with additional global attributes.

```
dataset.h5
│
├── train/ val/ test/          (same as Split Cache Schema)
│
└── Attributes
    ├── num_classes        int       — e.g. 11
    ├── sampling_rate      int       — e.g. 20480
    ├── signal_length      int       — e.g. 102400
    ├── generation_date    string    — ISO 8601
    ├── split_ratios       tuple
    └── rng_seed           int
```

## Reading HDF5 Data

### With CacheManager (loads into memory)

```python
from data.cache_manager import CacheManager

cache = CacheManager(cache_dir='./cache')
signals, labels, metadata = cache.load_cached_dataset('my_dataset')
# signals: np.ndarray (N, 102400)
# labels: np.ndarray (N,)
# metadata: Optional[List[Dict]]
```

### With StreamingHDF5Dataset (on-demand, memory-efficient)

```python
from data.streaming_hdf5_dataset import StreamingHDF5Dataset
from torch.utils.data import DataLoader

dataset = StreamingHDF5Dataset(
    hdf5_path='data/processed/dataset.h5',
    split='train',
    cache_size=128  # LRU cache for hot samples
)

loader = DataLoader(dataset, batch_size=32, num_workers=4)

for signal_batch, label_batch in loader:
    # signal_batch: torch.Tensor [32, 1, 102400]
    # label_batch: torch.Tensor [32]
    pass
```

### With ChunkedStreamingDataset (prefetching for HDDs)

```python
from data.streaming_hdf5_dataset import ChunkedStreamingDataset

dataset = ChunkedStreamingDataset(
    hdf5_path='data/processed/dataset.h5',
    split='train',
    chunk_size=256  # Read 256 samples at a time
)
```

### One-call DataLoader factory

```python
from data.streaming_hdf5_dataset import create_streaming_dataloaders

loaders = create_streaming_dataloaders(
    hdf5_path='data/processed/dataset.h5',
    batch_size=32,
    num_workers=4,
    use_chunked=False
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

### Direct h5py access

```python
import h5py

with h5py.File('data/processed/dataset.h5', 'r') as f:
    # List top-level groups
    print(list(f.keys()))  # ['train', 'val', 'test', 'metadata']

    # Read global attributes
    print(f.attrs['num_classes'])     # 11
    print(f.attrs['sampling_rate'])   # 20480

    # Read a split
    train_signals = f['train']['signals'][:]   # loads all into RAM
    train_labels = f['train']['labels'][:]

    # Read a single sample (lazy)
    first_signal = f['train']['signals'][0]    # reads only index 0

    # Read a slice
    batch = f['train']['signals'][10:42]       # reads indices 10–41
```

## Writing HDF5 Data

### Via CacheManager (flat)

```python
from data.cache_manager import CacheManager
import numpy as np

cache = CacheManager(cache_dir='data/cache')

cache.cache_dataset(
    signals=np.random.randn(500, 102400).astype(np.float32),
    labels=np.arange(500) % 11,
    metadata=[{'fault': 'sain', 'index': i} for i in range(500)],
    cache_name='experiment_01',
    compression='gzip',       # 'gzip', 'lzf', or None
    compression_opts=4         # gzip level 1–9
)
```

### Via CacheManager (with splits)

```python
cache.cache_dataset_with_splits(
    signals=signals,
    labels=labels,
    cache_name='experiment_01_split',
    split_ratios=(0.7, 0.15, 0.15),
    stratify=True,
    random_seed=42
)
```

### Via SignalGenerator

```python
from data.signal_generator import SignalGenerator
from config.data_config import DataConfig

config = DataConfig(num_signals_per_fault=130)
generator = SignalGenerator(config)
dataset = generator.generate_dataset()

paths = generator.save_dataset(
    dataset,
    output_dir='data/processed',
    format='hdf5',                           # 'mat', 'hdf5', or 'both'
    train_val_test_split=(0.7, 0.15, 0.15)
)
```

### Convenience function (minimal)

```python
from data.cache_manager import cache_dataset_simple, load_cached_dataset_simple
from pathlib import Path

cache_dataset_simple(signals, labels, Path('cache/quick.h5'))
sig, lab, meta = load_cached_dataset_simple(Path('cache/quick.h5'))
```

## Dataset Structure Within HDF5 Files

### Signal Datasets

- **Shape**: `(N, signal_length)` where `signal_length` is typically `102400` (5 seconds × 20480 Hz sampling rate).
- **Dtype**: `float32`
- **Compression**: gzip level 4 by default. Achieves `[PENDING BENCHMARKS]` compression ratio.

### Label Datasets

- **Shape**: `(N,)`
- **Dtype**: `int32` or `int64`
- **Encoding**: Integer labels `0–10` mapping to fault types:

| Label | Fault Type (French)        | English Translation |
| ----- | -------------------------- | ------------------- |
| 0     | `sain`                     | Healthy             |
| 1     | `desalignement`            | Misalignment        |
| 2     | `desequilibre`             | Imbalance           |
| 3     | `jeu`                      | Clearance           |
| 4     | `lubrification`            | Lubrication         |
| 5     | `cavitation`               | Cavitation          |
| 6     | `usure`                    | Wear                |
| 7     | `oilwhirl`                 | Oil Whirl           |
| 8     | `mixed_misalign_imbalance` | Mixed Fault 1       |
| 9     | `mixed_wear_lube`          | Mixed Fault 2       |
| 10    | `mixed_cavit_jeu`          | Mixed Fault 3       |

### Metadata Datasets

- **Shape**: `(N,)`
- **Dtype**: Variable-length UTF-8 strings (`h5py.string_dtype(encoding='utf-8')`)
- **Content**: JSON-serialized dictionaries, one per signal.

## Migration from .mat to HDF5

### Converting a directory of .mat files

```python
from data.matlab_importer import MatlabImporter, load_mat_dataset
from data.cache_manager import CacheManager

# Load all .mat files organized by fault type subdirectories
signals, labels, label_names = load_mat_dataset('data/raw/bearing_data')

# Cache as HDF5 with splits
cache = CacheManager(cache_dir='data/processed')
cache.cache_dataset_with_splits(
    signals=signals,
    labels=labels,
    cache_name='bearing_dataset',
    split_ratios=(0.7, 0.15, 0.15),
    stratify=True
)
```

### Converting individual .mat files

```python
from data.matlab_importer import MatlabImporter
from data.cache_manager import cache_dataset_simple
from pathlib import Path
import numpy as np

importer = MatlabImporter()
batch = importer.load_batch(Path('data/raw/matlab_signals'))
signals, labels = importer.extract_signals_and_labels(batch)

# Convert labels to integers
from utils.constants import FAULT_TYPES
label_to_int = {ft: i for i, ft in enumerate(FAULT_TYPES)}
int_labels = np.array([label_to_int.get(l, -1) for l in labels])

cache_dataset_simple(signals, int_labels, Path('data/processed/converted.h5'))
```

### Backward compatibility

- The default `SignalGenerator.save_dataset()` format remains `'mat'` — no breaking changes.
- Use `format='both'` during a transition period to generate both formats simultaneously.
- Use `format='hdf5'` once you are confident the HDF5 pipeline works correctly.

## Inspecting HDF5 Files

```python
import h5py

with h5py.File('data/processed/dataset.h5', 'r') as f:
    print("=== File Structure ===")
    print(f"Groups: {list(f.keys())}")
    print(f"Global attributes: {dict(f.attrs)}")

    for split in ['train', 'val', 'test']:
        if split in f:
            grp = f[split]
            print(f"\n--- {split} ---")
            print(f"  Signals: {grp['signals'].shape} {grp['signals'].dtype}")
            print(f"  Labels:  {grp['labels'].shape} {grp['labels'].dtype}")
            print(f"  Samples: {grp.attrs.get('num_samples', 'N/A')}")
```

## Compression Options

| Algorithm | `compression` | `compression_opts`     | Speed    | Ratio | Notes                           |
| --------- | ------------- | ---------------------- | -------- | ----- | ------------------------------- |
| gzip      | `'gzip'`      | `1`–`9` (default: `4`) | Moderate | Good  | Best general-purpose choice     |
| LZF       | `'lzf'`       | —                      | Fast     | Fair  | Better for read-heavy workloads |
| None      | `None`        | —                      | Fastest  | None  | Use for temporary caches        |

## Troubleshooting

### `ModuleNotFoundError: No module named 'h5py'`

```bash
pip install h5py>=3.8.0
```

### `ValueError: Split 'X' not found`

The HDF5 file was created with `cache_dataset()` (flat schema) but you're trying to access a split. Use `load_cached_dataset()` instead, or re-create the cache with `cache_dataset_with_splits()`.

```python
import h5py
with h5py.File('path.h5', 'r') as f:
    print("Available keys:", list(f.keys()))
```

### `PermissionError` on Windows

HDF5 files cannot be deleted while a file handle is open. Ensure all `h5py.File` objects are closed (use `with` statements) and that no `StreamingHDF5Dataset` instances are holding references.

### Slow random access

- Enable chunked prefetching: `ChunkedStreamingDataset(path, chunk_size=256)`
- Use SSD storage instead of HDD
- Reduce compression level: `compression_opts=1`

## Related Documentation

- [Storage README](STORAGE_README.md) — Module overview and API summary
- [Signal Generation (IDB 3.1)](SIGNAL_GENERATION_README.md) — Upstream signal producer
- [Data Loading (IDB 3.2)](DATA_LOADING_README.md) — Downstream data consumer
