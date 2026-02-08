# Shared Utilities

> Cross-cutting helper modules consumed by every domain in the LSTM_PFD project.

## Overview

The `utils/` package centralises reusable functions and classes that would otherwise be duplicated across modules. Every import is re-exported from `utils/__init__.py`, so consumers can write `from utils import get_device, set_seed, save_json`.

## Utility Catalog

| Utility            | File                     | Purpose                                                                                    | Key Exports                                                                                                                                                                  |
| ------------------ | ------------------------ | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Constants          | `constants.py`           | Project-wide magic numbers — signal params, fault types, training defaults, web-app limits | `SAMPLING_RATE`, `SIGNAL_LENGTH`, `NUM_CLASSES`, `FAULT_TYPES`, `FAULT_TYPE_TO_ID`, `get_fault_id()`, `get_fault_name()`, 100+ constants                                     |
| Device Manager     | `device_manager.py`      | PyTorch GPU/CPU selection, memory monitoring, multi-GPU support                            | `get_device()`, `move_to_device()`, `get_gpu_memory_usage()`, `clear_gpu_memory()`, `DeviceManager` (context manager)                                                        |
| File I/O           | `file_io.py`             | Pickle / JSON / YAML serialization, directory management, file operations                  | `save_pickle()`, `load_pickle()`, `save_json()`, `load_json()`, `save_yaml()`, `load_yaml()`, `ensure_dir()`, `list_files()`, `safe_save()`, `safe_load()`                   |
| Logging            | `logging.py`             | Structured logging with console + file outputs                                             | `setup_logging()`, `get_logger()`, `log_system_info()`                                                                                                                       |
| Logger (compat)    | `logger.py`              | Compatibility shim for dashboard components expecting `setup_logger()`                     | `setup_logger()`                                                                                                                                                             |
| Reproducibility    | `reproducibility.py`     | Seed control for Python, NumPy, and PyTorch                                                | `set_seed()`, `make_deterministic()`, `get_random_state()`, `restore_random_state()`                                                                                         |
| Timer              | `timer.py`               | Timing context manager, decorator, profiler, and benchmark utilities                       | `Timer`, `TimingStats`, `Profiler`, `time_function()`, `benchmark()`, `format_time()`, `get_global_profiler()`                                                               |
| Visualization      | `visualization_utils.py` | Matplotlib helpers — style management, figure saving, common plot types                    | `set_plot_style()`, `save_figure()`, `create_figure()`, `get_color_palette()`, `plot_time_series()`, `plot_spectrum()`, `plot_confusion_matrix()`, `plot_training_history()` |
| Checkpoint Manager | `checkpoint_manager.py`  | Save/load model checkpoints with top-k retention                                           | `CheckpointManager`                                                                                                                                                          |
| Early Stopping     | `early_stopping.py`      | Stop training when validation metric stalls                                                | `EarlyStopping`, `EarlyStoppingWithWarmup`                                                                                                                                   |

## Quick Start

```python
from utils import (
    set_seed, make_deterministic,
    get_device, move_to_device,
    save_json, load_json,
    get_logger, setup_logging,
    Timer, SAMPLING_RATE, FAULT_TYPES
)

# Reproducibility
set_seed(42)
make_deterministic()

# Logging
setup_logging()
logger = get_logger(__name__)
logger.info("Starting training...")

# Device management
device = get_device(prefer_gpu=True)
batch = move_to_device({'signals': tensor, 'labels': labels}, device)

# Timing
with Timer("forward_pass"):
    output = model(batch['signals'])

# File I/O
save_json({'accuracy': 0.95}, 'results/metrics.json')
```

## Module Details

### `constants.py` (629 lines)

The single source of truth for all numerical constants. Sections include:

| Section                  | Example Constants                                                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Signal Parameters        | `SAMPLING_RATE=20480`, `SIGNAL_LENGTH=102400`, `SIGNAL_DURATION=5.0`                                                              |
| Fault Classification     | `NUM_CLASSES=11`, `FAULT_TYPES` (list of 11 French names), `FAULT_TYPE_DISPLAY_NAMES` (English translations), `FAULT_LABELS_PINN` |
| Model Architecture       | `DEFAULT_CNN_CHANNELS=[32,64,128,256]`, `DEFAULT_RESNET_BLOCKS=[2,2,2,2]`, `DEFAULT_DROPOUT=0.3`                                  |
| Training Hyperparameters | `DEFAULT_LEARNING_RATE=0.001`, `DEFAULT_BATCH_SIZE=32`, `DEFAULT_NUM_EPOCHS=100`                                                  |
| Data Generation          | `DEFAULT_NUM_SIGNALS_PER_FAULT=100`, `SEVERITY_LEVELS`, `SEVERITY_RANGES`                                                         |
| File Paths               | `DEFAULT_CHECKPOINT_DIR`, `DEFAULT_LOG_DIR`, `DEFAULT_RESULTS_DIR`                                                                |
| Web Application          | Upload limits, pagination, cache TTL, rate limits, HTTP status codes, etc.                                                        |

Utility functions: `get_fault_id(name) → int`, `get_fault_name(id) → str`, `get_fault_display_name(name) → str`, `validate_signal_length(length) → bool`.

### `device_manager.py` (419 lines)

```python
device = get_device(prefer_gpu=True, gpu_id=None)       # Auto-select best device
gpus = get_available_gpus()                               # List GPU IDs
info = get_gpu_info()                                     # Detailed GPU dict
batch = move_to_device(data, device, non_blocking=False)  # Handles tensors, lists, dicts
mem = get_gpu_memory_usage()                              # {'allocated_gb', 'reserved_gb', ...}
clear_gpu_memory()                                        # Free cached memory
log_device_info()                                         # Logs GPU/CPU info
n = get_optimal_num_workers()                             # For DataLoader
synchronize_device()                                      # Wait for GPU ops

with DeviceManager(prefer_gpu=True) as device:
    model = MyModel().to(device)
```

### `checkpoint_manager.py` (434 lines)

```python
from utils.checkpoint_manager import CheckpointManager

mgr = CheckpointManager(
    checkpoint_dir=Path('checkpoints'),
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,
    mode='max',        # 'max' for accuracy, 'min' for loss
    save_top_k=3,
)

# During training
mgr.save_checkpoint(epoch=10, metric_value=0.95, metric_name='val_acc')

# Resume
checkpoint = mgr.load_checkpoint(path)     # Full state
checkpoint = mgr.load_best_checkpoint()    # Best only
mgr.load_weights_only(path)               # Inference / transfer learning

# Inspect
mgr.list_checkpoints()  # [(metric_value, path), ...]
mgr.get_best_metric()   # float
mgr.get_checkpoint_info(path)  # metadata dict
```

### `early_stopping.py` (375 lines)

```python
from utils.early_stopping import EarlyStopping, EarlyStoppingWithWarmup

es = EarlyStopping(patience=10, mode='max', min_delta=1e-4, restore_best_weights=True)

for epoch in range(num_epochs):
    val_acc = validate(model)
    if es(val_acc, model=model, epoch=epoch):
        print("Early stopping triggered!")
        break

# With warmup — ignores first N epochs
es2 = EarlyStoppingWithWarmup(patience=10, warmup_epochs=5)

# Checkpoint integration
state = es.state_dict()   # Save to checkpoint
es.load_state_dict(state) # Restore from checkpoint
```

### `timer.py` (452 lines)

```python
from utils import Timer, TimingStats, Profiler, time_function, benchmark

# Context manager
with Timer("training_loop"):
    train(model)

# Decorator
@Timer.decorator("inference")
def predict(model, x):
    return model(x)

# Accumulate stats
stats = TimingStats("forward_pass")
for batch in loader:
    with stats.timer():
        model(batch)
print(stats.summary())  # "forward_pass: mean=0.012s, median=0.011s, ..."
```

### `visualization_utils.py` (454 lines)

```python
from utils import set_plot_style, save_figure, create_figure, plot_time_series

set_plot_style('default')  # or 'seaborn', 'ggplot', 'dark'

fig, ax = create_figure(figsize=(12, 5))
plot_time_series(signal, fs=20480, ax=ax, title='Vibration Signal')
save_figure(fig, 'plots/signal.png', dpi=300)
```

Available palette names: `'default'`, `'vibrant'`, `'pastel'`, `'dark'`.

## Dependencies

- **External:** `torch`, `numpy`, `matplotlib`, `pyyaml`, `scipy`
- **Internal:** None (leaf module — no intra-project dependencies)

## Related Documentation

- [Integration README](../integration/README.md) — Uses `utils.constants` heavily
- [Configuration Guide](../config/CONFIGURATION_GUIDE.md) — IDB 4.4
