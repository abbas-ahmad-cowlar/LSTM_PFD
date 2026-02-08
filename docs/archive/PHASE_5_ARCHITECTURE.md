# Phase 5: Time-Frequency Analysis Architecture

## Overview
Phase 5 implements 2D CNN architectures operating on time-frequency representations (spectrograms, wavelets, Wigner-Ville) to capture frequency evolution patterns.

## Data Storage Architecture

### 1. MAT Files Location (1430 files)

```
data/
├── raw/                           # Store original 1430 .mat files here
│   ├── bearing_data/             # Main bearing fault signals
│   │   ├── normal/               # ~130 files
│   │   ├── ball_fault/           # ~130 files
│   │   ├── inner_race/           # ~130 files
│   │   ├── outer_race/           # ~130 files
│   │   ├── combined/             # ~130 files
│   │   ├── imbalance/            # ~130 files
│   │   ├── misalignment/         # ~130 files
│   │   ├── oil_whirl/            # ~130 files
│   │   ├── cavitation/           # ~130 files
│   │   ├── looseness/            # ~130 files
│   │   └── oil_deficiency/       # ~130 files
│   └── metadata/
│       ├── file_index.json       # Maps file_id -> fault_type, severity, etc.
│       └── dataset_stats.json    # Overall dataset statistics
├── processed/                     # Cached processed data
│   ├── signals_cache.h5          # All signals in HDF5 format
│   │                             # Structure: /fault_type/signal_id -> [102400]
│   ├── features_phase1.npz       # Classical ML features (Phase 1)
│   └── splits/
│       ├── train_indices.npy     # Training set indices
│       ├── val_indices.npy       # Validation set indices
│       └── test_indices.npy      # Test set indices
├── spectrograms/                  # Phase 5: Precomputed TFR
│   ├── stft/                     # STFT spectrograms
│   │   ├── train_stft.npz        # Shape: [N_train, 129, 400]
│   │   ├── val_stft.npz
│   │   └── test_stft.npz
│   ├── cwt/                      # CWT scalograms
│   │   ├── train_cwt.npz         # Shape: [N_train, 128, T]
│   │   ├── val_cwt.npz
│   │   └── test_cwt.npz
│   └── wvd/                      # Wigner-Ville distributions
│       ├── train_wvd.npz
│       ├── val_wvd.npz
│       └── test_wvd.npz
└── phase_5/                       # Phase 5 specific
    ├── tfr_config.json           # TFR generation parameters
    └── cache_metadata.json       # Cache validation info
```

### 2. Directory Size Estimates

- **raw/** (1430 MAT files): ~5-10 GB (depending on MAT format)
- **processed/signals_cache.h5**: ~2 GB (float32)
- **spectrograms/stft/**: ~600 MB (1430 × [129×400] × float32)
- **spectrograms/cwt/**: ~800 MB (higher resolution)
- **spectrograms/wvd/**: ~800 MB

**Total: ~12-15 GB**

## MATLAB Importer Integration

### Usage Workflow

```python
# 1. ONE-TIME: Import all 1430 MAT files into HDF5 cache
from data.matlab_importer import MATDatasetImporter

importer = MATDatasetImporter(
    mat_files_dir='data/raw/bearing_data/',
    output_cache='data/processed/signals_cache.h5'
)

# Batch import with progress bar
dataset_info = importer.import_all_mat_files(
    validate=True,           # Check signal quality
    generate_splits=True,    # Create train/val/test splits
    split_ratios=(0.7, 0.15, 0.15)
)

print(f"Imported {dataset_info['total_signals']} signals")
print(f"Classes: {dataset_info['class_distribution']}")
```

```python
# 2. NORMAL USE: Load cached signals (fast)
from data.dataset import BearingFaultDataset

dataset = BearingFaultDataset(
    cache_file='data/processed/signals_cache.h5',
    split='train'
)

# Access signal
signal, label = dataset[0]  # Returns (signal, label_int)
```

## Phase Independence Architecture

### Dependency Declaration

Each phase declares its dependencies in a config file:

```yaml
# config/phase_5_config.yaml
phase:
  name: "Phase 5: Time-Frequency Analysis"
  dependencies:
    required:
      - phase: "Phase 0"
        resources: ["data/processed/signals_cache.h5"]
        fallback: "auto_import_from_raw"
    optional:
      - phase: "Phase 1"
        resources: ["models/classical/best_rf.pkl"]
        purpose: "baseline_comparison"
      - phase: "Phase 2"
        resources: ["models/cnn/best_cnn_1d.pth"]
        purpose: "dual_stream_fusion"
```

### Standalone Execution

```python
# Phase 5 can run independently
from phase_5.runner import Phase5Runner

runner = Phase5Runner(config='config/phase_5_config.yaml')

# Automatically checks dependencies
runner.check_dependencies()
# Output:
#   ✓ Phase 0 cache found
#   ⚠ Phase 1 model not found (optional, skipping baseline comparison)
#   ✓ Phase 2 model found

# Run Phase 5
results = runner.run(
    mode='train',              # or 'evaluate', 'precompute_tfr'
    models=['resnet2d', 'efficientnet2d', 'dual_stream']
)
```

### Cache Validation & Regeneration

```python
# Automatic cache validation
from data.cache_validator import validate_cache

cache_valid = validate_cache(
    cache_path='data/processed/signals_cache.h5',
    expected_signals=1430,
    expected_shape=(102400,)
)

if not cache_valid:
    print("Cache invalid, regenerating from MAT files...")
    importer.import_all_mat_files()
```

## Implementation Stages

### Stage 1: Core Infrastructure (Files 1-10)

**Week 1: Time-Frequency Transforms**
1. `data/spectrogram_generator.py` - STFT implementation
2. `data/wavelet_transform.py` - CWT implementation
3. `data/wigner_ville.py` - WVD implementation
4. `data/tfr_dataset.py` - PyTorch Dataset for spectrograms

**Week 2: 2D CNN Models**
5. `models/spectrogram_cnn/resnet2d_spectrogram.py` - ResNet-2D
6. `models/spectrogram_cnn/efficientnet2d_spectrogram.py` - EfficientNet-2D
7. `models/spectrogram_cnn/__init__.py` - Package init

**Week 3: Training Infrastructure**
8. `training/spectrogram_trainer.py` - Spectrogram-specific trainer
9. `evaluation/spectrogram_evaluator.py` - Evaluation metrics
10. `scripts/train_spectrogram_cnn.py` - Training script

**Stage 1 Deliverable:** Can train ResNet-2D on STFT spectrograms end-to-end

### Stage 2: Advanced Features (Files 11-14)

**Week 4: Augmentation & Advanced Models**
11. `data/spectrogram_augmentation.py` - SpecAugment, MixUp
12. `models/spectrogram_cnn/dual_stream_cnn.py` - Time + Frequency fusion
13. `data/contrast_learning_tfr.py` - Contrastive learning (optional)

**Week 5: Evaluation & Visualization**
14. `evaluation/time_vs_frequency_comparison.py` - Systematic comparison
15. `visualization/spectrogram_plots.py` - Plotting utilities
16. `visualization/activation_maps_2d.py` - Grad-CAM for 2D

**Stage 2 Deliverable:** Full Phase 5 capabilities with comparison to Phase 1-4

## Data Loading Flow

### Offline Preprocessing (One-time, ~10 minutes)

```
┌──────────────────────────────────────────────────────────┐
│ 1. MAT Import (First Time Only)                          │
│    data/matlab_importer.py                               │
│    ├─ Scan data/raw/bearing_data/ (1430 files)          │
│    ├─ Load each .mat file                               │
│    ├─ Extract signal array [102400]                     │
│    ├─ Extract metadata (fault_type, severity, RPM, etc.)│
│    └─ Save to data/processed/signals_cache.h5           │
│         Time: ~5 minutes                                 │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ 2. Spectrogram Precomputation (Phase 5 specific)        │
│    scripts/precompute_spectrograms.py                    │
│    ├─ Load all signals from cache                       │
│    ├─ For each signal:                                  │
│    │   ├─ Compute STFT: [102400] → [129, 400]          │
│    │   ├─ Log-scale + normalize                         │
│    │   └─ Store in RAM                                  │
│    ├─ Save batch to data/spectrograms/stft/train.npz   │
│         Time: ~10 minutes for 1430 signals              │
└──────────────────────────────────────────────────────────┘
```

### Online Training (Every epoch, ~2 minutes/epoch)

```
┌──────────────────────────────────────────────────────────┐
│ 3. DataLoader (Training Loop)                            │
│    data/tfr_dataset.py                                   │
│    ├─ Load precomputed spectrograms (mmap)              │
│    ├─ Apply augmentation (SpecAugment)                  │
│    └─ Return batch [B, 1, 129, 400]                     │
│         Time: ~50ms per batch (B=32)                    │
└──────────────────────────────────────────────────────────┘
```

## Model Architecture Decisions

### 1. STFT Parameters
- **Window size**: 256 samples (nperseg)
- **Overlap**: 128 samples (50%)
- **Rationale**: 256 samples = 12.5ms at 20.48 kHz, captures 1-2 bearing rotation cycles at 1800 RPM
- **Output**: [129 freq bins, 400 time frames]

### 2. Transfer Learning
- **Source**: ImageNet pretrained ResNet-18
- **Adaptation**:
  - Conv1: 3 channels → 1 channel (grayscale spectrogram)
  - Rest of network: Keep pretrained weights
  - Fine-tune all layers with lr=1e-4
- **Expected gain**: +2-3% accuracy vs random init

### 3. Dual-Stream Architecture
```
Input Signal [B, 1, 102400]
      ├─────────────────────┬────────────────────┐
      │                     │                    │
 1D CNN Branch        Spectrogram Gen       (optional)
  [Phase 2 model]          │
      │                 2D CNN Branch
      │              [ResNet-2D/EfficientNet-2D]
      │                     │
      ├─────────────────────┴─────────────────────┐
      │                                            │
[B, 512] features                          [B, 512] features
      │                                            │
      └─────────────────── Concat ─────────────────┘
                            │
                       [B, 1024]
                            │
                         FC Layer
                            │
                        [B, 11] (predictions)
```

**When to use Dual-Stream:**
- If Phase 2 (1D CNN) already trained → reuse as time-domain branch
- Expected: +1-2% over best single-stream
- Trade-off: 2× inference time, 2× parameters

## Independent Phase Execution

### Scenario 1: Phase 5 Only (No Phase 0-4)

```bash
# User only has MAT files in data/raw/bearing_data/

# Step 1: Import MAT files
python scripts/import_mat_dataset.py \
    --mat_dir data/raw/bearing_data/ \
    --output data/processed/signals_cache.h5

# Step 2: Precompute spectrograms
python scripts/precompute_spectrograms.py \
    --signals_cache data/processed/signals_cache.h5 \
    --output_dir data/spectrograms/stft/ \
    --tfr_type stft

# Step 3: Train 2D CNN
python scripts/train_spectrogram_cnn.py \
    --config config/phase_5_config.yaml \
    --model resnet2d
```

### Scenario 2: Phase 5 After Phase 1-4

```bash
# Cache and models already exist from previous phases

# Step 1: Precompute spectrograms (uses existing cache)
python scripts/precompute_spectrograms.py

# Step 2: Train 2D CNN
python scripts/train_spectrogram_cnn.py --model resnet2d

# Step 3: Compare with Phase 1-4 models
python scripts/compare_all_phases.py \
    --phase1_model models/classical/best_rf.pkl \
    --phase2_model models/cnn/best_cnn_1d.pth \
    --phase3_model models/resnet/best_resnet_1d.pth \
    --phase5_model models/spectrogram_cnn/best_resnet2d.pth
```

### Scenario 3: Re-run Phase 5 Independently

```bash
# User wants to experiment with different TFR types

# CWT instead of STFT
python scripts/precompute_spectrograms.py --tfr_type cwt
python scripts/train_spectrogram_cnn.py --tfr_type cwt --model resnet2d

# Wigner-Ville
python scripts/precompute_spectrograms.py --tfr_type wvd
python scripts/train_spectrogram_cnn.py --tfr_type wvd --model resnet2d
```

## Configuration Management

### Phase 5 Config File

```yaml
# config/phase_5_config.yaml

data:
  mat_files_dir: "data/raw/bearing_data/"
  signals_cache: "data/processed/signals_cache.h5"
  spectrograms_dir: "data/spectrograms/"

  tfr_params:
    stft:
      nperseg: 256
      noverlap: 128
      window: 'hann'
      nfft: 256
    cwt:
      wavelet: 'morl'
      scales: 128
      scale_spacing: 'log'
    wvd:
      smoothing_window: 11

  normalization:
    method: 'log_db'          # '10*log10(power)' or 'standardize'
    per_sample: true          # Normalize each spectrogram independently

model:
  architecture: 'resnet2d'    # 'resnet2d', 'efficientnet2d', 'dual_stream'
  input_shape: [1, 129, 400]  # [C, H, W] for STFT
  num_classes: 11

  resnet2d:
    depth: 18                 # 18, 34, 50
    pretrained: true          # ImageNet transfer learning
    freeze_backbone: false

  dual_stream:
    time_branch: 'models/cnn/best_cnn_1d.pth'
    freq_branch: 'resnet2d'
    fusion_dim: 1024

training:
  batch_size: 32
  epochs: 100
  optimizer:
    type: 'adamw'
    lr: 1e-3
    weight_decay: 1e-4
  scheduler:
    type: 'cosine'
    T_max: 100
    eta_min: 1e-6
  augmentation:
    time_mask: 0.1            # SpecAugment
    freq_mask: 0.1
    mixup_alpha: 0.4

evaluation:
  metrics: ['accuracy', 'f1_macro', 'confusion_matrix', 'auc']
  compare_with: ['phase1', 'phase2', 'phase3']  # Baseline models
  robustness_tests: true
```

## Dependency Resolver

```python
# utils/dependency_resolver.py

class PhaseDependen cyResolver:
    """Automatically resolve and satisfy phase dependencies."""

    def __init__(self, phase_config):
        self.config = phase_config
        self.phase_name = phase_config['phase']['name']

    def check_dependencies(self):
        """Check if all required dependencies are satisfied."""
        results = {
            'required': {},
            'optional': {}
        }

        # Check required dependencies
        for dep in self.config['phase']['dependencies']['required']:
            resource_path = dep['resources'][0]
            exists = os.path.exists(resource_path)
            results['required'][dep['phase']] = {
                'satisfied': exists,
                'resource': resource_path,
                'fallback': dep.get('fallback')
            }

        # Check optional dependencies
        for dep in self.config['phase']['dependencies'].get('optional', []):
            resource_path = dep['resources'][0]
            exists = os.path.exists(resource_path)
            results['optional'][dep['phase']] = {
                'satisfied': exists,
                'purpose': dep['purpose']
            }

        return results

    def resolve_dependencies(self, auto_fix=True):
        """Attempt to automatically resolve missing dependencies."""
        dep_status = self.check_dependencies()

        for phase, status in dep_status['required'].items():
            if not status['satisfied']:
                if auto_fix and status['fallback']:
                    print(f"⚠ {phase} dependency missing, running fallback: {status['fallback']}")
                    self._run_fallback(status['fallback'])
                else:
                    raise DependencyError(
                        f"Required dependency missing: {status['resource']}\n"
                        f"Please run {phase} first or provide the resource."
                    )

    def _run_fallback(self, fallback_action):
        """Execute fallback action to satisfy dependency."""
        if fallback_action == 'auto_import_from_raw':
            from data.matlab_importer import MATDatasetImporter
            importer = MATDatasetImporter(...)
            importer.import_all_mat_files()
```

## Performance Considerations

### Memory Usage

- **Precomputed spectrograms in RAM**: ~600 MB (manageable)
- **GPU memory during training**: ~4 GB (batch_size=32, ResNet-18)
- **On-the-fly TFR computation**: 10× slower, but only ~1 GB RAM

**Recommendation**: Precompute spectrograms for fast training

### Training Time

- **Spectrogram precomputation**: ~10 min (one-time)
- **Training (100 epochs, ResNet-2D)**: ~2 hours (RTX 3080)
- **Inference**: ~5 ms per sample (50× faster than required 100ms)

### Disk I/O Optimization

```python
# Use memory-mapped arrays for large spectrogram files
spectrograms = np.load('data/spectrograms/stft/train.npz', mmap_mode='r')['spectrograms']
# Lazy loading: only load spectrograms when accessed
```

## Summary

This architecture ensures:
✅ **Clear data organization** - 1430 MAT files stored in `data/raw/`
✅ **Efficient caching** - HDF5 for signals, NPZ for spectrograms
✅ **Phase independence** - Can run Phase 5 standalone or with dependencies
✅ **Staged implementation** - Stage 1 (core) → Stage 2 (advanced)
✅ **MATLAB importer integration** - One-time batch import
✅ **Flexibility** - Easy to experiment with STFT, CWT, or WVD
✅ **Performance** - Precomputed spectrograms for fast training

**Next Steps:**
1. Place 1430 MAT files in `data/raw/bearing_data/`
2. Run `scripts/import_mat_dataset.py` (one-time)
3. Proceed with Phase 5 Stage 1 implementation
