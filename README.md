# LSTM_PFD: Advanced Bearing Fault Diagnosis System

![Project Status](https://img.shields.io/badge/Status-Phase%207%20Complete-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Modularity & Architecture](#-modularity--architecture)
- [Project Phases](#-project-phases)
- [Complete Setup Guide](#-complete-setup-guide)
- [Phase-by-Phase Execution Guide](#-phase-by-phase-execution-guide)
- [Data Flow & Dependencies](#-data-flow--dependencies)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Future Phases](#-future-phases)

---

## 🎯 Project Overview

**LSTM_PFD** is a comprehensive bearing fault diagnosis system that implements state-of-the-art machine learning and deep learning techniques. The project progresses from classical ML approaches to cutting-edge physics-informed neural networks (PINNs) and explainable AI (XAI), providing a complete research and production pipeline.

### Key Features

- **Multi-Modal Approach**: Classical ML, 1D CNNs, Advanced CNNs, Transformers, Time-Frequency Analysis
- **11 Fault Classes**: Normal, ball fault, inner race, outer race, combined, imbalance, misalignment, oil whirl, cavitation, looseness, oil deficiency
- **1430+ Samples**: Comprehensive MATLAB dataset with vibration signals
- **Physics-Informed Models**: PINN integration for enhanced accuracy
- **Explainable AI**: SHAP, LIME, CAVs, Integrated Gradients, PDP
- **Production-Ready**: Modular architecture, comprehensive testing, optimization tools

### Performance Highlights

| Phase | Approach | Target Accuracy | Key Innovation |
|-------|----------|----------------|----------------|
| 1 | Classical ML | 95.33% | Feature engineering (36 features → 15 via MRMR) |
| 2 | 1D CNN | 93-95% | Deep feature learning, multi-scale kernels |
| 3 | Advanced CNNs | 96-97% | ResNet, EfficientNet, NAS |
| 4 | Transformer | 96-97% | Self-attention, temporal modeling |
| 5 | Time-Frequency | 96-98% | 2D CNNs on spectrograms/scalograms |
| 6 | PINN | 97-98% | Physics-informed constraints |
| 7 | XAI | N/A | Interpretability & trust |

---

## 🧩 Modularity & Architecture

### Design Philosophy

The project is designed with **maximum modularity** in mind. Each phase can be:

1. ✅ **Executed Independently** - Run any phase in isolation with appropriate input data
2. ✅ **Connected Sequentially** - Build upon previous phases for enhanced results
3. ✅ **Composed Flexibly** - Mix and match components across phases
4. ✅ **Extended Easily** - Add new models/techniques without breaking existing code

### How Phases Connect

```
┌─────────────────────────────────────────────────────────────────┐
│                     SHARED FOUNDATION                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  Data Layer   │  │ Model Layer   │  │Training Layer │      │
│  │  • MAT Import │  │ • BaseModel   │  │ • Trainers    │      │
│  │  • HDF5 Cache │  │ • Factory     │  │ • Optimizers  │      │
│  │  • Datasets   │  │ • Registry    │  │ • Callbacks   │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
         ↓                  ↓                    ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PHASE 0-1  │    │  PHASE 2-4  │    │  PHASE 5-7  │
│  Classical  │───→│  Deep CNNs  │───→│  Advanced   │
│  ML & Data  │    │& Transform. │    │  PINN & XAI │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Shared Components

**All phases share these core utilities:**

- **`data/`** - Unified data storage (raw MAT files, processed HDF5, spectrograms)
- **`models/base_model.py`** - Common interface for all models
- **`models/model_factory.py`** - Centralized model creation and management
- **`training/`** - Reusable trainers, optimizers, schedulers, callbacks
- **`evaluation/`** - Common evaluation metrics and visualization
- **`features/`** - Feature extraction (used by Phase 1, reusable in others)
- **`utils/`** - Logging, checkpointing, reproducibility tools

### Phase-Specific Components

Each phase has its own:

- **Models** - `models/{classical,cnn,resnet,efficientnet,transformer,pinn}/`
- **Trainers** - Specialized training loops (e.g., `pinn_trainer.py`, `cnn_trainer.py`)
- **Scripts** - Execution scripts in `scripts/` (e.g., `train_cnn.py`, `train_spectrogram_cnn.py`)
- **Pipelines** - End-to-end workflows in `pipelines/` (e.g., `classical_ml_pipeline.py`)

### Running Phases: Independent vs Connected

#### Independent Execution

Each phase can run standalone if you have the right input data:

```python
# Phase 1: Just need raw signals
from pipelines.classical_ml_pipeline import ClassicalMLPipeline
pipeline = ClassicalMLPipeline()
results = pipeline.run(signals, labels)

# Phase 2: Just need raw signals
from models import create_cnn1d
from training.cnn_trainer import CNNTrainer
model = create_cnn1d(num_classes=11)
trainer = CNNTrainer(model)
trainer.train(train_loader, val_loader)

# Phase 5: Just need raw signals (generates spectrograms internally)
from scripts.train_spectrogram_cnn import main
main()  # Handles everything from signals to training

# Phase 6: Just need raw signals (adds physics constraints)
from training.pinn_trainer import PINNTrainer
trainer = PINNTrainer(model, physics_weight=0.1)
trainer.train(train_loader, val_loader)
```

#### Connected Execution

Phases can leverage outputs from previous phases:

```python
# Use Phase 1 features in Phase 2
features = phase1_pipeline.get_selected_features()  # 15 MRMR features
cnn_model = create_cnn1d_with_features(features)

# Use Phase 2-4 models in Phase 8 ensemble
cnn_model = load_pretrained('phase2_cnn.pth')
resnet_model = load_pretrained('phase3_resnet.pth')
transformer_model = load_pretrained('phase4_transformer.pth')
ensemble = create_voting_ensemble([cnn_model, resnet_model, transformer_model])

# Use Phase 6 PINN predictions in Phase 7 XAI
pinn_model = load_pretrained('phase6_pinn.pth')
explainer = SHAPExplainer(pinn_model)
explanations = explainer.explain(test_data)
```

---

## 📊 Project Phases

### Completed Phases (0-7)

| Phase | Name | Duration | Status | Key Deliverables |
|-------|------|----------|--------|------------------|
| **0** | **Foundation** | 30 days | ✅ Complete | Data pipeline, PyTorch infrastructure, base classes |
| **1** | **Classical ML** | 23 days | ✅ Complete | Feature engineering (36→15), RF/SVM/GB, 95.33% accuracy |
| **2** | **1D CNN** | 27 days | ✅ Complete | CNN architecture, multi-scale kernels, 93-95% accuracy |
| **3** | **Advanced CNNs** | 34 days | ✅ Complete | ResNet-18/34, EfficientNet, NAS, 96-97% accuracy |
| **4** | **Transformer** | 29 days | ✅ Complete | Self-attention, positional encoding, 96-97% accuracy |
| **5** | **Time-Frequency** | 14 days | ✅ Complete | STFT/CWT/WVD, 2D CNNs, dual-stream, 96-98% accuracy |
| **6** | **PINN** | 16 days | ✅ Complete | Physics-informed models, conservation laws, 97-98% accuracy |
| **7** | **XAI** | 12 days | ✅ Complete | SHAP, LIME, CAVs, IG, PDP, interactive dashboard |

### Upcoming Phases (8-10)

| Phase | Name | Duration | Status | Target Metric |
|-------|------|----------|--------|---------------|
| **8** | **Ensemble** | 10 days | 🔄 Planned | Voting, stacking, fusion → 98-99% accuracy |
| **9** | **Deployment** | 14 days | 🔄 Planned | Quantization, ONNX, API, Docker → <50ms latency |
| **10** | **QA & Integration** | 25 days | 🔄 Planned | Testing, benchmarking, documentation → Production-ready |

---

## 🚀 Complete Setup Guide

### Prerequisites

```bash
# System requirements
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (32GB recommended)
- 50GB+ disk space
```

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/LSTM_PFD.git
cd LSTM_PFD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Requirements.txt** (Core dependencies):
```txt
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
h5py>=3.8.0
pywavelets>=1.4.0
tqdm>=4.65.0
optuna>=3.1.0
shap>=0.41.0
lime>=0.2.0.1
captum>=0.6.0
plotly>=5.14.0
streamlit>=1.22.0
```

### Step 2: Directory Structure Setup

```bash
# Create required directories
mkdir -p data/{raw/bearing_data,processed,spectrograms/{stft,cwt,wvd}}
mkdir -p data/raw/bearing_data/{normal,ball_fault,inner_race,outer_race,combined,imbalance,misalignment,oil_whirl,cavitation,looseness,oil_deficiency}
mkdir -p checkpoints/{phase1,phase2,phase3,phase4,phase5,phase6,phase7}
mkdir -p logs results visualizations
```

### Step 3: Dataset Preparation

#### Option A: Using Existing MATLAB Dataset

```bash
# Place your 1430 MAT files in data/raw/bearing_data/
# Organize by fault type (130 files per class × 11 classes)

data/raw/bearing_data/
├── normal/          # 130 files
├── ball_fault/      # 130 files
├── inner_race/      # 130 files
├── outer_race/      # 130 files
├── combined/        # 130 files
├── imbalance/       # 130 files
├── misalignment/    # 130 files
├── oil_whirl/       # 130 files
├── cavitation/      # 130 files
├── looseness/       # 130 files
└── oil_deficiency/  # 130 files

# Import to HDF5 cache (one-time operation)
python scripts/import_mat_dataset.py \
    --mat_dir data/raw/bearing_data/ \
    --output data/processed/signals_cache.h5 \
    --split-ratios 0.7 0.15 0.15
```

**Expected output:**
```
Found 1430 MAT files
Loading MAT files... 100%|███████████████| 1430/1430
Loaded 1430 signals, shape: (1430, 102400)
Train: 1001 samples | Val: 215 samples | Test: 214 samples
✓ Cache saved to data/processed/signals_cache.h5
```

#### Option B: Using Public Datasets

```bash
# Case Western Reserve University (CWRU) dataset
wget https://engineering.case.edu/bearingdatacenter/download-data-file
# Follow instructions at: https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website

# Or use built-in downloader (if available)
python scripts/download_cwru.py --output data/raw/cwru/
```

### Step 4: Verify Installation

```bash
# Run verification script
python -c "
import torch
import numpy as np
from models import create_cnn1d, list_available_models

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Available models: {list_available_models()}')

# Test model creation
model = create_cnn1d(num_classes=11)
print(f'✓ Model created: {model.__class__.__name__}')
"
```

---

## 🏃 Phase-by-Phase Execution Guide

### Phase 0: Foundation (Completed)

**Purpose**: Establish data infrastructure and base classes

**Key Components**:
- `models/base_model.py` - Abstract base class for all models
- `models/model_factory.py` - Centralized model creation
- `data/` directory structure
- `utils/` logging and reproducibility tools

**Verification**:
```bash
# No specific execution needed - foundation is integrated throughout
python -c "from models import BaseModel; print('✓ Phase 0 foundation verified')"
```

---

### Phase 1: Classical Machine Learning

**Purpose**: Baseline performance using handcrafted features

**Key Components**:
- Feature extraction: 36 statistical/frequency features
- Feature selection: MRMR → 15 optimal features
- Models: SVM, Random Forest, Gradient Boosting
- Hyperparameter optimization: Bayesian, Grid, Random Search

**Files Created**: 31 files (see `PHASE_1_USAGE_GUIDE.md`)

#### Running Phase 1

**Option 1: Quick Start (End-to-End Pipeline)**

```bash
# Run complete pipeline with default settings
python -c "
from pipelines.classical_ml_pipeline import ClassicalMLPipeline
import numpy as np
import h5py

# Load data
with h5py.File('data/processed/signals_cache.h5', 'r') as f:
    signals = f['signals'][:]
    labels = f['labels'][:]

# Run pipeline
pipeline = ClassicalMLPipeline(random_state=42)
results = pipeline.run(
    signals=signals,
    labels=labels,
    fs=20480,
    optimize_hyperparams=True,
    n_trials=50,
    save_dir='results/phase1'
)

print(f\"Test Accuracy: {results['test_accuracy']:.4f}\")
print(f\"Best Model: {results['best_model_type']}\")
"
```

**Option 2: Step-by-Step (Manual Control)**

```python
# Step 1: Feature Extraction
from features.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(fs=20480)
features = extractor.extract_features(signals)  # (1430, 36)

# Step 2: Feature Selection
from features.feature_selector import FeatureSelector

selector = FeatureSelector(method='mrmr', n_features=15)
X_selected = selector.fit_transform(features, labels)

# Step 3: Train/Val/Test Split
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X_selected, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Step 4: Normalize
from features.feature_normalization import FeatureNormalizer

normalizer = FeatureNormalizer(method='zscore')
X_train_norm = normalizer.fit_transform(X_train)
X_val_norm = normalizer.transform(X_val)
X_test_norm = normalizer.transform(X_test)

# Step 5: Hyperparameter Optimization
from training.bayesian_optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(n_trials=50)
best_params = optimizer.optimize(X_train_norm, y_train, X_val_norm, y_val)

# Step 6: Train Best Model
from models.classical.model_selector import ModelSelector

selector = ModelSelector()
model = selector.create_model('random_forest', **best_params)
model.fit(X_train_norm, y_train)

# Step 7: Evaluate
from evaluation.evaluator import Evaluator

evaluator = Evaluator()
test_acc = evaluator.evaluate(model, X_test_norm, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

**Expected Results**:
- Training time: ~15-30 minutes (with Bayesian optimization)
- Test accuracy: **95-96%**
- Selected features: 15 (from 36 original)
- Best model: Random Forest or Gradient Boosting

**Outputs**:
- `results/phase1/feature_importance.png`
- `results/phase1/confusion_matrix.png`
- `results/phase1/best_model.pkl`
- `results/phase1/metrics.json`

---

### Phase 2: 1D Convolutional Neural Networks

**Purpose**: Deep learning baseline with 1D CNNs

**Key Components**:
- Multi-scale convolutional kernels (3, 5, 7)
- Batch normalization and dropout
- Advanced training: mixed precision, gradient clipping
- Data augmentation: noise, scaling, time shifting

**Files Created**: 25+ files (see `PHASE_2_USAGE_GUIDE.md`)

#### Running Phase 2

**Quick Start**:

```bash
# Train 1D CNN with default config
python scripts/train_cnn.py \
    --model cnn1d \
    --data-path data/processed/signals_cache.h5 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --checkpoint-dir checkpoints/phase2 \
    --log-dir logs/phase2
```

**Advanced Configuration**:

```python
from models import create_cnn1d
from training.cnn_trainer import CNNTrainer
from torch.utils.data import DataLoader
import torch

# Create model
model = create_cnn1d(
    input_length=102400,
    num_classes=11,
    hidden_sizes=[64, 128, 256],
    kernel_sizes=[7, 5, 3],
    dropout=0.5
)

# Setup trainer
trainer = CNNTrainer(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    mixed_precision=True
)

# Configure training
trainer.configure(
    optimizer='adam',
    lr=0.001,
    scheduler='cosine',
    T_max=100
)

# Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    checkpoint_dir='checkpoints/phase2',
    early_stopping_patience=15
)

# Evaluate
from scripts.evaluate_cnn import evaluate
results = evaluate(
    model=model,
    test_loader=test_loader,
    checkpoint='checkpoints/phase2/best_model.pth'
)
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

**Expected Results**:
- Training time: ~2-4 hours (GPU) / ~10-15 hours (CPU)
- Test accuracy: **93-95%**
- Model size: ~5-10M parameters

**Outputs**:
- `checkpoints/phase2/best_model.pth`
- `logs/phase2/training_curves.png`
- `results/phase2/confusion_matrix.png`

---

### Phase 3: Advanced CNN Architectures

**Purpose**: State-of-the-art CNN performance with ResNet, EfficientNet, NAS

**Key Components**:
- **ResNet-18/34**: Residual connections for deep networks
- **EfficientNet**: Compound scaling for efficiency
- **NAS (Neural Architecture Search)**: Automated architecture discovery
- Advanced training techniques: knowledge distillation, progressive resizing

**Files Created**: 40+ files (see `PHASE_3_USAGE_GUIDE.md`)

#### Running Phase 3

**ResNet Training**:

```bash
# ResNet-18
python scripts/train_cnn.py \
    --model resnet18 \
    --data-path data/processed/signals_cache.h5 \
    --epochs 150 \
    --batch-size 32 \
    --lr 0.0001 \
    --scheduler cosine \
    --checkpoint-dir checkpoints/phase3/resnet18

# ResNet-34 (deeper)
python scripts/train_cnn.py \
    --model resnet34 \
    --data-path data/processed/signals_cache.h5 \
    --epochs 150 \
    --batch-size 16 \
    --lr 0.0001 \
    --gradient-accumulation 2 \
    --checkpoint-dir checkpoints/phase3/resnet34
```

**EfficientNet Training**:

```python
from models.efficientnet import create_efficientnet_1d
from training.cnn_trainer import CNNTrainer

# Create EfficientNet model
model = create_efficientnet_1d(
    num_classes=11,
    width_mult=1.0,
    depth_mult=1.0,
    resolution_mult=1.0
)

# Train with progressive resizing
from training.progressive_resizing import ProgressiveResizingTrainer

trainer = ProgressiveResizingTrainer(
    model=model,
    stages=[
        {'input_length': 25600, 'epochs': 30, 'lr': 0.001},
        {'input_length': 51200, 'epochs': 30, 'lr': 0.0005},
        {'input_length': 102400, 'epochs': 40, 'lr': 0.0001},
    ]
)

history = trainer.train(train_loader, val_loader)
```

**NAS (Neural Architecture Search)**:

```python
from models.nas import NASSearcher

# Search for optimal architecture
searcher = NASSearcher(
    search_space='auto',
    num_trials=50,
    metric='accuracy'
)

best_arch = searcher.search(train_loader, val_loader)
print(f"Best architecture: {best_arch}")

# Train discovered architecture
from models.nas import build_nas_model
model = build_nas_model(best_arch, num_classes=11)
# ... train as usual
```

**Expected Results**:
- Training time: ~4-8 hours (ResNet), ~6-12 hours (EfficientNet), ~20-30 hours (NAS)
- Test accuracy: **96-97%** (ResNet), **96.5-97.5%** (EfficientNet)
- Model size: 5-15M parameters (ResNet), 3-10M (EfficientNet)

---

### Phase 4: Transformer Architecture

**Purpose**: Capture long-range temporal dependencies with self-attention

**Key Components**:
- Multi-head self-attention
- Positional encoding (sinusoidal)
- Transformer encoder blocks
- Specialized training for attention models

**Files**: See `Phase_4.md`

#### Running Phase 4

```bash
# Train Transformer
python scripts/train_cnn.py \
    --model transformer \
    --data-path data/processed/signals_cache.h5 \
    --epochs 150 \
    --batch-size 32 \
    --lr 0.0001 \
    --warmup-epochs 10 \
    --checkpoint-dir checkpoints/phase4
```

**Custom Transformer Configuration**:

```python
from models import create_transformer

model = create_transformer(
    input_length=102400,
    num_classes=11,
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_feedforward=1024,
    dropout=0.1
)

# Train with learning rate warmup
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(epoch):
    warmup_epochs = 10
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = LambdaLR(optimizer, warmup_lambda)

# ... train
```

**Expected Results**:
- Training time: ~6-10 hours (GPU)
- Test accuracy: **96-97%**
- Model size: ~10-20M parameters

---

### Phase 5: Time-Frequency Analysis

**Purpose**: 2D CNN on time-frequency representations (spectrograms)

**Key Components**:
- **STFT (Short-Time Fourier Transform)**: Time-frequency localization
- **CWT (Continuous Wavelet Transform)**: Multi-resolution analysis
- **WVD (Wigner-Ville Distribution)**: High-resolution time-frequency
- **Dual-Stream Architecture**: Combine time and frequency features
- 2D CNNs: ResNet-18/34/50, VGG, custom architectures

**Files**: See `PHASE_5_USAGE_GUIDE.md` and `PHASE_5_ARCHITECTURE.md`

#### Running Phase 5

**Step 1: Precompute Spectrograms** (Recommended for speed)

```bash
# Precompute all spectrogram types
python scripts/precompute_spectrograms.py \
    --cache-file data/processed/signals_cache.h5 \
    --output-dir data/spectrograms \
    --types stft cwt wvd \
    --num-workers 8

# This generates:
# - data/spectrograms/stft/ (~10GB)
# - data/spectrograms/cwt/ (~15GB)
# - data/spectrograms/wvd/ (~12GB)
```

**Step 2: Train Spectrogram CNN**

```bash
# STFT-based 2D CNN
python scripts/train_spectrogram_cnn.py \
    --spectrogram-type stft \
    --spectrogram-dir data/spectrograms/stft \
    --model resnet18 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --checkpoint-dir checkpoints/phase5/stft_resnet18

# CWT-based 2D CNN
python scripts/train_spectrogram_cnn.py \
    --spectrogram-type cwt \
    --spectrogram-dir data/spectrograms/cwt \
    --model resnet34 \
    --epochs 100 \
    --checkpoint-dir checkpoints/phase5/cwt_resnet34

# WVD-based 2D CNN
python scripts/train_spectrogram_cnn.py \
    --spectrogram-type wvd \
    --spectrogram-dir data/spectrograms/wvd \
    --model vgg16 \
    --epochs 100 \
    --checkpoint-dir checkpoints/phase5/wvd_vgg16
```

**Step 3: Dual-Stream Architecture** (Best Performance)

```python
from models.spectrogram_cnn import DualStreamCNN
from training.spectrogram_trainer import SpectrogramTrainer

# Create dual-stream model (1D temporal + 2D spectral)
model = DualStreamCNN(
    input_length=102400,
    num_classes=11,
    temporal_backbone='resnet18_1d',
    spectral_backbone='resnet18_2d',
    fusion_method='concat'
)

# Train
trainer = SpectrogramTrainer(model)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    checkpoint_dir='checkpoints/phase5/dual_stream'
)
```

**Expected Results**:
- Precomputation time: ~2-4 hours (parallel)
- Training time: ~3-6 hours per model
- Test accuracy: **96-98%** (STFT: 96-97%, CWT: 96.5-97.5%, WVD: 97-98%, Dual-Stream: 97.5-98%)

---

### Phase 6: Physics-Informed Neural Networks (PINN)

**Purpose**: Integrate physics knowledge for improved accuracy and interpretability

**Key Components**:
- Physics loss functions: energy conservation, momentum conservation
- Hybrid PINN: Combine data-driven and physics-based learning
- Multi-objective optimization
- Physics-constrained predictions

**Files**: See `models/pinn/`, `training/pinn_trainer.py`

#### Running Phase 6

**Step 1: Train Baseline Model** (No physics)

```python
from models import create_model

baseline_model = create_model('resnet18', num_classes=11)
# ... train as usual (see Phase 3)
```

**Step 2: Train PINN Model** (With physics constraints)

```bash
# PINN with energy conservation
python scripts/train_pinn.py \
    --base-model resnet18 \
    --physics-weight 0.1 \
    --conservation-laws energy momentum \
    --data-path data/processed/signals_cache.h5 \
    --epochs 150 \
    --checkpoint-dir checkpoints/phase6/pinn_energy

# PINN with all physics constraints
python scripts/train_pinn.py \
    --base-model resnet34 \
    --physics-weight 0.2 \
    --conservation-laws energy momentum angular_momentum \
    --bearing-params rpm=1500 load=10 \
    --epochs 150 \
    --checkpoint-dir checkpoints/phase6/pinn_full
```

**Custom PINN Configuration**:

```python
from models.hybrid_pinn import create_hybrid_pinn
from training.pinn_trainer import PINNTrainer
from training.physics_loss_functions import EnergyConservationLoss, MomentumConservationLoss

# Create PINN model
model = create_hybrid_pinn(
    base_model='resnet18',
    num_classes=11,
    physics_layers=[256, 128, 64],
    fusion_method='concat'
)

# Setup physics losses
physics_losses = {
    'energy': EnergyConservationLoss(weight=0.1),
    'momentum': MomentumConservationLoss(weight=0.05)
}

# Train
trainer = PINNTrainer(
    model=model,
    physics_losses=physics_losses,
    total_physics_weight=0.15
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=150,
    checkpoint_dir='checkpoints/phase6'
)
```

**Expected Results**:
- Training time: ~5-8 hours (slightly longer than baseline due to physics losses)
- Test accuracy: **97-98%** (0.5-1% improvement over baseline)
- Benefits: Better generalization, physically plausible predictions

---

### Phase 7: Explainable AI (XAI)

**Purpose**: Interpret model predictions and build trust

**Key Components**:
- **SHAP (SHapley Additive exPlanations)**: Feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local explanations
- **Integrated Gradients**: Attribution methods
- **CAVs (Concept Activation Vectors)**: Concept-based explanations
- **Partial Dependence Plots (PDP)**: Feature effects
- **Interactive Dashboard**: Streamlit-based visualization

**Files**: See `explainability/`

#### Running Phase 7

**Step 1: Generate Explanations**

```python
from explainability import SHAPExplainer, LIMEExplainer, IntegratedGradientsExplainer

# Load trained model (any phase)
model = load_pretrained('checkpoints/phase6/best_model.pth')

# SHAP explanations
shap_explainer = SHAPExplainer(model, background_data=X_train[:100])
shap_values = shap_explainer.explain(X_test, save_dir='results/phase7/shap')

# LIME explanations
lime_explainer = LIMEExplainer(model, num_features=20)
lime_explanations = lime_explainer.explain(X_test[:10], save_dir='results/phase7/lime')

# Integrated Gradients
ig_explainer = IntegratedGradientsExplainer(model)
attributions = ig_explainer.explain(X_test, save_dir='results/phase7/ig')
```

**Step 2: Concept Activation Vectors (CAVs)**

```python
from explainability.concept_activation_vectors import CAVAnalyzer

# Define concepts (e.g., "high frequency", "low frequency")
concepts = {
    'high_freq': X_train[high_freq_indices],
    'low_freq': X_train[low_freq_indices],
    'high_amplitude': X_train[high_amp_indices]
}

# Train CAVs
cav_analyzer = CAVAnalyzer(model)
cavs = cav_analyzer.train_cavs(concepts)

# Test concept importance
importance = cav_analyzer.tcav(X_test, cavs, target_class=1)
print(f"High frequency concept importance for ball fault: {importance['high_freq']:.3f}")
```

**Step 3: Interactive Dashboard**

```bash
# Launch Streamlit dashboard
streamlit run explainability/dashboard.py -- \
    --model checkpoints/phase6/best_model.pth \
    --data data/processed/signals_cache.h5

# Open http://localhost:8501 in browser
```

**Dashboard Features**:
- Upload custom signals for diagnosis
- Real-time predictions with confidence scores
- SHAP/LIME explanations for each prediction
- Feature importance visualization
- Confusion matrix and performance metrics
- Concept activation heatmaps

**Expected Results**:
- Explanation generation time: ~5-15 minutes per method
- Dashboard: Interactive, <2s response time per query
- Insights: Identify which signal regions/frequencies drive predictions

---

## 🔄 Data Flow & Dependencies

### Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA                                                      │
│    data/raw/bearing_data/*.mat (1430 MAT files, ~500MB)         │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. DATA IMPORT (scripts/import_mat_dataset.py)                  │
│    • Read MAT files → NumPy arrays                              │
│    • Validate signal quality (length, NaN, range)               │
│    • Generate train/val/test splits (0.7/0.15/0.15)             │
│    • Save to HDF5: data/processed/signals_cache.h5 (~1.2GB)     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
         ┌─────────────┴──────────────┐
         ↓                            ↓
┌─────────────────────┐    ┌─────────────────────┐
│ 3A. FEATURE EXTRACT │    │ 3B. SPECTROGRAMS    │
│  (Phase 1)          │    │  (Phase 5)          │
│  • 36 features      │    │  • STFT, CWT, WVD   │
│  • MRMR → 15        │    │  • 2D images        │
│  • Normalization    │    │  • Precomputed      │
└──────┬──────────────┘    └──────┬──────────────┘
       ↓                          ↓
┌─────────────────────┐    ┌─────────────────────┐
│ 4A. CLASSICAL ML    │    │ 4B. DEEP LEARNING   │
│  • SVM, RF, GB      │    │  • 1D CNNs (Ph 2-4) │
│  • 95-96% acc       │    │  • 2D CNNs (Ph 5)   │
└─────────────────────┘    │  • PINNs (Ph 6)     │
                           │  • 96-98% acc       │
                           └──────┬──────────────┘
                                  ↓
                           ┌─────────────────────┐
                           │ 5. XAI (Phase 7)    │
                           │  • SHAP, LIME, IG   │
                           │  • Interpretability │
                           └─────────────────────┘
```

### Phase Dependencies

| Phase | Depends On | Input Required | Output Generated |
|-------|-----------|----------------|------------------|
| 0 | None | N/A | Base classes, utilities |
| 1 | Phase 0 | Raw signals (HDF5) | Features, trained classical models |
| 2 | Phase 0 | Raw signals (HDF5) | Trained 1D CNN models |
| 3 | Phase 0, 2 | Raw signals (HDF5) | Trained ResNet/EfficientNet models |
| 4 | Phase 0, 2 | Raw signals (HDF5) | Trained Transformer models |
| 5 | Phase 0 | Raw signals OR precomputed spectrograms | Trained 2D CNN models |
| 6 | Phase 0, 2-5 | Raw signals + base model (optional) | Trained PINN models |
| 7 | Phase 0, any trained model | Trained model + test data | Explanations, dashboard |

**Key Insight**:
- **Phases 1-6 are largely independent** - each can train models from scratch
- **Phase 7 depends on any trained model** from Phases 1-6
- **Phases can be mixed** - e.g., use Phase 1 features in Phase 2 CNN, combine Phase 6 PINN with Phase 7 XAI

---

## 🎛️ Advanced Features

### Model Factory & Registry

Centralized model management:

```python
from models import (
    create_model, list_available_models,
    get_model_info, save_checkpoint, load_pretrained
)

# List all available models
print(list_available_models())
# ['cnn1d', 'resnet18', 'resnet34', 'efficientnet', 'transformer',
#  'hybrid_pinn', 'voting_ensemble', 'stacked_ensemble']

# Create any model by name
model = create_model('resnet34', num_classes=11, dropout=0.3)

# Get model info
info = get_model_info('resnet34')
print(f"Parameters: {info['params']}, FLOPs: {info['flops']}")

# Save checkpoint
save_checkpoint(model, 'checkpoints/my_model.pth', metadata={'accuracy': 0.97})

# Load pretrained
model = load_pretrained('checkpoints/my_model.pth')
```

### Custom Model Registration

Extend with your own models:

```python
from models import register_model

def create_my_custom_model(num_classes=11, **kwargs):
    # Your model implementation
    return MyCustomModel(num_classes, **kwargs)

# Register
register_model('my_custom_model', create_my_custom_model)

# Now use it like any other model
model = create_model('my_custom_model', num_classes=11)
```

### Ensemble Methods (Phase 8 - Upcoming)

```python
from models import create_voting_ensemble, create_stacked_ensemble

# Load multiple trained models
cnn_model = load_pretrained('checkpoints/phase2/best_model.pth')
resnet_model = load_pretrained('checkpoints/phase3/best_model.pth')
transformer_model = load_pretrained('checkpoints/phase4/best_model.pth')
pinn_model = load_pretrained('checkpoints/phase6/best_model.pth')

# Voting ensemble
voting_ensemble = create_voting_ensemble(
    models=[cnn_model, resnet_model, transformer_model, pinn_model],
    voting='soft',
    weights=[0.2, 0.3, 0.3, 0.2]
)

# Stacked ensemble
stacked_ensemble = create_stacked_ensemble(
    base_models=[cnn_model, resnet_model, transformer_model],
    meta_model='logistic_regression'
)

# Predict
predictions = voting_ensemble(X_test)
print(f"Ensemble Accuracy: {accuracy(predictions, y_test):.4f}")
```

### Hyperparameter Optimization

Multiple strategies available:

```python
# Bayesian Optimization (recommended)
from training.bayesian_optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(n_trials=100)
best_params = optimizer.optimize(
    model_fn=lambda **params: create_model('resnet18', **params),
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    search_space={
        'lr': (1e-5, 1e-2, 'log'),
        'dropout': (0.1, 0.5, 'uniform'),
        'weight_decay': (1e-6, 1e-3, 'log')
    }
)

# Grid Search
from training.grid_search import GridSearch

grid_search = GridSearch()
best_params = grid_search.search(
    param_grid={
        'lr': [0.001, 0.0001],
        'dropout': [0.3, 0.5],
        'batch_size': [32, 64]
    }
)

# Random Search
from training.random_search import RandomSearch

random_search = RandomSearch(n_iter=50)
best_params = random_search.search(...)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python scripts/train_cnn.py --batch-size 16  # instead of 32

# Enable gradient accumulation
python scripts/train_cnn.py --batch-size 16 --gradient-accumulation 2

# Use mixed precision (saves ~40% memory)
python scripts/train_cnn.py --mixed-precision

# Clear cache periodically
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. MAT File Loading Issues

**Error**: `ValueError: MAT file does not contain expected key`

**Solutions**:
```python
# Inspect MAT file structure
import scipy.io as sio
mat = sio.loadmat('data/raw/bearing_data/normal/file001.mat')
print(mat.keys())  # Check available keys

# Update key in import script
# Edit scripts/import_mat_dataset.py, line ~45
signal = mat['your_key_name']  # Update key name
```

#### 3. Spectrogram Precomputation Too Slow

**Issue**: Precomputing spectrograms takes >6 hours

**Solutions**:
```bash
# Use more workers (parallel processing)
python scripts/precompute_spectrograms.py --num-workers 16

# Precompute only one type at a time
python scripts/precompute_spectrograms.py --types stft
python scripts/precompute_spectrograms.py --types cwt

# Use lower resolution (faster but less accurate)
python scripts/precompute_spectrograms.py --resolution 128  # instead of 256
```

#### 4. Training Diverges / NaN Loss

**Symptoms**: Loss becomes NaN or explodes

**Solutions**:
```python
# Reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # instead of 0.001

# Enable gradient clipping
from torch.nn.utils import clip_grad_norm_
clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for NaN in input data
assert not torch.isnan(X_train).any(), "Input contains NaN!"

# Use more stable loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

#### 5. Phase 1 Feature Extraction Slow

**Issue**: Feature extraction takes >2 hours

**Solutions**:
```python
# Parallelize feature extraction
from joblib import Parallel, delayed

features = Parallel(n_jobs=8)(
    delayed(extractor.extract_features)(signal)
    for signal in signals
)

# Cache extracted features
import pickle
with open('data/processed/features_cache.pkl', 'wb') as f:
    pickle.dump(features, f)
```

### Performance Tuning

#### GPU Utilization

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Optimize for your GPU
# For 8GB GPU:
BATCH_SIZE=32
WORKERS=4

# For 16GB GPU:
BATCH_SIZE=64
WORKERS=8

# For 24GB+ GPU:
BATCH_SIZE=128
WORKERS=16

python scripts/train_cnn.py --batch-size $BATCH_SIZE --num-workers $WORKERS
```

#### Training Speed

```python
# Enable cuDNN benchmarking (5-10% speedup)
import torch
torch.backends.cudnn.benchmark = True

# Use mixed precision (2x speedup)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Prefetch data (reduce I/O bottleneck)
from torch.utils.data import DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    pin_memory=True,      # Faster CPU->GPU transfer
    prefetch_factor=2     # Prefetch 2 batches
)
```

---

## 🚀 Future Phases

### Phase 8: Ensemble Methods (10 days)

**Goals**:
- Combine best models from Phases 1-7
- Voting ensembles (soft/hard)
- Stacked ensembles with meta-learners
- Multi-level fusion (feature, decision, hybrid)
- Target: **98-99% accuracy**

**Implementation Plan**:
```python
# Planned API
from models import create_voting_ensemble, create_stacked_ensemble

ensemble = create_voting_ensemble(
    models=[phase2_cnn, phase3_resnet, phase4_transformer, phase6_pinn],
    weights='auto'  # Optimize weights automatically
)

# Advanced fusion
from models.fusion import MultiLevelFusion
fusion_model = MultiLevelFusion(
    feature_fusion='concat',
    decision_fusion='weighted_average',
    confidence_calibration=True
)
```

### Phase 9: Deployment (14 days)

**Goals**:
- Model quantization (INT8, FP16)
- ONNX export for cross-platform deployment
- REST API with FastAPI
- Docker containerization
- Real-time inference (<50ms latency)
- Edge deployment (Raspberry Pi, Jetson Nano)

**Implementation Plan**:
```bash
# Quantization
python scripts/quantize_model.py \
    --model checkpoints/phase8/ensemble.pth \
    --output checkpoints/phase9/ensemble_int8.pth \
    --precision int8

# ONNX export
python scripts/export_onnx.py \
    --model checkpoints/phase9/ensemble_int8.pth \
    --output models/ensemble.onnx

# Docker deployment
docker build -t lstm_pfd:v1 .
docker run -p 8000:8000 lstm_pfd:v1

# API endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, ..., 0.3]}'
```

### Phase 10: QA & Integration (25 days)

**Goals**:
- Comprehensive unit & integration tests
- Benchmarking suite (accuracy, speed, memory)
- Documentation finalization
- Code review and refactoring
- Continuous integration (CI/CD)
- Production deployment guide

**Deliverables**:
- Test coverage >90%
- Benchmark report comparing all phases
- Complete API documentation
- User manual and tutorials
- Production deployment guide

---

## 📚 Additional Resources

### Documentation Files

- **Phase Guides**:
  - `PHASE_1_USAGE_GUIDE.md` - Classical ML pipeline
  - `PHASE_2_USAGE_GUIDE.md` - 1D CNN training
  - `PHASE_3_USAGE_GUIDE.md` - Advanced CNNs (ResNet, EfficientNet, NAS)
  - `PHASE_5_USAGE_GUIDE.md` - Time-frequency analysis & spectrograms
  - `PHASE_5_ARCHITECTURE.md` - Detailed Phase 5 architecture

- **Phase Descriptions**:
  - `phase_0.md` - Foundation design
  - `phase_1.md` - Classical ML implementation details
  - `Phase_2.md` - 1D CNN architecture details
  - `Phase_3.md` - Advanced CNN architectures
  - `Phase_4.md` - Transformer implementation
  - `Phase_5.md` - Time-frequency analysis theory

### External Resources

- **Datasets**:
  - [Case Western Reserve University Bearing Data](https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website)
  - [Paderborn University Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter)

- **Papers**:
  - Zhang et al. (2019): "Deep Learning Algorithms for Bearing Fault Diagnosis"
  - Raissi et al. (2019): "Physics-informed neural networks"
  - Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions" (SHAP)

### Support

- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Contributing**: See `CONTRIBUTING.md` (coming in Phase 10)

---

## 📄 License

MIT License - See `LICENSE` file for details

---

## 🙏 Acknowledgments

- **Data Source**: MATLAB bearing fault dataset (1430 samples)
- **Frameworks**: PyTorch, scikit-learn, SHAP, LIME, Streamlit
- **Inspiration**: CWRU Bearing Data Center, Paderborn University dataset

---

## 📞 Contact

For questions or collaboration:
- GitHub Issues: [https://github.com/yourusername/LSTM_PFD/issues](https://github.com/yourusername/LSTM_PFD/issues)
- Email: your.email@example.com

---

**Last Updated**: November 2025 (Phase 7 Complete)

**Next Milestone**: Phase 8 - Ensemble Methods (Target: December 2025)
