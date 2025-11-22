# Milestone 3: CNN-LSTM Hybrid for Bearing Fault Diagnosis

**Combining Spatial Feature Extraction with Temporal Modeling**

This milestone represents the integration of CNN-based spatial feature extraction (Milestone 1) with LSTM-based temporal sequence modeling (Milestone 2) to create powerful hybrid architectures for bearing fault diagnosis.

---

## Table of Contents

- [Overview](#overview)
- [Why Combine CNN and LSTM?](#why-combine-cnn-and-lstm)
- [Key Innovation: Configurable Architecture](#key-innovation-configurable-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Hybrid Architectures](#hybrid-architectures)
- [Usage Examples](#usage-examples)
- [Performance Comparison](#performance-comparison)
- [Model Details](#model-details)
- [Training Pipeline](#training-pipeline)
- [Evaluation and Analysis](#evaluation-and-analysis)
- [Customization Guide](#customization-guide)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)
- [References](#references)

---

## Overview

### What is This Milestone?

Milestone 3 builds upon the strengths of both CNN (Milestone 1) and LSTM (Milestone 2) approaches by creating **hybrid architectures** that combine:

1. **CNN Feature Extraction**: Leveraging convolutional layers to extract spatial/local patterns from vibration signals
2. **LSTM Temporal Modeling**: Using recurrent layers to model temporal dependencies and sequential relationships

### The Hybrid Approach

```
Raw Signal → CNN Backbone → Feature Sequence → LSTM → Temporal Pooling → Classification
```

**Architecture Flow:**
1. **Input**: Raw vibration signal (102,400 samples)
2. **CNN Backbone**: Extracts rich spatial features at multiple temporal positions
3. **Feature Sequence**: CNN output is treated as a sequence of feature vectors
4. **LSTM Layer**: Models temporal dependencies across the feature sequence
5. **Temporal Pooling**: Aggregates temporal information (mean, max, or attention)
6. **Classification**: Final layers predict fault type (11 classes)

### Dataset

- **Source**: Bearing vibration data from Case Western Reserve University
- **Format**: MATLAB `.mat` files
- **Classes**: 11 fault types
  - Healthy (Normal)
  - Misalignment
  - Imbalance
  - Bearing Clearance
  - Lubrication Issue
  - Cavitation
  - Wear
  - Oil Whirl
  - Mixed Fault 1
  - Mixed Fault 2
  - Mixed Fault 3

- **Samples**: 1,430 total samples
- **Signal Length**: 102,400 samples per signal
- **Sampling Rate**: 20.48 kHz
- **Duration**: ~5 seconds per sample

---

## Why Combine CNN and LSTM?

### Complementary Strengths

| Aspect | CNN (Milestone 1) | LSTM (Milestone 2) | **Hybrid (Milestone 3)** |
|--------|-------------------|-------------------|--------------------------|
| **Feature Type** | Spatial/Local patterns | Temporal sequences | **Both spatial + temporal** |
| **Processing** | Parallel | Sequential | **Hierarchical** |
| **Strengths** | Local patterns, translation invariance | Long-range dependencies, order | **Combined advantages** |
| **Weaknesses** | Limited temporal modeling | Computationally expensive | Higher complexity |
| **Expected Accuracy** | 96-97% | 92-97% | **[TBD - To be determined]** |

### Why This Matters for Bearing Fault Diagnosis

**Bearing vibration signals have both spatial and temporal characteristics:**

1. **Spatial Patterns (CNN captures)**
   - Impulses and shock patterns
   - Frequency components
   - Local signal textures
   - Fault-specific signatures

2. **Temporal Dependencies (LSTM captures)**
   - Signal evolution over time
   - Phase relationships
   - Sequential patterns
   - Long-range correlations

3. **Combined Benefits (Hybrid captures)**
   - **Richer representations**: CNN extracts features, LSTM refines them
   - **Multi-scale analysis**: CNN for local, LSTM for global
   - **Robustness**: Leverages strengths of both approaches
   - **Better generalization**: More comprehensive feature learning

### Real-World Advantages

- **Complex fault patterns**: Better detection of mixed faults and early-stage degradation
- **Noisy environments**: More robust to noise through hierarchical processing
- **Variable operating conditions**: Better generalization across different speeds and loads
- **Transfer learning**: Pretrained CNNs can be combined with task-specific LSTMs

---

## Key Innovation: Configurable Architecture

### Design Philosophy

Instead of providing fixed hybrid models, **this milestone offers a configurable framework** where you can combine **ANY CNN backbone** with **ANY LSTM type**.

### Why Configurable?

1. **Flexibility**: Experiment with different CNN-LSTM combinations
2. **Scalability**: Easy to add new CNN or LSTM architectures
3. **Research-Friendly**: Rapidly test hypotheses about architecture choices
4. **Production-Ready**: Select optimal configuration based on your constraints

### Available CNN Backbones

Choose from any CNN architecture implemented in Milestone 1:

- **Basic CNNs**: `cnn1d` (lightweight, fast)
- **ResNet Family**: `resnet18`, `resnet34`, `resnet50` (balanced, robust)
- **EfficientNet Family**: `efficientnet_b0`, `efficientnet_b2`, `efficientnet_b4` (efficient, scalable)

### Available LSTM Types

Choose from LSTM architectures implemented in Milestone 2:

- **Vanilla LSTM**: `lstm` (~200K params, unidirectional)
- **Bidirectional LSTM**: `bilstm` (~400K params, bidirectional)

### Configuration Matrix

You can create **42+ different hybrid architectures** by combining:
- 7 CNN backbones × 2 LSTM types = **14 base combinations**
- Plus variations in LSTM hidden size, number of layers, pooling methods

**Example Combinations:**
```
ResNet34 + BiLSTM  (Recommended for accuracy)
EfficientNet-B2 + BiLSTM  (Recommended for efficiency)
ResNet18 + LSTM  (Recommended for speed)
CNN1D + LSTM  (Ultra-lightweight)
ResNet50 + BiLSTM  (Maximum capacity)
... and many more!
```

---

## Project Structure

```
milestone-3/
├── README.md                    # This file
├── QUICKSTART.md               # Quick start guide
├── DELIVERY_NOTES.md           # Delivery and setup notes
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── models/                     # Model architectures
│   ├── __init__.py            # Model factory (supports configurable hybrids)
│   ├── cnn/                   # CNN backbones (from Milestone 1)
│   │   ├── __init__.py
│   │   ├── base_cnn.py
│   │   ├── cnn_models.py      # Basic CNN architectures
│   │   ├── resnet.py          # ResNet variants
│   │   └── efficientnet.py    # EfficientNet variants
│   ├── lstm/                  # LSTM components (from Milestone 2)
│   │   ├── __init__.py
│   │   ├── base_lstm.py
│   │   └── lstm_models.py     # LSTM and BiLSTM
│   └── hybrid/                # ⭐ Hybrid architectures (NEW)
│       ├── __init__.py
│       └── hybrid_cnn_lstm.py # Configurable CNN-LSTM hybrid
│
├── data/                       # Data loading and preprocessing
│   ├── __init__.py
│   ├── cnn_dataloader.py      # CNN-style dataloaders
│   ├── lstm_dataloader.py     # LSTM-style dataloaders (if needed)
│   ├── preprocessing.py       # Signal preprocessing
│   └── augmentation.py        # Data augmentation
│
├── training/                   # Training infrastructure
│   ├── __init__.py
│   ├── cnn_trainer.py         # Trainer (works with hybrid models)
│   ├── optimizers.py          # Optimizer factory
│   ├── losses.py              # Loss functions
│   ├── schedulers.py          # Learning rate schedulers
│   └── metrics.py             # Evaluation metrics
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── reproducibility.py     # Random seed management
│   ├── device_manager.py      # GPU/CPU device handling
│   ├── early_stopping.py      # Early stopping
│   ├── checkpoint_manager.py  # Model checkpointing
│   └── logging.py             # Logging utilities
│
├── scripts/                    # Training and evaluation scripts
│   ├── train_hybrid.py        # ⭐ Train hybrid models
│   ├── evaluate_hybrid.py     # ⭐ Evaluate hybrid models
│   ├── export_model.py        # Export for deployment
│   └── visualize_results.py   # Visualization tools
│
├── notebooks/                  # Jupyter notebooks (optional)
│   └── hybrid_analysis.ipynb  # Interactive analysis
│
├── data/                       # Data directory (create this)
│   ├── raw/                   # Place .mat files here
│   │   └── bearing_data/      # Your bearing .mat files
│   └── processed/             # Processed data cache
│
└── results/                    # Training results
    ├── checkpoints/           # Model checkpoints
    │   └── hybrid/
    ├── logs/                  # Training logs
    ├── evaluation/            # Evaluation results
    └── visualizations/        # Plots and figures
```

---

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU training, optional but recommended)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: ~5 GB for data and models

### Step 1: Create Virtual Environment

```bash
cd milestones/milestone-3

# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n bearing-hybrid python=3.9
conda activate bearing-hybrid
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0 (for .mat file loading)
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

### Step 3: Prepare Data

```bash
# Create data directories
mkdir -p data/raw/bearing_data
mkdir -p data/processed

# Copy your .mat files to data/raw/bearing_data/
cp /path/to/your/*.mat data/raw/bearing_data/
```

### Step 4: Verify Installation

```bash
python scripts/validate_installation.py
```

---

## Quick Start

### Train a Recommended Hybrid Model

```bash
# Recommended Configuration 1: ResNet34 + BiLSTM
python scripts/train_hybrid.py --model recommended_1 --epochs 75

# Recommended Configuration 2: EfficientNet-B2 + BiLSTM
python scripts/train_hybrid.py --model recommended_2 --epochs 75

# Recommended Configuration 3: ResNet18 + LSTM
python scripts/train_hybrid.py --model recommended_3 --epochs 75
```

### Train a Custom Hybrid Model

```bash
# Custom: Any CNN + Any LSTM
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type resnet34 \
  --lstm-type bilstm \
  --lstm-hidden-size 256 \
  --lstm-num-layers 2 \
  --pooling mean \
  --epochs 75
```

### Evaluate a Trained Model

```bash
python scripts/evaluate_hybrid.py \
  --checkpoint results/checkpoints/hybrid/recommended_1/best_model.pth \
  --model recommended_1
```

For more detailed instructions, see [QUICKSTART.md](QUICKSTART.md).

---

## Hybrid Architectures

### Recommended Configuration 1: ResNet34 + BiLSTM

**Best for: High accuracy, balanced performance**

```python
from models import create_model

model = create_model('recommended_1')
```

**Specifications:**
- **CNN Backbone**: ResNet34 (pretrained architecture, adapted for 1D signals)
- **LSTM Type**: Bidirectional LSTM
- **LSTM Hidden Size**: 256
- **LSTM Layers**: 2
- **Pooling**: Mean pooling over temporal dimension
- **Total Parameters**: ~TBD
- **Model Size**: ~TBD MB
- **Expected Performance**: [TBD - To be determined after training]

**Characteristics:**
- Strong feature extraction with ResNet34
- Bidirectional temporal modeling
- Good balance of accuracy and efficiency
- Suitable for most applications

---

### Recommended Configuration 2: EfficientNet-B2 + BiLSTM

**Best for: Efficiency, mobile deployment**

```python
from models import create_model

model = create_model('recommended_2')
```

**Specifications:**
- **CNN Backbone**: EfficientNet-B2 (efficient scaling)
- **LSTM Type**: Bidirectional LSTM
- **LSTM Hidden Size**: 256
- **LSTM Layers**: 2
- **Pooling**: Mean pooling
- **Total Parameters**: ~TBD
- **Model Size**: ~TBD MB
- **Expected Performance**: [TBD - To be determined after training]

**Characteristics:**
- Efficient CNN architecture
- Smaller model size
- Faster inference
- Suitable for edge devices

---

### Recommended Configuration 3: ResNet18 + LSTM

**Best for: Speed, real-time applications**

```python
from models import create_model

model = create_model('recommended_3')
```

**Specifications:**
- **CNN Backbone**: ResNet18 (lightweight ResNet)
- **LSTM Type**: Unidirectional LSTM
- **LSTM Hidden Size**: 128
- **LSTM Layers**: 2
- **Pooling**: Mean pooling
- **Total Parameters**: ~TBD
- **Model Size**: ~TBD MB
- **Expected Performance**: [TBD - To be determined after training]

**Characteristics:**
- Fastest inference
- Lower memory footprint
- Good for real-time monitoring
- Suitable for resource-constrained environments

---

### Custom Configuration

**Best for: Research, experimentation, specific requirements**

```python
from models import create_model

model = create_model(
    'custom',
    cnn_type='resnet50',           # Any CNN backbone
    lstm_type='bilstm',            # lstm or bilstm
    lstm_hidden_size=512,          # LSTM hidden dimension
    lstm_num_layers=3,             # Number of LSTM layers
    pooling_method='attention',    # mean, max, last, or attention
    freeze_cnn=False               # Freeze CNN weights?
)
```

**Available CNN Backbones:**
- `cnn1d`: Basic 1D CNN (lightweight)
- `resnet18`: ResNet-18 (fast)
- `resnet34`: ResNet-34 (balanced)
- `resnet50`: ResNet-50 (powerful)
- `efficientnet_b0`: EfficientNet-B0 (efficient)
- `efficientnet_b2`: EfficientNet-B2 (balanced efficiency)
- `efficientnet_b4`: EfficientNet-B4 (high capacity)

**Available LSTM Types:**
- `lstm`: Unidirectional LSTM
- `bilstm`: Bidirectional LSTM

**Pooling Methods:**
- `mean`: Average pooling over temporal dimension
- `max`: Max pooling over temporal dimension
- `last`: Use last timestep output
- `attention`: Learnable attention mechanism

---

## Usage Examples

### Example 1: Train Recommended Model

```bash
python scripts/train_hybrid.py \
  --model recommended_1 \
  --data-dir data/raw/bearing_data \
  --epochs 75 \
  --batch-size 32 \
  --lr 0.001 \
  --optimizer adam \
  --scheduler cosine \
  --checkpoint-dir results/checkpoints/hybrid \
  --seed 42
```

### Example 2: Train Custom Hybrid with Specific Configuration

```bash
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type efficientnet_b0 \
  --lstm-type bilstm \
  --lstm-hidden-size 128 \
  --lstm-num-layers 2 \
  --pooling attention \
  --data-dir data/raw/bearing_data \
  --epochs 100 \
  --batch-size 64 \
  --mixed-precision \
  --seed 42
```

### Example 3: Train with Frozen CNN Backbone

```bash
# Train only LSTM layers, freeze CNN (transfer learning)
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type resnet34 \
  --lstm-type bilstm \
  --freeze-cnn \
  --epochs 50 \
  --lr 0.01
```

### Example 4: Evaluate Model

```bash
python scripts/evaluate_hybrid.py \
  --checkpoint results/checkpoints/hybrid/recommended_1/20240115_143022/best_model.pth \
  --model recommended_1 \
  --data-dir data/raw/bearing_data \
  --batch-size 64 \
  --output-dir results/evaluation/recommended_1
```

### Example 5: Python API Usage

```python
import torch
from models import create_model
from data.cnn_dataloader import create_cnn_dataloaders

# Create model
model = create_model(
    'custom',
    cnn_type='resnet34',
    lstm_type='bilstm',
    lstm_hidden_size=256,
    lstm_num_layers=2,
    pooling_method='mean'
)

# Load data
train_loader, val_loader, test_loader = create_cnn_dataloaders(
    data_dir='data/raw/bearing_data',
    batch_size=32
)

# Get model info
info = model.get_model_info()
print(f"Total parameters: {info['total_params']:,}")
print(f"CNN parameters: {info['cnn_params']:,}")
print(f"LSTM parameters: {info['lstm_params']:,}")

# Forward pass
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for signals, labels in train_loader:
    signals = signals.to(device)
    outputs = model(signals)
    print(f"Output shape: {outputs.shape}")  # [batch_size, 11]
    break
```

---

## Performance Comparison

### Expected Performance

| Model | Architecture | Parameters | Size | Accuracy | Inference Time | Notes |
|-------|-------------|------------|------|----------|----------------|-------|
| **CNN (M1)** | ResNet34 | ~21M | ~85 MB | 96-97% | ~TBD ms | Spatial features only |
| **LSTM (M2)** | BiLSTM | ~400K | ~1.6 MB | 92-97% | ~TBD ms | Temporal features only |
| **Hybrid (M3)** | ResNet34+BiLSTM | ~TBD | ~TBD MB | **[TBD]** | ~TBD ms | **Combined features** |

**Note**: All performance metrics for Milestone 3 are **placeholders** and will be updated after training and evaluation.

### When to Use Each Approach?

| Scenario | Recommended Approach | Reason |
|----------|---------------------|--------|
| **Maximum accuracy** | Hybrid (recommended_1) | Combines spatial + temporal |
| **Real-time monitoring** | CNN or Hybrid (recommended_3) | Fast inference |
| **Edge deployment** | Hybrid (recommended_2) | Efficient architecture |
| **Limited data** | CNN | Fewer parameters to learn |
| **Long signals** | LSTM or Hybrid | Better temporal modeling |
| **Mixed faults** | Hybrid | Richer representations |
| **Research/experimentation** | Hybrid (custom) | Configurable architecture |

---

## Model Details

### Architecture Components

#### 1. CNN Backbone

The CNN backbone serves as a **feature extractor**, processing the raw vibration signal and producing a sequence of feature vectors.

**Input**: `[batch_size, 1, 102400]` (raw signal)
**Output**: `[batch_size, sequence_length, feature_dim]` (feature sequence)

The CNN's convolutional and pooling layers naturally create a temporal sequence:
- Each position in the sequence corresponds to a temporal region in the input
- Feature dimension depends on the CNN architecture
- Sequence length depends on the CNN's downsampling factor

#### 2. LSTM Layer

The LSTM processes the feature sequence to model temporal dependencies.

**Input**: `[batch_size, sequence_length, feature_dim]`
**Output**: `[batch_size, sequence_length, hidden_size]` (or `2*hidden_size` for BiLSTM)

**Types:**
- **Unidirectional LSTM**: Processes sequence forward in time
- **Bidirectional LSTM**: Processes sequence both forward and backward

#### 3. Temporal Pooling

Aggregates the LSTM's output sequence into a fixed-size vector.

**Methods:**
- **Mean Pooling**: Average over temporal dimension
  ```python
  pooled = torch.mean(lstm_out, dim=1)
  ```
- **Max Pooling**: Maximum over temporal dimension
  ```python
  pooled, _ = torch.max(lstm_out, dim=1)
  ```
- **Last Timestep**: Use final timestep
  ```python
  pooled = lstm_out[:, -1, :]
  ```
- **Attention Pooling**: Learnable weighted average
  ```python
  attention_weights = softmax(W * lstm_out)
  pooled = sum(attention_weights * lstm_out)
  ```

#### 4. Classification Head

Final fully-connected layers for fault classification.

**Input**: `[batch_size, hidden_size]`
**Output**: `[batch_size, 11]` (logits for 11 fault types)

---

### Data Flow Example

```
Input Signal:        [32, 1, 102400]     # 32 signals, 102400 samples each
                            ↓
CNN Backbone:        [32, 512, 100]      # 100 feature vectors of dim 512
                            ↓
LSTM:                [32, 100, 256]      # Hidden size 256 (or 512 for BiLSTM)
                            ↓
Temporal Pooling:    [32, 256]           # Single vector per sample
                            ↓
Classifier:          [32, 11]            # 11 class logits
```

---

## Training Pipeline

### Training Script Arguments

```bash
python scripts/train_hybrid.py --help
```

**Model Arguments:**
- `--model`: Model configuration (recommended_1, recommended_2, recommended_3, custom)
- `--cnn-type`: CNN backbone type (for custom model)
- `--lstm-type`: LSTM type (for custom model)
- `--lstm-hidden-size`: LSTM hidden dimension
- `--lstm-num-layers`: Number of LSTM layers
- `--pooling`: Temporal pooling method
- `--freeze-cnn`: Freeze CNN weights (transfer learning)

**Data Arguments:**
- `--data-dir`: Directory containing .mat files
- `--batch-size`: Training batch size

**Training Arguments:**
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--optimizer`: Optimizer (adam, adamw, sgd)
- `--scheduler`: LR scheduler (cosine, step, plateau)
- `--mixed-precision`: Enable FP16 mixed precision

**Output Arguments:**
- `--checkpoint-dir`: Checkpoint save directory
- `--seed`: Random seed for reproducibility

### Training Features

#### Mixed Precision Training

Speeds up training on modern GPUs while reducing memory usage:

```bash
python scripts/train_hybrid.py --model recommended_1 --mixed-precision
```

**Benefits:**
- ~2x faster training
- ~50% less GPU memory
- Minimal accuracy impact

#### Learning Rate Scheduling

**Cosine Annealing** (default):
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

**Benefits:**
- Smooth learning rate decay
- Better convergence
- Avoids abrupt changes

#### Early Stopping

Automatically stops training when validation loss stops improving:

```python
patience = 15  # Stop if no improvement for 15 epochs
```

#### Checkpointing

Automatically saves:
- Best model (highest validation accuracy)
- Latest model (every epoch)
- Optimizer and scheduler state

```
results/checkpoints/hybrid/recommended_1/20240115_143022/
├── best_model.pth          # Best validation accuracy
├── checkpoint_epoch_10.pth # Latest checkpoint
└── training_history.json   # Training metrics
```

---

## Evaluation and Analysis

### Evaluation Script

```bash
python scripts/evaluate_hybrid.py \
  --checkpoint results/checkpoints/hybrid/recommended_1/best_model.pth \
  --model recommended_1 \
  --data-dir data/raw/bearing_data \
  --output-dir results/evaluation/recommended_1
```

### Evaluation Metrics

**Classification Metrics:**
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Top-k accuracy

**Outputs:**
- `classification_report.txt`: Detailed metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `per_class_accuracy.png`: Per-class performance
- `predictions.csv`: All predictions with ground truth

### Visualization Tools

```bash
python scripts/visualize_results.py \
  --checkpoint results/checkpoints/hybrid/recommended_1/best_model.pth \
  --model recommended_1 \
  --output-dir results/visualizations
```

**Generated Visualizations:**
- Training curves (loss, accuracy)
- Learning rate schedule
- Confusion matrix
- Per-class metrics
- Feature visualizations (t-SNE, UMAP)

---

## Customization Guide

### Adding a New CNN Backbone

1. Implement your CNN in `models/cnn/`:

```python
# models/cnn/my_custom_cnn.py
class MyCustomCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        # Your architecture here

    def forward(self, x):
        # Your forward pass
        return x
```

2. Register in `models/cnn/__init__.py`:

```python
from .my_custom_cnn import MyCustomCNN

def create_model(model_name, **kwargs):
    model_map = {
        # ...existing models...
        'my_custom_cnn': MyCustomCNN,
    }
    return model_map[model_name](**kwargs)
```

3. Use in hybrid model:

```bash
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type my_custom_cnn \
  --lstm-type bilstm
```

### Adding a New LSTM Variant

1. Implement in `models/lstm/lstm_models.py`:

```python
class MyCustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=11):
        super().__init__()
        # Your architecture here

    def forward(self, x):
        # Your forward pass
        return x
```

2. Register in `models/__init__.py` hybrid factory

3. Use in training:

```bash
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type resnet34 \
  --lstm-type my_custom_lstm
```

### Modifying Data Preprocessing

Edit `data/preprocessing.py`:

```python
def preprocess_signal(signal, sample_rate=20480):
    """Custom preprocessing logic"""
    # Normalization
    signal = (signal - signal.mean()) / signal.std()

    # Your custom preprocessing
    # ...

    return signal
```

### Custom Loss Functions

Edit `training/losses.py`:

```python
def create_loss_function(loss_name='cross_entropy', **kwargs):
    if loss_name == 'my_custom_loss':
        return MyCustomLoss(**kwargs)
    # ...
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
python scripts/train_hybrid.py --batch-size 16  # Instead of 32

# Use mixed precision
python scripts/train_hybrid.py --mixed-precision

# Use smaller model
python scripts/train_hybrid.py --model recommended_3  # Lighter model
```

#### 2. Data Loading Errors

**Error**: `FileNotFoundError: .mat files not found`

**Solution:**
```bash
# Check data directory structure
ls -la data/raw/bearing_data/*.mat

# Verify path in script
python scripts/train_hybrid.py --data-dir /absolute/path/to/bearing_data
```

#### 3. Slow Training

**Solutions:**
```bash
# Enable mixed precision
python scripts/train_hybrid.py --mixed-precision

# Increase batch size (if memory allows)
python scripts/train_hybrid.py --batch-size 64

# Use more workers for data loading
# Edit train_hybrid.py: num_workers=8 (default is 4)
```

#### 4. Model Convergence Issues

**Symptoms**: Validation accuracy not improving

**Solutions:**
```bash
# Adjust learning rate
python scripts/train_hybrid.py --lr 0.0001  # Lower LR

# Try different optimizer
python scripts/train_hybrid.py --optimizer adamw

# Increase training epochs
python scripts/train_hybrid.py --epochs 100

# Check for data issues
python scripts/validate_installation.py
```

#### 5. Import Errors

**Error**: `ModuleNotFoundError: No module named 'models'`

**Solution:**
```bash
# Ensure you're in the right directory
cd milestones/milestone-3

# Check Python path
python -c "import sys; print(sys.path)"

# The script should add project root to path automatically
# If not, you can set PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:/path/to/milestone-3"
```

---

## Technical Details

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB
- GPU: Not required (CPU training possible but slow)

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- Storage: 20 GB (for experiments)
- GPU: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3070, V100)

**Optimal:**
- CPU: 16+ cores
- RAM: 32 GB
- Storage: 50 GB SSD
- GPU: NVIDIA GPU with 16+ GB VRAM (e.g., A100, RTX 4090)

### Performance Benchmarks

**Training Time Estimates** (recommended_1, 75 epochs):
- **CPU only**: ~TBD hours
- **Single GPU (RTX 3080)**: ~TBD hours
- **Single GPU (A100)**: ~TBD minutes

**Inference Time** (single sample):
- **CPU**: ~TBD ms
- **GPU**: ~TBD ms

*Note: Actual times depend on hardware and will be updated after benchmarking.*

### Memory Usage

**Training Memory** (batch_size=32):
- **recommended_1**: ~TBD GB GPU memory
- **recommended_2**: ~TBD GB GPU memory
- **recommended_3**: ~TBD GB GPU memory

**Inference Memory** (batch_size=1):
- ~TBD MB

### Computational Complexity

**FLOPs (Floating Point Operations)**:
- CNN backbone: ~TBD GFLOPs
- LSTM layers: ~TBD GFLOPs
- Total: ~TBD GFLOPs per sample

---

## Relationship to Previous Milestones

### Building on Milestone 1 (CNN)

This milestone **reuses** CNN architectures from Milestone 1:
- Same CNN models (ResNet, EfficientNet, etc.)
- Same preprocessing pipeline
- Same data loading infrastructure

**Key difference**: Instead of using CNN for end-to-end classification, we use it as a feature extractor for the LSTM.

### Building on Milestone 2 (LSTM)

This milestone **reuses** LSTM architectures from Milestone 2:
- Same LSTM and BiLSTM implementations
- Same training infrastructure
- Same evaluation metrics

**Key difference**: Instead of processing raw signals directly, LSTM processes CNN-extracted features.

### Standalone Nature

While this milestone builds conceptually on Milestones 1 and 2, **it is completely standalone**:
- All necessary code is copied into `milestone-3/`
- No dependencies on other milestone folders
- Can be used independently

---

## Best Practices

### For Research and Experimentation

1. **Start with recommended configurations** to establish baselines
2. **Use custom configurations** to test hypotheses
3. **Track all experiments** with different random seeds
4. **Validate on held-out test set** only once

### For Production Deployment

1. **Train multiple models** with different seeds
2. **Ensemble predictions** for better reliability
3. **Quantize models** for faster inference (PyTorch JIT, ONNX)
4. **Monitor performance** on new data
5. **Retrain periodically** as new data arrives

### For Hyperparameter Tuning

1. **Fix random seed** for reproducibility
2. **Tune one parameter at a time** initially
3. **Use validation set** for selection
4. **Consider computational budget** when choosing architectures

---

## References

### Dataset

- Case Western Reserve University Bearing Data Center
- URL: https://engineering.case.edu/bearingdatacenter

### Deep Learning Frameworks

- PyTorch: https://pytorch.org/
- PyTorch Documentation: https://pytorch.org/docs/

### Related Papers

**CNNs for Vibration Analysis:**
- Zhang et al., "Deep Learning for Fault Diagnosis"
- Various ResNet and EfficientNet papers

**LSTMs for Time Series:**
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- Greff et al., "LSTM: A Search Space Odyssey" (2017)

**Hybrid Approaches:**
- Various hybrid CNN-LSTM architectures for time series

### Architecture References

**ResNet:**
- He et al., "Deep Residual Learning for Image Recognition" (2016)

**EfficientNet:**
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs" (2019)

**LSTM/BiLSTM:**
- Schuster & Paliwal, "Bidirectional Recurrent Neural Networks" (1997)

---

## Support and Contact

For questions or issues:

1. Check this README and QUICKSTART.md
2. Review DELIVERY_NOTES.md for setup instructions
3. Check the Troubleshooting section
4. Verify your installation with `python scripts/validate_installation.py`

---

## License

This project is provided for research and educational purposes. Please ensure compliance with data usage policies and licensing requirements for any production deployment.

---

## Acknowledgments

- **Case Western Reserve University** for providing the bearing dataset
- **PyTorch Team** for the excellent deep learning framework
- **Research Community** for CNN and LSTM architecture innovations

---

**Document Version**: 1.0
**Last Updated**: [To be filled]
**Milestone**: 3 - CNN-LSTM Hybrid for Bearing Fault Diagnosis
