# Quick Start Guide - Milestone 3: CNN-LSTM Hybrid

**Get up and running with hybrid models in 5 minutes!**

---

## Prerequisites

- Python 3.8+
- Your bearing `.mat` files ready
- Basic familiarity with command line

---

## 5-Minute Setup

### Step 1: Environment Setup (1 minute)

```bash
cd milestones/milestone-3

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data (1 minute)

```bash
# Create data directory
mkdir -p data/raw/bearing_data

# Copy your .mat files
cp /path/to/your/*.mat data/raw/bearing_data/

# Verify data
ls data/raw/bearing_data/*.mat | wc -l  # Should see your .mat files
```

### Step 3: Train Your First Model (2 minutes to start)

```bash
# Train recommended hybrid model (ResNet34 + BiLSTM)
python scripts/train_hybrid.py \
  --model recommended_1 \
  --epochs 75 \
  --batch-size 32
```

**Training will take several hours depending on your hardware.**

---

## Three Ways to Use Hybrid Models

### Option 1: Recommended Configurations (Easiest)

We provide 3 pre-configured hybrid models optimized for different use cases:

#### Configuration 1: Best Accuracy

```bash
python scripts/train_hybrid.py --model recommended_1 --epochs 75
```

- **Architecture**: ResNet34 + BiLSTM
- **Use case**: Maximum accuracy
- **Expected accuracy**: [TBD - To be determined after training]

#### Configuration 2: Best Efficiency

```bash
python scripts/train_hybrid.py --model recommended_2 --epochs 75
```

- **Architecture**: EfficientNet-B2 + BiLSTM
- **Use case**: Efficient training, smaller model
- **Expected accuracy**: [TBD - To be determined after training]

#### Configuration 3: Best Speed

```bash
python scripts/train_hybrid.py --model recommended_3 --epochs 75
```

- **Architecture**: ResNet18 + LSTM
- **Use case**: Fast inference, real-time
- **Expected accuracy**: [TBD - To be determined after training]

---

### Option 2: Custom Configuration (Flexible)

Mix and match **any CNN** with **any LSTM**:

```bash
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type resnet34 \
  --lstm-type bilstm \
  --lstm-hidden-size 256 \
  --lstm-num-layers 2 \
  --pooling mean \
  --epochs 75
```

**Available CNN Backbones:**
- `cnn1d` - Basic 1D CNN
- `resnet18`, `resnet34`, `resnet50` - ResNet variants
- `efficientnet_b0`, `efficientnet_b2`, `efficientnet_b4` - EfficientNet variants

**Available LSTM Types:**
- `lstm` - Unidirectional LSTM
- `bilstm` - Bidirectional LSTM

**Example Combinations:**

```bash
# Lightweight hybrid
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type cnn1d \
  --lstm-type lstm \
  --lstm-hidden-size 128 \
  --epochs 50

# High-capacity hybrid
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type resnet50 \
  --lstm-type bilstm \
  --lstm-hidden-size 512 \
  --epochs 100

# Efficient hybrid with attention pooling
python scripts/train_hybrid.py \
  --model custom \
  --cnn-type efficientnet_b0 \
  --lstm-type bilstm \
  --pooling attention \
  --epochs 75
```

---

### Option 3: Python API (Advanced)

For programmatic usage in your own scripts:

```python
import torch
from pathlib import Path
from models import create_model, list_available_cnn_backbones, list_available_lstm_types
from data.cnn_dataloader import create_cnn_dataloaders
from training.cnn_trainer import CNNTrainer
from training.optimizers import create_optimizer
from training.losses import create_loss_function
from utils.device_manager import get_device

# See available architectures
print("Available CNNs:", list_available_cnn_backbones())
print("Available LSTMs:", list_available_lstm_types())

# Create a recommended model
model = create_model('recommended_1')

# Or create a custom model
model = create_model(
    'custom',
    cnn_type='resnet34',
    lstm_type='bilstm',
    lstm_hidden_size=256,
    lstm_num_layers=2,
    pooling_method='mean'
)

# Get model information
info = model.get_model_info()
print(f"Total parameters: {info['total_params']:,}")
print(f"CNN parameters: {info['cnn_params']:,}")
print(f"LSTM parameters: {info['lstm_params']:,}")
print(f"Model size: {info['model_size_mb']:.2f} MB")

# Create dataloaders
train_loader, val_loader, test_loader = create_cnn_dataloaders(
    data_dir='data/raw/bearing_data',
    batch_size=32,
    num_workers=4
)

# Setup training
device = get_device()
optimizer = create_optimizer(model.parameters(), 'adam', lr=0.001)
criterion = create_loss_function('cross_entropy', num_classes=11)

# Create trainer
trainer = CNNTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    checkpoint_dir=Path('results/checkpoints/my_hybrid')
)

# Train
history = trainer.fit(num_epochs=75, save_best=True, verbose=True)

# Evaluate on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for signals, labels in test_loader:
        signals, labels = signals.to(device), labels.to(device)
        outputs = model(signals)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100. * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

---

## Training Options

### Basic Training

```bash
python scripts/train_hybrid.py --model recommended_1 --epochs 75
```

### Advanced Training Options

```bash
python scripts/train_hybrid.py \
  --model recommended_1 \
  --data-dir data/raw/bearing_data \
  --epochs 75 \
  --batch-size 32 \
  --lr 0.001 \
  --optimizer adam \
  --scheduler cosine \
  --mixed-precision \
  --checkpoint-dir results/checkpoints/hybrid \
  --seed 42
```

**Key Arguments:**

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--model` | Model configuration | `recommended_1` | `recommended_1`, `recommended_2`, `recommended_3`, `custom` |
| `--cnn-type` | CNN backbone (for custom) | `resnet34` | `cnn1d`, `resnet18/34/50`, `efficientnet_b0/b2/b4` |
| `--lstm-type` | LSTM type (for custom) | `bilstm` | `lstm`, `bilstm` |
| `--lstm-hidden-size` | LSTM hidden dimension | 256 | Any integer |
| `--lstm-num-layers` | Number of LSTM layers | 2 | Any integer |
| `--pooling` | Temporal pooling method | `mean` | `mean`, `max`, `last`, `attention` |
| `--freeze-cnn` | Freeze CNN weights | False | Flag (no value) |
| `--epochs` | Training epochs | 75 | Any integer |
| `--batch-size` | Batch size | 32 | Any integer |
| `--lr` | Learning rate | 0.001 | Any float |
| `--optimizer` | Optimizer | `adam` | `adam`, `adamw`, `sgd` |
| `--scheduler` | LR scheduler | `cosine` | `cosine`, `step`, `plateau`, `none` |
| `--mixed-precision` | Enable FP16 training | False | Flag (no value) |
| `--seed` | Random seed | 42 | Any integer |

---

## Evaluation

### Evaluate a Trained Model

```bash
python scripts/evaluate_hybrid.py \
  --checkpoint results/checkpoints/hybrid/recommended_1/20240115_143022/best_model.pth \
  --model recommended_1 \
  --data-dir data/raw/bearing_data \
  --batch-size 64 \
  --output-dir results/evaluation/recommended_1
```

**Outputs:**
- `classification_report.txt` - Detailed metrics
- `confusion_matrix.png` - Confusion matrix plot
- `per_class_accuracy.png` - Per-class performance
- `predictions.csv` - All predictions with labels

### Visualize Training Results

```bash
python scripts/visualize_results.py \
  --checkpoint results/checkpoints/hybrid/recommended_1/best_model.pth \
  --model recommended_1 \
  --output-dir results/visualizations
```

**Outputs:**
- Training loss and accuracy curves
- Learning rate schedule
- Confusion matrix
- Feature visualizations

---

## Common Workflows

### Workflow 1: Quick Experiment

```bash
# 1. Train with recommended config
python scripts/train_hybrid.py --model recommended_1 --epochs 50

# 2. Evaluate
python scripts/evaluate_hybrid.py \
  --checkpoint results/checkpoints/hybrid/recommended_1/[timestamp]/best_model.pth \
  --model recommended_1

# 3. Visualize
python scripts/visualize_results.py \
  --checkpoint results/checkpoints/hybrid/recommended_1/[timestamp]/best_model.pth \
  --model recommended_1
```

### Workflow 2: Compare Configurations

```bash
# Train all three recommended configs
python scripts/train_hybrid.py --model recommended_1 --epochs 75 --seed 42
python scripts/train_hybrid.py --model recommended_2 --epochs 75 --seed 42
python scripts/train_hybrid.py --model recommended_3 --epochs 75 --seed 42

# Compare results
# (Results will be in separate checkpoint directories)
```

### Workflow 3: Custom Architecture Search

```bash
# Try different CNN backbones
python scripts/train_hybrid.py --model custom --cnn-type resnet18 --lstm-type bilstm --seed 42
python scripts/train_hybrid.py --model custom --cnn-type resnet34 --lstm-type bilstm --seed 42
python scripts/train_hybrid.py --model custom --cnn-type efficientnet_b2 --lstm-type bilstm --seed 42

# Try different pooling methods
python scripts/train_hybrid.py --model custom --cnn-type resnet34 --pooling mean --seed 42
python scripts/train_hybrid.py --model custom --cnn-type resnet34 --pooling attention --seed 42
```

### Workflow 4: Transfer Learning

```bash
# First, train CNN backbone (from Milestone 1)
# Then freeze it and train only LSTM

python scripts/train_hybrid.py \
  --model custom \
  --cnn-type resnet34 \
  --lstm-type bilstm \
  --freeze-cnn \
  --epochs 50 \
  --lr 0.01
```

---

## Performance Optimization

### GPU Training (Recommended)

```bash
# Enable mixed precision for 2x speedup
python scripts/train_hybrid.py \
  --model recommended_1 \
  --mixed-precision \
  --epochs 75
```

**Benefits:**
- ~2x faster training
- ~50% less GPU memory
- Minimal accuracy impact

### CPU Training

```bash
# Will automatically use CPU if no GPU available
python scripts/train_hybrid.py --model recommended_1 --epochs 75
```

**Note**: CPU training will be significantly slower (10-50x depending on hardware).

### Batch Size Tuning

```bash
# Larger batch = faster training (if GPU memory allows)
python scripts/train_hybrid.py --batch-size 64 --mixed-precision

# Smaller batch = less memory
python scripts/train_hybrid.py --batch-size 16
```

---

## Troubleshooting

### Issue: Out of Memory

```bash
# Solution 1: Reduce batch size
python scripts/train_hybrid.py --batch-size 16

# Solution 2: Enable mixed precision
python scripts/train_hybrid.py --mixed-precision

# Solution 3: Use lighter model
python scripts/train_hybrid.py --model recommended_3
```

### Issue: Training Too Slow

```bash
# Solution 1: Enable mixed precision
python scripts/train_hybrid.py --mixed-precision

# Solution 2: Increase batch size
python scripts/train_hybrid.py --batch-size 64

# Solution 3: Use lighter model
python scripts/train_hybrid.py --model recommended_3
```

### Issue: No .mat Files Found

```bash
# Check data directory
ls data/raw/bearing_data/*.mat

# Specify correct path
python scripts/train_hybrid.py --data-dir /absolute/path/to/bearing_data
```

### Issue: Poor Accuracy

```bash
# Solution 1: Train longer
python scripts/train_hybrid.py --epochs 100

# Solution 2: Try different learning rate
python scripts/train_hybrid.py --lr 0.0001

# Solution 3: Try different model
python scripts/train_hybrid.py --model recommended_2

# Solution 4: Check data quality
python scripts/validate_installation.py
```

---

## Next Steps

1. **Train recommended models** to establish baselines
2. **Experiment with custom configurations** to find optimal architecture
3. **Evaluate on test set** to measure generalization
4. **Compare with Milestone 1 (CNN) and Milestone 2 (LSTM)** results
5. **Deploy best model** for production use

---

## Quick Reference

### List Available Models

```python
from models import list_available_cnn_backbones, list_available_lstm_types

print("CNNs:", list_available_cnn_backbones())
print("LSTMs:", list_available_lstm_types())
```

### Check Model Information

```python
from models import create_model

model = create_model('recommended_1')
info = model.get_model_info()

print(f"Parameters: {info['total_params']:,}")
print(f"Size: {info['model_size_mb']:.2f} MB")
```

### Monitor Training

Training progress is displayed in real-time:

```
Epoch [1/75] (45.23s)
  Train Loss: 2.1234, Train Acc: 45.67%
  Val Loss: 1.9876, Val Acc: 52.34%
  LR: 0.001000

Epoch [2/75] (44.89s)
  Train Loss: 1.8765, Train Acc: 58.91%
  Val Loss: 1.7234, Val Acc: 61.23%
  LR: 0.000995
...
```

Checkpoints are saved automatically:
- `results/checkpoints/hybrid/[model]/[timestamp]/best_model.pth`

---

## Resources

- **Full Documentation**: See [README.md](README.md)
- **Delivery Notes**: See [DELIVERY_NOTES.md](DELIVERY_NOTES.md)
- **Validation Script**: `python scripts/validate_installation.py`

---

## Summary

**Three simple steps to train a hybrid model:**

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Prepare data
mkdir -p data/raw/bearing_data
cp /path/to/*.mat data/raw/bearing_data/

# 3. Train
python scripts/train_hybrid.py --model recommended_1 --epochs 75
```

**That's it!** 🎉

For more details and advanced usage, see the full [README.md](README.md).
