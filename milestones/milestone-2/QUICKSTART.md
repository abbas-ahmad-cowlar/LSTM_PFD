# Quick Start Guide - LSTM Bearing Fault Diagnosis

This guide will get you up and running with LSTM-based bearing fault diagnosis in **under 10 minutes**.

---

## Prerequisites

- [ ] Python 3.8+
- [ ] NVIDIA GPU with CUDA (optional but recommended)
- [ ] 8GB+ RAM
- [ ] 1,430 .MAT files with vibration data

---

## Step-by-Step Setup

### 1. Installation (5 minutes)

```bash
# Navigate to milestone-2
cd milestone-2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'✓ PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

**Expected**: `✓ PyTorch 2.x.x | CUDA: True`

---

### 2. Prepare Data (Same as Milestone 1)

```
data/raw/bearing_data/
├── sain/ (130 samples)
├── desalignement/
├── desequilibre/
├── jeu/
├── lubrification/
├── cavitation/
├── usure/
├── oilwhirl/
├── mixed_misalign_imbalance/
├── mixed_wear_lube/
└── mixed_cavit_jeu/
```

---

### 3. Train Your First LSTM

#### Option A: Vanilla LSTM (Faster, Good Baseline)

```bash
python scripts/train_lstm.py \
    --model vanilla_lstm \
    --data-dir data/raw/bearing_data \
    --hidden-size 128 \
    --epochs 50 \
    --batch-size 32
```

**Training time**: 1-2 hours (GPU)
**Expected accuracy**: 92-95%

#### Option B: BiLSTM (Better Accuracy, Recommended)

```bash
python scripts/train_lstm.py \
    --model bilstm \
    --data-dir data/raw/bearing_data \
    --hidden-size 256 \
    --num-layers 2 \
    --epochs 75 \
    --batch-size 32 \
    --lr 0.001 \
    --scheduler cosine \
    --mixed-precision
```

**Training time**: 2-3 hours (GPU)
**Expected accuracy**: 94-97%

---

### 4. Evaluate Model

```bash
python scripts/evaluate_lstm.py \
    --model-checkpoint results/checkpoints/bilstm/best_model.pth \
    --data-dir data/raw/bearing_data \
    --output-dir results/evaluation/bilstm
```

**Outputs**:
- `classification_report.txt` - Precision/recall/F1
- `confusion_matrix.png` - Visual confusion matrix
- `per_class_metrics.csv` - Detailed metrics
- `model_summary.txt` - Model info

---

## Complete Workflow Example

```bash
# 1. Setup (one-time)
cd milestone-2
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Train BiLSTM (recommended)
python scripts/train_lstm.py \
    --model bilstm \
    --data-dir data/raw/bearing_data \
    --hidden-size 256 \
    --epochs 75 \
    --batch-size 32 \
    --mixed-precision \
    --checkpoint-dir results/checkpoints/bilstm

# 3. Evaluate
python scripts/evaluate_lstm.py \
    --model-checkpoint results/checkpoints/bilstm/best_model.pth \
    --data-dir data/raw/bearing_data

# 4. View results
cat results/evaluation/bilstm/classification_report.txt
```

---

## Python API Usage

```python
import torch
from models import create_model

# Load BiLSTM model
model = create_model('bilstm', num_classes=11, hidden_size=256)
checkpoint = torch.load('results/checkpoints/bilstm/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare signal [1, 1, 102400]
signal = torch.randn(1, 1, 102400).cuda()

# Predict
with torch.no_grad():
    output = model(signal)
    prediction = torch.argmax(output, dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, prediction].item()

print(f"Predicted class: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

---

## Common Issues

### Issue 1: CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train_lstm.py --model bilstm --batch-size 16

# Or use smaller model
python scripts/train_lstm.py --model bilstm --hidden-size 128
```

### Issue 2: Training Diverges (NaN)

```bash
# Enable gradient clipping
python scripts/train_lstm.py --gradient-clip 1.0 --lr 0.0001
```

### Issue 3: Slow Training

```bash
# Use mixed precision (2x speedup)
python scripts/train_lstm.py --mixed-precision

# Increase workers
python scripts/train_lstm.py --num-workers 8
```

---

## Comparison with Milestone 1 (CNN)

| Aspect | CNN (Milestone 1) | LSTM (Milestone 2) |
|--------|-------------------|---------------------|
| **Speed** | ⚡⚡⚡ Fastest | ⚡⚡ Moderate |
| **Memory** | Lower | Higher |
| **Strengths** | Local patterns | Temporal dependencies |
| **Best for** | Real-time | Offline analysis |
| **Training** | 1-4 hours | 2-5 hours |

**Use both for best results!** They provide complementary information.

---

## Next Steps

1. **Experiment**: Try different hidden sizes (64, 128, 256)
2. **Compare**: Compare LSTM results with your CNN from Milestone 1
3. **Tune**: Adjust learning rate and number of layers
4. **Visualize**: Explore training curves and confusion matrices

---

## Performance Expectations

| Model | Training Time* | Expected Accuracy | Parameters |
|-------|---------------|-------------------|------------|
| Vanilla LSTM (128) | 1-2 hours | 92-95% | ~200K |
| BiLSTM (128) | 2-3 hours | 94-96% | ~400K |
| BiLSTM (256) | 3-5 hours | 95-97% | ~1.5M |

*On NVIDIA RTX 3090. CPU is 10-20x slower.

---

## Support

- Main documentation: [README.md](README.md)
- Email: your.email@example.com

---

**You're ready to explore LSTM for fault diagnosis!** 🚀
