# Quick Start Guide - CNN Bearing Fault Diagnosis

This guide will walk you through getting the CNN-based bearing fault diagnosis system up and running in **under 10 minutes**.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] NVIDIA GPU with CUDA support (optional but recommended)
- [ ] 8GB+ RAM (16GB+ recommended)
- [ ] 1,430 .MAT files with vibration data

---

## Step-by-Step Setup

### 1. Installation (5 minutes)

```bash
# Navigate to the project directory
cd milestone-1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (choose your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed | CUDA available: {torch.cuda.is_available()}')"
```

**Expected output**:
```
✓ PyTorch 2.x.x installed | CUDA available: True
```

---

### 2. Prepare Your Data (2 minutes)

Organize your .MAT files in this structure:

```
data/raw/bearing_data/
├── sain/                    # Healthy bearings (130 samples)
│   ├── sample_001.mat
│   ├── sample_002.mat
│   └── ...
├── desalignement/           # Misalignment (~130 samples)
├── desequilibre/            # Imbalance (~130 samples)
├── jeu/                     # Bearing clearance (~130 samples)
├── lubrification/           # Lubrication (~130 samples)
├── cavitation/              # Cavitation (~130 samples)
├── usure/                   # Wear (~130 samples)
├── oilwhirl/                # Oil whirl (~130 samples)
├── mixed_misalign_imbalance/  # Mixed fault 1 (~130 samples)
├── mixed_wear_lube/         # Mixed fault 2 (~130 samples)
└── mixed_cavit_jeu/         # Mixed fault 3 (~130 samples)
```

**Quick validation:**

```bash
# Count files per fault type
find data/raw/bearing_data -name "*.mat" | wc -l
# Expected: 1430 (or close to it)

# Validate data structure
python scripts/import_mat_dataset.py \
    --mat-dir data/raw/bearing_data \
    --output data/processed/dataset_info.json \
    --validate
```

---

### 3. Train Your First Model (2 minutes to start)

#### Option A: Fast Baseline (CNN1D - 30 minutes training)

```bash
python scripts/train_cnn.py \
    --model cnn1d \
    --data-dir data/raw/bearing_data \
    --epochs 50 \
    --batch-size 32 \
    --checkpoint-dir results/checkpoints/cnn1d_baseline
```

**Expected accuracy**: 93-95%

#### Option B: High Accuracy (ResNet-34 - 2-4 hours training)

```bash
python scripts/train_cnn.py \
    --model resnet34 \
    --data-dir data/raw/bearing_data \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --optimizer adamw \
    --scheduler cosine \
    --augment \
    --mixed-precision \
    --checkpoint-dir results/checkpoints/resnet34_best
```

**Expected accuracy**: 96-97% ⭐

#### Option C: Efficient & Accurate (EfficientNet-B2 - 1-2 hours training)

```bash
python scripts/train_cnn.py \
    --model efficientnet_b2 \
    --data-dir data/raw/bearing_data \
    --epochs 75 \
    --batch-size 48 \
    --lr 0.001 \
    --augment \
    --mixed-precision \
    --checkpoint-dir results/checkpoints/efficientnet_b2
```

**Expected accuracy**: 96-97%, only 9M parameters ⭐

---

### 4. Monitor Training (Real-time)

While training is running, open another terminal:

```bash
# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start TensorBoard
tensorboard --logdir results/logs

# Open browser to: http://localhost:6006
```

You'll see:
- Training/validation loss curves
- Training/validation accuracy curves
- Learning rate schedule

---

### 5. Evaluate Your Model (1 minute)

After training completes:

```bash
python scripts/evaluate_cnn.py \
    --model-checkpoint results/checkpoints/resnet34_best/best_model.pth \
    --data-dir data/raw/bearing_data \
    --output-dir results/evaluation/resnet34 \
    --batch-size 128
```

**Outputs generated:**
- `classification_report.txt` - Precision, recall, F1 per class
- `confusion_matrix.png` - Visual confusion matrix
- `per_class_metrics.csv` - Detailed metrics table
- `model_summary.txt` - Model info and inference time

---

## Example Workflow

Here's a complete workflow from start to finish:

```bash
# 1. Setup (one-time)
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Validate data
python scripts/import_mat_dataset.py \
    --mat-dir data/raw/bearing_data \
    --output data/processed/dataset_info.json \
    --validate

# 3. Train ResNet-34 (recommended)
python scripts/train_cnn.py \
    --model resnet34 \
    --data-dir data/raw/bearing_data \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --optimizer adamw \
    --scheduler cosine \
    --augment \
    --mixup \
    --mixed-precision \
    --checkpoint-dir results/checkpoints/resnet34

# 4. Evaluate
python scripts/evaluate_cnn.py \
    --model-checkpoint results/checkpoints/resnet34/best_model.pth \
    --data-dir data/raw/bearing_data \
    --output-dir results/evaluation/resnet34

# 5. View results
cat results/evaluation/resnet34/classification_report.txt
open results/evaluation/resnet34/confusion_matrix.png  # or xdg-open on Linux
```

---

## Python API Usage

You can also use the models programmatically:

```python
import torch
from models import create_model
from data.matlab_importer import load_mat_signals
from utils.device_manager import get_device

# Load model
model = create_model('resnet34', num_classes=11)
checkpoint = torch.load('results/checkpoints/resnet34/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Move to GPU if available
device = get_device()
model = model.to(device)

# Load and preprocess signal
signal = load_mat_signals('path/to/signal.mat')
signal = torch.FloatTensor(signal).unsqueeze(0).to(device)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(signal)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

---

## Common Issues & Quick Fixes

### Issue 1: CUDA Out of Memory

```bash
# Solution: Reduce batch size
python scripts/train_cnn.py --model resnet34 --batch-size 32  # Instead of 64

# Or use mixed precision (reduces memory by 40%)
python scripts/train_cnn.py --model resnet34 --mixed-precision
```

### Issue 2: Training Too Slow

```bash
# Solution: Use more data workers
python scripts/train_cnn.py --num-workers 8  # Default is 4

# Or train a smaller model
python scripts/train_cnn.py --model efficientnet_b0  # Instead of resnet50
```

### Issue 3: Poor Accuracy

```bash
# Solution: Enable data augmentation
python scripts/train_cnn.py --augment --mixup

# Train for more epochs
python scripts/train_cnn.py --epochs 150

# Use a better model
python scripts/train_cnn.py --model resnet34  # or efficientnet_b2
```

---

## Next Steps

Once you have a trained model:

1. **Experiment with different architectures**: Try all models and compare
2. **Hyperparameter tuning**: Adjust learning rate, batch size, dropout
3. **Visualize results**: Use the visualization tools to understand predictions
4. **Deploy for inference**: Use `scripts/inference_cnn.py` for production

---

## Performance Expectations

| Model | Training Time* | Test Accuracy | When to Use |
|-------|---------------|---------------|-------------|
| **CNN1D** | 30 min | 93-95% | Quick baseline, fast inference |
| **ResNet-34** | 3-4 hours | 96-97% | Best accuracy |
| **EfficientNet-B2** | 1-2 hours | 96-97% | Balanced performance |

*On NVIDIA RTX 3090. CPU training will be 10-20x slower.

---

## Support

If you encounter issues not covered here:

1. Check the main [README.md](README.md) for detailed documentation
2. Review the [Troubleshooting](#troubleshooting) section in README.md
3. Contact: your.email@example.com

---

**You're all set! Happy fault diagnosis!** 🚀
