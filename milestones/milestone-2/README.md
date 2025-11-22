# LSTM-Based Bearing Fault Diagnosis System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Milestone](https://img.shields.io/badge/Milestone-2%2F4-blue)

**Long Short-Term Memory (LSTM) networks for bearing fault diagnosis** - capturing temporal dependencies and sequential patterns in vibration signals for predictive maintenance.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Why LSTM for Bearing Fault Diagnosis?](#-why-lstm-for-bearing-fault-diagnosis)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [LSTM Architectures](#-lstm-architectures)
- [Training Models](#-training-models)
- [Model Evaluation](#-model-evaluation)
- [Project Structure](#-project-structure)
- [Relationship to Milestone 1 (CNN)](#-relationship-to-milestone-1-cnn)
- [Performance Expectations](#-performance-expectations)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## 🎯 Overview

### What is This Milestone?

**Milestone 2** introduces **Long Short-Term Memory (LSTM) networks** for bearing fault diagnosis. Building on Milestone 1 (CNN-based diagnosis), this milestone explores how recurrent neural networks can capture **temporal dependencies** and **long-term patterns** in vibration signals.

### What Problem Does LSTM Solve?

While CNNs (from Milestone 1) excel at detecting **local patterns** and **spatial features**, LSTMs are specifically designed for **sequential data** where the order and temporal relationships matter:

- **Temporal Dependencies**: LSTMs can learn patterns that evolve over time
- **Long-Term Memory**: Remember important features from earlier in the sequence
- **Sequential Patterns**: Capture how vibration characteristics change during fault progression

### Dataset

Uses the same **1,430 vibration signal samples** from bearing test rigs:
- **11 fault types** (healthy + 10 fault conditions)
- **102,400 samples per signal** (5 seconds at 20.48 kHz)
- **Raw .MAT files** (no preprocessing required)

---

## 🔍 Why LSTM for Bearing Fault Diagnosis?

### Vibration Signals are Sequential

Bearing vibration signals are **time-series data** where:
1. **Order matters**: The sequence of vibration amplitudes contains diagnostic information
2. **Temporal patterns**: Faults create characteristic patterns that evolve over time
3. **Long-range dependencies**: Early signal behavior can influence later predictions

### Advantages of LSTM

1. **Sequential Processing**
   - Processes signals step-by-step, maintaining temporal context
   - Captures how vibration evolves from start to end of measurement

2. **Memory Mechanisms**
   - **Cell state**: Long-term memory of important patterns
   - **Gates**: Learn what information to remember, forget, or output
   - Can capture fault signatures that span the entire signal

3. **Temporal Feature Learning**
   - Automatically learns time-dependent features
   - No need for manual feature engineering
   - Discovers temporal patterns CNNs might miss

4. **Complementary to CNNs**
   - CNNs: Spatial/local patterns → "what" features are present
   - LSTMs: Temporal/sequential patterns → "when" and "how" features evolve

### When to Use LSTM vs CNN?

| Aspect | CNN (Milestone 1) | LSTM (Milestone 2) |
|--------|-------------------|---------------------|
| **Strengths** | Local patterns, fast inference | Temporal dependencies, sequential patterns |
| **Best for** | Stationary signals, frequency features | Time-varying signals, progression tracking |
| **Speed** | Faster (parallel processing) | Slower (sequential processing) |
| **Memory** | Lower | Higher (stores hidden states) |
| **Use case** | Real-time monitoring | Offline analysis, trend detection |

**Recommendation**: Use both! CNN and LSTM provide complementary information. Milestone 3 (CNN-LSTM Hybrid) will combine their strengths.

---

## ✨ Key Features

### 🤖 **LSTM Architectures**

- **Vanilla LSTM**: Unidirectional LSTM for forward sequential processing
- **Bidirectional LSTM (BiLSTM)**: Processes sequence in both directions for complete context

### 📊 **Same 11 Fault Types**

(Consistent with Milestone 1)

1. **Healthy (Sain)** - Normal bearing operation
2. **Misalignment (Désalignement)** - Shaft misalignment
3. **Imbalance (Déséquilibre)** - Rotor imbalance
4. **Bearing Clearance (Jeu)** - Excessive clearance
5. **Lubrication Issues** - Poor/degraded lubrication
6. **Cavitation** - Fluid cavitation damage
7. **Wear (Usure)** - General bearing wear
8. **Oil Whirl** - Oil-induced instability
9. **Mixed: Misalignment + Imbalance**
10. **Mixed: Wear + Lubrication**
11. **Mixed: Cavitation + Clearance**

### 🔧 **Training Features**

- **Direct .MAT file loading** (no preprocessing needed)
- **Flexible sequence processing** (full 102,400 samples or downsampled)
- **Mixed precision training** (FP16 for faster training)
- **Multiple optimizers** (Adam, AdamW, SGD, RMSprop)
- **Learning rate scheduling** (Cosine, Step, ReduceLROnPlateau)
- **Early stopping** to prevent overfitting
- **Gradient clipping** for stable training

### 📈 **Evaluation & Visualization**

- Confusion matrix and classification reports
- Per-class precision, recall, F1-score
- Training history plots
- LSTM hidden state visualization (optional)

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (recommended, CPU also supported)
- **8GB+ RAM** (16GB+ recommended for BiLSTM)

### Installation Steps

```bash
cd milestone-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

**Expected output**: `PyTorch 2.x.x | CUDA: True`

---

## 🏁 Quick Start

### Step 1: Prepare Data

Organize your 1,430 .MAT files (same as Milestone 1):

```
data/raw/bearing_data/
├── sain/
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

### Step 2: Train Vanilla LSTM

```bash
python scripts/train_lstm.py \
    --model vanilla_lstm \
    --data-dir data/raw/bearing_data \
    --hidden-size 128 \
    --num-layers 2 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

### Step 3: Train BiLSTM (Recommended)

```bash
python scripts/train_lstm.py \
    --model bilstm \
    --data-dir data/raw/bearing_data \
    --hidden-size 256 \
    --num-layers 2 \
    --epochs 75 \
    --batch-size 32 \
    --lr 0.001 \
    --mixed-precision \
    --checkpoint-dir results/checkpoints/bilstm
```

### Step 4: Evaluate

```bash
python scripts/evaluate_lstm.py \
    --model-checkpoint results/checkpoints/bilstm/best_model.pth \
    --data-dir data/raw/bearing_data
```

---

## 🧠 LSTM Architectures

### 1. Vanilla LSTM

**Unidirectional LSTM** that processes the signal from start to finish.

**Architecture:**
```
Input: [B, 1, 102400] (raw vibration signal)
  ↓
Reshape: [B, 102400, 1] (sequence of time steps)
  ↓
LSTM(hidden_size=128, num_layers=2)
  ↓
Global Average Pooling (across time)
  ↓
Fully Connected: 128 → 64 → 11 classes
```

**When to use:**
- Simpler model, faster training
- Lower memory requirements
- Good for real-time or online processing

**Parameters:**
- Hidden size: 128 (default)
- Num layers: 2
- ~200K parameters (with hidden_size=128)

**Training command:**
```bash
python scripts/train_lstm.py --model vanilla_lstm --hidden-size 128 --epochs 50
```

### 2. Bidirectional LSTM (BiLSTM)

**Bidirectional processing** - reads signal in both forward and backward directions.

**Architecture:**
```
Input: [B, 1, 102400]
  ↓
Reshape: [B, 102400, 1]
  ↓
BiLSTM(hidden_size=128, num_layers=2)
  Forward LSTM →
  Backward LSTM ←
  Concatenate: [forward, backward] = 256 features
  ↓
Global Average Pooling
  ↓
Fully Connected: 256 → 128 → 11 classes
```

**When to use:**
- Need complete context (past + future)
- Offline analysis (entire signal available)
- Better accuracy (typically 2-5% higher than vanilla)

**Parameters:**
- Hidden size: 128 per direction (256 total)
- Num layers: 2
- ~400K parameters (with hidden_size=128)

**Training command:**
```bash
python scripts/train_lstm.py --model bilstm --hidden-size 256 --epochs 75 --mixed-precision
```

### Model Comparison

| Model | Parameters | Memory | Speed | Accuracy | Use Case |
|-------|------------|--------|-------|----------|----------|
| **Vanilla LSTM** | ~200K | Lower | Faster | Good | Real-time, resource-constrained |
| **BiLSTM** | ~400K | Higher | Slower | Better | Offline, high-accuracy needed |

**Recommendation**: Start with **BiLSTM** for best accuracy, use **Vanilla LSTM** for deployment.

---

## 🎓 Training Models

### Training Script Arguments

```bash
python scripts/train_lstm.py [OPTIONS]

Required:
  --model {vanilla_lstm,bilstm}     LSTM architecture
  --data-dir PATH                   Directory with .MAT files

Model Architecture:
  --hidden-size INT                 LSTM hidden size (default: 128)
  --num-layers INT                  Number of LSTM layers (default: 2)
  --dropout FLOAT                   Dropout probability (default: 0.3)

Training:
  --epochs INT                      Number of epochs (default: 50)
  --batch-size INT                  Batch size (default: 32)
  --val-batch-size INT              Validation batch size (default: 64)

Optimization:
  --optimizer {adam,adamw,sgd,rmsprop}  Optimizer (default: adam)
  --lr FLOAT                        Learning rate (default: 0.001)
  --weight-decay FLOAT              Weight decay (default: 0.0001)
  --gradient-clip FLOAT             Gradient clipping max norm

Scheduling:
  --scheduler {cosine,step,plateau,none}
  --lr-patience INT                 Patience for ReduceLROnPlateau
  --early-stopping-patience INT     Early stopping patience (default: 15)

Performance:
  --mixed-precision                 Enable FP16 training
  --num-workers INT                 Data loading workers (default: 4)

Output:
  --checkpoint-dir PATH             Checkpoint directory
  --seed INT                        Random seed (default: 42)
```

### Training Examples

**1. Quick Training (Vanilla LSTM)**

```bash
python scripts/train_lstm.py \
    --model vanilla_lstm \
    --data-dir data/raw/bearing_data \
    --epochs 50 \
    --batch-size 32
```

**2. High Accuracy (BiLSTM)**

```bash
python scripts/train_lstm.py \
    --model bilstm \
    --data-dir data/raw/bearing_data \
    --hidden-size 256 \
    --num-layers 2 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --scheduler cosine \
    --mixed-precision \
    --early-stopping-patience 20
```

**3. Resource-Constrained (Small BiLSTM)**

```bash
python scripts/train_lstm.py \
    --model bilstm \
    --data-dir data/raw/bearing_data \
    --hidden-size 64 \
    --num-layers 1 \
    --epochs 75 \
    --batch-size 16 \
    --mixed-precision
```

### Expected Training Time

| Model | Hidden Size | Epochs | GPU (RTX 3090) | CPU (16 cores) |
|-------|-------------|--------|----------------|----------------|
| Vanilla LSTM | 128 | 50 | 1-2 hours | 10-15 hours |
| BiLSTM | 128 | 75 | 2-3 hours | 15-20 hours |
| BiLSTM | 256 | 100 | 3-5 hours | 25-35 hours |

**Note**: Training time depends on hardware. GPU highly recommended!

---

## 📊 Model Evaluation

### Evaluate Trained Model

```bash
python scripts/evaluate_lstm.py \
    --model-checkpoint results/checkpoints/bilstm/best_model.pth \
    --data-dir data/raw/bearing_data \
    --output-dir results/evaluation/bilstm
```

### Evaluation Outputs

1. **Classification Report** (`classification_report.txt`)
   - Precision, recall, F1-score per fault type
   - Overall accuracy

2. **Confusion Matrix** (`confusion_matrix.png`)
   - Heatmap of predicted vs. actual classes

3. **Per-Class Metrics** (`per_class_metrics.csv`)
   - Detailed performance metrics

4. **Model Summary** (`model_summary.txt`)
   - Model parameters
   - Inference time

---

## 📂 Project Structure

```
milestone-2/
├── README.md                  ⭐ This file
├── QUICKSTART.md              Quick start guide
├── DELIVERY_NOTES.md          Delivery summary
├── requirements.txt           Dependencies
├── .gitignore                 Git ignore rules
│
├── data/                      Data loading & processing
│   ├── __init__.py
│   ├── lstm_dataset.py        PyTorch Dataset for LSTM
│   ├── lstm_dataloader.py     DataLoader creation
│   ├── matlab_importer.py     .MAT file loading
│   ├── augmentation.py        Data augmentation
│   └── ...
│
├── models/                    LSTM architectures
│   ├── __init__.py            Model factory
│   └── lstm/
│       ├── __init__.py
│       └── lstm_models.py     Vanilla LSTM, BiLSTM
│
├── training/                  Training utilities
│   ├── __init__.py
│   ├── lstm_trainer.py        Training loop
│   ├── metrics.py             Evaluation metrics
│   ├── losses.py              Loss functions
│   └── optimizers.py          Optimizer creation
│
├── scripts/                   Executable scripts
│   ├── train_lstm.py          Training CLI
│   └── evaluate_lstm.py       Evaluation CLI
│
├── utils/                     Utilities
│   ├── constants.py           Project constants
│   ├── device_manager.py      GPU/CPU handling
│   ├── checkpoint_manager.py  Model saving
│   └── ...
│
├── visualization/             Plotting tools
│   ├── performance_plots.py   Metrics visualization
│   └── signal_plots.py        Signal analysis
│
└── results/                   Output directory
    ├── checkpoints/           Saved models
    ├── logs/                  Training logs
    └── evaluation/            Evaluation results
```

---

## 🔗 Relationship to Milestone 1 (CNN)

### What You Already Have

From **Milestone 1**, you have:
- ✅ CNN-based fault diagnosis (96-97% accuracy)
- ✅ 15+ CNN architectures (ResNet, EfficientNet, etc.)
- ✅ Fast inference (<30ms per sample)
- ✅ Excellent for real-time monitoring

### What Milestone 2 Adds

**LSTM-based approach** provides:
- ✅ Temporal pattern recognition
- ✅ Sequential dependency modeling
- ✅ Long-term memory of signal evolution
- ✅ Complementary features to CNN

### CNN vs LSTM - Quick Comparison

| Aspect | CNN (Milestone 1) | LSTM (Milestone 2) |
|--------|-------------------|---------------------|
| **Processing** | Parallel (convolution) | Sequential (time steps) |
| **Features** | Local patterns, frequency | Temporal dependencies, sequences |
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Moderate |
| **Memory** | Lower | Higher |
| **Accuracy** | 96-97% | TBD (typically competitive) |
| **Best for** | Real-time, edge devices | Offline analysis, research |

### Can I Use Both?

**Yes!** In fact, combining CNN and LSTM often gives the best results:

1. **Sequential approach**: Train both separately, compare results
2. **Ensemble**: Average predictions from both models
3. **Hybrid**: Use CNN for feature extraction + LSTM for sequence modeling (coming in Milestone 3!)

---

## 📈 Performance Expectations

### Typical Results

Based on similar LSTM implementations on bearing fault datasets:

| Model | Expected Accuracy | Training Time (GPU) | Parameters |
|-------|-------------------|---------------------|------------|
| Vanilla LSTM (128) | 92-95% | 1-2 hours | ~200K |
| BiLSTM (128) | 94-96% | 2-3 hours | ~400K |
| BiLSTM (256) | 95-97% | 3-5 hours | ~1.5M |

**Note**: Actual results depend on:
- Dataset quality
- Hyperparameter tuning
- Training duration
- Hardware

### Per-Class Performance

Expect similar per-class performance to Milestone 1 CNN:
- **Healthy, Imbalance**: 97-99% (easy to classify)
- **Misalignment, Oil Whirl**: 94-97% (moderate difficulty)
- **Mixed faults**: 92-95% (more challenging)

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python scripts/train_lstm.py --batch-size 16  # Instead of 32

# Use smaller hidden size
python scripts/train_lstm.py --hidden-size 64  # Instead of 128

# Enable mixed precision
python scripts/train_lstm.py --mixed-precision
```

#### 2. Training Diverges / NaN Loss

**Causes**: Exploding gradients (common with LSTMs)

**Solutions**:
```bash
# Enable gradient clipping
python scripts/train_lstm.py --gradient-clip 1.0

# Reduce learning rate
python scripts/train_lstm.py --lr 0.0001

# Use AdamW optimizer with weight decay
python scripts/train_lstm.py --optimizer adamw --weight-decay 0.01
```

#### 3. Very Slow Training

**Solutions**:
```bash
# Use mixed precision (2x speedup)
python scripts/train_lstm.py --mixed-precision

# Increase workers
python scripts/train_lstm.py --num-workers 8

# Use smaller model
python scripts/train_lstm.py --hidden-size 64 --num-layers 1
```

#### 4. Poor Accuracy (<85%)

**Check these:**
1. Data normalization enabled? (default: yes)
2. Sufficient training epochs? (try 75-100)
3. Learning rate too high/low? (try 0.001 to 0.0001)
4. Early stopping too aggressive? (increase patience)

**Try:**
```bash
python scripts/train_lstm.py \
    --model bilstm \
    --hidden-size 256 \
    --epochs 100 \
    --lr 0.0005 \
    --scheduler cosine \
    --early-stopping-patience 20
```

---

## 💡 Tips for Best Results

### 1. Start Simple, Then Scale

```bash
# First: Vanilla LSTM, small size
python scripts/train_lstm.py --model vanilla_lstm --hidden-size 64 --epochs 30

# Then: BiLSTM, medium size
python scripts/train_lstm.py --model bilstm --hidden-size 128 --epochs 50

# Finally: BiLSTM, large size
python scripts/train_lstm.py --model bilstm --hidden-size 256 --epochs 100 --mixed-precision
```

### 2. Hyperparameter Tuning

Key hyperparameters to tune (in order of importance):

1. **Hidden size**: 64, 128, 256
2. **Number of layers**: 1, 2, 3
3. **Learning rate**: 0.0001, 0.0005, 0.001
4. **Dropout**: 0.2, 0.3, 0.4

### 3. Use Mixed Precision

Always use `--mixed-precision` on modern GPUs (Volta/Turing/Ampere):
- 2x faster training
- 40% less memory
- No accuracy loss

### 4. Monitor Training

Check for these signs of healthy training:
- ✅ Training loss steadily decreasing
- ✅ Validation loss decreasing (may plateau)
- ✅ Gap between train/val loss < 10%
- ✅ Validation accuracy improving

---

## 📄 Citation

If you use this system in your research or project, please cite:

```bibtex
@software{lstm_bearing_fault_diagnosis_2025,
  title = {LSTM-Based Bearing Fault Diagnosis System},
  author = {Your Name},
  year = {2025},
  note = {Milestone 2: LSTM implementation for temporal fault pattern recognition},
  url = {https://github.com/abbas-ahmad-cowlar/bearing-fault-diagnosis}
}
```

---

## 📞 Support

For questions or issues:

**Email**: your.email@example.com

**Common questions**:
- Installation: Run `python validate_installation.py`
- Training issues: See [Troubleshooting](#troubleshooting)
- Compare with CNN: See [Relationship to Milestone 1](#relationship-to-milestone-1-cnn)

---

## 🙏 Acknowledgments

- **PyTorch Team**: Deep learning framework
- **Research Community**: LSTM architectures for time-series
- **Milestone 1**: CNN baseline for comparison

---

**Last Updated**: November 2025

**Version**: 1.0.0 (Milestone 2 - LSTM Implementation Complete)

---

<div align="center">

### 🚀 Ready to Get Started?

[Installation](#-installation) | [Quick Start](#-quick-start) | [Training](#-training-models)

**Milestone 2 of 4 - Sequential Pattern Recognition** 🧠

</div>
