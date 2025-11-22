# LSTM PFD Quick Start Guide

**Welcome!** This guide will take you from zero to a fully-functional bearing fault diagnosis system in 11 phases.

> **👋 Complete Beginner?** This guide assumes you have no prior experience with the system. We'll walk through everything step-by-step, from installation to deploying a production-ready AI system that achieves 98-99% accuracy.

---

## 📖 Table of Contents

- [What is LSTM_PFD?](#what-is-lstm_pfd)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Phase 0: Foundation & Data Preparation](#phase-0-foundation--data-preparation)
- [Phase 1: Classical Machine Learning](#phase-1-classical-machine-learning)
- [Phase 2: Deep Learning with 1D CNNs](#phase-2-deep-learning-with-1d-cnns)
- [Phase 3: Advanced CNN Architectures](#phase-3-advanced-cnn-architectures)
- [Phase 4: Transformer Models](#phase-4-transformer-models)
- [Phase 5: Time-Frequency Analysis](#phase-5-time-frequency-analysis)
- [Phase 6: Physics-Informed Neural Networks](#phase-6-physics-informed-neural-networks)
- [Phase 7: Explainable AI](#phase-7-explainable-ai)
- [Phase 8: Ensemble Methods](#phase-8-ensemble-methods)
- [Phase 9: Production Deployment](#phase-9-production-deployment)
- [Phase 10: Testing & Quality Assurance](#phase-10-testing--quality-assurance)
- [Phase 11: Enterprise Dashboard](#phase-11-enterprise-dashboard)
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)

---

## What is LSTM_PFD?

**LSTM_PFD** (Long Short-Term Memory - Predictive Fault Diagnosis) is a complete, production-ready system for diagnosing bearing faults in rotating machinery using advanced machine learning and deep learning techniques.

### What Problem Does It Solve?

Bearing failures cause **80% of unplanned downtime** in industrial machinery (motors, pumps, turbines). This system:

- **Detects faults early** before catastrophic failure occurs
- **Classifies 11 different fault types** with 98-99% accuracy
- **Provides explainable predictions** so engineers understand why a fault was detected
- **Deploys in production** with <50ms inference time

### Who Is This For?

- **Researchers** exploring state-of-the-art fault diagnosis techniques
- **Engineers** implementing predictive maintenance systems
- **Data Scientists** learning time-series classification and deep learning
- **Companies** deploying AI-driven condition monitoring

### What Will You Build?

By the end of this guide, you'll have:

- ✅ **Trained 20+ different AI models** (classical ML, CNNs, transformers, ensembles)
- ✅ **Achieved 98-99% accuracy** on fault classification
- ✅ **Deployed a REST API** for real-time predictions
- ✅ **Built an enterprise dashboard** for managing experiments
- ✅ **Implemented explainable AI** to interpret predictions

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux, macOS, or Windows 10+ |
| **Python** | 3.8 or higher |
| **RAM** | 16GB (32GB recommended) |
| **Disk Space** | 50GB free |
| **CPU** | 4 cores (8 cores recommended) |
| **GPU** | Optional but recommended (NVIDIA with CUDA 11.8+) |

### Recommended Setup

- **GPU**: NVIDIA RTX 3080 or better (for faster training)
- **RAM**: 32GB (for processing large datasets)
- **SSD**: 100GB+ for fast data access

### Software Dependencies

You'll need:
- Git
- Python 3.8+ with pip
- (Optional) NVIDIA CUDA Toolkit 11.8+ if using GPU
- (Optional) Docker for deployment

---

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/LSTM_PFD.git
cd LSTM_PFD

# Verify you're in the right directory
ls
# You should see: README.md, requirements.txt, data/, models/, etc.
```

### Step 2: Create Virtual Environment

**Why?** Isolates project dependencies from your system Python.

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate     # On Windows

# Verify activation (you should see (venv) in your prompt)
which python  # Should point to venv/bin/python
```

### Step 3: Install PyTorch

**Important:** Install PyTorch first with the correct CUDA version for your system.

```bash
# For GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower training, not recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch version: 2.1.0+cu118
CUDA available: True  # (or False if using CPU)
```

### Step 4: Install Project Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install testing dependencies (optional, for Phase 10)
pip install -r requirements-test.txt

# Install deployment dependencies (optional, for Phase 9)
pip install -r requirements-deployment.txt
```

This will install ~50 packages including:
- NumPy, SciPy, Pandas (data processing)
- Scikit-learn (classical ML)
- PyTorch (deep learning)
- FastAPI, Uvicorn (REST API)
- Plotly Dash (dashboard)
- SHAP, LIME (explainability)
- And more...

### Step 5: Verify Installation

```bash
# Run verification script
python -c "
from models import list_available_models
import torch
print('✅ Installation successful!')
print(f'Available models: {len(list_available_models())}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

**Expected output:**
```
✅ Installation successful!
Available models: 23
PyTorch: 2.1.0+cu118
CUDA: True
```

---

## Project Overview

### The 11 Phases

This project is structured into 11 phases, each building on the previous:

| Phase | Name | Purpose | Accuracy | Duration |
|-------|------|---------|----------|----------|
| **0** | Foundation | Data pipeline & infrastructure | N/A | 1-2 hours |
| **1** | Classical ML | Baseline models (SVM, Random Forest) | 95-96% | 30 min |
| **2** | 1D CNNs | Deep learning baseline | 93-95% | 2-3 hours |
| **3** | Advanced CNNs | ResNet, EfficientNet | 96-97% | 3-4 hours |
| **4** | Transformers | Self-attention models | 96-97% | 4-6 hours |
| **5** | Time-Frequency | Spectrograms + 2D CNNs | 96-98% | 3-4 hours |
| **6** | PINN | Physics-informed networks | 97-98% | 4-5 hours |
| **7** | XAI | Explainable AI | N/A | 1-2 hours |
| **8** | Ensemble | Combine multiple models | 98-99% | 3-4 hours |
| **9** | Deployment | Production optimization | N/A | 2-3 hours |
| **10** | QA & Testing | Quality assurance | N/A | 1 hour |
| **11** | Dashboard | Web-based interface | N/A | 2-3 hours |

**Total Time**: ~2-3 days (depending on hardware and training epochs)

### How to Use This Guide

**For Complete Beginners:**
- Follow phases sequentially (0 → 11)
- Read all explanations before running commands
- Don't skip Phase 0 (foundation)

**For Experienced Users:**
- Jump to specific phases as needed
- Each phase can run independently (with Phase 0 data)
- Refer to detailed USAGE_GUIDES/ for advanced options

---

## Phase 0: Foundation & Data Preparation

### What is Phase 0?

Phase 0 sets up the **data pipeline and infrastructure**. You'll either:
- **Option A**: Import existing bearing data (if you have 1,430 MAT files)
- **Option B**: Generate synthetic data (for testing/learning)

**⏱️ Time Required**: 1-2 hours (mostly waiting for data generation)

### Create Directory Structure

```bash
# Create all required directories
mkdir -p data/raw/bearing_data/{normal,ball_fault,inner_race,outer_race,combined,imbalance,misalignment,oil_whirl,cavitation,looseness,oil_deficiency}
mkdir -p data/processed
mkdir -p data/spectrograms/{stft,cwt,wvd}
mkdir -p checkpoints/{phase1,phase2,phase3,phase4,phase5,phase6,phase7,phase8,phase9}
mkdir -p logs results visualizations
mkdir -p models
```

### Option A: Import Existing MAT Files

**If you have bearing vibration data in MATLAB .mat files:**

1. **Organize your data:**
   - Place your MAT files in `data/raw/bearing_data/` subdirectories
   - ~130 files per fault type
   - Each file should contain a vibration signal (102,400 samples @ 20.48 kHz)

2. **Import to HDF5 cache:**

```bash
python scripts/import_mat_dataset.py \
    --mat_dir data/raw/bearing_data/ \
    --output data/processed/signals_cache.h5 \
    --split-ratios 0.7 0.15 0.15
```

**Expected output:**
```
Found 1430 MAT files
Loading MAT files... 100%|████████████| 1430/1430
✓ Loaded 1430 signals
✓ Train: 1001 samples | Val: 215 samples | Test: 214 samples
✓ Cache saved to data/processed/signals_cache.h5
```

### Option B: Generate Synthetic Data (Recommended for Learning)

**If you don't have data, generate synthetic bearing signals:**

```python
# Create generate_data.py
cat > generate_data.py << 'EOF'
import numpy as np
from data.signal_generator import SignalGenerator
from config.data_config import DataConfig

# Configure data generation
config = DataConfig(
    num_signals_per_fault=130,  # 130 × 11 classes = 1,430 total
    signal_length=102400,        # 5 seconds @ 20.48 kHz
    fs=20480,                    # Sampling frequency
    rng_seed=42                  # For reproducibility
)

# Generate dataset
generator = SignalGenerator(config)
dataset = generator.generate_dataset()

# Save to HDF5
import h5py
with h5py.File('data/processed/signals_cache.h5', 'w') as f:
    f.create_dataset('signals', data=dataset['signals'])
    f.create_dataset('labels', data=dataset['labels'])
    f.create_dataset('metadata', data=str(dataset['metadata']))

print(f"✓ Generated {len(dataset['signals'])} signals")
print(f"✓ Saved to data/processed/signals_cache.h5")
EOF

# Run it
python generate_data.py
```

**Expected output:**
```
✓ Generated 1430 signals
✓ Saved to data/processed/signals_cache.h5
```

### Verify Data

```bash
# Check the cache file
python -c "
import h5py
with h5py.File('data/processed/signals_cache.h5', 'r') as f:
    print(f'Signals: {f[\"signals\"].shape}')
    print(f'Labels: {f[\"labels\"].shape}')
    print(f'Unique classes: {len(set(f[\"labels\"][:]))}')
"
```

**Expected output:**
```
Signals: (1430, 102400)
Labels: (1430,)
Unique classes: 11
```

**✅ Phase 0 Complete!** You now have a dataset ready for training.

---

## Phase 1: Classical Machine Learning

### What is Phase 1?

Phase 1 establishes a **baseline** using traditional machine learning:
- Extract 36 hand-crafted features from vibration signals
- Use MRMR to select best 15 features
- Train Random Forest, SVM, Gradient Boosting classifiers

**🎯 Target Accuracy**: 95-96%
**⏱️ Time Required**: 30 minutes

### Quick Start (Command Line)

```bash
# Run the complete classical ML pipeline
python scripts/train_classical_ml.py \
    --data data/processed/signals_cache.h5 \
    --output results/phase1/ \
    --optimize-hyperparams \
    --n-trials 50
```

**What this does:**
1. Loads data from HDF5 cache
2. Extracts 36 features (time domain, frequency domain, wavelets, etc.)
3. Selects best 15 features using MRMR
4. Trains 4 models: SVM, Random Forest, Neural Network, Gradient Boosting
5. Optimizes hyperparameters with Bayesian optimization (50 trials)
6. Saves best model and results

**Expected output:**
```
Extracting features... ━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:15
Selecting features (MRMR)... ✓ 15 features selected
Training Random Forest... ✓ Val Acc: 95.3%
Training SVM... ✓ Val Acc: 94.8%
Training Neural Network... ✓ Val Acc: 93.2%
Training Gradient Boosting... ✓ Val Acc: 94.5%

🏆 Best Model: Random Forest
   Test Accuracy: 95.3%
   F1 Score: 0.951

✓ Model saved: results/phase1/best_model.pkl
```

### Python API (Advanced)

```python
from pipelines.classical_ml_pipeline import ClassicalMLPipeline

# Initialize pipeline
pipeline = ClassicalMLPipeline(random_state=42)

# Load data
import h5py
with h5py.File('data/processed/signals_cache.h5', 'r') as f:
    signals = f['signals'][:]
    labels = f['labels'][:]

# Run pipeline
results = pipeline.run(
    signals=signals,
    labels=labels,
    fs=20480,
    optimize_hyperparams=True,
    n_trials=50
)

print(f"Best Model: {results['best_model']}")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

**✅ Phase 1 Complete!** You have a 95% accurate baseline model.

**📚 More Details**: See `USAGE_GUIDES/PHASE_1_USAGE_GUIDE.md`

---

## Phase 2: Deep Learning with 1D CNNs

### What is Phase 2?

Phase 2 introduces **deep learning** for end-to-end learning:
- No manual feature engineering required
- CNN learns optimal features automatically
- Multi-scale kernels capture different patterns

**🎯 Target Accuracy**: 93-95%
**⏱️ Time Required**: 2-3 hours (GPU) or 10-15 hours (CPU)

### Quick Start

```bash
# Train baseline 1D CNN
python scripts/train_cnn.py \
    --model cnn1d \
    --data-path data/processed/signals_cache.h5 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --checkpoint-dir checkpoints/phase2
```

**Training progress:**
```
Epoch 1/100: train_loss=2.234, train_acc=0.342, val_acc=0.445 ━━━━━ 2min
Epoch 10/100: train_loss=0.523, train_acc=0.893, val_acc=0.921 ━━━━━ 2min
...
Epoch 100/100: train_loss=0.015, train_acc=0.997, val_acc=0.947 ━━━━━ 2min

✓ Best validation accuracy: 94.7% (epoch 94)
✓ Test accuracy: 94.3%
✓ Model saved: checkpoints/phase2/best_cnn1d.pth
```

### Train Attention-Based CNN (Better Performance)

```bash
python scripts/train_cnn.py \
    --model attention \
    --data-path data/processed/signals_cache.h5 \
    --epochs 150 \
    --batch-size 32 \
    --mixed-precision \
    --early-stopping \
    --checkpoint-dir checkpoints/phase2/attention
```

**✅ Phase 2 Complete!** You have a deep learning model with 94-95% accuracy.

**📚 More Details**: See `USAGE_GUIDES/PHASE_2_USAGE_GUIDE.md`

---

## Phase 3: Advanced CNN Architectures

### What is Phase 3?

Phase 3 applies **state-of-the-art CNN architectures**:
- ResNet-18, ResNet-34, ResNet-50
- SE-ResNet (Squeeze-and-Excitation)
- EfficientNet (compound scaling)
- Wide ResNet

**🎯 Target Accuracy**: 96-97%
**⏱️ Time Required**: 3-4 hours

### Train ResNet-18

```bash
python scripts/train_cnn.py \
    --model resnet18 \
    --data-path data/processed/signals_cache.h5 \
    --epochs 150 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/phase3/resnet18
```

### Train ResNet-34 (Recommended)

```bash
python scripts/train_cnn.py \
    --model resnet34 \
    --data-path data/processed/signals_cache.h5 \
    --epochs 150 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/phase3/resnet34
```

**Expected result**: 96.5-96.8% test accuracy

### Train EfficientNet-B3 (Best Balance)

```bash
python scripts/train_cnn.py \
    --model efficientnet_b3 \
    --data-path data/processed/signals_cache.h5 \
    --epochs 150 \
    --checkpoint-dir checkpoints/phase3/efficientnet
```

**✅ Phase 3 Complete!** You have multiple models with 96-97% accuracy.

**📚 More Details**: See `USAGE_GUIDES/PHASE_3_USAGE_GUIDE.md`

---

## Phase 4: Transformer Models

### What is Phase 4?

Phase 4 applies **self-attention mechanisms**:
- Transformer encoder architecture
- Captures long-range temporal dependencies
- Patch-based processing of vibration signals

**🎯 Target Accuracy**: 96-97%
**⏱️ Time Required**: 4-6 hours

### Train Transformer

```bash
python scripts/train_transformer.py \
    --data-path data/processed/signals_cache.h5 \
    --d-model 256 \
    --nhead 8 \
    --num-layers 6 \
    --epochs 100 \
    --batch-size 32 \
    --warmup-epochs 10 \
    --checkpoint-dir checkpoints/phase4
```

**Important**: Transformers require learning rate warmup for stable training!

**Expected result**: 96.5% test accuracy

**✅ Phase 4 Complete!** You have a transformer model with attention visualization.

**📚 More Details**: See `USAGE_GUIDES/PHASE_4_USAGE_GUIDE.md`

---

## Phase 5: Time-Frequency Analysis

### What is Phase 5?

Phase 5 converts signals to **spectrograms**:
- STFT (Short-Time Fourier Transform)
- CWT (Continuous Wavelet Transform)
- WVD (Wigner-Ville Distribution)
- Train 2D CNNs on time-frequency images

**🎯 Target Accuracy**: 96-98%
**⏱️ Time Required**: 3-4 hours

### Step 1: Precompute Spectrograms

```bash
# Generate STFT spectrograms (recommended)
python scripts/precompute_spectrograms.py \
    --signals_cache data/processed/signals_cache.h5 \
    --output_dir data/spectrograms/stft/ \
    --tfr_type stft \
    --nperseg 256 \
    --noverlap 128
```

**Expected output:**
```
Generating STFT spectrograms... ━━━━━━━━━━━━━ 100% 0:02:15
✓ Saved 1430 spectrograms
✓ Shape: (129, 400) per spectrogram
```

### Step 2: Train 2D CNN on Spectrograms

```bash
python scripts/train_spectrogram_cnn.py \
    --model resnet2d \
    --spectrogram-dir data/spectrograms/stft/ \
    --epochs 100 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/phase5
```

**Expected result**: 96.8-97.4% test accuracy

**✅ Phase 5 Complete!** You have models that analyze time-frequency patterns.

**📚 More Details**: See `USAGE_GUIDES/PHASE_5_USAGE_GUIDE.md`

---

## Phase 6: Physics-Informed Neural Networks

### What is Phase 6?

Phase 6 adds **physics knowledge** to neural networks:
- Energy conservation constraints
- Momentum conservation constraints
- Bearing dynamics equations

**🎯 Target Accuracy**: 97-98%
**⏱️ Time Required**: 4-5 hours

### Train PINN

```bash
python scripts/train_pinn.py \
    --base-model resnet34 \
    --data-path data/processed/signals_cache.h5 \
    --physics-losses energy momentum bearing \
    --epochs 150 \
    --checkpoint-dir checkpoints/phase6
```

**What makes PINN different:**
- Better generalization to unseen conditions
- More physically plausible predictions
- Improved performance on complex/combined faults

**Expected result**: 97.2-97.8% test accuracy

**✅ Phase 6 Complete!** You have physics-informed models with better generalization.

**📚 More Details**: See `USAGE_GUIDES/PHASE_6_USAGE_GUIDE.md`

---

## Phase 7: Explainable AI

### What is Phase 7?

Phase 7 makes predictions **interpretable**:
- SHAP: Game-theory based feature attribution
- LIME: Local interpretable explanations
- Integrated Gradients: Neural network attribution
- Grad-CAM: CNN activation visualization

**⏱️ Time Required**: 1-2 hours

### Generate SHAP Explanations

```bash
python scripts/explain_prediction.py \
    --model checkpoints/phase6/best_pinn.pth \
    --signal-index 0 \
    --method shap \
    --output results/phase7/
```

### Launch Interactive XAI Dashboard

```bash
streamlit run explainability/xai_dashboard.py
```

Then open browser to `http://localhost:8501` to explore explanations interactively.

**✅ Phase 7 Complete!** You can now explain why your model makes predictions.

**📚 More Details**: See `USAGE_GUIDES/PHASE_7_USAGE_GUIDE.md`

---

## Phase 8: Ensemble Methods

### What is Phase 8?

Phase 8 combines **multiple models**:
- Voting ensemble (soft/hard voting)
- Stacked ensemble (meta-learner)
- Mixture of Experts (dynamic selection)
- Boosting ensemble

**🎯 Target Accuracy**: 98-99% ⭐
**⏱️ Time Required**: 3-4 hours

### Create Voting Ensemble

```python
from models.ensemble.voting_ensemble import VotingEnsemble
import torch

# Load your best models from previous phases
model_cnn = torch.load('checkpoints/phase2/best_cnn1d.pth')
model_resnet34 = torch.load('checkpoints/phase3/resnet34.pth')
model_transformer = torch.load('checkpoints/phase4/transformer.pth')
model_pinn = torch.load('checkpoints/phase6/best_pinn.pth')

# Create ensemble
ensemble = VotingEnsemble(
    models=[model_cnn, model_resnet34, model_transformer, model_pinn],
    voting='soft',
    weights=[0.2, 0.3, 0.25, 0.25]
)

# Evaluate
# ... (load test data)
# accuracy = evaluate(ensemble, test_loader)
```

### Train Stacked Ensemble (Best Performance)

```bash
python scripts/train_stacked_ensemble.py \
    --base-models checkpoints/phase3/resnet34.pth checkpoints/phase4/transformer.pth checkpoints/phase6/best_pinn.pth \
    --meta-learner xgboost \
    --output checkpoints/phase8/stacked_ensemble.pth
```

**Expected result**: 98.2-98.5% test accuracy 🎉

**✅ Phase 8 Complete!** You have achieved state-of-the-art performance (98-99%)!

**📚 More Details**: See `USAGE_GUIDES/PHASE_8_USAGE_GUIDE.md`

---

## Phase 9: Production Deployment

### What is Phase 9?

Phase 9 optimizes models for **production**:
- Model quantization (INT8, FP16) - 4x smaller, 3x faster
- ONNX export for cross-platform deployment
- REST API with FastAPI
- Docker containerization

**⏱️ Time Required**: 2-3 hours

### Step 1: Quantize Model

```bash
# Quantize to INT8 (4x smaller, 3x faster)
python scripts/quantize_model.py \
    --model checkpoints/phase8/stacked_ensemble.pth \
    --output checkpoints/phase9/model_int8.pth \
    --quantization-type dynamic
```

**Result:**
```
✓ Original size: 47.2 MB
✓ Quantized size: 11.8 MB
✓ Speedup: 2.96x
✓ Accuracy loss: <0.5%
```

### Step 2: Export to ONNX

```bash
python scripts/export_onnx.py \
    --model checkpoints/phase9/model_int8.pth \
    --output models/model.onnx \
    --validate \
    --optimize
```

### Step 3: Start REST API

```bash
# Set environment variables
export MODEL_PATH=checkpoints/phase9/model_int8.pth
export DEVICE=cuda  # or 'cpu'

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Test API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, ..., 0.3], "return_probabilities": true}'
```

### Step 5: Deploy with Docker

```bash
# Build Docker image
docker build -t lstm_pfd:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  lstm_pfd:latest

# Or use docker-compose
docker-compose up -d
```

**✅ Phase 9 Complete!** You have a production-ready API with <50ms latency.

**📚 More Details**: See `USAGE_GUIDES/Phase_9_DEPLOYMENT_GUIDE.md`

---

## Phase 10: Testing & Quality Assurance

### What is Phase 10?

Phase 10 ensures **quality and reliability**:
- 50+ unit tests
- 11 integration tests
- Performance benchmarks
- CI/CD pipeline with GitHub Actions
- 90%+ code coverage

**⏱️ Time Required**: 1 hour

### Run All Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html
```

### Run Benchmarks

```bash
python tests/benchmarks/benchmark_suite.py \
    --model checkpoints/phase9/model_int8.pth \
    --output benchmark_results.json
```

**Expected benchmark results:**
- Feature extraction: ~8.5ms per signal
- Model inference (INT8): ~15.3ms per signal
- API latency (P95): <68ms
- Throughput: 65 samples/second

**✅ Phase 10 Complete!** You have comprehensive test coverage and quality assurance.

**📚 More Details**: See `USAGE_GUIDES/Phase_10_QA_INTEGRATION_GUIDE.md`

---

## Phase 11: Enterprise Dashboard

### What is Phase 11?

Phase 11 provides a **web-based interface**:
- Interactive experiment management
- Real-time training monitoring
- Explainable AI visualizations
- Hyperparameter optimization campaigns
- User authentication and role-based access
- Production monitoring and alerting

**⏱️ Time Required**: 2-3 hours (mostly setup)

### Quick Start with Docker (Recommended)

```bash
cd dash_app

# Copy environment template
cp .env.example .env

# Start all services (dashboard + PostgreSQL + Redis)
docker-compose up
```

**Access dashboard at**: `http://localhost:8050`

**Default login:**
- Username: `admin`
- Password: `admin` (change this in production!)

### Local Development Setup

```bash
cd dash_app

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from database.connection import init_database; init_database()"

# Seed initial data
python -c "from database.seed_data import seed_initial_data; seed_initial_data()"

# Start Celery worker (in separate terminal)
celery -A tasks.celery_app worker --loglevel=info

# Start dashboard
python app.py
```

### Dashboard Features

**1. Home Page** (`/`)
- System overview with quick stats
- Recent experiments
- Health monitoring gauges

**2. Data Explorer** (`/data-explorer`)
- Browse and filter signals
- t-SNE visualization
- Statistical summaries

**3. New Experiment** (`/experiment/new`)
- Configuration wizard for 20+ models
- Hyperparameter tuning
- Launch training jobs

**4. Monitor Training** (`/experiment/<id>/monitor`)
- Real-time progress updates
- Live loss/accuracy curves
- Pause/resume/stop controls

**5. View Results** (`/experiment/<id>/results`)
- Confusion matrix
- Per-class metrics
- Export reports (PDF, CSV)

**6. XAI Dashboard** (`/xai`)
- SHAP explanations
- LIME explanations
- Integrated Gradients
- Grad-CAM visualizations

**7. HPO Campaigns** (`/hpo`)
- Bayesian optimization
- Grid/random search
- Parallel coordinates plots

**✅ Phase 11 Complete!** You have a full enterprise dashboard for ML operations.

**📚 More Details**: See `USAGE_GUIDES/PHASE_11_USAGE_GUIDE.md`

---

## Next Steps

Congratulations! You've completed all 11 phases and built a production-ready bearing fault diagnosis system. Here's what you can do next:

### 1. Experiment with Your Own Data

If you used synthetic data, try importing real bearing vibration data:

```bash
python scripts/import_mat_dataset.py \
    --mat_dir /path/to/your/bearing/data/ \
    --output data/processed/real_data.h5
```

Then retrain models with `--data-path data/processed/real_data.h5`

### 2. Deploy to Production

- Set up on a cloud server (AWS, Azure, GCP)
- Configure HTTPS with Let's Encrypt
- Set up monitoring with Prometheus + Grafana
- Configure autoscaling for high traffic

### 3. Customize for Your Use Case

- Add new fault types to classification
- Modify signal preprocessing
- Integrate with existing SCADA systems
- Build custom dashboards

### 4. Research and Development

- Experiment with new architectures
- Try different ensemble strategies
- Explore domain adaptation techniques
- Publish your findings

### 5. Contribute to the Project

- Report bugs on GitHub
- Submit feature requests
- Contribute code improvements
- Help improve documentation

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Problem:** GPU runs out of memory during training

**Solution:**
```bash
# Reduce batch size
python scripts/train_cnn.py --batch-size 16  # instead of 32

# Or enable gradient accumulation
python scripts/train_cnn.py --batch-size 16 --gradient-accumulation 2

# Or use mixed precision
python scripts/train_cnn.py --mixed-precision
```

#### 2. Training Diverges (Loss → NaN)

**Problem:** Loss becomes NaN during training

**Solution:**
```bash
# Reduce learning rate
python scripts/train_cnn.py --lr 0.0001  # instead of 0.001

# Enable gradient clipping
python scripts/train_cnn.py --grad-clip 1.0
```

#### 3. Low Accuracy (<90%)

**Possible causes:**
- Not enough data (need 1000+ samples per class)
- Data not normalized
- Learning rate too high/low
- Model overfitting or underfitting

**Solution:**
- Check data quality with visualization tools
- Verify signal preprocessing
- Tune hyperparameters with Phase 11 HPO
- Try different model architectures

#### 4. Slow Training

**Problem:** Training takes too long

**Solution:**
```bash
# Use GPU instead of CPU
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Enable mixed precision training
python scripts/train_cnn.py --mixed-precision

# Use fewer epochs for initial testing
python scripts/train_cnn.py --epochs 30  # instead of 150

# Use smaller model first
python scripts/train_cnn.py --model efficientnet_b0  # instead of b3
```

#### 5. Docker Issues

**Problem:** Docker containers fail to start

**Solution:**
```bash
# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up

# Rebuild image
docker-compose build --no-cache
```

### Getting Help

If you're stuck:

1. **Check the detailed usage guides** in `USAGE_GUIDES/` for your phase
2. **Search GitHub Issues**: https://github.com/yourusername/LSTM_PFD/issues
3. **Review the main README**: Most questions are answered there
4. **Open a new GitHub Issue** with:
   - What you're trying to do
   - What error you're getting
   - Your system specs (OS, Python version, GPU)
   - Relevant code/commands

---

## Summary

You've learned how to:

✅ **Install and set up** the LSTM_PFD system
✅ **Prepare data** (generate synthetic or import real data)
✅ **Train 20+ AI models** across 8 different approaches
✅ **Achieve 98-99% accuracy** with ensemble methods
✅ **Explain predictions** with SHAP, LIME, and other XAI tools
✅ **Deploy to production** with Docker and REST API
✅ **Monitor and manage** experiments with the enterprise dashboard

**Total Achievement:**
- 🎯 **State-of-the-art accuracy** (98-99%)
- ⚡ **Fast inference** (<50ms)
- 🔍 **Explainable predictions**
- 🚀 **Production-ready deployment**
- 🖥️ **Enterprise dashboard**

### Key Takeaways

1. **Start simple** (Phase 1 classical ML) before trying deep learning
2. **Each phase builds on the previous** - don't skip Phase 0
3. **Ensembles are powerful** - combining models (Phase 8) gives best accuracy
4. **Explainability matters** - XAI (Phase 7) builds trust in predictions
5. **Production requires optimization** - quantization (Phase 9) makes models deployable

---

## Additional Resources

### Documentation
- **Phase-specific guides**: `USAGE_GUIDES/PHASE_X_USAGE_GUIDE.md`
- **Architecture details**: `phase-plan/` directory
- **API docs**: `api/README.md`
- **Dashboard docs**: `dash_app/README.md`

### Example Scripts
- Training: `scripts/train_*.py`
- Evaluation: `scripts/evaluate_*.py`
- Deployment: `scripts/quantize_model.py`, `scripts/export_onnx.py`

### Configuration
- Data config: `config/data_config.py`
- Model config: `config/model_config.py`
- Training config: `config/training_config.py`

---

**🎉 Congratulations on completing the LSTM_PFD Quick Start Guide!**

You now have a production-ready bearing fault diagnosis system. Happy diagnosing! 🔧⚙️

---

**Last Updated**: November 2025
**Version**: 1.0.0
**Project**: LSTM_PFD - Advanced Bearing Fault Diagnosis System
