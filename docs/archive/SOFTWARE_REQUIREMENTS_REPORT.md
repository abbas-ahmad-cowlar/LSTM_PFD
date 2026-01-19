> [!WARNING]
> **Archived Document**
> This document is historical and may be outdated.
> For current information, see the main documentation.
>
> *Archived on: 2026-01-20*
> *Reason: Superseded by consolidated documentation*
# Software Requirements Report for LSTM_PFD

**Generated:** November 2025  
**System:** Windows 10 (Build 26200)  
**Python:** 3.14.0

---

## Executive Summary

**Status:** 12/34 requirements installed (35%)  
**Critical Missing:** PyTorch (required for all ML operations)  
**Optional Missing:** PostgreSQL CLI, Redis CLI (can use Docker instead)

---

## ✅ INSTALLED SOFTWARE

### System Requirements
- ✅ **Python 3.14.0** (Required: 3.8+) - **EXCEEDS REQUIREMENT**
- ✅ **Git 2.52.0** - Version control
- ✅ **Docker 28.3.3** - Containerization
- ✅ **Docker Compose v2.39.2** - Multi-container orchestration

### Core Python Packages (Installed)
- ✅ **numpy 2.3.4** - Scientific computing
- ✅ **scipy 1.16.3** - Signal processing
- ✅ **pandas 2.3.3** - Data manipulation
- ✅ **scikit-learn 1.7.2** - Machine learning
- ✅ **h5py 3.15.1** - HDF5 file support
- ✅ **matplotlib 3.10.7** - Plotting
- ✅ **plotly 6.3.1** - Interactive visualizations
- ✅ **seaborn 0.13.2** - Statistical plots
- ✅ **pytest 9.0.0** - Testing framework

---

## ❌ MISSING SOFTWARE

### Critical Requirements (Must Install)

#### 1. PyTorch ⚠️ **CRITICAL**
- **Status:** NOT INSTALLED
- **Required:** 2.0.0+
- **Why Critical:** All deep learning models require PyTorch
- **Install Command:**
  ```bash
  # For GPU (if you have NVIDIA GPU):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  
  # For CPU only (works but slower):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

#### 2. Core Python Packages
- **tqdm** - Progress bars
- **xgboost** - Gradient boosting (Phase 1, 8)
- **optuna** - Hyperparameter optimization
- **shap** - Explainable AI (Phase 7)
- **lime** - Explainable AI (Phase 7)
- **captum** - Model interpretability (Phase 7)

#### 3. Dashboard Packages (Required for Phase 11)
- **dash** - Plotly Dash framework
- **dash-bootstrap-components** - UI components
- **flask** - Web framework
- **sqlalchemy** - Database ORM
- **psycopg2-binary** - PostgreSQL driver
- **redis** - Caching
- **celery** - Background task queue

#### 4. API/Deployment Packages (Required for Phase 9)
- **fastapi** - REST API framework
- **uvicorn** - ASGI server
- **onnx** - Model export format
- **onnxruntime** - ONNX inference

#### 5. Testing Packages (Optional but Recommended)
- **pytest-cov** - Coverage reporting

### Optional Requirements

#### PostgreSQL CLI
- **Status:** NOT INSTALLED (but not required)
- **Why Optional:** Docker Compose can provide PostgreSQL container
- **Alternative:** Use `docker-compose up` in `packages/dashboard/` directory

#### Redis CLI
- **Status:** NOT INSTALLED (but not required)
- **Why Optional:** Docker Compose can provide Redis container
- **Alternative:** Use `docker-compose up` in `packages/dashboard/` directory

#### NVIDIA GPU/CUDA
- **Status:** NOT DETECTED
- **Why Optional:** CPU training works but is 10-20x slower
- **Impact:** Training will take longer but is fully functional
- **Note:** If you have an NVIDIA GPU, install CUDA Toolkit separately

---

## 📋 Installation Checklist

### Step 1: Install PyTorch (CRITICAL - Do This First!)

```bash
# Check if you have NVIDIA GPU first
# If yes, use cu118. If no, use cpu.

# For GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install Core Requirements

```bash
# Install main project requirements
pip install -r requirements.txt
```

This will install:
- tqdm, xgboost, optuna
- shap, lime, captum
- pywavelets, streamlit
- And other core packages

### Step 3: Install Dashboard Requirements (If Using Dashboard)

```bash
# Install dashboard-specific requirements
pip install -r packages/dashboard/requirements.txt
```

This will install:
- dash, dash-bootstrap-components
- flask, sqlalchemy, psycopg2-binary
- redis, celery
- And other dashboard packages

### Step 4: Install Deployment Requirements (If Using API)

```bash
# Install API/deployment requirements
pip install -r requirements-deployment.txt
```

This will install:
- fastapi, uvicorn
- onnx, onnxruntime
- And other deployment packages

### Step 5: Install Testing Requirements (Optional)

```bash
# Install testing requirements
pip install -r requirements-test.txt
```

This will install:
- pytest-cov
- black, isort, flake8 (code quality)
- And other testing tools

---

## 🚀 Quick Installation (All-in-One)

If you want to install everything at once:

```bash
# 1. Install PyTorch first (choose GPU or CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install all requirements
pip install -r requirements.txt
pip install -r packages/dashboard/requirements.txt
pip install -r requirements-deployment.txt
pip install -r requirements-test.txt
```

**Estimated time:** 10-30 minutes depending on internet speed

---

## 🐳 Docker Alternative (Recommended for Dashboard)

If you're using the dashboard (Phase 11), you can use Docker instead of installing PostgreSQL and Redis locally:

```bash
cd dash_app
docker-compose up
```

This will automatically:
- Start PostgreSQL container
- Start Redis container
- Start Celery worker
- Start Dash application

**No need to install PostgreSQL or Redis CLI tools!**

---

## 📊 Requirements by Use Case

### For CLI Training (Phases 1-8)
**Required:**
- ✅ Python 3.8+ (you have 3.14.0)
- ❌ PyTorch (CRITICAL - must install)
- ❌ Core packages from `requirements.txt`

**Optional:**
- ❌ CUDA/GPU (faster training but not required)

### For Dashboard (Phase 11)
**Required:**
- ✅ Python 3.8+ (you have 3.14.0)
- ✅ Docker & Docker Compose (you have both)
- ❌ PyTorch (CRITICAL)
- ❌ Dashboard packages from `packages/dashboard/requirements.txt`

**Optional:**
- ❌ PostgreSQL CLI (Docker provides it)
- ❌ Redis CLI (Docker provides it)

### For API (Phase 9)
**Required:**
- ✅ Python 3.8+ (you have 3.14.0)
- ❌ PyTorch (CRITICAL)
- ❌ API packages from `requirements-deployment.txt`

---

## ⚠️ Important Notes

1. **Python Version:** You have Python 3.14.0, which is newer than required (3.8+). This should work fine, but if you encounter compatibility issues, consider using Python 3.10 or 3.11.

2. **GPU Support:** No NVIDIA GPU detected. Training will work on CPU but will be slower. For faster training, consider:
   - Using a cloud GPU (Google Colab, AWS, etc.)
   - Installing an NVIDIA GPU and CUDA Toolkit

3. **Docker:** You have Docker installed, which is great! You can use it to run PostgreSQL and Redis without installing them locally.

4. **Installation Order:** Always install PyTorch FIRST before other packages, as some packages depend on it.

---

## ✅ Verification After Installation

After installing requirements, verify everything works:

```bash
# 1. Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 2. Check core packages
python -c "import numpy, scipy, pandas, sklearn, h5py; print('Core packages OK')"

# 3. Check dashboard packages (if installed)
python -c "import dash, flask, sqlalchemy; print('Dashboard packages OK')"

# 4. Check API packages (if installed)
python -c "import fastapi, uvicorn; print('API packages OK')"
```

---

## 📞 Next Steps

1. **Install PyTorch** (most critical)
2. **Install requirements.txt** (core functionality)
3. **Choose your path:**
   - CLI training → You're ready after steps 1-2
   - Dashboard → Also install `packages/dashboard/requirements.txt` and use Docker
   - API → Also install `requirements-deployment.txt`

4. **Follow START_HERE.md** for your next steps

---

## 🔄 Re-run Check

To check your installation status again, run:

```bash
python check_requirements.py
```

This will show you what's still missing.

---

**Last Updated:** November 2025  
**Report Generated By:** Automated requirements checker

