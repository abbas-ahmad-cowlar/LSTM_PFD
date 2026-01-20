# LSTM_PFD: Advanced Bearing Fault Diagnosis System

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-98--99%25-brightgreen)

**A state-of-the-art, production-ready bearing fault diagnosis system** implementing cutting-edge ML/DL techniques for predictive maintenance.

> 🚀 **Complete beginner?** Start with **[QUICKSTART.md](QUICKSTART.md)** — Zero to 98-99% accuracy in 11 phases.

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Phases](#-project-phases)
- [Quick Start](#-quick-start)
- [Model Performance](#-model-performance)
- [Enterprise Dashboard](#-enterprise-dashboard)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---

## 🎯 Project Overview

**LSTM_PFD** is a comprehensive bearing fault diagnosis system progressing from classical ML to physics-informed deep learning.

### What Problem Does This Solve?

**Bearing failures** cause unplanned downtime in rotating machinery. This system:

- **Detects faults early** before catastrophic failure
- **Classifies 11 fault types** with 98-99% accuracy
- **Provides explainable predictions** via SHAP/LIME
- **Deploys in production** with <50ms latency

### Who Is This For?

| Audience            | Use Case                                      |
| ------------------- | --------------------------------------------- |
| **Researchers**     | Exploring advanced fault diagnosis techniques |
| **Engineers**       | Implementing predictive maintenance           |
| **Data Scientists** | Learning time-series classification           |
| **Companies**       | Deploying AI-driven condition monitoring      |

---

## ✨ Key Features

### 🤖 Multiple Model Architectures

| Category         | Models                       | Accuracy   |
| ---------------- | ---------------------------- | ---------- |
| Classical ML     | SVM, Random Forest, XGBoost  | 95-96%     |
| Deep Learning    | 1D CNN, ResNet, EfficientNet | 96-97%     |
| Transformers     | Self-attention, hybrid       | 96-97%     |
| Time-Frequency   | STFT/CWT/WVD + 2D CNNs       | 96-98%     |
| Physics-Informed | PINN with domain constraints | 97-98%     |
| Ensemble         | Voting, stacking, MoE        | **98-99%** |

### 🔍 Explainable AI (XAI)

- **SHAP** — Feature importance and attribution
- **LIME** — Local interpretable explanations
- **Integrated Gradients** — Neural network attribution
- **Attention Visualization** — Transformer interpretability

### 🚀 Production-Ready

- Model Quantization (INT8/FP16)
- ONNX Export
- REST API (FastAPI)
- Docker deployment
- 90%+ test coverage

### 📊 11 Fault Types

Normal, Ball Fault, Inner Race, Outer Race, Combined, Imbalance, Misalignment, Oil Whirl, Cavitation, Looseness, Oil Deficiency

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SHARED FOUNDATION (Phase 0)                  │
│  Data Layer (MAT/HDF5) │ Model Layer (Factory) │ Training Layer │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                        DATA-DRIVEN MODELS                        │
│  Phase 1: Classical ML → Phase 2: 1D CNNs → Phase 3: ResNet →   │
│  Phase 4: Transformers                                           │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ADVANCED TECHNIQUES                          │
│  Phase 5: Time-Freq → Phase 6: PINN → Phase 7: XAI →            │
│  Phase 8: Ensemble (98-99%)                                      │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION DEPLOYMENT                        │
│  Phase 9: Quantization/ONNX/API → Phase 10: QA/CI-CD →          │
│  Phase 11: Enterprise Dashboard                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Raw MAT Files → HDF5 Cache → Model Training → Quantization → REST API
                    ↓
              Feature Extraction / Spectrograms / Physics Constraints
```

---

## 📊 Project Phases

All 11 phases are **complete and production-ready**:

| Phase  | Name           | Accuracy   | Key Innovation                        |
| ------ | -------------- | ---------- | ------------------------------------- |
| **0**  | Foundation     | N/A        | Data pipeline, PyTorch infrastructure |
| **1**  | Classical ML   | 95-96%     | Feature engineering (MRMR, 36→15)     |
| **2**  | 1D CNN         | 93-95%     | Multi-scale kernels                   |
| **3**  | Advanced CNNs  | 96-97%     | ResNet, EfficientNet                  |
| **4**  | Transformer    | 96-97%     | Self-attention                        |
| **5**  | Time-Frequency | 96-98%     | STFT/CWT/WVD spectrograms             |
| **6**  | PINN           | 97-98%     | Physics-informed constraints          |
| **7**  | XAI            | N/A        | SHAP, LIME, Integrated Gradients      |
| **8**  | Ensemble       | **98-99%** | Voting, stacking, MoE                 |
| **9**  | Deployment     | N/A        | Quantization, ONNX, Docker            |
| **10** | QA             | N/A        | 90% coverage, CI/CD                   |
| **11** | Dashboard      | N/A        | Enterprise web UI                     |

**Total Development**: 264 days (~9 months) | **Status**: 🎉 Production Ready

---

## 🚀 Quick Start

### Choose Your Path

| Path                   | Guide                                                     | Description                   |
| ---------------------- | --------------------------------------------------------- | ----------------------------- |
| **GUI** (No Code)      | [GUI_QUICKSTART.md](packages/dashboard/GUI_QUICKSTART.md) | Train via web dashboard       |
| **CLI** (Full Control) | [QUICKSTART.md](QUICKSTART.md)                            | 11-phase implementation guide |
| **Dashboard**          | [USAGE_PHASE_11.md](docs/USAGE_PHASE_11.md)               | Complete dashboard features   |

### 30-Second Installation

```bash
# Clone and setup
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
cd LSTM_PFD
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'✓ PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

### Environment Configuration (Dashboard)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Generate secure secrets
python -c 'import secrets; print("SECRET_KEY=" + secrets.token_hex(32))' >> .env
python -c 'import secrets; print("JWT_SECRET_KEY=" + secrets.token_hex(32))' >> .env

# 3. Configure DATABASE_URL in .env
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

> ⚠️ **Security**: Never commit `.env`. Use different secrets for dev/staging/prod.

---

## 📈 Model Performance

### Accuracy Progression

| Phase | Model            | Test Accuracy |
| ----- | ---------------- | ------------- |
| 1     | Random Forest    | 95.33%        |
| 3     | ResNet-34        | 96.8%         |
| 5     | CWT + ResNet     | 97.4%         |
| 6     | PINN             | 97.8%         |
| 8     | Stacked Ensemble | **98.4%**     |

### Inference Performance

| Model            | Latency    | Model Size  |
| ---------------- | ---------- | ----------- |
| ResNet-34 (FP32) | 45.2ms     | 47.2 MB     |
| ResNet-34 (INT8) | **15.3ms** | **11.8 MB** |
| Ensemble (INT8)  | 48.5ms     | 59.0 MB     |

✅ **Target achieved**: All models <50ms latency

---

## 🖥️ Enterprise Dashboard

**Location**: [`packages/dashboard/`](packages/dashboard/)

### Why Use the Dashboard?

- 🚫 **No Coding Required** — Train models via web UI
- 📊 **Real-time Monitoring** — Live training progress
- 🔍 **Explainable AI** — Interactive SHAP, LIME, Grad-CAM
- 🎯 **HPO Campaigns** — Automated hyperparameter optimization
- 🔐 **Enterprise Security** — JWT auth, rate limiting

### Key Pages

| Page              | Description                       |
| ----------------- | --------------------------------- |
| `/data-explorer`  | Dataset browser with t-SNE        |
| `/experiments`    | Experiment management             |
| `/experiment/new` | Configuration wizard (20+ models) |
| `/xai`            | SHAP, LIME, IG visualizations     |
| `/hpo`            | Hyperparameter optimization       |
| `/health`         | System health monitoring          |

### Quick Start

```bash
cd packages/dashboard
cp ../.env.example .env
# Configure .env with DATABASE_URL, SECRET_KEY, JWT_SECRET_KEY
docker-compose up
# Access at http://localhost:8050
```

📖 **Full Documentation**: [Dashboard README](packages/dashboard/README.md) | [Usage Guide](docs/USAGE_PHASE_11.md)

---

## 📚 Documentation

### Phase Guides

All usage guides are in [`docs/user-guide/phases/`](docs/user-guide/phases/):

- Phase 1-8: Model training guides
- Phase 9: [Deployment Guide](docs/user-guide/phases/Phase_9_DEPLOYMENT_GUIDE.md)
- Phase 10: [QA Integration Guide](docs/user-guide/phases/Phase_10_QA_INTEGRATION_GUIDE.md)
- Phase 11: [Dashboard Usage](docs/user-guide/phases/PHASE_11_USAGE_GUIDE.md)

### Other Resources

| Resource        | Location                                                     |
| --------------- | ------------------------------------------------------------ |
| API Reference   | [docs/API_REFERENCE.md](docs/API_REFERENCE.md)               |
| Deployment      | [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)         |
| Troubleshooting | [docs/troubleshooting/](docs/troubleshooting/)               |
| Contributing    | [CONTRIBUTING.md](CONTRIBUTING.md)                           |
| HDF5 Migration  | [docs/HDF5_MIGRATION_GUIDE.md](docs/HDF5_MIGRATION_GUIDE.md) |

### MkDocs Site

```bash
pip install -r docs/requirements.txt
mkdocs serve
# Open http://localhost:8000
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -r requirements-test.txt
pytest -v
black . && isort . && flake8 .
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 📖 Citation

```bibtex
@software{lstm_pfd_2025,
  author = {Syed Abbas Ahmad},
  title = {LSTM_PFD: Advanced Bearing Fault Diagnosis System},
  year = {2025},
  url = {https://github.com/abbas-ahmad-cowlar/LSTM_PFD}
}
```

---

## 🌟 Key Highlights

- ✅ **98-99% Accuracy** — State-of-the-art performance
- ✅ **11 Fault Types** — Comprehensive coverage
- ✅ **<50ms Latency** — Production-ready
- ✅ **Explainable** — SHAP, LIME, attention
- ✅ **90%+ Test Coverage** — Well-tested
- ✅ **Full Documentation** — 10+ guides

---

## 📞 Contact

- [GitHub Issues](https://github.com/abbas-ahmad-cowlar/LSTM_PFD/issues)
- [GitHub Discussions](https://github.com/abbas-ahmad-cowlar/LSTM_PFD/discussions)

---

**Status**: 🎉 **PRODUCTION READY** — All 11 phases complete!

<div align="center">

### 🚀 Ready to Get Started?

[Quick Start](#-quick-start) | [Documentation](#-documentation) | [Dashboard](#-enterprise-dashboard)

**Built with ❤️ for Predictive Maintenance**

</div>
