# 🎯 START HERE: Your Entry Point to LSTM_PFD

**Last Updated:** November 2025  
**Purpose:** Guide you to exactly where to start based on your goal

---

## 📋 Quick Decision Tree

**What do you want to do?**

1. **🚀 Run the system end-to-end** → Start at [Section 1: Quick Start](#1-quick-start)
2. **🔍 Understand the codebase architecture** → Start at [Section 2: Codebase Overview](#2-codebase-overview)
3. **🛠️ Develop new features** → Start at [Section 3: Development Entry Points](#3-development-entry-points)
4. **📊 Use the dashboard (no coding)** → Start at [Section 4: Dashboard Usage](#4-dashboard-usage)
5. **🔬 Train models via CLI** → Start at [Section 5: CLI Workflow](#5-cli-workflow)

---

## 1. Quick Start

### If you want to run everything immediately:

**Step 1: Read the Quick Start Guide**
- 📄 **File:** `QUICKSTART.md` (in root directory)
- **What it covers:** Complete 11-phase workflow from installation to deployment
- **Time:** 30 minutes to read, 2-3 days to complete all phases

**Step 2: Set up environment**
```bash
# 1. Clone repository (if not already done)
cd LSTM_PFD

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt
```

**Step 3: Generate or import data**
- **Option A (Synthetic):** Use `data/signal_generator.py` - See `QUICKSTART.md` Phase 0
- **Option B (Real data):** Use `scripts/import_mat_dataset.py` if you have MAT files

**Step 4: Choose your path:**
- **GUI (No coding):** → `dash_app/app.py` - See [Section 4](#4-dashboard-usage)
- **CLI (Command line):** → `scripts/` directory - See [Section 5](#5-cli-workflow)

---

## 2. Codebase Overview

### Project Structure (High-Level)

```
LSTM_PFD/
├── 📁 data/              # Data generation, loading, preprocessing
│   ├── signal_generator.py    ⭐ START HERE for data generation
│   ├── dataset.py             ⭐ START HERE for data loading
│   └── matlab_importer.py     For importing real data
│
├── 📁 models/            # All model architectures
│   ├── model_factory.py       ⭐ START HERE to understand model creation
│   ├── classical/            Phase 1: Random Forest, SVM, etc.
│   ├── cnn/                   Phase 2-3: CNNs, ResNet, EfficientNet
│   ├── transformer/           Phase 4: Self-attention models
│   ├── pinn/                  Phase 6: Physics-informed networks
│   └── ensemble/              Phase 8: Voting, stacking, MoE
│
├── 📁 training/          # Training infrastructure
│   ├── trainer.py             ⭐ START HERE for training logic
│   ├── cnn_trainer.py         CNN-specific training
│   ├── pinn_trainer.py         PINN training
│   └── bayesian_optimizer.py  Hyperparameter optimization
│
├── 📁 dash_app/          # Enterprise Dashboard (Phase 11)
│   ├── app.py                 ⭐ MAIN ENTRY POINT for web UI
│   ├── layouts/                Page layouts
│   ├── callbacks/              Event handlers
│   ├── services/               Business logic
│   └── models/                 Database models
│
├── 📁 api/                # REST API (Phase 9)
│   ├── main.py                 ⭐ MAIN ENTRY POINT for API server
│   ├── config.py              API configuration
│   └── schemas.py             Request/response schemas
│
├── 📁 scripts/            # Command-line scripts
│   ├── train_*.py              Training scripts for each phase
│   ├── evaluate_*.py           Evaluation scripts
│   └── import_mat_dataset.py   Data import script
│
├── 📁 pipelines/          # End-to-end pipelines
│   └── classical_ml_pipeline.py ⭐ Example pipeline
│
└── 📁 config/             # Configuration files
    ├── data_config.py          Data generation config
    ├── model_config.py         Model architecture config
    └── training_config.py      Training hyperparameters
```

### Key Entry Points by Use Case

| Use Case | Entry Point | Documentation |
|----------|-------------|---------------|
| **Web Dashboard** | `dash_app/app.py` | `dash_app/README.md` |
| **REST API** | `api/main.py` | `docs/API_REFERENCE.md` |
| **Data Generation** | `data/signal_generator.py` | `QUICKSTART.md` Phase 0 |
| **Model Training (CLI)** | `scripts/train_*.py` | `USAGE_GUIDES/PHASE_X_USAGE_GUIDE.md` |
| **Model Creation** | `models/model_factory.py` | `models/__init__.py` |

---

## 3. Development Entry Points

### If you want to understand the codebase for development:

#### 3.1 Understanding Data Flow

**Start here:**
1. **`data/signal_generator.py`** (915 lines)
   - How synthetic bearing signals are generated
   - Physics-based models for 11 fault types
   - Signal parameters (fs=20480 Hz, T=5s, length=102400)

2. **`data/dataset.py`**
   - HDF5 dataset loading
   - Train/val/test splits
   - PyTorch Dataset implementation

3. **`data/matlab_importer.py`**
   - Importing real MAT files
   - Data validation and preprocessing

#### 3.2 Understanding Model Architecture

**Start here:**
1. **`models/model_factory.py`**
   - Central model creation system
   - Model registry pattern
   - How to create any model by name

2. **`models/base_model.py`**
   - Base class for all models
   - Common interface

3. **Specific model implementations:**
   - `models/cnn/cnn_1d.py` - Phase 2 baseline
   - `models/resnet/resnet_1d.py` - Phase 3 advanced CNNs
   - `models/transformer/signal_transformer.py` - Phase 4 transformers
   - `models/pinn/hybrid_pinn.py` - Phase 6 physics-informed

#### 3.3 Understanding Training

**Start here:**
1. **`training/trainer.py`**
   - Base training loop
   - Callbacks, metrics, checkpointing

2. **`training/cnn_trainer.py`**
   - CNN-specific training with data augmentation
   - Mixed precision, gradient clipping

3. **`training/pinn_trainer.py`**
   - Physics loss integration
   - Multi-objective optimization

#### 3.4 Understanding Dashboard Architecture

**Start here:**
1. **`dash_app/app.py`** (86 lines)
   - Main entry point
   - Flask + Dash setup
   - Route registration

2. **`dash_app/callbacks/__init__.py`**
   - How callbacks are registered
   - Page routing logic

3. **`dash_app/services/`**
   - Business logic layer
   - Database interactions
   - Cache management

4. **`dash_app/layouts/`**
   - UI components
   - Page layouts

#### 3.5 Understanding API

**Start here:**
1. **`api/main.py`** (378 lines)
   - FastAPI application setup
   - Endpoint definitions
   - Inference engine initialization

2. **`deployment/inference.py`**
   - Optimized inference engine
   - Model loading and caching
   - Batch processing

---

## 4. Dashboard Usage

### If you want to use the web interface (no coding):

**Entry Point:** `dash_app/app.py`

**Quick Start:**
```bash
cd dash_app

# 1. Set up environment (REQUIRED)
cp .env.example .env
# Edit .env and set:
# - DATABASE_URL (PostgreSQL connection)
# - SECRET_KEY (generate with: python -c "import secrets; print(secrets.token_hex(32))")
# - JWT_SECRET_KEY (generate similarly)

# 2. Start with Docker (Recommended)
docker-compose up

# OR start locally
python app.py
```

**Access:** `http://localhost:8050`

**Documentation:**
- **Quick Start:** `dash_app/GUI_QUICKSTART.md` (30-minute tutorial)
- **Complete Guide:** `docs/USAGE_PHASE_11.md` (850+ lines)
- **Dashboard README:** `dash_app/README.md`

**Key Pages:**
- `/` - Home dashboard
- `/data-explorer` - Browse datasets
- `/experiment/new` - Create training experiment
- `/experiment/<id>/monitor` - Monitor training
- `/xai` - Explainable AI dashboard

---

## 5. CLI Workflow

### If you want to train models via command line:

**Entry Points:** Scripts in `scripts/` directory

**Phase-by-Phase Workflow:**

**Phase 0: Data Generation**
```bash
# Generate synthetic data
python -c "
from data.signal_generator import SignalGenerator
from config.data_config import DataConfig
config = DataConfig(num_signals_per_fault=130)
generator = SignalGenerator(config)
dataset = generator.generate_dataset()
generator.save_dataset(dataset, format='hdf5', output_dir='data/processed')
"
```

**Phase 1: Classical ML**
```bash
python scripts/train_classical_ml.py \
    --data data/processed/dataset.h5 \
    --output results/phase1/
```

**Phase 2: 1D CNN**
```bash
python scripts/train_cnn.py \
    --model cnn1d \
    --data-path data/processed/dataset.h5 \
    --epochs 100 \
    --checkpoint-dir checkpoints/phase2
```

**Phase 3: Advanced CNNs**
```bash
python scripts/train_cnn.py \
    --model resnet34 \
    --data-path data/processed/dataset.h5 \
    --epochs 150 \
    --checkpoint-dir checkpoints/phase3/resnet34
```

**Complete workflow:** See `QUICKSTART.md` for all phases

**Documentation per phase:**
- Phase 1: `USAGE_GUIDES/PHASE_1_USAGE_GUIDE.md`
- Phase 2: `USAGE_GUIDES/PHASE_2_USAGE_GUIDE.md`
- ... (and so on for all 11 phases)

---

## 6. Recommended Learning Path

### For Complete Beginners:

1. **Read:** `QUICKSTART.md` (comprehensive guide)
   - Explains every concept from scratch
   - Step-by-step instructions
   - What each phase does and why
   - Note: For historical reference, see `docs/archive/COMPLETE_BEGINNER_GUIDE.md`

2. **Follow:** `QUICKSTART.md`
   - Execute commands as you read
   - Build understanding through practice

3. **Explore:** Start with Phase 0 (data generation)
   - Understand the data format
   - Generate your first dataset
   - Verify it works

4. **Progress:** Move through phases sequentially
   - Each phase builds on the previous
   - Don't skip Phase 0!

### For Experienced ML Engineers:

1. **Read:** `README.md` (1,113 lines)
   - High-level architecture
   - Performance metrics
   - Quick reference

2. **Explore:** Key files in order:
   - `data/signal_generator.py` - Understand data
   - `models/model_factory.py` - Understand models
   - `training/trainer.py` - Understand training
   - `dash_app/app.py` - Understand dashboard

3. **Run:** Start with Phase 8 (Ensemble)
   - Best accuracy (98-99%)
   - Combines all previous phases
   - Production-ready

### For Software Developers:

1. **Read:** `dash_app/README.md`
   - Dashboard architecture
   - Development guidelines
   - Adding new features

2. **Explore:** Dashboard code structure:
   - `dash_app/app.py` - Entry point
   - `dash_app/callbacks/` - Event handlers
   - `dash_app/services/` - Business logic
   - `dash_app/models/` - Database models

3. **Understand:** Three-layer architecture
   - Presentation (layouts/callbacks)
   - Service (business logic)
   - Data (database/files)

---

## 7. Key Files to Read First

### Must-Read Files (in order):

1. **`README.md`** ⭐⭐⭐
   - Project overview
   - All 11 phases explained
   - Quick start instructions

2. **`QUICKSTART.md`** ⭐⭐⭐
   - Step-by-step guide
   - All commands you need
   - Expected outputs

3. **`dash_app/README.md`** ⭐⭐ (if using dashboard)
   - Dashboard features
   - Architecture
   - Development guide

4. **`docs/FINAL_REPORT.md`** ⭐⭐
   - Project summary
   - Performance results
   - Technical details

### Important Code Files:

1. **`data/signal_generator.py`** - How data is created
2. **`models/model_factory.py`** - How models are created
3. **`training/trainer.py`** - How training works
4. **`dash_app/app.py`** - Dashboard entry point
5. **`api/main.py`** - API entry point

---

## 8. Common Starting Scenarios

### Scenario A: "I want to train a model right now"

**Path:**
1. Read `QUICKSTART.md` Phase 0 (data generation)
2. Generate dataset: `python generate_data.py`
3. Train Phase 1: `python scripts/train_classical_ml.py --data data/processed/dataset.h5`
4. Check results in `results/phase1/`

**Time:** 1-2 hours

### Scenario B: "I want to understand the codebase"

**Path:**
1. Read `README.md` (30 min)
2. Read `docs/FINAL_REPORT.md` (20 min)
3. Explore `data/signal_generator.py` (30 min)
4. Explore `models/model_factory.py` (20 min)
5. Explore `training/trainer.py` (30 min)

**Time:** 2-3 hours

### Scenario C: "I want to use the dashboard"

**Path:**
1. Read `dash_app/GUI_QUICKSTART.md` (30 min)
2. Set up environment (see Section 4)
3. Start dashboard: `cd dash_app && python app.py`
4. Follow GUI tutorial in dashboard

**Time:** 1 hour setup + ongoing use

### Scenario D: "I want to add a new feature"

**Path:**
1. Read `dash_app/README.md` Development section
2. Understand architecture (Section 3.4)
3. Find similar feature in codebase
4. Follow pattern to add new feature

**Time:** Depends on feature complexity

---

## 9. Documentation Map

### By Goal:

| Goal | Primary Doc | Secondary Docs |
|------|-------------|----------------|
| **Learn everything** | `QUICKSTART.md` | `README.md` |
| **Quick start** | `QUICKSTART.md` | `README.md` |
| **Use dashboard** | `dash_app/GUI_QUICKSTART.md` | `dash_app/README.md`, `docs/USAGE_PHASE_11.md` |
| **Understand architecture** | `README.md` | `docs/FINAL_REPORT.md`, `dash_app/README.md` |
| **Develop features** | `dash_app/README.md` | `CONTRIBUTING.md`, `docs/IMPLEMENTATION_PLAN.md` |
| **Deploy production** | `docs/DEPLOYMENT_GUIDE.md` | `USAGE_GUIDES/Phase_9_DEPLOYMENT_GUIDE.md` |
| **Phase-specific** | `USAGE_GUIDES/PHASE_X_USAGE_GUIDE.md` | `phase-plan/Phase_X.md` |

### By Phase:

- **Phase 0:** `QUICKSTART.md` Phase 0 section
- **Phase 1:** `USAGE_GUIDES/PHASE_1_USAGE_GUIDE.md`
- **Phase 2:** `USAGE_GUIDES/PHASE_2_USAGE_GUIDE.md`
- **Phase 3:** `USAGE_GUIDES/PHASE_3_USAGE_GUIDE.md`
- **Phase 4:** `USAGE_GUIDES/PHASE_4_USAGE_GUIDE.md`
- **Phase 5:** `USAGE_GUIDES/PHASE_5_USAGE_GUIDE.md`
- **Phase 6:** `USAGE_GUIDES/PHASE_6_USAGE_GUIDE.md`
- **Phase 7:** `USAGE_GUIDES/PHASE_7_USAGE_GUIDE.md`
- **Phase 8:** `USAGE_GUIDES/PHASE_8_USAGE_GUIDE.md`
- **Phase 9:** `USAGE_GUIDES/Phase_9_DEPLOYMENT_GUIDE.md`
- **Phase 10:** `USAGE_GUIDES/Phase_10_QA_INTEGRATION_GUIDE.md`
- **Phase 11:** `USAGE_GUIDES/PHASE_11_USAGE_GUIDE.md`

---

## 10. Next Steps After Starting

### Once you've chosen your entry point:

1. **Set up environment** (if not done)
   - Virtual environment
   - Install dependencies
   - Configure `.env` (for dashboard)

2. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

3. **Generate or import data**
   - Follow Phase 0 instructions
   - Verify dataset exists

4. **Run your first experiment**
   - Dashboard: Use GUI wizard
   - CLI: Run Phase 1 script

5. **Explore results**
   - Check output directories
   - View visualizations
   - Understand metrics

---

## 11. Troubleshooting

### "Where do I start?" → This file!

### "I'm getting errors" → Check:
- `QUICKSTART.md` Troubleshooting section
- `dash_app/README.md` Troubleshooting section
- GitHub Issues: https://github.com/abbas-ahmad-cowlar/LSTM_PFD/issues

### "I don't understand X" → Read:
- `QUICKSTART.md` for detailed explanations
- Phase-specific usage guides
- Code comments in relevant files
- Historical reference: `docs/archive/COMPLETE_BEGINNER_GUIDE.md`

---

## 12. Summary: Your Exact Starting Point

**Based on your query, here's exactly where to start:**

### 🎯 **RECOMMENDED STARTING POINT:**

1. **Read this file** (`START_HERE.md`) - ✅ You're here!

2. **Read `README.md`** (30 minutes)
   - Understand project overview
   - See all 11 phases
   - Get high-level architecture

3. **Read `QUICKSTART.md`** (1 hour)
   - Step-by-step instructions
   - All commands you need
   - Expected outputs

4. **Choose your path:**
   - **Dashboard:** Follow Section 4 → `dash_app/GUI_QUICKSTART.md`
   - **CLI:** Follow Section 5 → Execute Phase 0 commands

5. **Start with Phase 0 (Data Generation)**
   - This is the foundation
   - Everything else depends on it
   - See `QUICKSTART.md` Phase 0 section

### 📁 **Key Files to Open:**

1. `README.md` - Project overview
2. `QUICKSTART.md` - Step-by-step guide
3. `data/signal_generator.py` - Data generation (if doing Phase 0)
4. `dash_app/app.py` - Dashboard entry (if using GUI)
5. `scripts/train_classical_ml.py` - First training script (if using CLI)

---

## 🚀 Ready to Start?

**Your next action:**

```bash
# 1. Open and read README.md
cat README.md

# 2. Open and read QUICKSTART.md
cat QUICKSTART.md

# 3. Follow Phase 0 instructions in QUICKSTART.md
```

**Or if you prefer GUI:**

```bash
# 1. Read GUI_QUICKSTART.md
cat dash_app/GUI_QUICKSTART.md

# 2. Set up dashboard
cd dash_app
cp .env.example .env
# Edit .env with your settings
python app.py
```

---

**Good luck! You've got this! 🎉**

For questions, see the documentation files listed above or check GitHub Issues.

