# Getting Started

> Installation, configuration, and first run for LSTM-PFD.

## Prerequisites

| Requirement | Minimum                              | Recommended   |
| ----------- | ------------------------------------ | ------------- |
| Python      | 3.8+                                 | 3.10          |
| CUDA        | 11.8+ (optional)                     | 12.x with GPU |
| RAM         | 8 GB                                 | 16 GB+        |
| GPU VRAM    | 4 GB (optional)                      | 6 GB+         |
| OS          | Windows 10 / Ubuntu 20.04 / macOS 12 | —             |
| PostgreSQL  | 13+ (for dashboard)                  | 15+           |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
cd LSTM_PFD
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install PyTorch

Choose your platform:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python check_requirements.py  # Validates all dependencies
```

## Environment Configuration

### For Dashboard Usage

The dashboard requires PostgreSQL and environment variables:

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` and set at minimum:

```
DATABASE_URL=postgresql://user:password@localhost:5432/lstm_pfd
SECRET_KEY=<generate-with-secrets.token_hex(32)>
JWT_SECRET_KEY=<generate-with-secrets.token_hex(32)>
```

Generate secure keys:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

> ⚠️ **Security**: Never commit `.env` to version control. The `.gitignore` already excludes it.

### For CLI / Research Usage

No database or `.env` is required for pure ML training and research scripts. You can use the data pipeline, training, and evaluation directly.

## First Run

### Option A: Dashboard (No Code)

```bash
cd packages/dashboard
python app.py
# Open http://localhost:8050
```

### Option B: CLI Training

```bash
# Example: run with default config
python scripts/research/run_experiment.py --config config/default.yaml
```

> 📖 See [scripts/research/EXPERIMENT_GUIDE.md](../scripts/research/EXPERIMENT_GUIDE.md) for detailed experiment setup.

## Common Issues

| Issue                         | Solution                                                             |
| ----------------------------- | -------------------------------------------------------------------- |
| `ModuleNotFoundError: torch`  | Ensure virtual environment is activated and PyTorch is installed     |
| CUDA out of memory            | Reduce batch size in config, or use CPU mode                         |
| PostgreSQL connection refused | Ensure PostgreSQL is running and `DATABASE_URL` is correct in `.env` |
| `ImportError: lime`           | Run `pip install lime>=0.2.0.1` (known version sensitivity)          |

See also: `python check_pytorch_cuda.py` for GPU diagnostics.

## Next Steps

- **Explore models**: [packages/core/models/README.md](../packages/core/models/README.md)
- **Understand data pipeline**: [data/STORAGE_README.md](../data/STORAGE_README.md)
- **Learn the architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **Set up development**: [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Full documentation map**: [docs/index.md](index.md)
