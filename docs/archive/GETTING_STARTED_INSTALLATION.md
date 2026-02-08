# Installation

Complete installation guide for LSTM PFD on Windows, Linux, and macOS.

## Prerequisites

| Requirement | Minimum | Recommended |
| ----------- | ------- | ----------- |
| Python      | 3.8     | 3.10        |
| RAM         | 8 GB    | 16 GB       |
| GPU VRAM    | -       | 6 GB+       |
| CUDA        | -       | 11.8+       |

## Quick Install

=== "Windows"

    ```powershell
    # Clone repository
    git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
    cd LSTM_PFD

    # Create virtual environment
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    # Install PyTorch with CUDA
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install dependencies
    pip install -r requirements.txt
    ```

=== "Linux/macOS"

    ```bash
    # Clone repository
    git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
    cd LSTM_PFD

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate

    # Install PyTorch with CUDA (Linux) or CPU (macOS)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install dependencies
    pip install -r requirements.txt
    ```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

Expected output:

```
PyTorch 2.1.0+cu118 | CUDA: True
```

## Dashboard Setup

The enterprise dashboard requires environment configuration:

```bash
# Copy example config
cp packages/dashboard/.env.example packages/dashboard/.env

# Generate secure secrets
python -c "import secrets; print(secrets.token_hex(32))"
# Copy output to SECRET_KEY and JWT_SECRET_KEY in .env

# Edit database URL (SQLite for development)
# DATABASE_URL=sqlite:///./lstm_dashboard.db
```

## Next Steps

- [Quick Start](quickstart.md) - Run your first model
- [Configuration](configuration.md) - Customize settings
- [First Experiment](first-experiment.md) - Train a model
