# Reproducibility Package

This directory contains everything needed to reproduce our research results.

## Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
cd LSTM_PFD

# 2. Create environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run experiments
python reproducibility/scripts/run_all.py
```

## Directory Structure

```
reproducibility/
├── README.md              # This file
├── config/
│   └── pinn_optimal.yaml  # Best hyperparameters
├── scripts/
│   ├── set_seeds.py       # Seed management
│   ├── run_all.py         # Run all experiments
│   └── generate_tables.py # Create paper tables
└── results/               # Experiment outputs (gitignored)
```

## Fixed Seeds

All experiments use the following seeds for reproducibility:

| Library       | Seed |
| ------------- | ---- |
| Python random | 42   |
| NumPy         | 42   |
| PyTorch       | 42   |
| CUDA          | 42   |

## Data

Dataset is tracked with DVC. To download:

```bash
dvc pull
```

Or generate synthetic data:

```bash
python scripts/run_phase0.py
```

## Key Results

| Model        | Accuracy  | F1-Score  |
| ------------ | --------- | --------- |
| CNN Baseline | 94.2%     | 0.941     |
| PINN (ours)  | **98.1%** | **0.980** |

## Citation

```bibtex
@software{lstm_pfd_2025,
  author = {Ahmad, Syed Abbas},
  title = {LSTM PFD: Physics-Informed Deep Learning for Bearing Fault Diagnosis},
  year = {2025},
  url = {https://github.com/abbas-ahmad-cowlar/LSTM_PFD}
}
```
