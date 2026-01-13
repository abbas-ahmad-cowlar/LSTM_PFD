# Quick Start

Get LSTM PFD running in 5 minutes.

## 1. Generate Sample Data

```bash
python -c "
from data.signal_generator import SignalGenerator
from config.data_config import DataConfig

config = DataConfig(num_signals_per_fault=50)
generator = SignalGenerator(config)
dataset = generator.generate_dataset()
paths = generator.save_dataset(dataset, format='hdf5')
print(f'✓ Generated {len(dataset[\"signals\"])} samples')
print(f'✓ Saved to {paths[\"hdf5\"]}')
"
```

## 2. Train a Model

=== "Dashboard (No Code)"

    ```bash
    cd packages/dashboard
    python app.py
    ```

    Open http://localhost:8050 → Experiments → New Experiment

=== "Command Line"

    ```bash
    python scripts/train_cnn.py \
        --model resnet18 \
        --data-path data/processed/signals_cache.h5 \
        --epochs 50 \
        --batch-size 64
    ```

## 3. Evaluate Results

```bash
python scripts/evaluate_model.py \
    --checkpoint checkpoints/best.pth \
    --data-path data/processed/signals_cache.h5
```

Expected output:

```
Test Accuracy: 96.4%
Confusion Matrix saved to: results/confusion_matrix.png
```

## Next Steps

- [Full Installation Guide](installation.md)
- [Phase-by-Phase Tutorial](../user-guide/phases/overview.md)
- [Dashboard Guide](../user-guide/dashboard/overview.md)
