# First Experiment

This tutorial guides you through training your first bearing fault detection model.

## Overview

By the end of this tutorial, you will:

1. Generate a synthetic dataset
2. Train a ResNet-18 model
3. Evaluate on test data
4. Generate explainability visualizations

## Step 1: Generate Data

```python
from data.signal_generator import SignalGenerator
from config.data_config import DataConfig

# Configure dataset
config = DataConfig(
    num_signals_per_fault=100,
    sampling_rate=20480,
    signal_duration=5.0
)

# Generate
generator = SignalGenerator(config)
dataset = generator.generate_dataset()
paths = generator.save_dataset(dataset, format='hdf5')
```

## Step 2: Train Model

```bash
python scripts/train_cnn.py \
    --model resnet18 \
    --data-path data/processed/signals_cache.h5 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001
```

## Step 3: Evaluate

```bash
python scripts/evaluate_model.py \
    --checkpoint checkpoints/resnet18/best.pth
```

## Step 4: Explain

```bash
python scripts/explain.py \
    --checkpoint checkpoints/resnet18/best.pth \
    --method shap
```

## Expected Results

| Metric            | Expected Value |
| ----------------- | -------------- |
| Test Accuracy     | 96-97%         |
| Training Time     | ~30 min (GPU)  |
| Inference Latency | <50ms          |

## Next Steps

- [Phase 2: Advanced CNNs](../user-guide/phases/phase-2.md)
- [XAI Dashboard](../user-guide/dashboard/xai.md)
