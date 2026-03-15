# Colab GPU Training Pipeline

Train all 35 models on Google Colab with T4 GPU in ~2.5 hours.

## Quick Start (copy-paste into Colab cells)

### Cell 1: Clone & Setup
```python
!git clone https://github.com/YOUR_USERNAME/LSTM_PFD.git
%cd LSTM_PFD
!git checkout fix/master-plan
!bash scripts/colab/01_setup.sh
```

### Cell 2: Generate Data (~1 min)
```python
!python scripts/colab/02_generate_data.py
```

### Cell 3-8: Train Models (run one cell at a time to track progress)
```python
# Batch 1: CNN models (~15 min)
!python scripts/colab/03_train_batch1_cnn.py

# Batch 2: ResNet models (~40 min)
!python scripts/colab/04_train_batch2_resnet.py

# Batch 3: Transformer models (~30 min)
!python scripts/colab/05_train_batch3_transformer.py

# Batch 4: EfficientNet models (~20 min)
!python scripts/colab/06_train_batch4_efficientnet.py

# Batch 5: Hybrid models (~20 min)
!python scripts/colab/07_train_batch5_hybrid.py

# Batch 6: Advanced models (~25 min)
!python scripts/colab/08_train_batch6_advanced.py
```

### Cell 9: Push Results to GitHub
```python
!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add results/ logs/ implementation_plan.md
!git commit -m "feat(P3): training results for 35 models on Colab T4"
!git push
```

### Cell 10: Save Checkpoints to Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r checkpoints/ /content/drive/MyDrive/LSTM_PFD_checkpoints/
```

## Model Batches

| Script | Models | Count | Est. Time |
|---|---|---|---|
| `03_train_batch1_cnn.py` | CNN1D, AttentionCNN, MultiScaleCNN variants | 5 | ~15 min |
| `04_train_batch2_resnet.py` | ResNet18/34/50, WideResNet, SE-ResNet | 10 | ~40 min |
| `05_train_batch3_transformer.py` | SignalTransformer, ViT, PatchTST, TSMixer | 6 | ~30 min |
| `06_train_batch4_efficientnet.py` | EfficientNet B0-B3 (B4-B7 optional) | 4-8 | ~20 min |
| `07_train_batch5_hybrid.py` | CNN-LSTM, CNN-TCN, CNN-Transformer | 4 | ~20 min |
| `08_train_batch6_advanced.py` | PINN, Spectrogram2D, Contrastive | 6 | ~25 min |

## Output Structure

```
checkpoints/{model_key}/{model_key}_best.pt    # Best model checkpoint
results/{model_key}_results.json               # Per-model results
results/batch_{batch_name}.json                # Batch summary
```

## Customization

All scripts accept `--epochs`, `--batch-size`, and `--lr` arguments:
```python
!python scripts/colab/03_train_batch1_cnn.py --epochs 50 --batch-size 64 --lr 5e-4
```
