## **PHASE 2: 1D Convolutional Neural Network Implementation**

### Phase Objective
Implement and train a 1D CNN architecture for end-to-end learning from raw vibration signals, bypassing manual feature engineering. Achieve performance comparable to classical ML baseline (target: 93-96% test accuracy) while establishing the foundation for more advanced deep learning models in subsequent phases.

### Complete File List (24 files)

#### **1. CNN Architecture (5 files)**

**`models/cnn/cnn_1d.py`** *(Enhanced from Phase 0)*
- **Purpose**: Main 1D CNN architecture with configurable depth
- **Key Classes**:
  - `CNN1D(BaseModel)`: Core CNN architecture
  - `ConvBlock`: Reusable Conv1D-BN-ReLU-Dropout-Pool block
- **Architecture**:
  ```
  Input [B, 1, 102400]  # Batch, Channels, Time samples
  ├─ ConvBlock1: Conv1D(1→32, k=64, s=4) → [B, 32, 25600]
  ├─ ConvBlock2: Conv1D(32→64, k=32, s=2) → [B, 64, 12800]
  ├─ ConvBlock3: Conv1D(64→128, k=16, s=2) → [B, 128, 6400]
  ├─ ConvBlock4: Conv1D(128→256, k=8, s=2) → [B, 256, 3200]
  ├─ ConvBlock5: Conv1D(256→512, k=4, s=2) → [B, 512, 1600]
  ├─ GlobalAvgPool → [B, 512]
  ├─ FC1: 512 → 256, ReLU, Dropout(0.5)
  └─ FC2: 256 → 11 (num_classes)
  ```
- **Key Functions**:
  - `forward(x)`: Forward pass
  - `get_intermediate_features(x, layer_name)`: Extract features at specific layer
  - `count_parameters()`: ~1.2M parameters
- **Hyperparameters**:
  - Kernel sizes: [64, 32, 16, 8, 4]
  - Strides: [4, 2, 2, 2, 2]
  - Dropout: 0.5
  - Batch normalization: After each conv layer
- **Dependencies**: `torch.nn`, `models/base_model.py`

**`models/cnn/conv_blocks.py`**
- **Purpose**: Modular convolutional blocks for reusability
- **Key Classes**:
  - `ConvBlock1D(nn.Module)`: Conv-BN-ReLU-Dropout-Pool
  - `ResidualConvBlock1D(nn.Module)`: Conv block with skip connection
  - `SeparableConv1D(nn.Module)`: Depthwise separable convolution (efficient)
- **Key Functions**:
  - `forward(x)`: Standard forward pass
- **Design Rationale**: 
  - Modular blocks enable architecture search (Phase 4)
  - Residual blocks prevent gradient vanishing
  - Separable conv reduces parameters by ~9×
- **Dependencies**: `torch.nn`

**`models/cnn/attention_mechanisms.py`**
- **Purpose**: Attention modules to focus on discriminative time regions
- **Key Classes**:
  - `SelfAttention1D(nn.Module)`: Channel/spatial attention
  - `SEBlock(nn.Module)`: Squeeze-and-Excitation (recalibrate channels)
  - `CBAM(nn.Module)`: Convolutional Block Attention Module
- **Key Functions**:
  - `forward(x)`: Attention-weighted features
- **Usage**: Insert after conv layers: `x = attention_module(x)`
- **Benefit**: +1-2% accuracy from report benchmarks
- **Dependencies**: `torch.nn`

**`models/cnn/pooling_layers.py`**
- **Purpose**: Advanced pooling beyond MaxPool/AvgPool
- **Key Classes**:
  - `AdaptiveAvgPool1D(nn.Module)`: Adaptive pooling to fixed output size
  - `StochasticPooling(nn.Module)`: Random sampling during training (regularization)
  - `AttentionPooling(nn.Module)`: Learnable weighted pooling
- **Key Functions**:
  - `forward(x)`: Pooled output
- **Dependencies**: `torch.nn`

**`models/cnn/model_variants.py`**
- **Purpose**: CNN architecture variations for experimentation
- **Key Classes**:
  - `ShallowCNN(CNN1D)`: 3-layer lightweight (fast baseline)
  - `DeepCNN(CNN1D)`: 10-layer deep network
  - `WideResidualCNN(CNN1D)`: Wide layers with residual connections
- **Key Functions**:
  - Same interface as `CNN1D.forward(x)`
- **Usage**: `model = create_model('DeepCNN', config)`
- **Dependencies**: `cnn_1d.py`, `conv_blocks.py`

#### **2. Data Preprocessing for CNN (4 files)**

**`data/cnn_transforms.py`**
- **Purpose**: Signal preprocessing specific to CNN input requirements
- **Key Classes**:
  - `ToTensor1D(object)`: Convert NumPy array → torch.Tensor
  - `Normalize1D(object)`: Z-score normalization per sample
  - `RandomCrop1D(object)`: Extract random subsequence (data augmentation)
  - `RandomAmplitudeScale(object)`: Multiply by [0.8, 1.2]
  - `AddGaussianNoise(object)`: Inject noise with probability p
- **Key Functions**:
  - `__call__(signal)`: Apply transformation
- **Dependencies**: `torch`, `numpy`

**`data/cnn_dataset.py`**
- **Purpose**: PyTorch Dataset for CNN training (raw signals, no feature extraction)
- **Key Classes**:
  - `RawSignalDataset(torch.utils.data.Dataset)`: Load signals directly
- **Key Functions**:
  - `__getitem__(idx)`: Returns `(signal [1, T], label [int])`
  - `__len__()`: Dataset size
- **Difference from Phase 0**: No feature extraction, returns raw waveforms
- **Dependencies**: `torch.utils.data`, `data/dataset.py`

**`data/cnn_dataloader.py`**
- **Purpose**: Optimized DataLoaders for CNN training
- **Key Functions**:
  - `create_cnn_dataloaders(dataset, config)`: Returns train/val/test loaders
  - `collate_fn(batch)`: Stack signals into batch tensor [B, 1, T]
- **Optimizations**:
  - Pin memory: True (faster GPU transfer)
  - Num workers: 4 (parallel data loading)
  - Persistent workers: True (reduce initialization overhead)
- **Dependencies**: `torch.utils.data`, `cnn_dataset.py`

**`data/signal_augmentation.py`**
- **Purpose**: Advanced augmentation techniques for CNNs
- **Key Classes**:
  - `SignalAugmenter`: Orchestrates multiple augmentations
- **Key Functions**:
  - `time_warp(signal, warp_factor)`: Non-linear time stretching
  - `frequency_mask(signal, fs, mask_param)`: Zero out frequency bands (SpecAugment-style)
  - `time_mask(signal, mask_param)`: Zero out time segments
  - `mixup(signal1, signal2, alpha=0.4)`: Convex combination of signals and labels
- **Benefit**: +2-3% accuracy from data augmentation literature
- **Dependencies**: `numpy`, `scipy.signal`

#### **3. CNN Training Infrastructure (5 files)**

**`training/cnn_trainer.py`**
- **Purpose**: CNN-specific training loop with optimizations
- **Key Classes**:
  - `CNNTrainer(Trainer)`: Extends base trainer with CNN-specific logic
- **Key Functions**:
  - `train_epoch(dataloader)`: Training loop with mixed precision
  - `validate_epoch(dataloader)`: Validation loop
  - `_compute_loss(outputs, targets)`: Cross-entropy with label smoothing
  - `_update_lr_scheduler(epoch)`: Cosine annealing schedule
- **Optimizations**:
  - Mixed precision training (torch.cuda.amp)
  - Gradient clipping (max_norm=1.0)
  - Gradient accumulation for large effective batch size
- **Dependencies**: `torch`, `training/trainer.py`, `training/losses.py`

**`training/cnn_losses.py`**
- **Purpose**: Loss functions tailored for CNN training
- **Key Classes**:
  - `LabelSmoothingCrossEntropy(nn.Module)`: Regularization via soft labels
  - `FocalLoss(nn.Module)`: Address class imbalance (focus on hard examples)
  - `SupConLoss(nn.Module)`: Supervised contrastive loss (optional)
- **Key Functions**:
  - `forward(logits, targets)`: Compute loss
- **Usage**: `loss = LabelSmoothingCrossEntropy(smoothing=0.1)(logits, targets)`
- **Dependencies**: `torch.nn`

**`training/cnn_schedulers.py`**
- **Purpose**: Learning rate schedules for CNN optimization
- **Key Functions**:
  - `create_cosine_scheduler(optimizer, T_max, eta_min)`: Cosine annealing
  - `create_step_scheduler(optimizer, step_size, gamma)`: Step decay
  - `create_warmup_scheduler(optimizer, warmup_epochs)`: Linear warmup
- **Recommended**: Warmup (5 epochs) → Cosine annealing (remaining epochs)
- **Dependencies**: `torch.optim.lr_scheduler`

**`training/cnn_callbacks.py`**
- **Purpose**: Callbacks specific to CNN training
- **Key Classes**:
  - `LearningRateMonitor(Callback)`: Log LR to MLflow
  - `GradientMonitor(Callback)`: Track gradient norms (detect vanishing/exploding)
  - `ActivationMonitor(Callback)`: Visualize layer activations
- **Key Functions**:
  - `on_batch_end(batch, logs)`: Hook for monitoring
- **Dependencies**: `training/callbacks.py`, `mlflow`

**`training/cnn_optimizer.py`**
- **Purpose**: Optimizer configurations for CNN
- **Key Functions**:
  - `create_adam_optimizer(model_params, lr, weight_decay)`: Adam with decoupled weight decay (AdamW)
  - `create_sgd_optimizer(model_params, lr, momentum, nesterov)`: SGD with Nesterov momentum
- **Recommended**: AdamW(lr=1e-3, weight_decay=1e-4)
- **Dependencies**: `torch.optim`

#### **4. CNN Evaluation (4 files)**

**`evaluation/cnn_evaluator.py`**
- **Purpose**: Evaluate trained CNN on test set
- **Key Classes**:
  - `CNNEvaluator(ModelEvaluator)`: Extends base evaluator
- **Key Functions**:
  - `evaluate(model, test_loader)`: Full evaluation
  - `compute_per_class_metrics(preds, targets)`: Precision/recall/F1 per class
  - `generate_classification_report()`: Summary report
- **Additional Metrics**:
  - Inference time per sample
  - GPU memory usage
  - FLOPs count
- **Dependencies**: `evaluation/evaluator.py`, `torch`

**`evaluation/cnn_interpretability.py`**
- **Purpose**: Explain CNN predictions (address black-box concern)
- **Key Classes**:
  - `GradCAM1D`: Gradient-weighted Class Activation Mapping for 1D signals
  - `IntegratedGradients1D`: Attribution method
- **Key Functions**:
  - `generate_gradcam(model, signal, target_class)`: Heatmap of important regions
  - `generate_attribution_map(model, signal, target_class)`: Pixel-level importance
- **Usage**: Visualize which time regions contribute to fault classification
- **Dependencies**: `torch`, `captum` (PyTorch interpretability library)

**`evaluation/cnn_robustness.py`**
- **Purpose**: Robustness testing specific to CNN (beyond classical tests)
- **Key Functions**:
  - `test_adversarial_robustness(model, test_loader, epsilon)`: FGSM attacks
  - `test_input_corruption(model, test_loader, corruption_types)`: Blur, noise, etc.
- **Corruption Types**: 
  - Gaussian noise
  - Impulse noise
  - Shot noise (Poisson)
  - Motion blur (simulated sensor vibration)
- **Dependencies**: `evaluation/robustness_tester.py`, `torch`

**`evaluation/cnn_visualization.py`**
- **Purpose**: Visualize CNN internals
- **Key Functions**:
  - `plot_feature_maps(model, signal, layer_name)`: Visualize conv layer activations
  - `plot_filters(model, layer_name)`: Visualize learned conv filters
  - `plot_training_curves(history)`: Loss/accuracy over epochs
- **Dependencies**: `matplotlib`, `torch`

#### **5. Experiment Management (3 files)**

**`experiments/cnn_experiment.py`**
- **Purpose**: Orchestrate full CNN training experiment
- **Key Classes**:
  - `CNNExperiment`: Manages experiment lifecycle
- **Key Functions**:
  - `setup_experiment(config)`: Initialize model, data, trainer
  - `run_training()`: Train model
  - `run_evaluation()`: Evaluate on test set
  - `log_results_to_mlflow()`: Log metrics, artifacts
- **Dependencies**: `mlflow`, `training/cnn_trainer.py`, `evaluation/cnn_evaluator.py`

**`experiments/cnn_hparam_search.py`**
- **Purpose**: Hyperparameter search for CNN
- **Key Functions**:
  - `run_hyperparameter_search(config, n_trials)`: Optuna-based tuning
  - `objective(trial)`: Define search space
- **Search Space**:
  - Learning rate: [1e-4, 1e-2] (log scale)
  - Batch size: [16, 32, 64, 128]
  - Dropout: [0.3, 0.5, 0.7]
  - Weight decay: [1e-5, 1e-3] (log scale)
  - Number of conv layers: [4, 6, 8]
- **Dependencies**: `optuna`, `experiments/cnn_experiment.py`

**`experiments/cnn_ablation_study.py`**
- **Purpose**: Ablation studies to understand component contributions
- **Key Functions**:
  - `ablate_data_augmentation(config)`: Train with/without augmentation
  - `ablate_batch_normalization(config)`: Remove BN layers
  - `ablate_dropout(config)`: Train without dropout
  - `ablate_attention(config)`: Remove attention modules
- **Output**: Table showing impact of each component (e.g., "-2.1% accuracy without BN")
- **Dependencies**: `experiments/cnn_experiment.py`

#### **6. Utilities (3 files)**

**`utils/cnn_utils.py`**
- **Purpose**: Helper functions for CNN development
- **Key Functions**:
  - `count_parameters(model)`: Total trainable parameters
  - `compute_flops(model, input_size)`: Computational cost
  - `visualize_model_architecture(model)`: Generate architecture diagram
- **Dependencies**: `torch`, `torchsummary`

**`utils/checkpoint_manager.py`**
- **Purpose**: Manage model checkpoints during training
- **Key Classes**:
  - `CheckpointManager`: Save/load best models
- **Key Functions**:
  - `save_checkpoint(model, optimizer, epoch, metrics)`: Save state dict
  - `load_checkpoint(checkpoint_path)`: Restore training state
  - `save_best_model(model, metric, threshold)`: Save if metric improves
- **Dependencies**: `torch`, `pathlib`

**`utils/early_stopping.py`**
- **Purpose**: Early stopping to prevent overfitting
- **Key Classes**:
  - `EarlyStopping`: Monitor validation loss
- **Key Functions**:
  - `should_stop(val_loss)`: Returns True if patience exceeded
- **Parameters**: patience=10, min_delta=0.001
- **Dependencies**: None (pure Python)

### Architecture Decisions

**1. 1D Convolution vs. 2D Convolution (Spectrogram)**
- **Decision**: Use 1D convolution on raw signals
- **Rationale**:
  - More parameter-efficient (1D kernels vs. 2D)
  - Directly learns from waveform (no hand-crafted spectrogram)
  - Faster training (no STFT computation per sample)
- **Alternative Considered**: 2D CNN on spectrograms (Phase 3 explores this)

**2. Large Receptive Field via Strided Convolutions**
- **Decision**: Use stride 4 in first layer, stride 2 thereafter
- **Rationale**:
  - Input is 102,400 samples (5 sec × 20.48 kHz)
  - Need to downsample quickly to manageable size
  - Large strides increase receptive field
- **Trade-off**: Some aliasing, but acceptable for fault diagnosis

**3. Global Average Pooling Instead of Flatten**
- **Decision**: Use AdaptiveAvgPool1D before FC layers
- **Rationale**:
  - Reduces overfitting (fewer parameters in FC layer)
  - Invariant to small input size changes
  - Standard in modern CNNs (ResNet, EfficientNet)

**4. Batch Normalization After Every Conv Layer**
- **Decision**: Conv → BN → ReLU (not Conv → ReLU → BN)
- **Rationale**: 
  - BN before activation is standard practice
  - Stabilizes training (allows higher learning rates)
  - Reduces internal covariate shift

**5. Label Smoothing for Regularization**
- **Decision**: Use smoothing factor ε = 0.1
- **Rationale**:
  - Prevents overconfident predictions
  - Improves calibration (ECE in report was 0.1267, want to reduce)
  - Negligible accuracy cost (< 0.5%) from literature

**6. Mixed Precision Training**
- **Decision**: Use torch.cuda.amp for FP16 training
- **Rationale**:
  - 2-3× speedup on modern GPUs (Tensor Cores)
  - Reduces memory usage (allows larger batches)
  - Numerical stability with gradient scaling
- **Requirement**: CUDA-capable GPU (RTX 20xx+, V100, A100)

### Data Flow

```
┌────────────────────────────────────────────────────────────┐
│              CNN TRAINING PIPELINE (Phase 2)                │
└────────────────────────────────────────────────────────────┘

1. DATA LOADING
   ┌──────────────────────────────────────────────────────┐
   │ data/cnn_dataset.py (RawSignalDataset)               │
   │  ├─ Load signals from HDF5 cache (Phase 0 output)   │
   │  ├─ No feature extraction (raw waveforms)            │
   │  └─ Apply transforms (normalize, augment)            │
   │         ↓                                             │
   │ data/cnn_dataloader.py                                │
   │  ├─ Batch signals: [B, 1, 102400]                   │
   │  ├─ Pin memory for fast GPU transfer                │
   │  └─ Parallel loading (num_workers=4)                 │
   └──────────────────────────────────────────────────────┘
                        ↓

2. MODEL FORWARD PASS
   ┌──────────────────────────────────────────────────────┐
   │ models/cnn/cnn_1d.py                                  │
   │                                                       │
   │ Input: [B, 1, 102400]                                 │
   │  ↓                                                    │
   │ ConvBlock1: [B, 32, 25600]  (k=64, s=4)             │
   │  ↓                                                    │
   │ ConvBlock2: [B, 64, 12800]  (k=32, s=2)             │
   │  ↓                                                    │
   │ ConvBlock3: [B, 128, 6400]  (k=16, s=2)             │
   │  ↓                                                    │
   │ ConvBlock4: [B, 256, 3200]  (k=8, s=2)              │
   │  ↓                                                    │
   │ ConvBlock5: [B, 512, 1600]  (k=4, s=2)              │
   │  ↓                                                    │
   │ GlobalAvgPool: [B, 512]                              │
   │  ↓                                                    │
   │ FC1: [B, 256] + ReLU + Dropout(0.5)                 │
   │  ↓                                                    │
   │ FC2: [B, 11] (logits)                                │
   │         ↓                                             │
   │ Output: Logits [B, 11]                                │
   └──────────────────────────────────────────────────────┘
                        ↓

3. LOSS COMPUTATION
   ┌──────────────────────────────────────────────────────┐
   │ training/cnn_losses.py                                │
   │  ├─ LabelSmoothingCrossEntropy(logits, targets)     │
   │  └─ Loss: scalar                                     │
   └──────────────────────────────────────────────────────┘
                        ↓

4. BACKPROPAGATION
   ┌──────────────────────────────────────────────────────┐
   │ training/cnn_trainer.py                               │
   │  ├─ Scaled backward pass (mixed precision)          │
   │  ├─ Gradient clipping (max_norm=1.0)                │
   │  ├─ Optimizer step (AdamW)                           │
   │  └─ LR scheduler step (Cosine annealing)            │
   └──────────────────────────────────────────────────────┘
                        ↓

5. VALIDATION LOOP (every N epochs)
   ┌──────────────────────────────────────────────────────┐
   │ training/cnn_trainer.py                               │
   │  ├─ Forward pass on validation set (no grad)        │
   │  ├─ Compute validation loss, accuracy               │
   │  └─ Trigger callbacks (checkpoint, early stop)      │
   └──────────────────────────────────────────────────────┘
                        ↓

6. EXPERIMENT LOGGING
   ┌──────────────────────────────────────────────────────┐
   │ experiments/experiment_manager.py (MLflow)            │
   │  ├─ Log epoch metrics (loss, accuracy)              │
   │  ├─ Log learning rate                                │
   │  ├─ Log hyperparameters                              │
   │  └─ Save model checkpoint                            │
   └──────────────────────────────────────────────────────┘
                        ↓

7. FINAL EVALUATION
   ┌──────────────────────────────────────────────────────┐
   │ evaluation/cnn_evaluator.py                           │
   │  ├─ Load best checkpoint                             │
   │  ├─ Predict on test set                              │
   │  ├─ Compute metrics (accuracy, F1, confusion matrix) │
   │  └─ Generate classification report                   │
   │         ↓                                             │
   │ Output: Test accuracy, per-class metrics             │
   │         Confusion matrix, ROC curves                 │
   └──────────────────────────────────────────────────────┘
```

### Integration Points

**1. With Phase 0 (Data Infrastructure)**
- **Input**: Signals from `data/cache_manager.py` (HDF5 cache)
- **Interface**: `RawSignalDataset` loads signals without feature extraction
- **Difference**: Phase 1 used extracted features, Phase 2 uses raw signals

**2. With Phase 1 (Classical ML Baseline)**
- **Comparison**: Benchmark CNN vs. Random Forest (Phase 1 best: 95.33%)
- **Target**: Match or exceed classical ML accuracy
- **Visualization**: Side-by-side confusion matrices, ROC curves

**3. With Future Phases**
- **Phase 3**: CNN features serve as input to Transformer
- **Phase 6**: CNN backbone used in hybrid physics-informed models
- **Phase 8**: CNN predictions combined in ensemble

**4. With MLflow**
- **Logging**: All CNN experiments tracked in MLflow
- **Artifacts**: Model checkpoints, training curves, confusion matrices
- **Comparison**: Compare CNN variants (shallow vs. deep, with/without attention)

### Testing Strategy

**1. Unit Tests**

**`tests/test_cnn_model.py`**
```python
def test_cnn_forward_pass():
    """Test CNN forward pass with dummy input."""
    model = CNN1D(num_classes=11)
    x = torch.randn(2, 1, 102400)  # Batch of 2 signals
    output = model(x)
    assert output.shape == (2, 11), "Output shape mismatch"

def test_cnn_gradient_flow():
    """Ensure gradients flow through all layers."""
    model = CNN1D(num_classes=11)
    x = torch.randn(2, 1, 102400, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    # Check input gradient computed
    assert x.grad is not None
```

**`tests/test_cnn_transforms.py`**
```python
def test_random_crop():
    """Test random crop augmentation."""
    signal = np.random.randn(102400)
    transform = RandomCrop1D(crop_size=10000)
    cropped = transform(signal)
    assert len(cropped) == 10000

def test_mixup():
    """Test mixup augmentation."""
    signal1 = np.ones(1000)
    signal2 = np.zeros(1000)
    mixed, lambda_ = mixup(signal1, signal2, alpha=0.4)
    # Mixed signal should be between 0 and 1
    assert 0 <= mixed.mean() <= 1
```

**2. Integration Tests**

**`tests/test_cnn_training.py`**
```python
def test_cnn_training_loop():
    """Test full training loop runs without errors."""
    # Small dummy dataset
    dataset = DummyBearingDataset(n_samples=50)
    train_loader = DataLoader(dataset, batch_size=8)
    
    # Model and trainer
    model = CNN1D(num_classes=11)
    trainer = CNNTrainer(model, config)
    
    # Train for 2 epochs
    trainer.fit(num_epochs=2, train_loader=train_loader)
    
    # Check model trained
    assert trainer.epoch == 2
```

**3. Convergence Tests**

**`tests/test_cnn_convergence.py`**
```python
def test_cnn_overfits_small_dataset():
    """Ensure CNN can overfit (sanity check)."""
    # Tiny dataset (10 samples)
    dataset = DummyBearingDataset(n_samples=10)
    train_loader = DataLoader(dataset, batch_size=10)
    
    model = CNN1D(num_classes=11)
    trainer = CNNTrainer(model, config)
    
    # Train until convergence
    trainer.fit(num_epochs=100, train_loader=train_loader)
    
    # Should achieve 100% training accuracy
    train_acc = trainer.evaluate(train_loader)
    assert train_acc > 0.99, "Model failed to overfit small dataset"
```

**4. Comparison Tests**

**`tests/test_cnn_vs_classical.py`**
```python
def test_cnn_matches_classical_baseline():
    """Ensure CNN achieves similar accuracy to classical ML."""
    # Load standard test set
    test_dataset = load_standard_test_set()
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load trained CNN
    cnn_model = load_best_cnn_checkpoint()
    cnn_accuracy = evaluate_model(cnn_model, test_loader)
    
    # Compare to classical baseline (95.33% from Phase 1)
    classical_accuracy = 0.9533
    
    # Allow 3% margin (92.33% minimum)
    assert cnn_accuracy >= 0.9233, \
        f"CNN accuracy ({cnn_accuracy:.2%}) below acceptable threshold"
```

### Acceptance Criteria

**Phase 2 Complete When:**

✅ **CNN model trains successfully**
- Forward pass completes without errors
- Backward pass computes gradients correctly
- Model converges on training set (> 95% train accuracy after 50 epochs)

✅ **Achieves target accuracy on test set**
- **Minimum**: 93% test accuracy (within 2.5% of classical baseline)
- **Target**: 95% test accuracy (matches classical ML)
- **Stretch Goal**: 97% test accuracy (surpasses classical ML)

✅ **Per-class performance acceptable**
- Per-class recall ≥ 85% for at least 9/11 classes
- Mixed fault classes (challenge cases from Phase 1) improved accuracy

✅ **Training efficiency acceptable**
- Training time: < 2 hours for 100 epochs on single GPU (RTX 3080 or better)
- Inference time: < 50 ms per sample (faster than 100ms required for deployment)
- GPU memory usage: < 8 GB (fits on consumer GPUs)

✅ **Robustness comparable to classical ML**
- Sensor noise test: accuracy drop ≤ 20% (vs. 16.82% for classical ML)
- Missing features test: N/A for end-to-end models (different paradigm)
- Temporal drift test: accuracy drop ≤ 5%
- Adversarial robustness: < 10% accuracy drop under FGSM attack (ε=0.1)

✅ **Interpretability demonstrated**
- Grad-CAM visualizations show CNN focuses on fault-relevant time regions
- Activation visualizations confirm hierarchical feature learning
- Can explain misclassifications with attribution maps

✅ **Reproducibility validated**
- Same hyperparameters → same accuracy (±0.5%)
- Saved checkpoint loads correctly and reproduces results
- Config file alone sufficient to reproduce experiment

✅ **Comparison with classical ML documented**
- Side-by-side confusion matrix comparison (CNN vs. Random Forest)
- Accuracy comparison table across all 11 classes
- Error analysis: Which faults does CNN handle better/worse?

✅ **MLflow logging functional**
- All experiments tracked with hyperparameters
- Training curves (loss, accuracy) logged every epoch
- Best model checkpoint saved as artifact
- Confusion matrix, ROC curves saved as images

✅ **Documentation complete**
- README explaining CNN architecture choices
- Jupyter notebook demonstrating CNN training from scratch
- API documentation for all CNN modules

### Estimated Effort

**Time Breakdown:**
- CNN architecture (5 files): 4 days
  - `cnn_1d.py`: 1 day (core architecture)
  - `conv_blocks.py`: 1 day (modular blocks)
  - `attention_mechanisms.py`: 1 day (SE, CBAM modules)
  - `pooling_layers.py`: 0.5 days
  - `model_variants.py`: 0.5 days

- Data preprocessing (4 files): 2 days
  - `cnn_transforms.py`: 1 day
  - `cnn_dataset.py`: 0.5 days
  - `cnn_dataloader.py`: 0.25 days
  - `signal_augmentation.py`: 0.25 days

- Training infrastructure (5 files): 3 days
  - `cnn_trainer.py`: 1 day (mixed precision, gradient clipping)
  - `cnn_losses.py`: 0.5 days
  - `cnn_schedulers.py`: 0.5 days
  - `cnn_callbacks.py`: 0.5 days
  - `cnn_optimizer.py`: 0.5 days

- Evaluation (4 files): 3 days
  - `cnn_evaluator.py`: 1 day
  - `cnn_interpretability.py`: 1 day (Grad-CAM, Integrated Gradients)
  - `cnn_robustness.py`: 0.5 days
  - `cnn_visualization.py`: 0.5 days

- Experiment management (3 files): 2 days
  - `cnn_experiment.py`: 1 day
  - `cnn_hparam_search.py`: 0.5 days
  - `cnn_ablation_study.py`: 0.5 days

- Utilities (3 files): 1 day
- Testing (unit, integration, convergence): 4 days
- Hyperparameter tuning (find best config): 3 days
- Documentation: 2 days
- Buffer for debugging: 3 days

**Total: ~27 days (1.3 months) for Phase 2**

**Complexity**: ⭐⭐⭐⭐☆ (High)
- Deep learning requires GPU setup, debugging
- Hyperparameter tuning is time-consuming
- Interpretability methods (Grad-CAM) need careful implementation

**Dependencies**: Phase 0 (data), Phase 1 (baseline comparison)

**Risk**: Medium-High
- May not match classical ML accuracy on first attempt (need tuning)
- GPU availability/configuration issues
- Mixed precision training may have numerical instabilities

---
