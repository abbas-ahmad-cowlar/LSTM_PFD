## **PHASE 3: ResNet-1D and Advanced CNN Architectures**

### Phase Objective
Implement state-of-the-art 1D CNN architectures (ResNet-18 adapted for signals, EfficientNet-inspired models) to push accuracy beyond Phase 2 baseline. Explore architecture search to find optimal depth/width trade-offs. Target: 96-98% test accuracy through deeper networks with residual connections and efficient scaling.

### Complete File List (22 files)

#### **1. ResNet Architecture (5 files)**

**`models/resnet/resnet_1d.py`**
- **Purpose**: ResNet-18 adapted for 1D vibration signals
- **Key Classes**:
  - `ResNet1D(BaseModel)`: Main ResNet architecture
  - `BasicBlock1D(nn.Module)`: Residual block with 2 conv layers + skip connection
  - `Bottleneck1D(nn.Module)`: Bottleneck block (1x1 → 3x3 → 1x1 conv)
- **Architecture**:
  ```
  Input [B, 1, 102400]
  ├─ Conv1: 1→64, k=64, s=4 → [B, 64, 25600]
  ├─ MaxPool: k=4, s=4 → [B, 64, 6400]
  ├─ Layer1: 2× BasicBlock, 64 channels → [B, 64, 6400]
  ├─ Layer2: 2× BasicBlock, 128 channels, s=2 → [B, 128, 3200]
  ├─ Layer3: 2× BasicBlock, 256 channels, s=2 → [B, 256, 1600]
  ├─ Layer4: 2× BasicBlock, 512 channels, s=2 → [B, 512, 800]
  ├─ AdaptiveAvgPool → [B, 512]
  └─ FC: 512 → 11
  ```
- **Key Functions**:
  - `forward(x)`: Forward pass
  - `_make_layer(block, channels, num_blocks, stride)`: Build residual layer
- **Parameters**: ~2.5M (deeper than Phase 2 CNN)
- **Dependencies**: `torch.nn`, `models/base_model.py`

**`models/resnet/residual_blocks.py`**
- **Purpose**: Reusable residual block variants
- **Key Classes**:
  - `BasicBlock1D(nn.Module)`: Standard residual block
    ```python
    x_identity = x
    out = Conv(x) → BN → ReLU → Conv → BN
    out = out + x_identity  # Skip connection
    out = ReLU(out)
    ```
  - `Bottleneck1D(nn.Module)`: Efficient bottleneck design
    ```python
    x_identity = x
    out = Conv1x1(x) → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN
    out = out + x_identity
    out = ReLU(out)
    ```
  - `PreActBlock1D(nn.Module)`: Pre-activation variant (BN-ReLU-Conv)
- **Key Functions**:
  - `forward(x)`: Residual forward pass
- **Design Rationale**:
  - Bottleneck reduces parameters (256→64→64→256 vs. 256→256→256)
  - Pre-activation improves gradient flow
- **Dependencies**: `torch.nn`

**`models/resnet/resnet_variants.py`**
- **Purpose**: ResNet-34, ResNet-50 for scalability study
- **Key Classes**:
  - `ResNet34_1D(ResNet1D)`: Deeper variant (34 layers)
  - `ResNet50_1D(ResNet1D)`: Even deeper with bottlenecks (50 layers)
- **Parameters**:
  - ResNet-34: ~5M parameters
  - ResNet-50: ~10M parameters
- **Usage**: Compare depth vs. accuracy trade-off
- **Dependencies**: `resnet_1d.py`, `residual_blocks.py`

**`models/resnet/se_resnet.py`**
- **Purpose**: ResNet with Squeeze-and-Excitation (SE) blocks
- **Key Classes**:
  - `SEResNet1D(ResNet1D)`: ResNet + SE modules
  - `SEBlock(nn.Module)`: Channel-wise attention
    ```python
    # Squeeze: Global average pooling
    squeeze = AdaptiveAvgPool(x)  # [B, C, T] → [B, C, 1]
    # Excitation: 2-layer FC
    excitation = FC(squeeze) → ReLU → FC → Sigmoid  # [B, C, 1]
    # Recalibration
    out = x * excitation  # Channel-wise multiplication
    ```
- **Benefit**: +1-2% accuracy from literature (ImageNet results)
- **Dependencies**: `resnet_1d.py`

**`models/resnet/wide_resnet.py`**
- **Purpose**: Wide ResNet (fewer layers, wider channels)
- **Key Classes**:
  - `WideResNet1D(ResNet1D)`: 16-layer network with 8× wider channels
- **Architecture**: Instead of [64, 128, 256, 512], use [128, 256, 512, 1024]
- **Trade-off**: More parameters but shallower (faster training)
- **Dependencies**: `resnet_1d.py`

#### **2. EfficientNet-Inspired Models (4 files)**

**`models/efficientnet/efficientnet_1d.py`**
- **Purpose**: EfficientNet compound scaling for 1D signals
- **Key Classes**:
  - `EfficientNet1D(BaseModel)`: Scaled CNN architecture
  - `MBConvBlock(nn.Module)`: Mobile inverted bottleneck
    ```python
    # Expansion
    x_exp = Conv1x1(x, expand_ratio * in_channels)
    # Depthwise conv
    x_dw = DepthwiseConv(x_exp)
    # Squeeze-Excitation
    x_se = SEBlock(x_dw)
    # Projection
    out = Conv1x1(x_se, out_channels)
    # Skip connection (if stride=1 and same channels)
    if stride == 1 and in_channels == out_channels:
        out = out + x
    ```
- **Compound Scaling**: Scale depth, width, resolution together
  - Depth: α = 1.2 (20% more layers)
  - Width: β = 1.1 (10% wider channels)
  - Resolution: γ = 1.15 (15% longer input signals)
  - Constraint: α × β² × γ² ≈ 2 (2× FLOPs)
- **Key Functions**:
  - `forward(x)`: Forward pass
  - `_calculate_scaling(phi)`: Compute depth/width multipliers
- **Parameters**: EfficientNet-B0: ~1M, EfficientNet-B3: ~5M
- **Dependencies**: `torch.nn`, `models/base_model.py`

**`models/efficientnet/mbconv_block.py`**
- **Purpose**: Mobile inverted bottleneck convolution
- **Key Classes**:
  - `MBConvBlock(nn.Module)`: Core building block
  - `DepthwiseSeparableConv1D(nn.Module)`: Efficient convolution
- **Efficiency Gain**: Depthwise separable conv uses 8-9× fewer parameters
- **Dependencies**: `torch.nn`

**`models/efficientnet/efficient_attention.py`**
- **Purpose**: Efficient attention mechanisms for EfficientNet
- **Key Classes**:
  - `EfficientChannelAttention(nn.Module)`: Lightweight SE variant
  - `CoordinateAttention(nn.Module)`: Spatial + channel attention
- **Dependencies**: `torch.nn`

**`models/efficientnet/efficientnet_variants.py`**
- **Purpose**: EfficientNet-B0 through B7 variants
- **Key Classes**:
  - `EfficientNetB0_1D` through `EfficientNetB7_1D`
- **Scaling Factors**:
  - B0: baseline (φ=0)
  - B3: φ=1.8 (balanced, recommended)
  - B7: φ=3.1 (largest, 20M parameters)
- **Dependencies**: `efficientnet_1d.py`

#### **3. Architecture Search (4 files)**

**`models/nas/neural_architecture_search.py`**
- **Purpose**: Automated architecture search using DARTS (Differentiable Architecture Search)
- **Key Classes**:
  - `SearchableCell(nn.Module)`: Cell with multiple candidate operations
  - `MixedOperation(nn.Module)`: Weighted sum of operations (conv3, conv5, pool, skip)
  - `NASController`: Search algorithm
- **Search Space**:
  - Operations: {Conv k=3, Conv k=5, Conv k=7, MaxPool, AvgPool, Identity}
  - Connections: Which layers connect to which
- **Key Functions**:
  - `search(train_loader, val_loader, epochs)`: Run architecture search
  - `derive_discrete_architecture()`: Extract final architecture from continuous weights
- **Search Time**: ~1 day on single GPU (amortized across many experiments)
- **Dependencies**: `torch.nn`, `torch.optim`

**`models/nas/search_space.py`**
- **Purpose**: Define search space for NAS
- **Key Classes**:
  - `SearchSpace`: Enumerate possible architectures
- **Key Functions**:
  - `sample_architecture()`: Random architecture for random search
  - `mutate_architecture(arch)`: Evolutionary search
- **Dependencies**: None

**`models/nas/darts_trainer.py`**
- **Purpose**: Training loop for DARTS
- **Key Classes**:
  - `DARTSTrainer(Trainer)`: Bi-level optimization (architecture + weights)
- **Key Functions**:
  - `train_epoch()`: Alternate between updating weights and architecture params
- **Dependencies**: `training/trainer.py`, `neural_architecture_search.py`

**`models/nas/architecture_evaluator.py`**
- **Purpose**: Evaluate discovered architectures
- **Key Functions**:
  - `evaluate_architecture(arch, train_loader, val_loader)`: Train from scratch
  - `rank_architectures(arch_list)`: Compare multiple discovered architectures
- **Dependencies**: `models/base_model.py`

#### **4. Hybrid Architectures (3 files)**

**`models/hybrid/cnn_lstm.py`**
- **Purpose**: CNN feature extractor + LSTM for temporal modeling
- **Key Classes**:
  - `CNNLSTM(BaseModel)`: CNN backbone + bidirectional LSTM
- **Architecture**:
  ```
  Input [B, 1, 102400]
    ↓
  CNN Backbone (ResNet-18) → [B, 512, 800]  # Feature maps
    ↓
  Permute: [B, 512, 800] → [B, 800, 512]  # Time steps, features
    ↓
  BiLSTM: 512 → 256 hidden → [B, 800, 512]  # 256×2 = 512 (bidirectional)
    ↓
  Attention Pooling: [B, 800, 512] → [B, 512]  # Weighted average over time
    ↓
  FC: 512 → 11
  ```
- **Rationale**: LSTM captures long-term temporal dependencies CNN might miss
- **Dependencies**: `torch.nn`, `resnet/resnet_1d.py`

**`models/hybrid/cnn_tcn.py`**
- **Purpose**: CNN + Temporal Convolutional Network (alternative to LSTM)
- **Key Classes**:
  - `CNNTCN(BaseModel)`: CNN + dilated causal convolutions
  - `TCNBlock(nn.Module)`: Dilated conv with exponentially increasing dilation
- **TCN Advantages over LSTM**:
  - Parallelizable (no sequential dependency)
  - Larger receptive field with dilation
  - Faster training
- **Dependencies**: `torch.nn`, `resnet/resnet_1d.py`

**`models/hybrid/multiscale_cnn.py`**
- **Purpose**: Multi-scale CNN processing signals at multiple resolutions
- **Key Classes**:
  - `MultiScaleCNN(BaseModel)`: Parallel branches with different kernel sizes
- **Architecture**:
  ```
  Input [B, 1, 102400]
    ↓
  ┌─────────────┬─────────────┬─────────────┐
  Branch 1      Branch 2      Branch 3      (parallel)
  k=64, s=4     k=32, s=4     k=16, s=4
  [B, 64, ...]  [B, 64, ...]  [B, 64, ...]
  └─────────────┴─────────────┴─────────────┘
    ↓ Concatenate
  [B, 192, ...]  # Fused features
    ↓ Further convolutions
  [B, 11]
  ```
- **Benefit**: Captures both fine-grained and coarse-grained patterns
- **Dependencies**: `torch.nn`, `models/base_model.py`

#### **5. Training Enhancements (3 files)**

**`training/advanced_augmentation.py`**
- **Purpose**: Stronger augmentation for deeper models
- **Key Functions**:
  - `cutmix(signal1, signal2, alpha)`: Cut-and-paste augmentation
  - `adversarial_augmentation(model, signal, epsilon)`: FGSM-based augmentation
  - `autoaugment_policy()`: Learned augmentation policy
- **CutMix**: 
  ```python
  # Cut a segment from signal2, paste into signal1
  lambda_ = np.random.beta(alpha, alpha)
  cut_length = int(lambda_ * len(signal1))
  cut_start = np.random.randint(0, len(signal1) - cut_length)
  signal1[cut_start:cut_start+cut_length] = signal2[cut_start:cut_start+cut_length]
  # Mixed label: lambda_ * y1 + (1-lambda_) * y2
  ```
- **Dependencies**: `numpy`, `torch`

**`training/knowledge_distillation.py`**
- **Purpose**: Train smaller models using larger teacher models
- **Key Classes**:
  - `DistillationTrainer(Trainer)`: Knowledge distillation training loop
- **Key Functions**:
  - `distillation_loss(student_logits, teacher_logits, targets, T, alpha)`:
    ```python
    # Soft targets from teacher
    soft_loss = KL_divergence(
        softmax(student_logits / T), 
        softmax(teacher_logits / T)
    ) * T^2
    # Hard targets (true labels)
    hard_loss = CrossEntropy(student_logits, targets)
    # Combined loss
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    ```
- **Usage**: Train ResNet-50 (teacher) → Distill to ResNet-18 (student)
- **Benefit**: Student often matches teacher accuracy with fewer parameters
- **Dependencies**: `torch.nn`, `training/trainer.py`

**`training/progressive_resizing.py`**
- **Purpose**: Train with progressively longer signals
- **Key Functions**:
  - `train_with_progressive_resizing(model, config)`:
    ```python
    # Stage 1: Short signals (fast training)
    train(model, signal_length=25600, epochs=30)
    # Stage 2: Medium signals
    train(model, signal_length=51200, epochs=20)
    # Stage 3: Full signals
    train(model, signal_length=102400, epochs=50)
    ```
- **Benefit**: Faster initial convergence, better final accuracy
- **Dependencies**: `training/trainer.py`

#### **6. Evaluation and Comparison (3 files)**

**`evaluation/architecture_comparison.py`**
- **Purpose**: Systematic comparison of all architectures
- **Key Functions**:
  - `compare_architectures(model_dict, test_loader)`: 
    ```python
    results = {}
    for name, model in model_dict.items():
        acc = evaluate(model, test_loader)
        params = count_parameters(model)
        flops = compute_flops(model)
        results[name] = {
            'accuracy': acc, 
            'params': params, 
            'flops': flops,
            'inference_time': time_per_sample
        }
    return pd.DataFrame(results).T
    ```
  - `plot_accuracy_vs_params(results)`: Pareto frontier visualization
- **Output**: Table comparing ResNet-18, ResNet-50, EfficientNet-B3, etc.
- **Dependencies**: `evaluation/evaluator.py`, `pandas`, `matplotlib`

**`evaluation/error_analysis.py`**
- **Purpose**: Deep-dive into misclassifications
- **Key Functions**:
  - `analyze_misclassifications(model, test_loader)`: 
    - Which fault pairs are confused most?
    - Are errors severity-dependent (incipient vs. severe)?
    - Do certain noise types cause more errors?
  - `find_hard_examples(model, test_loader)`: Samples with low confidence
  - `compare_errors_across_models(model_list, test_loader)`: 
    - Do different models make the same mistakes?
    - Complementary errors suggest ensemble potential
- **Dependencies**: `evaluation/evaluator.py`, `numpy`

**`evaluation/ensemble_voting.py`**
- **Purpose**: Combine predictions from multiple Phase 3 models
- **Key Functions**:
  - `soft_voting(model_predictions, weights)`: Weighted average of probabilities
  - `hard_voting(model_predictions)`: Majority vote
  - `stacking(model_predictions, meta_learner)`: Train meta-learner on predictions
- **Usage**: Ensemble ResNet-18 + ResNet-50 + EfficientNet-B3
- **Expected Gain**: +1-2% accuracy over best individual model
- **Dependencies**: `torch`, `sklearn.linear_model`

### Architecture Decisions

**1. ResNet Over Plain Deep CNN**
- **Decision**: Use residual connections for networks > 20 layers
- **Rationale**:
  - Plain deep CNNs suffer from vanishing gradients
  - ResNet skip connections enable training 50+ layer networks
  - Proven on ImageNet, adapts well to 1D signals
- **Evidence**: ResNet-18 expected to outperform 18-layer plain CNN by 2-3%

**2. EfficientNet Compound Scaling**
- **Decision**: Scale depth, width, resolution together (not independently)
- **Rationale**:
  - Deeper networks need wider channels to increase capacity
  - Longer input signals need more layers to process
  - Compound scaling found optimal by EfficientNet paper (grid search)
- **Implementation**: Use pre-defined φ values from EfficientNet paper

**3. Mobile Inverted Bottleneck (MBConv) for Efficiency**
- **Decision**: Use depthwise separable convolutions in EfficientNet
- **Rationale**:
  - 8-9× fewer parameters than standard convolution
  - Similar accuracy with much faster inference
  - Critical for deployment on edge devices (future Phase 9)
- **Trade-off**: Slightly harder to train (needs more epochs)

**4. LSTM vs. TCN for Temporal Modeling**
- **Decision**: Implement both, compare empirically
- **Rationale**:
  - LSTM is standard for sequences but slow to train
  - TCN is newer, faster, but less validated for fault diagnosis
  - Bearing fault diagnosis may benefit from long-term dependencies (LSTM) or local patterns (TCN)
- **Evaluation Metric**: Accuracy and training time

**5. Architecture Search (DARTS) as Exploration Tool**
- **Decision**: Use DARTS for one-time architecture discovery, not production
- **Rationale**:
  - Expensive (1 day GPU time)
  - May discover novel architectures for bearing fault diagnosis
  - Treat as research tool, not part of standard pipeline
- **Risk Mitigation**: Run DARTS in parallel with manual architecture design

**6. Knowledge Distillation for Compression**
- **Decision**: Train large teacher → distill to small student
- **Rationale**:
  - Large models (ResNet-50) achieve best accuracy but slow inference
  - Small models (ResNet-18) fast but lower accuracy
  - Distillation bridges gap: ResNet-18 (student) can match ResNet-34 accuracy
- **Use Case**: Deploy distilled ResNet-18 for production

### Data Flow

```
┌────────────────────────────────────────────────────────────┐
│         RESNET/EFFICIENTNET TRAINING (Phase 3)              │
└────────────────────────────────────────────────────────────┘

1. DATA LOADING (same as Phase 2)
   ┌──────────────────────────────────────────────────────┐
   │ data/cnn_dataloader.py                                │
   │  ├─ Load raw signals [B, 1, 102400]                  │
   │  ├─ Apply stronger augmentation (CutMix, etc.)       │
   │  └─ Batch for GPU                                     │
   └──────────────────────────────────────────────────────┘
                        ↓

2. MODEL SELECTION
   ┌──────────────────────────────────────────────────────┐
   │ models/model_factory.py                               │
   │  ├─ User selects: 'ResNet18_1D'                      │
   │  ├─ Instantiate model from config                    │
   │  └─ Load pretrained weights (if available)           │
   └──────────────────────────────────────────────────────┘
                        ↓

3. RESNET-18 FORWARD PASS
   ┌──────────────────────────────────────────────────────┐
   │ models/resnet/resnet_1d.py                            │
   │                                                       │
   │ Input: [B, 1, 102400]                                 │
   │  ↓                                                    │
   │ Conv1 + MaxPool: [B, 64, 6400]                       │
   │  ↓                                                    │
   │ Layer1 (2× BasicBlock): [B, 64, 6400]               │
   │  ├─ Block1: Conv-BN-ReLU-Conv-BN + Skip             │
   │  └─ Block2: Conv-BN-ReLU-Conv-BN + Skip             │
   │  ↓                                                    │
   │ Layer2 (2× BasicBlock, stride=2): [B, 128, 3200]    │
   │  ↓                                                    │
   │ Layer3 (2× BasicBlock, stride=2): [B, 256, 1600]    │
   │  ↓                                                    │
   │ Layer4 (2× BasicBlock, stride=2): [B, 512, 800]     │
   │  ↓                                                    │
   │ AdaptiveAvgPool: [B, 512]                            │
   │  ↓                                                    │
   │ FC: [B, 11]                                           │
   └──────────────────────────────────────────────────────┘
                        ↓

4. TRAINING LOOP (same as Phase 2, with enhancements)
   ┌──────────────────────────────────────────────────────┐
   │ training/cnn_trainer.py                               │
   │  ├─ Forward pass                                      │
   │  ├─ Compute loss (label smoothing + CutMix)         │
   │  ├─ Backward pass (mixed precision)                  │
   │  ├─ Optimizer step (AdamW)                           │
   │  └─ LR scheduler (cosine annealing with warmup)     │
   └──────────────────────────────────────────────────────┘
                        ↓

5. ARCHITECTURE COMPARISON
   ┌──────────────────────────────────────────────────────┐
   │ evaluation/architecture_comparison.py                 │
   │  ├─ Train: ResNet-18, ResNet-34, ResNet-50          │
   │  ├─ Train: EfficientNet-B0, B3                       │
   │  ├─ Train: CNN-LSTM, CNN-TCN                         │
   │  └─ Compare: accuracy, params, FLOPs, inference time │
   │         ↓                                             │
   │ Output: Comparison table + Pareto frontier plot      │
   └──────────────────────────────────────────────────────┘
                        ↓

6. ENSEMBLE CONSTRUCTION
   ┌──────────────────────────────────────────────────────┐
   │ evaluation/ensemble_voting.py                         │
   │  ├─ Load best 3 models (e.g., ResNet-50, EffNet-B3, │
   │  │  WideResNet)                                       │
   │  ├─ Soft voting: Avg(prob1, prob2, prob3)           │
   │  └─ Evaluate ensemble on test set                    │
   │         ↓                                             │
   │ Output: Ensemble accuracy (target: 97-98%)           │
   └──────────────────────────────────────────────────────┘
```

### Integration Points

**1. With Phase 2 (Baseline CNN)**
- **Comparison**: ResNet-18 expected to outperform Phase 2 CNN by 1-2%
- **Shared Code**: Use same `cnn_trainer.py`, `cnn_losses.py`
- **Difference**: Phase 3 models are deeper with residual connections

**2. With Phase 1 (Classical ML)**
- **Benchmark**: Target is to exceed Random Forest's 95.33% accuracy
- **Complementarity**: Classical ML + Deep Ensemble (Phase 8)

**3. With Phase 4 (Transformer)**
- **Feature Extraction**: ResNet backbone can extract features for Transformer
- **Comparison**: ResNet (convolutional) vs. Transformer (attention)

**4. With Phase 6 (Physics-Informed)**
- **Backbone**: ResNet-18 serves as feature extractor for PINN
- **Hybrid**: Combine ResNet features with physics-based features

**5. With Phase 9 (Edge Deployment)**
- **Model Selection**: EfficientNet-B0 or distilled ResNet-18 for edge devices
- **Optimization**: Quantization, pruning of Phase 3 models

### Testing Strategy

**1. Unit Tests**

**`tests/test_resnet.py`**
```python
def test_resnet18_forward():
    """Test ResNet-18 forward pass."""
    model = ResNet1D(num_layers=18, num_classes=11)
    x = torch.randn(2, 1, 102400)
    output = model(x)
    assert output.shape == (2, 11)

def test_residual_connection():
    """Verify skip connection adds identity."""
    block = BasicBlock1D(in_channels=64, out_channels=64)
    x = torch.randn(2, 64, 1000)
    output = block(x)
    # Output should be different from input (conv applied)
    # but gradients should flow through skip connection
    assert output.shape == x.shape
```

**`tests/test_efficientnet.py`**
```python
def test_mbconv_block():
    """Test mobile inverted bottleneck block."""
    block = MBConvBlock(in_channels=32, out_channels=64, expand_ratio=6)
    x = torch.randn(2, 32, 1000)
    output = block(x)
    assert output.shape == (2, 64, 1000)  # Assuming stride=1
```

**2. Architecture Validation Tests**

**`tests/test_architecture_search.py`**
```python
def test_darts_search():
    """Test DARTS architecture search runs."""
    # Small dataset for fast test
    train_loader, val_loader = create_dummy_loaders(n_samples=50)
    
    # Run 2 epochs of search
    nas_controller = NASController()
    best_arch = nas_controller.search(
        train_loader, val_loader, epochs=2
    )
    
    # Check valid architecture returned
    assert best_arch is not None
    assert 'operations' in best_arch
```

**3. Comparison Tests**

**`tests/test_model_comparison.py`**
```python
def test_resnet_vs_plain_cnn():
    """ResNet should outperform plain CNN of same depth."""
    test_loader = load_standard_test_set()
    
    # Train both
    resnet18 = train_model('ResNet18_1D', config)
    plain_cnn_18layer = train_model('PlainCNN18Layer', config)
    
    # Evaluate
    resnet_acc = evaluate(resnet18, test_loader)
    plain_acc = evaluate(plain_cnn_18layer, test_loader)
    
    # ResNet should be better (allowing 1% margin for randomness)
    assert resnet_acc >= plain_acc - 0.01
```

**4. Ensemble Tests**

**`tests/test_ensemble.py`**
```python
def test_ensemble_improves_accuracy():
    """Ensemble should outperform individual models."""
    test_loader = load_standard_test_set()
    
    # Individual models
    model1 = load_trained_model('ResNet18')
    model2 = load_trained_model('ResNet50')
    model3 = load_trained_model('EfficientNetB3')
    
    acc1 = evaluate(model1, test_loader)
    acc2 = evaluate(model2, test_loader)
    acc3 = evaluate(model3, test_loader)
    
    best_individual = max(acc1, acc2, acc3)
    
    # Ensemble
    ensemble_acc = evaluate_ensemble([model1, model2, model3], test_loader)
    
    # Ensemble should be at least as good as best individual
    assert ensemble_acc >= best_individual - 0.005  # Tiny margin
```

### Acceptance Criteria

**Phase 3 Complete When:**

✅ **ResNet models train successfully**
- ResNet-18, ResNet-34, ResNet-50 converge without gradient issues
- Residual connections verified (gradients flow through skip connections)
- Deeper models (ResNet-50) achieve higher accuracy than shallower (ResNet-18)

✅ **Achieves target accuracy improvements**
- **ResNet-18**: 95-96% test accuracy (1-2% over Phase 2 CNN)
- **ResNet-50 or EfficientNet-B3**: 96-97% test accuracy
- **Ensemble**: 97-98% test accuracy (stretch goal)

✅ **EfficientNet models validated**
- EfficientNet-B0, B3, B7 train successfully
- Compound scaling improves accuracy as φ increases
- Parameter efficiency verified (EfficientNet-B0 < 2M params, matches ResNet-18 accuracy)

✅ **Hybrid models functional**
- CNN-LSTM and CNN-TCN train without errors
- Performance compared to pure CNN (may be similar or slightly better)

✅ **Architecture search completes**
- DARTS discovers non-trivial architecture (not just stacking same block)
- Discovered architecture achieves competitive accuracy (within 2% of ResNet-18)
- Search process documented (which operations selected, why)

✅ **Comprehensive comparison documented**
- Table comparing 8+ architectures: accuracy, parameters, FLOPs, inference time
- Pareto frontier plot (accuracy vs. efficiency)
- Error analysis: Which faults does ResNet handle better than Phase 2 CNN?

✅ **Knowledge distillation working**
- Student model (ResNet-18) trained with teacher (ResNet-50)
- Student achieves 95-96% accuracy (within 1% of teacher's 97%)
- Student is 2-3× faster inference than teacher

✅ **Ensemble validated**
- 3-model ensemble (e.g., ResNet-50 + EfficientNet-B3 + WideResNet)
- Ensemble accuracy > best individual model accuracy
- Soft voting outperforms hard voting

✅ **Robustness maintained or improved**
- Sensor noise test: ≤ 20% accuracy drop (same or better than Phase 2)
- Adversarial robustness: ≤ 10% drop under FGSM (ε=0.1)
- Deeper models are not more brittle

✅ **MLflow tracking complete**
- All architecture experiments logged
- Comparison dashboard showing all models
- Best models saved as artifacts

✅ **Documentation complete**
- README explaining ResNet adaptation to 1D signals
- Jupyter notebook: "Training ResNet-18 for Bearing Fault Diagnosis"
- Architecture comparison report (PDF)

### Estimated Effort

**Time Breakdown:**
- ResNet architecture (5 files): 4 days
  - `resnet_1d.py`: 1 day
  - `residual_blocks.py`: 1 day
  - Variants (SE-ResNet, WideResNet): 1 day
  - Debugging gradient flow: 1 day

- EfficientNet (4 files): 4 days
  - `efficientnet_1d.py`: 1.5 days (complex scaling logic)
  - `mbconv_block.py`: 1 day
  - Variants (B0-B7): 1 day
  - Testing depthwise separable conv: 0.5 days

- Architecture search (4 files): 5 days
  - `neural_architecture_search.py` (DARTS): 2 days (complex)
  - `darts_trainer.py`: 1 day
  - Running search: 1 day GPU time + 1 day analysis
  - Deriving final architecture: 1 day

- Hybrid models (3 files): 3 days
  - CNN-LSTM: 1 day
  - CNN-TCN: 1 day
  - Multi-scale CNN: 1 day

- Training enhancements (3 files): 2 days
  - CutMix, adversarial augmentation: 1 day
  - Knowledge distillation: 1 day

- Evaluation and comparison (3 files): 3 days
  - `architecture_comparison.py`: 1 day
  - `error_analysis.py`: 1 day
  - `ensemble_voting.py`: 1 day

- Training all models: 5 days
  - Train ResNet-18, 34, 50: 1 day each = 3 days
  - Train EfficientNet variants: 1 day
  - Train hybrid models: 1 day

- Testing (unit, comparison, ensemble): 3 days
- Documentation: 2 days
- Buffer for debugging: 3 days

**Total: ~34 days (1.7 months) for Phase 3**

**Complexity**: ⭐⭐⭐⭐⭐ (Very High)
- Many architectures to implement and train
- Architecture search (DARTS) is algorithmically complex
- Ensemble construction requires careful tuning
- Comparison study requires systematic evaluation

**Dependencies**: Phase 0 (data), Phase 2 (baseline CNN)

**Risk**: High
- Architecture search may not discover better architectures than manual design
- Ensemble may not improve over best individual model
- Training many large models is time/GPU-intensive

---
