
## **PHASE 4: Transformer Architecture for Time-Series**

### Phase Objective
Implement Transformer encoder architecture adapted for vibration signal classification, leveraging self-attention mechanisms to capture long-range temporal dependencies that CNNs may miss. Explore patch-based embeddings and compare attention patterns to ResNet activations. Target: Match or exceed ResNet-50 accuracy (96-97%) while providing interpretable attention maps showing which time regions drive fault classification.

### Complete File List (20 files)

#### **1. Transformer Core (5 files)**

**`models/transformer/signal_transformer.py`**
- **Purpose**: Main Transformer encoder for fault diagnosis
- **Key Classes**:
  - `SignalTransformer(BaseModel)`: Complete Transformer architecture
  - `PatchEmbedding(nn.Module)`: Convert signal to sequence of patches
  - `PositionalEncoding(nn.Module)`: Learnable or sinusoidal position embeddings
- **Architecture**:
  ```
  Input: [B, 1, 102400]
  
  1. Patch Embedding:
     ├─ Divide signal into patches: 102400 / 512 = 200 patches
     ├─ Each patch: [512] samples
     ├─ Linear projection: 512 → d_model (e.g., 256)
     └─ Output: [B, 200, 256]  # (batch, seq_len, embed_dim)
  
  2. Positional Encoding:
     ├─ Add learnable position embeddings
     └─ Output: [B, 200, 256]
  
  3. Transformer Encoder (6 layers):
     For each layer:
       ├─ Multi-Head Self-Attention (8 heads)
       │   ├─ Q, K, V = Linear(x)
       │   ├─ Attention(Q, K, V) = softmax(QK^T / √d_k) V
       │   └─ Concatenate heads, project
       ├─ Add & Norm (residual + LayerNorm)
       ├─ Feed-Forward Network (d_model → 4*d_model → d_model)
       └─ Add & Norm
     Output: [B, 200, 256]
  
  4. Classification Head:
     ├─ Global average pooling over sequence: [B, 200, 256] → [B, 256]
     └─ FC: 256 → 11
  ```
- **Key Functions**:
  - `forward(x)`: Full forward pass
  - `get_attention_weights(x, layer_idx)`: Extract attention maps for interpretability
- **Hyperparameters**:
  - d_model: 256 (embedding dimension)
  - n_heads: 8 (multi-head attention)
  - n_layers: 6 (transformer blocks)
  - d_ff: 1024 (feedforward hidden dim, 4 × d_model)
  - dropout: 0.1
- **Parameters**: ~5M (comparable to ResNet-34)
- **Dependencies**: `torch.nn`, `models/base_model.py`

**`models/transformer/patch_embedding.py`**
- **Purpose**: Convert 1D signal to sequence of patch embeddings
- **Key Classes**:
  - `PatchEmbedding1D(nn.Module)`: Patch extraction + projection
- **Patching Strategies**:
  ```python
  # Strategy 1: Non-overlapping patches
  patches = signal.reshape(B, -1, patch_size)  # [B, n_patches, patch_size]
  embeddings = linear(patches)  # [B, n_patches, d_model]
  
  # Strategy 2: Overlapping patches (stride < patch_size)
  patches = unfold(signal, kernel_size=patch_size, stride=stride)
  embeddings = linear(patches)
  
  # Strategy 3: Convolutional embedding (1D conv with large kernel)
  embeddings = Conv1D(in_channels=1, out_channels=d_model, 
                      kernel_size=patch_size, stride=patch_size)(signal)
  ```
- **Key Functions**:
  - `forward(x)`: [B, 1, T] → [B, n_patches, d_model]
- **Recommended**: patch_size=512, stride=512 (non-overlapping)
- **Dependencies**: `torch.nn`

**`models/transformer/positional_encoding.py`**
- **Purpose**: Add position information to patch embeddings
- **Key Classes**:
  - `LearnablePositionalEncoding(nn.Module)`: Trainable embeddings
  - `SinusoidalPositionalEncoding(nn.Module)`: Fixed sin/cos encoding (Vaswani et al.)
  - `RelativePositionalEncoding(nn.Module)`: Relative positions (more robust)
- **Sinusoidal Formula** (original Transformer):
  ```python
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```
- **Key Functions**:
  - `forward(x)`: Add positional encoding to input
- **Recommendation**: Use learnable for flexibility
- **Dependencies**: `torch.nn`

**`models/transformer/multi_head_attention.py`**
- **Purpose**: Multi-head self-attention mechanism
- **Key Classes**:
  - `MultiHeadAttention(nn.Module)`: Parallel attention heads
- **Implementation**:
  ```python
  class MultiHeadAttention(nn.Module):
      def __init__(self, d_model, n_heads):
          self.d_k = d_model // n_heads
          self.n_heads = n_heads
          self.q_linear = nn.Linear(d_model, d_model)
          self.k_linear = nn.Linear(d_model, d_model)
          self.v_linear = nn.Linear(d_model, d_model)
          self.out_linear = nn.Linear(d_model, d_model)
      
      def forward(self, x, mask=None):
          B, seq_len, d_model = x.shape
          
          # Linear projections and split into heads
          Q = self.q_linear(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
          K = self.k_linear(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
          V = self.v_linear(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
          # Q, K, V: [B, n_heads, seq_len, d_k]
          
          # Scaled dot-product attention
          scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
          # scores: [B, n_heads, seq_len, seq_len]
          
          if mask is not None:
              scores = scores.masked_fill(mask == 0, -1e9)
          
          attn_weights = F.softmax(scores, dim=-1)  # Attention probabilities
          attn_output = torch.matmul(attn_weights, V)  # [B, n_heads, seq_len, d_k]
          
          # Concatenate heads
          attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, d_model)
          output = self.out_linear(attn_output)
          
          return output, attn_weights  # Return weights for visualization
  ```
- **Key Functions**:
  - `forward(x, mask)`: Compute attention output and weights
- **Dependencies**: `torch.nn`, `torch.nn.functional`

**`models/transformer/transformer_encoder_layer.py`**
- **Purpose**: Single Transformer encoder block
- **Key Classes**:
  - `TransformerEncoderLayer(nn.Module)`: Attention + FFN + residual
- **Implementation**:
  ```python
  class TransformerEncoderLayer(nn.Module):
      def __init__(self, d_model, n_heads, d_ff, dropout):
          self.attention = MultiHeadAttention(d_model, n_heads)
          self.norm1 = nn.LayerNorm(d_model)
          self.ffn = nn.Sequential(
              nn.Linear(d_model, d_ff),
              nn.GELU(),
              nn.Dropout(dropout),
              nn.Linear(d_ff, d_model),
              nn.Dropout(dropout)
          )
          self.norm2 = nn.LayerNorm(d_model)
          self.dropout = nn.Dropout(dropout)
      
      def forward(self, x):
          # Multi-head attention with residual
          attn_output, attn_weights = self.attention(x)
          x = self.norm1(x + self.dropout(attn_output))
          
          # Feed-forward network with residual
          ffn_output = self.ffn(x)
          x = self.norm2(x + ffn_output)
          
          return x, attn_weights
  ```
- **Key Functions**:
  - `forward(x)`: Apply full encoder layer
- **Dependencies**: `multi_head_attention.py`

#### **2. Transformer Variants (4 files)**

**`models/transformer/vision_transformer_1d.py`**
- **Purpose**: Vision Transformer (ViT) adapted for 1D signals
- **Key Classes**:
  - `ViT1D(SignalTransformer)`: ViT with cls token
- **Architecture Modification**:
  ```python
  # Standard Transformer:
  Input patches → Transformer → Global avg pool → FC
  
  # ViT:
  [CLS] token + Input patches → Transformer → [CLS] output → FC
  ```
  - Prepend learnable [CLS] token to sequence
  - Use [CLS] token output for classification (instead of avg pooling)
- **Benefit**: [CLS] token learns task-relevant global representation
- **Dependencies**: `signal_transformer.py`

**`models/transformer/performer.py`**
- **Purpose**: Performer (efficient attention) for long signals
- **Key Classes**:
  - `Performer1D(SignalTransformer)`: Linear-complexity attention
- **Problem with Standard Attention**:
  - Attention complexity: O(n²) where n = sequence length
  - For 200 patches: 200×200 = 40,000 attention computations
  - Becomes prohibitive for longer signals
- **Performer Solution**:
  - Approximate attention with random features
  - Complexity: O(n) (linear in sequence length)
  - Slight accuracy trade-off (~1%) for 10× speedup
- **Key Functions**:
  - `forward(x)`: Same interface as standard Transformer
- **Dependencies**: `torch.nn`, `performer-pytorch` library

**`models/transformer/temporal_fusion_transformer.py`**
- **Purpose**: Temporal Fusion Transformer (TFT) with gating mechanisms
- **Key Classes**:
  - `TFT1D(SignalTransformer)`: TFT adapted for fault diagnosis
  - `GatedResidualNetwork(nn.Module)`: Gating for feature selection
- **Additions to Standard Transformer**:
  - Variable selection: Learn which patches are important
  - Gated residual connections: Adaptive skip connections
  - Static covariate encoder: Incorporate operating condition metadata
- **Use Case**: When metadata available (load, speed, temperature)
- **Dependencies**: `signal_transformer.py`

**`models/transformer/informer.py`**
- **Purpose**: Informer (efficient long-sequence Transformer)
- **Key Classes**:
  - `Informer1D(SignalTransformer)`: ProbSparse attention
- **ProbSparse Attention**:
  - Select top-k most important queries (not all queries attend to all keys)
  - Complexity: O(n log n)
  - Better for very long signals (> 50,000 samples)
- **Dependencies**: `signal_transformer.py`

#### **3. Training Enhancements for Transformer (3 files)**

**`training/transformer_trainer.py`**
- **Purpose**: Transformer-specific training loop
- **Key Classes**:
  - `TransformerTrainer(Trainer)`: Extends base trainer
- **Transformer-Specific Considerations**:
  ```python
  # Learning rate warmup (critical for Transformers)
  def lr_schedule(epoch):
      if epoch < warmup_epochs:
          return (epoch + 1) / warmup_epochs  # Linear warmup
      else:
          return cosine_annealing(epoch - warmup_epochs)
  
  # Gradient clipping (prevent exploding gradients)
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  
  # Label smoothing (Transformers benefit more than CNNs)
  loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
  ```
- **Key Functions**:
  - `train_epoch(dataloader)`: Training with warmup schedule
  - `_update_lr(epoch)`: Custom LR schedule
- **Dependencies**: `training/trainer.py`

**`training/transformer_augmentation.py`**
- **Purpose**: Augmentation specific to patch-based models
- **Key Functions**:
  - `patch_dropout(patches, drop_prob=0.1)`: Randomly drop patches (regularization)
  - `patch_mixup(patches1, patches2, alpha)`: Mix patches from two signals
  - `temporal_shift_patches(patches, shift)`: Shift patches temporally
- **Patch Dropout**: Similar to DropBlock in CNNs
  ```python
  # Randomly drop 10% of patches during training
  mask = torch.rand(B, n_patches, 1) > drop_prob
  patches = patches * mask
  ```
- **Dependencies**: `numpy`, `torch`

**`training/transformer_schedulers.py`**
- **Purpose**: Learning rate schedules tailored for Transformers
- **Key Functions**:
  - `create_warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs)`:
    ```python
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
    ```
  - `create_noam_schedule(optimizer, d_model, warmup_steps)`: Original Transformer schedule
- **Dependencies**: `torch.optim.lr_scheduler`

#### **4. Interpretability and Visualization (4 files)**

**`evaluation/attention_visualization.py`**
- **Purpose**: Visualize attention maps to understand model decisions
- **Key Functions**:
  - `plot_attention_heatmap(attention_weights, signal, patch_size)`:
    ```python
    # attention_weights: [n_heads, n_patches, n_patches]
    # Average over heads
    avg_attention = attention_weights.mean(dim=0)  # [n_patches, n_patches]
    
    # For each patch, show which other patches it attends to
    for patch_idx in range(n_patches):
        attn_scores = avg_attention[patch_idx, :]  # Attention from patch_idx
        # Visualize as heatmap overlaid on signal
        plt.plot(signal)
        for i, score in enumerate(attn_scores):
            start_sample = i * patch_size
            end_sample = (i + 1) * patch_size
            plt.axvspan(start_sample, end_sample, alpha=score, color='red')
    ```
  - `plot_attention_rollout(model, signal)`: Aggregate attention across layers
  - `find_most_attended_patches(attention_weights)`: Identify key time regions
- **Use Case**: 
  - Explain why model classified signal as "misalignment"
  - Show attention focuses on 2X harmonic regions (expected for misalignment)
- **Dependencies**: `matplotlib`, `torch`

**`evaluation/transformer_interpretability.py`**
- **Purpose**: Attribution methods for Transformers
- **Key Functions**:
  - `attention_rollout(model, signal, target_class)`: Aggregate attention from all layers
    ```python
    # Recursively multiply attention matrices
    rollout = attention_layer1 @ attention_layer2 @ ... @ attention_layer6
    # rollout[i, j] = importance of patch j for patch i
    ```
  - `attention_flow(model, signal)`: Visualize information flow through layers
  - `patch_attribution(model, signal, target_class)`: Which patches contribute to prediction
- **Dependencies**: `torch`, `attention_visualization.py`

**`evaluation/compare_attention_vs_gradcam.py`**
- **Purpose**: Compare Transformer attention to CNN Grad-CAM
- **Key Functions**:
  - `compare_interpretability(transformer_model, cnn_model, signal, label)`:
    ```python
    # Transformer: Extract attention weights
    attn_weights = transformer_model.get_attention_weights(signal, layer_idx=5)
    attn_importance = attn_weights.mean(dim=0)  # Average over heads
    
    # CNN: Compute Grad-CAM
    gradcam_heatmap = generate_gradcam(cnn_model, signal, label)
    
    # Visualize side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(signal);  ax1.imshow(attn_importance, alpha=0.5)  # Attention
    ax2.plot(signal);  ax2.imshow(gradcam_heatmap, alpha=0.5)  # Grad-CAM
    ```
  - `quantify_agreement(attn_map, gradcam_map)`: Correlation between methods
- **Expected Finding**: Both methods should highlight similar time regions
- **Dependencies**: `attention_visualization.py`, `evaluation/cnn_interpretability.py`

**`visualization/attention_dashboard.py`**
- **Purpose**: Interactive dashboard for attention exploration
- **Key Classes**:
  - `AttentionDashboard`: Streamlit app for attention visualization
- **Features**:
  - Upload signal → see attention heatmap
  - Slider to select layer (layer 1-6)
  - Hover over patch → see attention distribution
  - Compare attention for different fault types
- **Dependencies**: `streamlit`, `plotly`, `attention_visualization.py`

#### **5. Hybrid CNN-Transformer (2 files)**

**`models/hybrid/cnn_transformer.py`**
- **Purpose**: Combine CNN feature extraction with Transformer
- **Key Classes**:
  - `CNNTransformer(BaseModel)`: CNN backbone + Transformer encoder
- **Architecture**:
  ```
  Input: [B, 1, 102400]
    ↓
  CNN Backbone (ResNet-18, remove final FC):
    ├─ Conv layers
    └─ Output: [B, 512, 800]  # Feature maps
    ↓
  Permute: [B, 512, 800] → [B, 800, 512]  # (batch, seq_len, channels)
    ↓
  Transformer Encoder (4 layers, d_model=512):
    ├─ Self-attention over spatial locations
    └─ Output: [B, 800, 512]
    ↓
  Global Avg Pool: [B, 800, 512] → [B, 512]
    ↓
  FC: [B, 512] → [B, 11]
  ```
- **Rationale**:
  - CNN: Inductive bias for local patterns (good for low-level features)
  - Transformer: Model long-range dependencies (good for high-level reasoning)
  - Combines strengths of both
- **Expected Performance**: 97-98% accuracy (best of both worlds)
- **Dependencies**: `models/resnet/resnet_1d.py`, `models/transformer/signal_transformer.py`

**`models/hybrid/perceiver.py`**
- **Purpose**: Perceiver architecture (cross-attention between latents and signal)
- **Key Classes**:
  - `Perceiver1D(BaseModel)`: Perceiver for signals
- **Architecture**:
  ```
  Latent array: [n_latents, d_latent]  # e.g., [64, 512]
  Input signal patches: [n_patches, d_model]  # [200, 256]
  
  Cross-Attention:
    Q = Latents
    K, V = Input patches
    Output: [n_latents, d_latent]  # Bottleneck representation
  
  Self-Attention (Transformer):
    Process latents: [n_latents, d_latent]
  
  Classification:
    Pool latents → FC → [11]
  ```
- **Benefit**: O(n_latents × n_patches) instead of O(n_patches²)
- **Use Case**: Very long signals (> 100,000 samples)
- **Dependencies**: `torch.nn`, `models/base_model.py`

#### **6. Evaluation (2 files)**

**`evaluation/transformer_evaluator.py`**
- **Purpose**: Evaluate Transformer models
- **Key Classes**:
  - `TransformerEvaluator(ModelEvaluator)`: Extends base evaluator
- **Additional Metrics**:
  - Attention entropy: How focused is attention? (low entropy = focused)
  - Patch importance scores: Which patches are most critical?
  - Layer-wise attention analysis: How does attention evolve through layers?
- **Key Functions**:
  - `evaluate(model, test_loader)`: Standard evaluation
  - `analyze_attention_patterns(model, test_loader)`: Statistical attention analysis
- **Dependencies**: `evaluation/evaluator.py`, `attention_visualization.py`

**`evaluation/transformer_vs_resnet.py`**
- **Purpose**: Systematic comparison Transformer vs. ResNet
- **Key Functions**:
  - `compare_architectures(transformer, resnet, test_loader)`:
    ```python
    results = {
        'Accuracy': {
            'Transformer': transformer_acc,
            'ResNet': resnet_acc
        },
        'Per-class F1': {
            'Transformer': transformer_f1_per_class,
            'ResNet': resnet_f1_per_class
        },
        'Inference Time': {
            'Transformer': transformer_time,
            'ResNet': resnet_time
        },
        'Parameters': {
            'Transformer': count_parameters(transformer),
            'ResNet': count_parameters(resnet)
        }
    }
    return pd.DataFrame(results)
    ```
  - `plot_comparison_radar(results)`: Radar chart showing trade-offs
- **Expected Findings**:
  - Transformer: Slightly better on mixed faults (long-range dependencies)
  - ResNet: Slightly faster inference (no attention computation)
  - Similar accuracy (~0.5% difference)
- **Dependencies**: `evaluation/evaluator.py`, `pandas`, `matplotlib`

### Architecture Decisions

**1. Patch Size Selection**
- **Decision**: Use patch_size=512 (non-overlapping)
- **Rationale**:
  - 102,400 samples / 512 = 200 patches (manageable sequence length)
  - Each patch covers ~25ms of signal at 20.48 kHz (captures fault cycles)
  - Non-overlapping avoids redundancy, faster training
- **Alternative**: Overlapping patches (stride=256) for 400 patches (more detail, slower)

**2. Positional Encoding: Learnable vs. Sinusoidal**
- **Decision**: Use learnable positional embeddings
- **Rationale**:
  - Signals have irregular temporal patterns (not natural language)
  - Learnable embeddings can adapt to fault-specific patterns
  - Minimal parameter overhead (200 patches × 256 dim = 51k params)
- **Fallback**: Sinusoidal if overfitting occurs

**3. Number of Transformer Layers**
- **Decision**: 6 layers (standard Transformer depth)
- **Rationale**:
  - Deeper than 6 layers risks overfitting on 1,430 samples
  - Shallower than 6 layers may not capture complex patterns
  - 6 layers proven effective in ViT, BERT
- **Experiment**: Try 4, 6, 8 layers, compare validation accuracy

**4. Multi-Head Attention: 8 Heads**
- **Decision**: Use 8 attention heads
- **Rationale**:
  - d_model=256 / 8 heads = 32 dim per head (not too small)
  - Multiple heads capture different temporal relationships
  - Standard in Transformer literature
- **Constraint**: n_heads must divide d_model evenly

**5. Classification Token vs. Global Average Pooling**
- **Decision**: Start with global average pooling, experiment with [CLS] token
- **Rationale**:
  - Global avg pool is simpler, fewer hyperparameters
  - [CLS] token (ViT-style) may be better but requires tuning
  - Easy to swap implementations, test both
- **Evaluation**: Compare accuracy, use better one

**6. Warmup Schedule (Critical for Transformers)**
- **Decision**: 5-epoch linear warmup, then cosine annealing
- **Rationale**:
  - Transformers sensitive to learning rate at start of training
  - Without warmup: training diverges or converges slowly
  - Warmup stabilizes gradients in early epochs
- **Evidence**: Required in original "Attention is All You Need" paper

### Data Flow

```
┌────────────────────────────────────────────────────────────┐
│         TRANSFORMER TRAINING PIPELINE (Phase 4)             │
└────────────────────────────────────────────────────────────┘

1. DATA LOADING (same as Phases 2-3)
   ┌──────────────────────────────────────────────────────┐
   │ data/cnn_dataloader.py                                │
   │  └─ Load raw signals [B, 1, 102400]                  │
   └──────────────────────────────────────────────────────┘
                        ↓

2. PATCH EMBEDDING
   ┌──────────────────────────────────────────────────────┐
   │ models/transformer/patch_embedding.py                 │
   │                                                       │
   │ Input: [B, 1, 102400]                                 │
   │  ↓                                                    │
   │ Reshape: [B, 200, 512]  # 200 patches, 512 samples each
   │  ↓                                                    │
   │ Linear projection: [B, 200, 512] → [B, 200, 256]    │
   │         (d_model=256)                                 │
   │  ↓                                                    │
   │ Add positional encoding: [B, 200, 256]               │
   │         (learnable embeddings)                        │
   └──────────────────────────────────────────────────────┘
                        ↓

3. TRANSFORMER ENCODER (6 layers)
   ┌──────────────────────────────────────────────────────┐
   │ models/transformer/signal_transformer.py              │
   │                                                       │
   │ For layer in [1..6]:                                  │
   │   ┌────────────────────────────────────────────────┐ │
   │   │ Multi-Head Self-Attention (8 heads):          │ │
   │   │  ├─ Q, K, V = Linear(x)                       │ │
   │   │  ├─ Attention = softmax(QK^T / √32) V        │ │
   │   │  └─ Output: [B, 200, 256]                     │ │
   │   │                                                 │ │
   │   │ Add & Norm: x + Attention(x), LayerNorm       │ │
   │   │                                                 │ │
   │   │ Feed-Forward:                                  │ │
   │   │  ├─ FC: 256 → 1024 → 256                     │ │
   │   │  └─ GELU, Dropout                             │ │
   │   │                                                 │ │
   │   │ Add & Norm: x + FFN(x), LayerNorm             │ │
   │   └────────────────────────────────────────────────┘ │
   │ Output after 6 layers: [B, 200, 256]                 │
   └──────────────────────────────────────────────────────┘
                        ↓

4. CLASSIFICATION HEAD
   ┌──────────────────────────────────────────────────────┐
   │ Global Average Pool:                                  │
   │   [B, 200, 256] → [B, 256]  # Average over patches  │
   │         ↓                                             │
   │ Fully Connected:                                      │
   │   [B, 256] → [B, 11]  # Class logits                 │
   └──────────────────────────────────────────────────────┘
                        ↓

5. ATTENTION VISUALIZATION (during evaluation)
   ┌──────────────────────────────────────────────────────┐
   │ evaluation/attention_visualization.py                 │
   │                                                       │
   │ Extract attention weights from each layer:            │
   │   attention_weights: [B, n_heads, n_patches, n_patches]
   │         ↓                                             │
   │ Average over heads: [B, n_patches, n_patches]        │
   │         ↓                                             │
   │ Visualize as heatmap:                                 │
   │   - Rows: Query patches                               │
   │   - Columns: Key patches                              │
   │   - Color: Attention score                            │
   │         ↓                                             │
   │ Interpret: Which time regions are important for       │
   │            classifying this signal?                   │
   └──────────────────────────────────────────────────────┘
```

### Integration Points

**1. With Phase 3 (ResNet)**
- **Comparison**: Transformer vs. ResNet-50 accuracy, interpretability
- **Ensemble**: Combine Transformer + ResNet predictions (Phase 8)
- **Hybrid**: Use ResNet as CNN backbone in CNN-Transformer model

**2. With Phase 2 (Baseline CNN)**
- **Progression**: Plain CNN → ResNet → Transformer (increasing complexity)
- **Benchmark**: Transformer should match ResNet-50 (96-97% accuracy)

**3. With Phase 5 (Time-Frequency Analysis)**
- **Input**: Transformer can process spectrogram patches (2D patches flattened to 1D)
- **Comparison**: Raw signal Transformer vs. Spectrogram Transformer

**4. With Phase 6 (Physics-Informed)**
- **Attention as Physics**: Attention weights can be constrained to focus on fault-relevant frequencies
- **Hybrid**: Physics features + Transformer learned features

**5. With Phase 7 (XAI)**
- **Interpretability**: Attention visualization is core XAI method
- **Comparison**: Attention vs. SHAP vs. Grad-CAM

### Testing Strategy

**1. Unit Tests**

**`tests/test_transformer.py`**
```python
def test_patch_embedding():
    """Test patch embedding converts signal to patches correctly."""
    patch_emb = PatchEmbedding1D(patch_size=512, d_model=256)
    x = torch.randn(2, 1, 102400)
    patches = patch_emb(x)
    assert patches.shape == (2, 200, 256)  # 200 patches, 256-dim embeddings

def test_multi_head_attention():
    """Test multi-head attention output shape."""
    attn = MultiHeadAttention(d_model=256, n_heads=8)
    x = torch.randn(2, 200, 256)  # (batch, seq_len, d_model)
    output, attn_weights = attn(x)
    assert output.shape == (2, 200, 256)
    assert attn_weights.shape == (2, 8, 200, 200)  # (batch, heads, seq_len, seq_len)

def test_transformer_forward():
    """Test full Transformer forward pass."""
    model = SignalTransformer(d_model=256, n_heads=8, n_layers=6)
    x = torch.randn(2, 1, 102400)
    output = model(x)
    assert output.shape == (2, 11)  # 11 classes
```

**2. Attention Validation Tests**

**`tests/test_attention_visualization.py`**
```python
def test_attention_weights_sum_to_one():
    """Attention weights should sum to 1 (valid probability distribution)."""
    model = SignalTransformer()
    x = torch.randn(1, 1, 102400)
    _ = model(x)
    attn_weights = model.get_attention_weights(x, layer_idx=0)
    
    # Sum over key dimension should be 1
    attn_sum = attn_weights.sum(dim=-1)
    torch.testing.assert_allclose(attn_sum, torch.ones_like(attn_sum), rtol=1e-5)

def test_attention_focuses_on_relevant_regions():
    """For known fault signal, attention should focus on expected regions."""
    # Load misalignment signal (known to have strong 2X harmonic)
    signal = load_test_signal('misalignment_moderate.mat')
    model = load_trained_transformer()
    
    attn_weights = model.get_attention_weights(signal, layer_idx=5)
    important_patches = find_most_attended_patches(attn_weights, top_k=10)
    
    # Check if important patches correspond to 2X harmonic regions
    # (Requires domain knowledge of where 2X harmonic appears in time domain)
    # This is a qualitative test - manual inspection required
    plot_signal_with_attention(signal, important_patches)
```

**3. Comparison Tests**

**`tests/test_transformer_vs_resnet.py`**
```python
def test_transformer_matches_resnet_accuracy():
    """Transformer should achieve similar accuracy to ResNet-50."""
    test_loader = load_standard_test_set()
    
    transformer = load_trained_model('SignalTransformer')
    resnet50 = load_trained_model('ResNet50_1D')
    
    transformer_acc = evaluate(transformer, test_loader)
    resnet_acc = evaluate(resnet50, test_loader)
    
    # Allow 2% difference
    assert abs(transformer_acc - resnet_acc) < 0.02, \
        f"Transformer ({transformer_acc:.2%}) vs ResNet ({resnet_acc:.2%}) differ by > 2%"
```

**4. Interpretability Tests**

**`tests/test_interpretability_agreement.py`**
```python
def test_attention_and_gradcam_agree():
    """Attention and Grad-CAM should highlight similar regions."""
    signal = load_test_signal('oil_whirl_severe.mat')
    transformer = load_trained_transformer()
    resnet = load_trained_resnet()
    
    # Get attention map
    attn_map = get_attention_importance(transformer, signal)
    
    # Get Grad-CAM
    gradcam_map = generate_gradcam(resnet, signal, target_class='oilwhirl')
    
    # Compute correlation
    correlation = np.corrcoef(attn_map.flatten(), gradcam_map.flatten())[0, 1]
    
    # Should be moderately correlated (> 0.5)
    assert correlation > 0.5, \
        f"Attention and Grad-CAM poorly correlated ({correlation:.2f})"
```

### Acceptance Criteria

**Phase 4 Complete When:**

✅ **Transformer model trains successfully**
- Forward pass completes without errors
- Attention weights sum to 1 (valid probability distribution)
- Model converges on training set (> 95% train accuracy)
- Warmup schedule prevents training divergence

✅ **Achieves target accuracy**
- **Transformer**: 96-97% test accuracy (matches ResNet-50)
- **CNN-Transformer hybrid**: 97-98% test accuracy (best overall)
- **Per-class recall**: ≥ 85% for at least 10/11 classes

✅ **Attention visualization functional**
- Can extract attention weights from any layer
- Heatmap visualization implemented
- Attention rollout (aggregate across layers) working
- Dashboard for interactive exploration

✅ **Interpretability validated**
- Attention focuses on fault-relevant time regions (qualitative check)
- For misalignment: Attention highlights 2X harmonic regions
- For oil whirl: Attention highlights sub-synchronous regions (0.42-0.48X)
- Attention and Grad-CAM show moderate agreement (correlation > 0.5)

✅ **Comparison with ResNet documented**
- Side-by-side accuracy table
- Inference time comparison (Transformer likely slower due to attention)
- Interpretability comparison (attention vs. Grad-CAM)
- Error analysis: Does Transformer handle different faults better?

✅ **Transformer variants explored**
- ViT (with [CLS] token) vs. standard Transformer tested
- Performer (efficient attention) achieves similar accuracy with faster training
- CNN-Transformer hybrid outperforms pure Transformer

✅ **Robustness maintained**
- Sensor noise test: ≤ 20% accuracy drop
- Adversarial robustness: ≤ 10% drop under FGSM
- Transformer not more brittle than CNNs

✅ **MLflow tracking complete**
- All Transformer experiments logged
- Attention visualizations saved as artifacts
- Comparison with ResNet documented

✅ **Documentation complete**
- README explaining Transformer adaptation to signals
- Jupyter notebook: "Training Transformer for Fault Diagnosis"
- Attention visualization tutorial
- Comparison report: Transformer vs. ResNet vs. Hybrid

### Estimated Effort

**Time Breakdown:**
- Transformer core (5 files): 5 days
  - `signal_transformer.py`: 1 day
  - `patch_embedding.py`: 0.5 days
  - `positional_encoding.py`: 0.5 days
  - `multi_head_attention.py`: 1.5 days (complex)
  - `transformer_encoder_layer.py`: 1 day
  - Debugging attention: 0.5 days

- Transformer variants (4 files): 3 days
  - ViT adaptation: 1 day
  - Performer (efficient attention): 1 day
  - TFT, Informer: 1 day (optional, lower priority)

- Training enhancements (3 files): 2 days
  - Warmup scheduler: 0.5 days
  - Transformer-specific augmentation: 1 day
  - Custom trainer: 0.5 days

- Interpretability (4 files): 4 days
  - Attention visualization: 2 days
  - Attention rollout: 1 day
  - Comparison with Grad-CAM: 0.5 days
  - Interactive dashboard: 0.5 days

- Hybrid models (2 files): 2 days
  - CNN-Transformer: 1 day
  - Perceiver: 1 day

- Evaluation (2 files): 2 days
  - Transformer evaluator: 1 day
  - Comparison with ResNet: 1 day

- Training Transformer models: 3 days
  - Hyperparameter tuning (warmup, lr, layers): 1 day
  - Train standard Transformer: 1 day
  - Train variants (ViT, CNN-Transformer): 1 day

- Testing (unit, attention, interpretability): 3 days
- Documentation: 2 days
- Buffer for debugging: 3 days

**Total: ~29 days (1.4 months) for Phase 4**

**Complexity**: ⭐⭐⭐⭐⭐ (Very High)
- Transformer architecture is complex (attention mechanism)
- Attention visualization requires careful implementation
- Warmup schedule is critical (easy to get wrong)
- Interpretability comparison (attention vs. Grad-CAM) is research-level

**Dependencies**: Phase 0 (data), Phase 2 (CNN baseline), Phase 3 (ResNet for comparison and hybrid)

**Risk**: High
- Transformer may not outperform ResNet (attention may not help for signals)
- Attention visualization may not be interpretable (random patterns)
- Training Transformers is finicky (sensitive to hyperparameters)
- Large memory footprint (attention matrices grow quadratically)

