
## **PHASE 5: Time-Frequency Analysis & Spectrogram-Based Deep Learning**

### Phase Objective
Implement 2D CNN architectures operating on time-frequency representations (spectrograms, wavelets, Wigner-Ville) to capture frequency evolution patterns that 1D CNNs may miss. Compare time-domain vs. frequency-domain learning. Target: 96-98% accuracy, particularly improved performance on frequency-modulated faults (oil whirl, cavitation).

### Complete File List (14 files)

#### **1. Time-Frequency Transforms (4 files)**

**`data/spectrogram_generator.py`**
- **Purpose**: Generate spectrograms from vibration signals
- **Key Functions**:
  - `generate_stft_spectrogram(signal, fs, nperseg=256, noverlap=128)`:
    ```python
    f, t, Sxx = scipy.signal.stft(signal, fs, nperseg=nperseg, noverlap=noverlap)
    # Sxx: complex spectrogram [n_freq, n_time]
    power_spectrogram = np.abs(Sxx) ** 2  # Power spectral density
    log_spectrogram = 10 * np.log10(power_spectrogram + 1e-10)  # dB scale
    return log_spectrogram, f, t
    ```
  - `generate_mel_spectrogram(signal, fs, n_mels=128)`: Mel-scaled (perceptually-weighted)
  - `normalize_spectrogram(spectrogram)`: Normalize to [0, 1] or [-1, 1]
- **Output Shape**: [n_freq, n_time] e.g., [129, 400] for 5-second signal
- **Design Choice**: STFT window 256 samples (~12.5ms) captures bearing fault frequencies
- **Dependencies**: `scipy.signal.stft`, `librosa` (for Mel)

**`data/wavelet_transform.py`**
- **Purpose**: Continuous Wavelet Transform (CWT) scalograms
- **Key Functions**:
  - `generate_cwt_scalogram(signal, fs, wavelet='morl', scales=128)`:
    ```python
    scales = np.logspace(1, 4, 128)  # Logarithmic scale spacing
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 1/fs)
    scalogram = np.abs(coefficients)  # Time-frequency energy
    return scalogram, frequencies
    ```
  - `wavelet_denoising(signal, wavelet='db4', level=5)`: Denoise before CWT
- **Benefit**: Better time-frequency resolution than STFT for transients
- **Dependencies**: `pywt` (PyWavelets)

**`data/wigner_ville.py`**
- **Purpose**: Wigner-Ville distribution (high resolution, cross-terms)
- **Key Functions**:
  - `generate_wvd(signal, fs)`: Compute WVD
    ```python
    # Wigner-Ville: W(t, f) = ∫ x(t+τ/2) x*(t-τ/2) e^(-j2πfτ) dτ
    # Pseudo-Wigner-Ville to suppress cross-terms
    wvd = tftb.processing.WignerVilleDistribution(signal)
    wvd_result = wvd.run()
    return np.abs(wvd_result)
    ```
  - `smooth_wvd(wvd)`: Smoothing to reduce cross-term artifacts
- **Use Case**: Signals with rapid frequency changes (chirps in cavitation)
- **Dependencies**: `tftb` (Time-Frequency Toolbox)

**`data/tfr_dataset.py`**
- **Purpose**: PyTorch Dataset for time-frequency representations
- **Key Classes**:
  - `SpectrogramDataset(torch.utils.data.Dataset)`: Loads precomputed spectrograms
  - `OnTheFlyTFRDataset`: Compute TFR on-the-fly (slower but flexible)
- **Key Functions**:
  - `__getitem__(idx)`: Returns `(spectrogram [C, H, W], label)`
    - C=1 (grayscale), H=n_freq, W=n_time
- **Caching Strategy**: Precompute spectrograms, save as `.npz` (10× faster training)
- **Dependencies**: `torch.utils.data`, `spectrogram_generator.py`

#### **2. 2D CNN Architectures (3 files)**

**`models/spectrogram_cnn/resnet2d_spectrogram.py`**
- **Purpose**: ResNet-18/34/50 for spectrogram classification
- **Key Classes**:
  - `ResNet2DSpectrogram(nn.Module)`: Standard ResNet with 2D convolutions
- **Architecture Adaptation**:
  ```python
  # Standard ResNet: Input [B, 3, 224, 224] (RGB images)
  # Our adaptation: Input [B, 1, 129, 400] (grayscale spectrograms)
  
  # Modify first conv layer:
  self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # 1 input channel
  
  # Rest of ResNet unchanged (residual blocks, etc.)
  ```
- **Training**: Transfer learning from ImageNet weights (conv1 reinitialized)
- **Expected Performance**: 95-97% (similar to 1D ResNet)
- **Dependencies**: `torch.nn`, `torchvision.models.resnet`

**`models/spectrogram_cnn/efficientnet2d_spectrogram.py`**
- **Purpose**: EfficientNet for spectrograms (parameter-efficient)
- **Key Classes**:
  - `EfficientNet2DSpectrogram(nn.Module)`: EfficientNet-B0/B3
- **Architecture**: Same as Phase 3 EfficientNet, but 2D conv kernels
- **Benefit**: 5-10× fewer parameters than ResNet for similar accuracy
- **Dependencies**: `torch.nn`, `efficientnet_pytorch`

**`models/spectrogram_cnn/dual_stream_cnn.py`**
- **Purpose**: Two-stream network processing time-domain + frequency-domain
- **Key Classes**:
  - `DualStreamCNN(nn.Module)`: Parallel time + frequency branches
- **Architecture**:
  ```
  Input Signal [B, 1, 102400]
    ├─ Branch 1: 1D CNN (time domain) → [B, 512]
    └─ Branch 2: Spectrogram → 2D CNN → [B, 512]
          ↓ Concatenate
        [B, 1024]
          ↓ FC
        [B, 11]
  ```
- **Rationale**: Combine complementary time/frequency features
- **Expected Gain**: +1-2% over single-stream models
- **Dependencies**: `models/cnn/cnn_1d.py`, `resnet2d_spectrogram.py`

#### **3. Spectrogram Augmentation (2 files)**

**`data/spectrogram_augmentation.py`**
- **Purpose**: Augmentation specific to spectrograms
- **Key Functions**:
  - `time_mask(spectrogram, mask_width)`: SpecAugment time masking
  - `frequency_mask(spectrogram, mask_width)`: SpecAugment frequency masking
  - `time_warp(spectrogram, W)`: Non-linear time warping
  - `mixup_spectrograms(spec1, spec2, alpha)`: Spectrogram MixUp
- **SpecAugment** (from speech recognition):
  ```python
  # Randomly mask time bins
  t0 = random.randint(0, n_time - mask_width)
  spectrogram[:, t0:t0+mask_width] = spectrogram.mean()
  
  # Randomly mask frequency bins
  f0 = random.randint(0, n_freq - mask_width)
  spectrogram[f0:f0+mask_width, :] = spectrogram.mean()
  ```
- **Dependencies**: `numpy`

**`data/contrast_learning_tfr.py`**
- **Purpose**: Contrastive learning for time-frequency representations
- **Key Functions**:
  - `generate_positive_pairs(signal)`: Two augmented views of same signal
  - `contrastive_loss(z1, z2, temperature)`: SimCLR-style loss
- **Usage**: Self-supervised pretraining (optional, if limited labeled data)
- **Dependencies**: `torch`

#### **4. Training & Evaluation (3 files)**

**`training/spectrogram_trainer.py`**
- **Purpose**: Training loop for spectrogram models
- **Key Classes**:
  - `SpectrogramTrainer(Trainer)`: Extends base trainer
- **Spectrogram-Specific Considerations**:
  - Data augmentation on spectrograms (SpecAugment)
  - Learning rate schedule (same as CNNs: warmup + cosine)
  - Mixed precision training (spectrograms are larger than 1D signals)
- **Dependencies**: `training/trainer.py`

**`evaluation/spectrogram_evaluator.py`**
- **Purpose**: Evaluate spectrogram models
- **Key Functions**:
  - `evaluate(model, test_loader)`: Standard evaluation
  - `visualize_predictions(model, signal)`: 
    ```python
    # Generate spectrogram
    spec = generate_stft_spectrogram(signal, fs)
    
    # Predict
    pred_class, probs = model.predict(spec)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(signal)  # Time-domain signal
    ax2.imshow(spec, aspect='auto', origin='lower')  # Spectrogram
    ax2.set_title(f'Predicted: {pred_class}')
    ```
- **Dependencies**: `evaluation/evaluator.py`, `matplotlib`

**`evaluation/time_vs_frequency_comparison.py`**
- **Purpose**: Systematic comparison of time-domain vs. frequency-domain models
- **Key Functions**:
  - `compare_time_vs_frequency(models_dict, test_loader)`:
    ```python
    results = {
        '1D CNN (time)': evaluate(models['cnn_1d'], test_loader_time),
        'ResNet-1D (time)': evaluate(models['resnet_1d'], test_loader_time),
        'ResNet-2D (spectrogram)': evaluate(models['resnet_2d'], test_loader_spec),
        'Dual-Stream': evaluate(models['dual_stream'], test_loader_both)
    }
    return pd.DataFrame(results)
    ```
  - `analyze_per_fault_performance(results)`: Which faults benefit from frequency domain?
- **Expected Finding**: Frequency-modulated faults (oil whirl, cavitation) improve with spectrograms
- **Dependencies**: `pandas`, `evaluation/evaluator.py`

#### **5. Visualization (2 files)**

**`visualization/spectrogram_plots.py`**
- **Purpose**: Plotting utilities for spectrograms
- **Key Functions**:
  - `plot_spectrogram_comparison(signal, fs)`:
    ```python
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # STFT spectrogram
    stft_spec, f, t = generate_stft_spectrogram(signal, fs)
    axes[0].imshow(stft_spec, aspect='auto', origin='lower', extent=[0, 5, 0, fs/2])
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('STFT Spectrogram')
    
    # CWT scalogram
    cwt_spec, cwt_freqs = generate_cwt_scalogram(signal, fs)
    axes[1].imshow(cwt_spec, aspect='auto', origin='lower', extent=[0, 5, cwt_freqs[0], cwt_freqs[-1]])
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('CWT Scalogram')
    
    # Wigner-Ville
    wvd = generate_wvd(signal, fs)
    axes[2].imshow(wvd, aspect='auto', origin='lower')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_title('Wigner-Ville Distribution')
    axes[2].set_xlabel('Time (s)')
    ```
  - `plot_fault_spectrograms_grid(signals_by_fault, fs)`: 11×3 grid (11 faults, 3 TFRs)
- **Dependencies**: `matplotlib`, `data/spectrogram_generator.py`

**`visualization/activation_maps_2d.py`**
- **Purpose**: Visualize 2D CNN activations on spectrograms
- **Key Functions**:
  - `plot_conv_filters_2d(model, layer_name)`: Visualize learned 2D filters
  - `plot_feature_maps_2d(model, spectrogram, layer_name)`: Activation maps
  - `grad_cam_2d(model, spectrogram, target_class)`: Grad-CAM for spectrograms
- **Usage**: Understand what patterns 2D CNN learns (edges, blobs, frequency bands)
- **Dependencies**: `torch`, `matplotlib`

### Architecture Decisions

**1. STFT vs. CWT vs. Wigner-Ville**
- **Decision**: Primarily use STFT, optionally compare CWT
- **Rationale**:
  - STFT is standard, well-understood, fast to compute
  - CWT offers better time-frequency resolution but 5× slower
  - Wigner-Ville has cross-term artifacts, harder to interpret
- **Experiment**: Train models on all three, compare accuracy

**2. Spectrogram Normalization**
- **Decision**: Log-scale (dB) + standardization
- **Rationale**:
  - Log-scale compresses dynamic range (human perception-aligned)
  - Standardization (mean=0, std=1) helps neural network training
  ```python
  log_spec = 10 * np.log10(power_spec + 1e-10)
  normalized_spec = (log_spec - log_spec.mean()) / log_spec.std()
  ```

**3. Transfer Learning from ImageNet**
- **Decision**: Initialize ResNet-2D with ImageNet weights, fine-tune
- **Rationale**:
  - ImageNet features (edges, textures) transfer to spectrograms
  - Faster convergence, better accuracy with limited data
  - Conv1 reinitialized (1 channel vs. 3), rest transferred
- **Expected Gain**: +2-3% accuracy vs. random initialization

**4. Dual-Stream vs. Single-Stream**
- **Decision**: Implement both, compare systematically
- **Rationale**:
  - Dual-stream combines complementary information (time + frequency)
  - Single-stream simpler, fewer parameters
  - Literature mixed on whether fusion helps (task-dependent)
- **Evaluation**: If dual-stream < 1% better than best single-stream, not worth complexity

**5. Spectrogram Resolution**
- **Decision**: STFT with nperseg=256, noverlap=128
- **Rationale**:
  - 256 samples = 12.5ms window at 20.48 kHz (captures 1-2 rotation cycles at 60 Hz)
  - 50% overlap balances time resolution vs. computation
  - Output: [129, 400] spectrogram (manageable size for CNNs)
- **Sensitivity Analysis**: Try nperseg=128, 256, 512

### Data Flow

```
┌────────────────────────────────────────────────────────────┐
│      SPECTROGRAM-BASED PIPELINE (Phase 5)                   │
└────────────────────────────────────────────────────────────┘

1. SPECTROGRAM GENERATION (offline, precomputed)
   ┌──────────────────────────────────────────────────────┐
   │ data/spectrogram_generator.py                         │
   │                                                       │
   │ For each signal in dataset (1,430 signals):          │
   │   ├─ Load signal: [102400] samples                   │
   │   ├─ Compute STFT: scipy.signal.stft(nperseg=256)   │
   │   ├─ Convert to power: |STFT|²                       │
   │   ├─ Log-scale: 10*log10(power)                      │
   │   ├─ Normalize: (spec - mean) / std                  │
   │   └─ Save: spectrograms.npz [1430, 129, 400]        │
   │                                                       │
   │ Time: ~10 minutes for 1,430 signals                  │
   └──────────────────────────────────────────────────────┘
                        ↓

2. DATASET LOADING
   ┌──────────────────────────────────────────────────────┐
   │ data/tfr_dataset.py                                   │
   │   ├─ Load precomputed spectrograms                   │
   │   ├─ Apply augmentation (SpecAugment)               │
   │   └─ Return: [B, 1, 129, 400] (batch of spectrograms)
   └──────────────────────────────────────────────────────┘
                        ↓

3. 2D CNN FORWARD PASS
   ┌──────────────────────────────────────────────────────┐
   │ models/spectrogram_cnn/resnet2d_spectrogram.py        │
   │                                                       │
   │ Input: [B, 1, 129, 400]                               │
   │  ↓                                                    │
   │ Conv2D(1→64, k=7, s=2): [B, 64, 65, 200]            │
   │  ↓                                                    │
   │ Residual Blocks (2D): ...                            │
   │  ↓                                                    │
   │ GlobalAvgPool2D: [B, 512]                            │
   │  ↓                                                    │
   │ FC: [B, 11]                                           │
   └──────────────────────────────────────────────────────┘
                        ↓

4. COMPARISON WITH TIME-DOMAIN MODELS
   ┌──────────────────────────────────────────────────────┐
   │ evaluation/time_vs_frequency_comparison.py            │
   │                                                       │
   │ Test Set Evaluation:                                  │
   │   ├─ 1D CNN (Phase 2): 93-95%                       │
   │   ├─ ResNet-1D (Phase 3): 96-97%                    │
   │   ├─ ResNet-2D (Phase 5): 95-97%                    │
   │   └─ Dual-Stream: 97-98% (best)                     │
   │                                                       │
   │ Per-Fault Analysis:                                   │
   │   ├─ Oil whirl: Spectrogram +3% (frequency-modulated)│
   │   ├─ Cavitation: Spectrogram +2% (high-freq bursts) │
   │   └─ Misalignment: Time-domain same (harmonic-based)│
   └──────────────────────────────────────────────────────┘
```

### Integration Points

**1. With Phase 3 (ResNet-1D)**
- **Comparison**: ResNet-2D (spectrogram) vs. ResNet-1D (raw signal)
- **Architecture Reuse**: Same residual block structure, just 2D convolutions

**2. With Phase 4 (Transformer)**
- **Spectrogram Transformer**: Treat spectrogram as image, apply ViT
- **Comparison**: Transformer on raw signal vs. Transformer on spectrogram

**3. With Phase 1 (Feature Engineering)**
- **Validation**: Spectrogram CNN should learn spectral features automatically
- **Comparison**: CNN-learned features vs. hand-crafted spectral features

**4. With Phase 8 (Ensemble)**
- **Multi-Modal Fusion**: Ensemble 1D CNN + 2D CNN + Transformer
- **Diversity**: Time-domain and frequency-domain models make different errors

### Testing Strategy

**1. Unit Tests**

```python
def test_spectrogram_generation():
    """Test spectrogram has expected shape."""
    signal = np.random.randn(102400)
    fs = 20480
    spec, f, t = generate_stft_spectrogram(signal, fs, nperseg=256, noverlap=128)
    assert spec.shape == (129, 400)  # Expected output shape

def test_resnet2d_forward():
    """Test 2D ResNet forward pass."""
    model = ResNet2DSpectrogram(num_classes=11)
    spec = torch.randn(2, 1, 129, 400)  # Batch of 2 spectrograms
    output = model(spec)
    assert output.shape == (2, 11)
```

**2. Comparison Tests**

```python
def test_spectrogram_improves_frequency_modulated_faults():
    """Spectrogram models should excel on oil whirl, cavitation."""
    test_loader_oilwhirl = load_test_subset(fault_type='oilwhirl')
    
    resnet_1d = load_trained_model('ResNet18_1D')
    resnet_2d = load_trained_model('ResNet18_2D_Spectrogram')
    
    acc_1d = evaluate(resnet_1d, test_loader_oilwhirl)
    acc_2d = evaluate(resnet_2d, test_loader_oilwhirl)
    
    # 2D should be at least as good, ideally 2-3% better
    assert acc_2d >= acc_1d - 0.01, f"Spectrogram ({acc_2d:.2%}) worse than time-domain ({acc_1d:.2%})"
```

### Acceptance Criteria

**Phase 5 Complete When:**

✅ **Spectrogram generation working**
- STFT, CWT, WVD implementations functional
- Precomputed spectrograms cached (10 min for 1,430 signals)
- Spectrograms visually plausible (frequency structure visible)

✅ **2D CNN models train successfully**
- ResNet-2D, EfficientNet-2D converge
- Transfer learning from ImageNet speeds up training
- Dual-stream model trains without errors

✅ **Performance targets met**
- **ResNet-2D (spectrogram)**: 95-97% accuracy
- **Dual-Stream (time + frequency)**: 97-98% accuracy (best overall)
- **Per-fault improvement**: Oil whirl, cavitation +2-3% with spectrograms

✅ **Systematic comparison documented**
- Table: 1D CNN vs. ResNet-1D vs. ResNet-2D vs. Dual-Stream
- Per-fault analysis: Which faults benefit from frequency domain?
- Visualization: Spectrograms of all 11 fault types

✅ **Robustness validated**
- Spectrogram models maintain robustness (sensor noise, drift)
- SpecAugment improves generalization (+1-2% accuracy)

✅ **Documentation complete**
- Tutorial: "Time-Frequency Analysis for Fault Diagnosis"
- Comparison report: Time vs. Frequency domain learning

### Estimated Effort

**Time Breakdown:**
- Time-frequency transforms (4 files): 3 days
- 2D CNN architectures (3 files): 2 days
- Augmentation (2 files): 1 day
- Training & evaluation (3 files): 2 days
- Visualization (2 files): 1 day
- Training models: 2 days
- Testing: 2 days
- Documentation: 1 day

**Total: ~14 days (2.5 weeks) for Phase 5**

**Complexity**: ⭐⭐⭐☆☆ (Moderate)

---
