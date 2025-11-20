## **PHASE 6: Physics-Informed Neural Networks (PINN)**

### Phase Objective
Incorporate domain knowledge of bearing dynamics directly into neural network architecture and loss functions. Constrain model to respect physical laws (bearing frequency equations, Sommerfeld number relationships). Target: 97-98% accuracy with improved sample efficiency and better extrapolation to unseen operating conditions.

### Complete File List (12 files)

#### **1. Physics Models (3 files)**

**`models/physics/bearing_dynamics.py`**
- **Purpose**: Encode bearing fault physics as differentiable functions
- **Key Functions**:
  - `characteristic_frequencies(rpm, bearing_params)`:
    ```python
    def characteristic_frequencies(rpm, n_balls, ball_diameter, pitch_diameter):
        # Fundamental Train Frequency (FTF)
        ftf = (rpm / 60) * (1 - (ball_diameter / pitch_diameter) * np.cos(contact_angle)) / 2
        
        # Ball Pass Frequency Outer Race (BPFO)
        bpfo = (n_balls * rpm / 60) * (1 - (ball_diameter / pitch_diameter) * np.cos(contact_angle)) / 2
        
        # Ball Pass Frequency Inner Race (BPFI)
        bpfi = (n_balls * rpm / 60) * (1 + (ball_diameter / pitch_diameter) * np.cos(contact_angle)) / 2
        
        # Ball Spin Frequency (BSF)
        bsf = (pitch_diameter * rpm / (2 * ball_diameter * 60)) * (1 - (ball_diameter / pitch_diameter)**2 * np.cos(contact_angle)**2)
        
        return {'FTF': ftf, 'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf}
    ```
  - `sommerfeld_number(load, speed, viscosity, clearance, radius)`:
    ```python
    S = (viscosity * speed) / load * (radius / clearance) ** 2
    return S
    ```
  - `reynolds_number(speed, clearance, viscosity)`:
    ```python
    Re = (speed * clearance) / viscosity
    return Re
    ```
- **Dependencies**: `numpy`, `torch` (for autodiff)

**`models/physics/fault_signatures.py`**
- **Purpose**: Expected frequency signatures for each fault type
- **Key Functions**:
  - `get_fault_signature(fault_type, rpm)`:
    ```python
    signatures = {
        'misalignment': [1*f0, 2*f0, 3*f0],  # Harmonics at 1X, 2X, 3X
        'imbalance': [1*f0],  # Strong 1X
        'oil_whirl': [0.42*f0, 0.43*f0, 0.48*f0],  # Sub-synchronous
        'cavitation': [1500, 2000, 2500],  # High-frequency bursts
        # ...
    }
    return signatures[fault_type]
    ```
  - `compute_expected_spectrum(fault_type, rpm, amplitude)`: Generate ideal spectrum
- **Dependencies**: `numpy`

**`models/physics/operating_conditions.py`**
- **Purpose**: Calculate valid operating condition ranges
- **Key Functions**:
  - `validate_operating_point(load, speed, temp)`: Check if thermodynamically valid
  - `predict_film_thickness(load, speed, viscosity)`: Lubrication theory
  - `check_laminar_turbulent(reynolds_number)`: Flow regime classification
- **Dependencies**: `numpy`

#### **2. PINN Architectures (4 files)**

**`models/pinn/hybrid_pinn.py`**
- **Purpose**: Combine CNN features with physics-based features
- **Key Classes**:
  - `HybridPINN(nn.Module)`: Dual-branch architecture
- **Architecture**:
  ```
  Input Signal [B, 1, 102400]
    ├─ Data Branch: CNN → [B, 512] (learned features)
    └─ Physics Branch: 
         ├─ Extract operating conditions (load, speed, temp) from metadata
         ├─ Compute: Sommerfeld number, Reynolds number, characteristic freqs
         ├─ FC layers: [B, 10] → [B, 64] (physics features)
         └─ Output: [B, 64]
           ↓ Concatenate
         [B, 512 + 64] = [B, 576]
           ↓ Fusion FC
         [B, 256] → [B, 11]
  ```
- **Key Functions**:
  - `forward(signal, metadata)`: Dual input (signal + operating conditions)
  - `extract_physics_features(metadata)`: Compute Sommerfeld, Reynolds, etc.
- **Dependencies**: `torch.nn`, `models/cnn/cnn_1d.py`, `models/physics/bearing_dynamics.py`

**`models/pinn/physics_constrained_cnn.py`**
- **Purpose**: CNN with physics-based loss constraints
- **Key Classes**:
  - `PhysicsConstrainedCNN(nn.Module)`: Standard CNN + physics loss
- **Physics Loss**:
  ```python
  def physics_loss(predicted_class, signal_fft, metadata):
      # Extract dominant frequencies from FFT
      predicted_freqs = find_peaks(signal_fft)
      
      # Get expected frequencies for predicted class
      expected_freqs = get_fault_signature(predicted_class, metadata['rpm'])
      
      # Penalize mismatch
      freq_error = torch.abs(predicted_freqs - expected_freqs).sum()
      return freq_error
  
  # Total loss
  total_loss = cross_entropy_loss + lambda_physics * physics_loss
  ```
- **Key Functions**:
  - `forward(signal)`: Standard CNN forward
  - `compute_physics_loss(signal, pred_class, metadata)`: Frequency constraint
- **Dependencies**: `torch.nn`, `models/cnn/cnn_1d.py`

**`models/pinn/knowledge_graph_pinn.py`**
- **Purpose**: Encode fault relationships as graph, use GNN
- **Key Classes**:
  - `KnowledgeGraphPINN(nn.Module)`: CNN + Graph Neural Network
- **Knowledge Graph**:
  ```
  Nodes: Fault types (11 nodes)
  Edges: Physical relationships
    - wear → lubrication (wear degrades oil)
    - clearance → cavitation (clearance enables cavitation)
    - misalignment → imbalance (coupling effect)
  
  Node Features: Characteristic frequencies, expected severity progression
  ```
- **Architecture**:
  ```
  Signal → CNN → [B, 512]
    ↓
  Graph Convolution: Aggregate neighboring fault information
    ↓
  Classification: [B, 11]
  ```
- **Benefit**: Leverages fault relationships (reduces mixed fault confusion)
- **Dependencies**: `torch.nn`, `torch_geometric`

**`models/pinn/multitask_pinn.py`**
- **Purpose**: Multi-task learning (classify fault + predict operating conditions)
- **Key Classes**:
  - `MultitaskPINN(nn.Module)`: Shared encoder, multiple heads
- **Architecture**:
  ```
  Signal → Shared CNN Encoder → [B, 512]
    ├─ Task 1: Fault Classification → [B, 11]
    ├─ Task 2: Speed Regression → [B, 1]
    ├─ Task 3: Load Regression → [B, 1]
    └─ Task 4: Severity Classification → [B, 4]
  ```
- **Rationale**: Auxiliary tasks (speed/load prediction) regularize feature learning
- **Loss**: Weighted sum of task losses
- **Dependencies**: `torch.nn`, `models/cnn/cnn_1d.py`

#### **3. Training Infrastructure (2 files)**

**`training/pinn_trainer.py`**
- **Purpose**: Training loop with physics loss
- **Key Classes**:
  - `PINNTrainer(Trainer)`: Extends base trainer
- **Key Functions**:
  - `compute_loss(outputs, targets, signal, metadata)`:
    ```python
    # Standard classification loss
    ce_loss = cross_entropy(outputs, targets)
    
    # Physics loss
    phys_loss = self.model.compute_physics_loss(signal, outputs, metadata)
    
    # Combined
    total_loss = ce_loss + self.config.lambda_physics * phys_loss
    return total_loss
    ```
  - `_update_lambda_physics(epoch)`: Gradually increase physics loss weight
- **Dependencies**: `training/trainer.py`

**`training/physics_loss_functions.py`**
- **Purpose**: Various physics-based loss terms
- **Key Functions**:
  - `frequency_consistency_loss(signal_fft, predicted_class, metadata)`: Penalize incorrect dominant frequencies
  - `sommerfeld_consistency_loss(predicted_severity, metadata)`: Severity should match operating conditions
  - `temporal_smoothness_loss(predictions_sequence)`: Predictions shouldn't jump erratically
- **Dependencies**: `torch`, `models/physics/bearing_dynamics.py`

#### **4. Evaluation (2 files)**

**`evaluation/pinn_evaluator.py`**
- **Purpose**: Evaluate PINN models
- **Key Functions**:
  - `evaluate_with_physics_metrics(model, test_loader)`:
    ```python
    # Standard metrics
    accuracy = compute_accuracy(preds, targets)
    
    # Physics-aware metrics
    freq_consistency = compute_frequency_consistency(preds, signals, metadata)
    operating_condition_extrapolation = test_on_unseen_conditions(model, ood_loader)
    
    return {'accuracy': accuracy, 'freq_consistency': freq_consistency, 'ood_accuracy': ood_accuracy}
    ```
  - `test_sample_efficiency(model, train_sizes)`: Train on [50, 100, 200, 500] samples, compare to baseline
- **Expected Finding**: PINN achieves 90% accuracy with 50% less data than baseline CNN
- **Dependencies**: `evaluation/evaluator.py`, `models/physics/bearing_dynamics.py`

**`evaluation/physics_interpretability.py`**
- **Purpose**: Visualize physics constraints
- **Key Functions**:
  - `plot_learned_vs_expected_frequencies(model, test_samples)`:
    ```python
    # For each test sample:
    #   - Extract dominant frequencies from signal
    #   - Get expected frequencies for true fault type
    #   - Get model prediction
    #   - Plot: [expected, observed, predicted] frequency distributions
    ```
  - `visualize_knowledge_graph(kg_pinn_model)`: Plot fault relationship graph with learned edge weights
- **Dependencies**: `matplotlib`, `networkx`

#### **5. Experiment (1 file)**

**`experiments/pinn_ablation.py`**
- **Purpose**: Ablation study on physics components
- **Key Functions**:
  - `ablate_physics_loss(config)`: Train with/without physics loss
  - `ablate_physics_features(config)`: Train with/without Sommerfeld/Reynolds features
  - `ablate_knowledge_graph(config)`: Train with/without fault relationship graph
- **Output**: Table showing impact of each physics component
  ```
  | Configuration           | Accuracy | Sample Efficiency |
  |-------------------------|----------|-------------------|
  | Baseline CNN            | 95.3%    | 1000 samples      |
  | + Physics Loss          | 96.1%    | 800 samples       |
  | + Physics Features      | 96.8%    | 700 samples       |
  | + Knowledge Graph       | 97.2%    | 600 samples       |
  ```
- **Dependencies**: `experiments/cnn_experiment.py`

### Architecture Decisions

**1. Physics Loss Weight Scheduling**
- **Decision**: Start λ_physics = 0, linearly increase to 0.5 over 20 epochs
- **Rationale**: Early training focuses on learning basic patterns, then enforces physics
- **Alternative**: Fixed weight (may hurt early convergence)

**2. Operating Condition Metadata**
- **Decision**: Include load, speed, temperature as auxiliary inputs
- **Rationale**: Sommerfeld and Reynolds numbers depend on these conditions
- **Challenge**: Existing dataset may not have metadata → use nominal values or augment

**3. Knowledge Graph Construction**
- **Decision**: Manually construct graph based on domain knowledge
- **Rationale**: Fault relationships well-understood (wear → lubrication, etc.)
- **Alternative**: Learn graph structure (requires more data)

**4. Multi-Task Learning Tasks**
- **Decision**: Primary task (fault classification) + auxiliary (speed/severity prediction)
- **Rationale**: Auxiliary tasks provide additional supervision signal
- **Risk**: Auxiliary task difficulty may hurt primary task (careful loss weighting needed)

### Data Flow

```
┌────────────────────────────────────────────────────────────┐
│             PINN TRAINING PIPELINE (Phase 6)                 │
└────────────────────────────────────────────────────────────┘

1. DATA AUGMENTATION (add operating conditions)
   ┌──────────────────────────────────────────────────────┐
   │ If metadata not available:                            │
   │   ├─ Sample operating conditions:                     │
   │   │   - Load: 30-100% rated                           │
   │   │   - Speed: 3000-4000 RPM                          │
   │   │   - Temp: 40-80°C                                 │
   │   └─ Compute physics features:                        │
   │       - Sommerfeld number                             │
   │       - Reynolds number                               │
   │       - Characteristic frequencies                    │
   └──────────────────────────────────────────────────────┘
                        ↓

2. HYBRID PINN FORWARD PASS
   ┌──────────────────────────────────────────────────────┐
   │ models/pinn/hybrid_pinn.py                            │
   │                                                       │
   │ Input: Signal [B, 1, 102400] + Metadata [B, 3]       │
   │                                                       │
   │ Data Branch:                                          │
   │   Signal → CNN → [B, 512]                            │
   │                                                       │
   │ Physics Branch:                                       │
   │   Metadata → Compute Sommerfeld, Reynolds → FC → [B, 64]
   │                                                       │
   │ Fusion:                                               │
   │   Concatenate [B, 512+64] → FC → [B, 11]            │
   └──────────────────────────────────────────────────────┘
                        ↓

3. LOSS COMPUTATION
   ┌──────────────────────────────────────────────────────┐
   │ training/pinn_trainer.py                              │
   │                                                       │
   │ Classification Loss:                                  │
   │   L_CE = CrossEntropy(predictions, targets)          │
   │                                                       │
   │ Physics Loss:                                         │
   │   ├─ Extract FFT(signal)                             │
   │   ├─ Find dominant frequencies                        │
   │   ├─ Get expected frequencies for predicted class    │
   │   └─ L_physics = |dominant - expected|               │
   │                                                       │
   │ Total Loss:                                           │
   │   L_total = L_CE + λ_physics * L_physics             │
   └──────────────────────────────────────────────────────┘
                        ↓

4. EVALUATION (sample efficiency)
   ┌──────────────────────────────────────────────────────┐
   │ evaluation/pinn_evaluator.py                          │
   │                                                       │
   │ Train on varying dataset sizes:                       │
   │   ├─ 50 samples: PINN 87%, Baseline 72%             │
   │   ├─ 100 samples: PINN 91%, Baseline 82%            │
   │   ├─ 500 samples: PINN 96%, Baseline 94%            │
   │   └─ Full (1430): PINN 97%, Baseline 95%            │
   │                                                       │
   │ Conclusion: PINN needs 50% less data for same accuracy
   └──────────────────────────────────────────────────────┘
```

### Integration Points

**1. With Phase 0 (Data Generator)**
- **Enhancement**: Modify `signal_generator.py` to output operating condition metadata
- **Sommerfeld Calculation**: Already present (Section 7.5 of report)

**2. With Phase 3 (ResNet)**
- **Backbone**: Use ResNet-18 as CNN backbone in Hybrid PINN

**3. With Phase 7 (XAI)**
- **Physics Interpretability**: Explain predictions using physics constraints
- **Feature Attribution**: Which physics features (Sommerfeld, Reynolds) drive predictions?

**4. With Phase 8 (Ensemble)**
- **Diversity**: PINN makes different errors than pure data-driven models
- **Ensemble**: Combine PINN + ResNet + Transformer

### Acceptance Criteria

**Phase 6 Complete When:**

✅ **Physics models implemented**
- Bearing dynamics equations (characteristic frequencies, Sommerfeld, Reynolds)
- Fault signature database functional
- Operating condition validator working

✅ **PINN architectures train successfully**
- Hybrid PINN converges
- Physics loss decreases during training
- Multi-task PINN learns all tasks simultaneously

✅ **Performance targets met**
- **Hybrid PINN**: 97-98% accuracy (best overall)
- **Sample efficiency**: 90% accuracy with 50% less data than baseline
- **Out-of-distribution**: 85% accuracy on unseen operating conditions

✅ **Physics constraints validated**
- Frequency consistency: Predictions align with expected fault frequencies
- Sommerfeld consistency: Severity predictions match operating conditions
- Knowledge graph: Fault relationships reduce mixed fault confusion

✅ **Ablation study complete**
- Quantify impact of physics loss, physics features, knowledge graph
- Document sample efficiency gains

✅ **Documentation complete**
- Tutorial: "Building Physics-Informed Neural Networks for Fault Diagnosis"
- Comparison: PINN vs. pure data-driven models

### Estimated Effort

**Time Breakdown:**
- Physics models (3 files): 3 days
- PINN architectures (4 files): 4 days
- Training infrastructure (2 files): 2 days
- Evaluation (2 files): 2 days
- Experiments (1 file): 2 days
- Testing: 2 days
- Documentation: 1 day

**Total: ~16 days (3 weeks) for Phase 6**

**Complexity**: ⭐⭐⭐⭐☆ (High) - Requires domain expertise

---
