## **PHASE 8: Ensemble Learning & Multi-Modal Fusion**

### Phase Objective
Combine predictions from all previous models (classical ML, CNNs, ResNet, Transformer, PINN, spectrogram models) using advanced ensemble techniques. Implement late fusion, early fusion, and learned fusion strategies. Target: 98-99% accuracy through diversity.

### Complete File List (8 files)

#### **1. Ensemble Methods (4 files)**

**`models/ensemble/voting_ensemble.py`**
- **Purpose**: Soft/hard voting across models
- **Key Functions**:
  - `soft_voting(predictions_list, weights=None)`:
    ```python
    # Average probability distributions
    ensemble_probs = np.average(predictions_list, axis=0, weights=weights)
    ensemble_pred = np.argmax(ensemble_probs, axis=-1)
    return ensemble_pred, ensemble_probs
    ```
  - `hard_voting(predictions_list)`: Majority vote
  - `optimize_ensemble_weights(models, val_loader)`: Find optimal weights via grid search
- **Dependencies**: `numpy`, `sklearn`

**`models/ensemble/stacking_ensemble.py`**
- **Purpose**: Meta-learner trained on base model predictions
- **Architecture**:
  ```
  Base Models: [ResNet-1D, Transformer, PINN, ResNet-2D, Random Forest]
    ↓ Generate predictions on validation set
  Meta-Features: [B, 5 × 11] = [B, 55]  # 5 models × 11 class probabilities
    ↓ Meta-Learner (Logistic Regression or MLP)
  Final Prediction: [B, 11]
  ```
- **Key Functions**:
  - `train_stacking(base_models, meta_learner, train_loader, val_loader)`:
    ```python
    # Step 1: Generate meta-features
    meta_features = []
    for model in base_models:
        preds = model.predict_proba(val_loader)
        meta_features.append(preds)
    meta_features = np.concatenate(meta_features, axis=-1)
    
    # Step 2: Train meta-learner
    meta_learner.fit(meta_features, val_labels)
    ```
- **Dependencies**: `sklearn`, `torch`

**`models/ensemble/boosting_ensemble.py`**
- **Purpose**: Boosting (train models sequentially, focus on hard examples)
- **Key Classes**:
  - `AdaptiveBoosting`: Train model_t+1 on samples model_t got wrong
- **Key Functions**:
  - `train_boosting(base_model_class, train_loader, n_iterations=5)`:
    ```python
    models = []
    sample_weights = np.ones(len(train_loader.dataset))
    
    for iteration in range(n_iterations):
        # Train model on weighted samples
        model = base_model_class()
        model.fit(train_loader, sample_weights)
        models.append(model)
        
        # Increase weights for misclassified samples
        predictions = model.predict(train_loader)
        errors = (predictions != labels)
        sample_weights[errors] *= 2
        sample_weights /= sample_weights.sum()
    
    return models
    ```
- **Dependencies**: `torch`, `numpy`

**`models/ensemble/mixture_of_experts.py`**
- **Purpose**: Gating network selects expert for each sample
- **Architecture**:
  ```
  Signal → Gating Network → [B, n_experts]  # Which expert to use
                ↓
  Experts: [Expert1(signal), Expert2(signal), ..., ExpertN(signal)]
                ↓
  Weighted Sum: ∑ gate_weight_i × expert_i_prediction
  ```
- **Experts**: Specialize in different fault types
  - Expert 1: Trained on frequency-modulated faults (oil whirl, cavitation)
  - Expert 2: Trained on harmonic faults (misalignment, imbalance)
  - Expert 3: Trained on mixed faults
- **Dependencies**: `torch.nn`

#### **2. Multi-Modal Fusion (2 files)**

**`models/fusion/early_fusion.py`**
- **Purpose**: Concatenate features from multiple domains before classification
- **Architecture**:
  ```
  Signal:
    ├─ Time-domain features (Phase 1): [B, 36]
    ├─ CNN features (Phase 2): [B, 512]
    ├─ Transformer features (Phase 4): [B, 512]
    └─ Physics features (Phase 6): [B, 64]
          ↓ Concatenate
        [B, 36+512+512+64] = [B, 1124]
          ↓ FC layers
        [B, 256] → [B, 11]
  ```
- **Benefit**: Joint representation learning
- **Challenge**: High-dimensional feature space (overfitting risk)

**`models/fusion/late_fusion.py`**
- **Purpose**: Combine final predictions (same as voting ensemble)
- **Key Functions**:
  - `late_fusion(model_predictions, fusion_method='weighted_average')`:
    ```python
    if fusion_method == 'weighted_average':
        return weighted_average(model_predictions, learned_weights)
    elif fusion_method == 'max':
        return np.max(model_predictions, axis=0)  # Take most confident
    elif fusion_method == 'product':
        return np.prod(model_predictions, axis=0)  # Product rule
    ```

#### **3. Evaluation (2 files)**

**`evaluation/ensemble_evaluator.py`**
- **Purpose**: Comprehensive ensemble evaluation
- **Key Functions**:
  - `evaluate_ensemble_diversity(models, test_loader)`:
    ```python
    # Measure agreement/disagreement between models
    predictions = [model.predict(test_loader) for model in models]
    
    # Pairwise disagreement
    disagreement_matrix = np.zeros((len(models), len(models)))
    for i, j in combinations(range(len(models)), 2):
        disagreement = (predictions[i] != predictions[j]).mean()
        disagreement_matrix[i, j] = disagreement
    
    # Higher disagreement = more diversity = better ensemble potential
    return disagreement_matrix.mean()
    ```
  - `evaluate_ensemble_performance(ensemble, test_loader)`: Accuracy, confusion matrix

**`experiments/ensemble_comparison.py`**
- **Purpose**: Compare all ensemble methods
- **Output**:
  ```
  | Ensemble Method        | Test Accuracy | Ensemble Diversity |
  |------------------------|---------------|--------------------|
  | Best Single Model      | 97.2%         | N/A                |
  | Soft Voting            | 97.8%         | 0.15               |
  | Stacking               | 98.1%         | 0.18               |
  | Mixture of Experts     | 98.3%         | 0.22               |
  | Early Fusion           | 97.9%         | N/A                |
  ```

### Acceptance Criteria

✅ **Ensemble outperforms best individual**
- **Target**: 98-99% accuracy (1-2% above best single model)

✅ **Diversity metrics positive**
- Models disagree on 15-25% of samples (good diversity)

✅ **Ensemble reduces mixed fault errors**
- Mixed fault F1-score improves by 3-5%

✅ **Comparison complete**
- Voting vs. Stacking vs. MoE documented

### Estimated Effort

**Total: ~10 days (2 weeks) for Phase 8**

---
