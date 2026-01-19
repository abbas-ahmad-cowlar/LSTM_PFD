> [!WARNING]
> **Archived Document**
> This document is historical and may be outdated.
> For current information, see the main documentation.
>
> *Archived on: 2026-01-20*
> *Reason: Superseded by consolidated documentation*
## **PHASE 1: Classical ML Baseline Enhancement**

### Phase Objective
Port and enhance the existing MATLAB classical ML pipeline (`pipeline.m` → Python), achieving feature parity with improved modularity, then extend with advanced ensemble techniques and automated hyperparameter optimization. Establish performance baseline (95.33% accuracy) for deep learning comparison.

### Complete File List (28 files)

#### **1. Feature Engineering Module (7 files)**

**`features/feature_extractor.py`**
- **Purpose**: Main orchestrator for 36-feature extraction (Section 8.2)
- **Key Classes**:
  - `FeatureExtractor`: Manages all feature extraction
- **FeatureExtractor Class Methods**:
  - `extract_features(signal)`: Returns 36-dim feature vector (numpy array)
  - `extract_time_domain_features(signal)`: Returns dict with 7 time features
  - `extract_frequency_domain_features(signal)`: Returns dict with 12 frequency features
- **Module-Level Functions** (in separate files):
  - `extract_time_domain_features(signal)`: 7 time features (time_domain.py)
  - `extract_frequency_domain_features(signal, fs)`: 12 frequency features (frequency_domain.py)
  - `extract_envelope_features(signal, fs)`: 4 envelope features (envelope_analysis.py)
  - `extract_wavelet_features(signal, fs)`: 7 wavelet features (wavelet_features.py)
  - `extract_bispectrum_features(signal)`: 6 bispectrum features (bispectrum.py)
- **Dependencies**: `numpy`, `scipy.signal`, `scipy.fft`, `pywt`

**`features/time_domain.py`**
- **Purpose**: Time-domain statistical features (Section 8.2.1)
- **Key Functions**:
  - `compute_rms(signal)`: Root mean square
  - `compute_kurtosis(signal)`: 4th moment (impulsiveness)
  - `compute_skewness(signal)`: 3rd moment (asymmetry)
  - `compute_crest_factor(signal)`: peak-to-RMS ratio
  - `compute_shape_factor(signal)`: RMS/mean ratio
  - `compute_impulse_factor(signal)`: peak/mean ratio
  - `compute_clearance_factor(signal)`: peak/sqrt(mean(abs))
- **Returns**: Dict with 7 features
- **Dependencies**: `numpy`, `scipy.stats`

**`features/frequency_domain.py`**
- **Purpose**: Frequency-domain features (Section 8.2.3)
- **Key Functions**:
  - `compute_fft(signal, fs)`: Power spectral density
  - `compute_dominant_frequency(psd, freqs)`: Peak frequency
  - `compute_spectral_centroid(psd, freqs)`: Center of mass
  - `compute_spectral_entropy(psd)`: Shannon entropy of spectrum
  - `compute_band_energy(psd, freqs, band_range)`: Energy in frequency band
  - `compute_harmonic_ratios(psd, freqs, f0)`: 2X/1X, 3X/1X ratios
- **Returns**: Dict with 12 frequency-domain features
- **Dependencies**: `numpy`, `scipy.signal`, `scipy.fft`

**`features/envelope_analysis.py`**
- **Purpose**: Hilbert envelope features (Section 8.2.2)
- **Key Functions**:
  - `compute_envelope(signal)`: Hilbert transform → |signal_analytic|
  - `extract_envelope_features(envelope)`: RMS, kurtosis, peak, modulation freq
- **Returns**: Dict with 4 features
- **Dependencies**: `numpy`, `scipy.signal.hilbert`

**`features/wavelet_features.py`**
- **Purpose**: Wavelet transform features (Section 8.2.5)
- **Key Functions**:
  - `compute_dwt_energy(signal, wavelet='db4', level=5)`: Discrete wavelet energy
  - `compute_wavelet_kurtosis(signal)`: Kurtosis of wavelet coefficients
  - `compute_cepstral_peak_ratio(signal, fs)`: Cepstrum analysis
  - `compute_quefrency_centroid(cepstrum)`: Cepstrum center of mass
- **Returns**: Dict with 7 features
- **Dependencies**: `pywt` (PyWavelets), `numpy`

**`features/bispectrum.py`**
- **Purpose**: Higher-order spectral analysis (Section 8.2.6)
- **Key Functions**:
  - `compute_bispectrum(signal)`: Third-order spectrum
  - `compute_bispectrum_peak(bispec)`: Peak value in bispectrum
  - `compute_phase_coupling(signal)`: Quadratic phase coupling indicator
- **Returns**: Dict with 6 features
- **Dependencies**: `numpy`, `scipy.signal`

**`features/feature_selector.py`**
- **Purpose**: Post-split MRMR feature selection (Section 8.4)
- **Key Classes**:
  - `FeatureSelector`: MRMR implementation
- **Key Functions**:
  - `fit(X_train, y_train)`: Compute mutual information, select top 15
  - `transform(X)`: Apply feature mask
  - `get_selected_features()`: Return feature names/indices
- **Returns**: Selected feature indices
- **Dependencies**: `sklearn.feature_selection`, `numpy`

#### **2. Classical ML Models (6 files)**

**`models/classical/svm_classifier.py`**
- **Purpose**: Support Vector Machine with ECOC (Section 9.2)
- **Key Classes**:
  - `SVMClassifier`: Wrapper for sklearn SVM
- **Key Functions**:
  - `train(X_train, y_train, hyperparams)`: Train with given hyperparams
  - `predict(X_test)`: Return class predictions
  - `predict_proba(X_test)`: Return probability estimates
- **Hyperparameters**: Box constraint C, kernel scale γ
- **Dependencies**: `sklearn.svm.SVC`, `sklearn.multiclass.OutputCodeClassifier`

**`models/classical/random_forest.py`**
- **Purpose**: Random Forest ensemble (Section 9.3)
- **Key Classes**:
  - `RandomForestClassifier`: Wrapper for sklearn RF
- **Key Functions**:
  - `train(X_train, y_train, hyperparams)`: Train forest
  - `predict(X_test)`: Predictions
  - `get_feature_importances()`: Gini importances
- **Hyperparameters**: n_estimators, min_leaf_size, max_depth
- **Dependencies**: `sklearn.ensemble.RandomForestClassifier`

**`models/classical/neural_network.py`**
- **Purpose**: Multi-layer perceptron (Section 9.4)
- **Key Classes**:
  - `MLPClassifier`: 3-layer MLP
- **Architecture**: 36 → 20 → 10 → 11 (from report)
- **Key Functions**:
  - `train(X_train, y_train, hyperparams)`: Train with backprop
  - `predict(X_test)`: Predictions
- **Hyperparameters**: Learning rate, dropout, epochs
- **Dependencies**: `sklearn.neural_network.MLPClassifier` or PyTorch

**`models/classical/gradient_boosting.py`**
- **Purpose**: Gradient boosting trees (mentioned in report Table 7)
- **Key Classes**:
  - `GradientBoostingClassifier`: Wrapper for sklearn GBM
- **Key Functions**:
  - `train(X_train, y_train, hyperparams)`: Train boosted trees
  - `predict(X_test)`: Predictions
- **Hyperparameters**: Learning rate, n_estimators, max_depth
- **Dependencies**: `sklearn.ensemble.GradientBoostingClassifier`

**`models/classical/stacked_ensemble.py`** ⚠️ *Not integrated - manual use only*
- **Purpose**: Meta-learner combining base models (Section 9.1)
- **Status**: NOT trained by ModelSelector (only SVM, RF, NN, GBM are auto-trained)
- **Key Classes**:
  - `StackedEnsemble`: Stacking with logistic regression meta-learner
- **Key Functions**:
  - `train(X_train, y_train, base_models)`: Train meta-learner
  - `predict(X_test)`: Aggregate base model predictions
- **Dependencies**: `sklearn.linear_model.LogisticRegression`

**`models/classical/model_selector.py`**
- **Purpose**: Automated model selection based on validation accuracy
- **Key Functions**:
  - `train_all_models(X_train, y_train, X_val, y_val)`: Train all classical models
  - `select_best_model(models, metric='accuracy')`: Return best performer
  - `cross_validate_models(X, y, cv=5)`: K-fold CV comparison
- **Dependencies**: All classical model classes

#### **3. Hyperparameter Optimization (3 files)**

**`training/bayesian_optimizer.py`**
- **Purpose**: Bayesian optimization for hyperparameters (Section 9.5)
- **Key Classes**:
  - `BayesianOptimizer`: Optuna-based optimizer
- **Key Functions**:
  - `optimize(model_class, X_train, y_train, X_val, y_val, n_trials)`: Run optimization
  - `suggest_hyperparameters(trial)`: Define search space
  - `objective(trial)`: Objective function (validation accuracy)
- **Search Spaces**:
  - SVM: C ∈ [0.1, 100] (log scale), γ ∈ [0.01, 10]
  - RF: n_estimators ∈ [50, 200], min_leaf_size ∈ [1, 20]
  - NN: learning_rate ∈ [1e-4, 1e-1] (log scale), dropout ∈ [0.1, 0.5]
- **Dependencies**: `optuna`, `sklearn`

**`training/grid_search.py`** ⚠️ *Not integrated - manual use only*
- **Purpose**: Grid search for exhaustive hyperparameter search
- **Status**: NOT used in pipeline (only BayesianOptimizer is integrated)
- **Key Functions**:
  - `grid_search(model_class, X_train, y_train, param_grid, cv=5)`: Grid search
- **Usage**: When search space is discrete/small
- **Dependencies**: `sklearn.model_selection.GridSearchCV`

**`training/random_search.py`** ⚠️ *Not integrated - manual use only*
- **Purpose**: Random search as baseline optimizer
- **Status**: NOT used in pipeline (only BayesianOptimizer is integrated)
- **Key Functions**:
  - `random_search(model_class, X_train, y_train, param_distributions, n_iter)`: Random sampling
- **Dependencies**: `sklearn.model_selection.RandomizedSearchCV`

#### **4. Feature Engineering Enhancements (4 files)**

**`features/advanced_features.py`** ⚠️ *Not integrated - manual use only*
- **Purpose**: Optional 16 advanced features (Section 8.3) - expensive to compute
- **Status**: Implemented but NOT integrated in FeatureExtractor or pipeline
- **Usage**: Can be imported and used manually for research purposes
- **Key Functions**:
  - `extract_cwt_features(signal)`: Continuous wavelet transform energy
  - `extract_wpt_features(signal)`: Wavelet packet decomposition
  - `compute_lyapunov_exponent(signal)`: Chaos indicator
  - `compute_sample_entropy(signal)`: Irregularity measure
  - `compute_dfa(signal)`: Detrended fluctuation analysis
- **Returns**: Dict with 16 features
- **Computational Cost**: ~10× slower than base features (from guide)
- **Dependencies**: `nolds` (nonlinear dynamics library), `pywt`, `numpy`

**`features/feature_normalization.py`**
- **Purpose**: Standardization and normalization
- **Key Classes**:
  - `FeatureNormalizer`: Z-score normalization
- **Key Functions**:
  - `fit(X_train)`: Compute mean/std from training data
  - `transform(X)`: Apply normalization
  - `inverse_transform(X)`: Revert normalization
- **Dependencies**: `sklearn.preprocessing.StandardScaler`

**`features/feature_validator.py`** ⚠️ *Not integrated - manual use only*
- **Purpose**: Validate extracted features
- **Status**: NOT called in pipeline (no automatic validation)
- **Key Functions**:
  - `validate_feature_vector(features, expected_dim=36)`: Check shape/NaN/Inf
  - `check_feature_distribution(features)`: Warn if constant features
- **Dependencies**: `numpy`

**`features/feature_importance.py`** ⚠️ *Not integrated - manual use only*
- **Purpose**: Analyze feature importance from trained models
- **Status**: NOT used in pipeline (results don't include importance analysis)
- **Key Functions**:
  - `get_random_forest_importances(rf_model)`: Gini importances
  - `get_permutation_importances(model, X_val, y_val)`: Permutation importance
  - `plot_feature_importances(importances, feature_names)`: Visualization
- **Dependencies**: `sklearn.inspection`, `matplotlib`

#### **4. Pipeline Integration (4 files)**

**`pipelines/classical_ml_pipeline.py`**
- **Purpose**: End-to-end classical ML pipeline (replaces `pipeline.m`)
- **Key Classes**:
  - `ClassicalMLPipeline`: Orchestrates feature extraction → training → evaluation
- **Key Functions**:
  - `run_full_pipeline(config)`: Main entry point
  - `_extract_features(dataset)`: Apply feature extraction
  - `_select_features(X_train, y_train)`: MRMR selection
  - `_train_models(X_train, y_train)`: Train all models with hyperparam tuning
  - `_evaluate_models(models, X_test, y_test)`: Test set evaluation
- **Dependencies**: All feature/model/evaluation modules

**`pipelines/feature_pipeline.py`**
- **Purpose**: Standalone feature extraction pipeline
- **Key Functions**:
  - `extract_dataset_features(signals, fs, save_path)`: Batch feature extraction
  - `load_extracted_features(path)`: Load precomputed features
- **Usage**: Separate feature extraction from training for caching
- **Dependencies**: `features/`, `utils/file_io.py`

**`pipelines/matlab_compat.py`**
- **Purpose**: Compatibility layer with MATLAB code
- **Key Functions**:
  - `convert_matlab_features_to_python(mat_file)`: Import MATLAB feature matrix
  - `export_python_features_to_matlab(features, save_path)`: Export for MATLAB
- **Dependencies**: `scipy.io`, `numpy`

**`pipelines/pipeline_validator.py`**
- **Purpose**: Validate pipeline correctness
- **Key Functions**:
  - `validate_pipeline_output(results, expected_accuracy=0.92)`: Check against baseline
  - `compare_with_matlab_baseline(python_results, matlab_results)`: Numerical comparison
- **Dependencies**: `numpy`, `scipy.stats`

#### **5. Visualization (4 files)**

**`visualization/feature_visualization.py`**
- **Purpose**: Feature correlation and distribution plots (Figures 4, 5)
- **Key Functions**:
  - `plot_correlation_matrix(features, feature_names)`: Heatmap (Fig 4)
  - `plot_feature_distributions(features, labels, feature_names)`: Boxplots (Fig 5)
  - `plot_tsne_clusters(features, labels)`: t-SNE visualization (Fig 6)
- **Dependencies**: `matplotlib`, `seaborn`, `sklearn.manifold.TSNE`

**`visualization/performance_plots.py`**
- **Purpose**: Model performance visualizations (Figures 7, 8)
- **Key Functions**:
  - `plot_model_comparison(results, model_names)`: Bar chart (Fig 7)
  - `plot_confusion_matrix(cm, class_names, normalize=True)`: Confusion matrix (Fig 8)
  - `plot_roc_curves(roc_data, class_names)`: One-vs-rest ROC (Fig 9)
- **Dependencies**: `matplotlib`, `seaborn`

**`visualization/signal_plots.py`**
- **Purpose**: Signal examples (Figures 2, 3)
- **Key Functions**:
  - `plot_signal_examples(signals, labels, fs)`: Time/freq/spectrogram (Fig 2, 3)
- **Dependencies**: `matplotlib`, `scipy.signal`

**`visualization/dashboard.py`**
- **Purpose**: Interactive dashboard for exploration
- **Key Classes**:
  - `ExperimentDashboard`: Streamlit-based dashboard
- **Features**:
  - Upload signals, visualize features
  - Compare model predictions
  - Explore confusion matrices
- **Dependencies**: `streamlit`, `plotly`

### Architecture Decisions

**1. Scikit-learn for Classical ML**
- **Decision**: Use sklearn instead of porting MATLAB exactly
- **Rationale**:
  - sklearn is industry-standard, well-tested
  - Better integration with Python ecosystem
  - MATLAB's fitcecoc ≈ sklearn's OutputCodeClassifier
- **Validation**: Train both, ensure accuracy difference < 1%

**2. Feature Extraction Modularity**
- **Decision**: Separate files for each feature domain (time, freq, wavelet)
- **Rationale**:
  - Easier testing (unit test each domain)
  - Selective computation (skip expensive advanced features)
  - Clear separation of concerns
- **Trade-off**: Slightly more overhead than monolithic function

**3. Bayesian Optimization (Optuna) over Grid Search**
- **Decision**: Use Optuna for hyperparameter tuning
- **Rationale**:
  - More efficient than grid search (50 trials vs. 1000s evaluations)
  - Built-in pruning of unpromising trials
  - Easily parallelize trials
- **From Report**: Existing pipeline uses 50 Bayesian iterations (Section 9.5)

**4. Post-Split Feature Selection**
- **Decision**: Apply MRMR after train/test split (not before)
- **Rationale**: Prevent data leakage (Section 8.4, Innovation #5 in report)
- **Impact**: Slightly lower accuracy but scientifically rigorous

**5. Separate Feature Extraction Pipeline**
- **Decision**: Allow pre-computing features, saving to disk
- **Rationale**:
  - Feature extraction takes ~3 minutes for 1400 signals
  - Training multiple models doesn't need to recompute features
  - Cache features as `.npz` or `.h5` files
- **Benefit**: Faster experimentation

### Data Flow

```
┌───────────────────────────────────────────────────────────┐
│         CLASSICAL ML PIPELINE (Phase 1)                    │
└───────────────────────────────────────────────────────────┘

1. FEATURE EXTRACTION
   ┌─────────────────────────────────────────────────────┐
   │ Input: Raw signals [N, T] from Phase 0              │
   │         ↓                                            │
   │ features/feature_extractor.py                        │
   │  ├─ time_domain.py → 7 features                    │
   │  ├─ frequency_domain.py → 12 features              │
   │  ├─ envelope_analysis.py → 4 features              │
   │  ├─ wavelet_features.py → 7 features               │
   │  └─ bispectrum.py → 6 features                     │
   │         ↓                                            │
   │ Output: Feature matrix [N, 36]                       │
   │         Saved to: features_train.npz                 │
   └─────────────────────────────────────────────────────┘
                        ↓

2. FEATURE SELECTION
   ┌─────────────────────────────────────────────────────┐
   │ Input: Features [N_train, 36], labels [N_train]     │
   │         ↓                                            │
   │ features/feature_selector.py (MRMR)                 │
   │  ├─ Compute mutual information I(f; y)             │
   │  ├─ Rank features by relevance                     │
   │  └─ Select top 15 features                         │
   │         ↓                                            │
   │ Output: Feature mask [15], selected features [N, 15]│
   └─────────────────────────────────────────────────────┘
                        ↓

3. NORMALIZATION
   ┌─────────────────────────────────────────────────────┐
   │ Input: Features [N, 15]                              │
   │         ↓                                            │
   │ features/feature_normalization.py                    │
   │  ├─ Fit on training data (compute μ, σ)            │
   │  ├─ Transform train/val/test                        │
   │         ↓                                            │
   │ Output: Normalized features [N, 15]                  │
   └─────────────────────────────────────────────────────┘
                        ↓

4. HYPERPARAMETER OPTIMIZATION
   ┌─────────────────────────────────────────────────────┐
   │ For each model (SVM, RF, NN, GBM):                  │
   │         ↓                                            │
   │ training/bayesian_optimizer.py                       │
   │  ├─ Optuna trial loop (50 iterations)              │
   │  ├─ Train model with trial hyperparams             │
   │  ├─ Evaluate on validation set                     │
   │  └─ Select best hyperparams                        │
   │         ↓                                            │
   │ Output: Optimized hyperparameters (dict)            │
   └─────────────────────────────────────────────────────┘
                        ↓

5. FINAL MODEL TRAINING
   ┌─────────────────────────────────────────────────────┐
   │ Input: Best hyperparams, full training data         │
   │         ↓                                            │
   │ models/classical/[svm|rf|nn|gbm]_classifier.py      │
   │  ├─ Train with optimized hyperparams               │
   │  ├─ Save trained model                             │
   │         ↓                                            │
   │ Output: Trained models (saved to disk)              │
   │         best_model.pkl                               │
   └─────────────────────────────────────────────────────┘
                        ↓

6. MODEL SELECTION
   ┌─────────────────────────────────────────────────────┐
   │ Input: All trained models                            │
   │         ↓                                            │
   │ models/classical/model_selector.py                   │
   │  ├─ Evaluate all models on validation set          │
   │  ├─ Compare accuracy, F1, AUC                      │
   │  └─ Select best model (Random Forest = 95.81%)    │
   │         ↓                                            │
   │ Output: Best model pointer                          │
   └─────────────────────────────────────────────────────┘
                        ↓

7. FINAL EVALUATION
   ┌─────────────────────────────────────────────────────┐
   │ Input: Best model, test data                         │
   │         ↓                                            │
   │ evaluation/evaluator.py                              │
   │  ├─ Predict on test set                            │
   │  ├─ Compute accuracy, F1, confusion matrix         │
   │  └─ Generate classification report                 │
   │         ↓                                            │
   │ Output: Evaluation metrics (JSON)                    │
   │         Test accuracy: 95.33%                        │
   └─────────────────────────────────────────────────────┘
```

### Integration Points

**1. With Phase 0 (Data Generation)**
- **Input**: Signals from `data/cache_manager.py` (HDF5 file)
- **Interface**: `BearingFaultDataset` loads signals, passes to feature extractor
- **Testing**: Verify feature extraction works on Phase 0 generated signals

**2. With Existing MATLAB Code**
- **Compatibility**: `pipelines/matlab_compat.py` converts MATLAB feature matrices
- **Validation**: Load MATLAB features, train models, compare accuracy to report (95.33%)
- **Use Case**: Users with existing MATLAB data can import

**3. With Deep Learning (Future Phases)**
- **Baseline**: Classical ML results serve as baseline for DL comparison (Section 11.9)
- **Hybrid Models**: Phase 6 will use classical features as additional input to DL models
- **Ensemble**: Phase 8 may combine classical + DL predictions

**4. With Visualization**
- **Figures**: Reproduce all figures from report (Figures 2-9)
- **Dashboard**: Interactive exploration of feature importances, confusion matrices
- **MLflow**: Log plots as artifacts

### Testing Strategy

**1. Unit Tests**

**`tests/test_feature_extraction.py`**
```python
def test_time_domain_features():
    """Test time-domain feature extraction."""
    signal = np.random.randn(10000)  # Mock signal
    features = time_domain.extract_time_domain(signal)
    assert len(features) == 7
    assert 'RMS' in features
    assert not np.isnan(features['Kurtosis'])

def test_feature_selector():
    """Test MRMR feature selection."""
    X = np.random.rand(100, 36)  # Mock features
    y = np.random.randint(0, 11, 100)  # Mock labels
    selector = FeatureSelector()
    selector.fit(X, y)
    selected_indices = selector.get_selected_features()
    assert len(selected_indices) == 15
```

**`tests/test_classical_models.py`**
```python
def test_random_forest_training():
    """Test Random Forest trains without errors."""
    X_train = np.random.rand(100, 15)
    y_train = np.random.randint(0, 11, 100)
    rf = RandomForestClassifier()
    rf.train(X_train, y_train, hyperparams={'n_estimators': 100})
    assert rf.model is not None
    
    # Test prediction
    X_test = np.random.rand(20, 15)
    preds = rf.predict(X_test)
    assert len(preds) == 20
```

**2. Integration Tests**

**`tests/test_full_pipeline.py`**
```python
def test_end_to_end_pipeline():
    """Test full classical ML pipeline."""
    # Generate small dataset
    dataset = generate_small_test_dataset(n_samples=100)
    
    # Run pipeline
    pipeline = ClassicalMLPipeline(config)
    results = pipeline.run_full_pipeline(dataset)
    
    # Check outputs
    assert 'best_model' in results
    assert results['test_accuracy'] > 0.8  # Lower threshold for small data
    assert 'confusion_matrix' in results
```

**3. MATLAB Parity Tests**

**`tests/test_matlab_parity.py`**
```python
def test_feature_extraction_parity():
    """Ensure Python features match MATLAB features."""
    # Load same signal in both MATLAB and Python
    signal = load_test_signal('test_signal.mat')
    
    # Python features
    python_features = FeatureExtractor().extract_features(signal, fs=20480)
    
    # MATLAB features (precomputed)
    matlab_features = load_matlab_features('test_signal_features.mat')
    
    # Compare (allow 1% numerical difference)
    np.testing.assert_allclose(
        python_features, matlab_features, rtol=0.01, atol=0.01
    )
```

**4. Performance Regression Tests**

**`tests/test_performance_regression.py`**
```python
def test_accuracy_regression():
    """Ensure accuracy doesn't drop below baseline."""
    # Load standard test set
    test_dataset = load_standard_test_set()
    
    # Train model
    model = train_random_forest(config)
    
    # Evaluate
    accuracy = evaluate_model(model, test_dataset)
    
    # Assert accuracy >= 93% (allowing 2% margin from 95.33%)
    assert accuracy >= 0.93, f"Accuracy dropped to {accuracy:.2%}"
```

### Acceptance Criteria

**Phase 1 Complete When:**

✅ **Feature extraction working**
- Extracts 36 features from signals
- Features match MATLAB output within 1%
- Handles all 1,430 signals in < 5 minutes

✅ **Feature selection operational**
- MRMR selects 15 features
- Selection only uses training data (no leakage)
- Selected features include top-ranked from report (envelope modulation freq, spectral centroid)

✅ **All classical models train successfully**
- SVM, Random Forest, Neural Network, Gradient Boosting
- Hyperparameter optimization completes in < 15 minutes per model
- Models save/load correctly

✅ **Achieves baseline performance**
- Random Forest: ≥ 95% validation accuracy
- Random Forest: ≥ 93% test accuracy (allowing 2% margin)
- Per-class recall: ≥ 85% for at least 10/11 classes

✅ **Reproduces report figures**
- Figure 4: Feature correlation heatmap
- Figure 5: Feature distributions by class
- Figure 6: t-SNE visualization
- Figure 7: Model comparison bar chart
- Figure 8: Confusion matrix

✅ **MATLAB parity validated**
- Feature extraction: < 1% difference
- Model predictions: Same best model selected (Random Forest)
- Accuracy: Within 2% of reported 95.33%

✅ **Pipeline integration**
- `ClassicalMLPipeline` runs end-to-end without manual intervention
- Config-driven (no hardcoded parameters)
- Logs to MLflow correctly

✅ **Documentation complete**
- README explaining feature extraction process
- API documentation for all modules
- Example notebook demonstrating usage

### Estimated Effort

**Time Breakdown:**
- Feature extraction module (7 files): 5 days
  - Time-domain: 0.5 days
  - Frequency-domain: 1 day (FFT, spectral features)
  - Envelope analysis: 0.5 days
  - Wavelet features: 1 day (PyWavelets integration)
  - Bispectrum: 1 day (complex)
  - Feature selector: 0.5 days
  - Feature extractor orchestrator: 0.5 days
  
- Classical ML models (6 files): 3 days
  - SVM, RF, NN, GBM wrappers: 2 days
  - Ensemble stacking: 0.5 days
  - Model selector: 0.5 days
  
- Hyperparameter optimization (3 files): 2 days
  - Bayesian optimizer (Optuna): 1 day
  - Grid/random search: 1 day
  
- Pipeline integration (4 files): 2 days
  - Classical ML pipeline: 1 day
  - MATLAB compatibility: 0.5 days
  - Validators: 0.5 days
  
- Visualization (4 files): 3 days
  - Feature plots: 1 day
  - Performance plots: 1 day
  - Signal plots: 0.5 days
  - Dashboard (Streamlit): 0.5 days
  
- Testing (MATLAB parity, unit tests): 3 days
- Documentation: 2 days
- Buffer for debugging: 3 days

**Total: ~23 days (1 month) for Phase 1**

**Complexity**: ⭐⭐⭐☆☆ (Moderate-High)
- Feature engineering requires domain knowledge
- MATLAB parity testing is tedious
- Bayesian optimization integration needs care

**Dependencies**: Phase 0 (data generation)

**Risk**: Medium
- MATLAB numerical differences may be tricky to resolve
- Optuna integration may have edge cases

---
