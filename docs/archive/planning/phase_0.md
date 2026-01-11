## **PHASE 0: Project Foundation & Deep Learning Infrastructure**

### Phase Objective
Establish the complete project structure, deep learning framework integration, and foundational utilities that all subsequent phases will depend upon. Create a robust, modular architecture supporting both classical ML (existing) and deep learning (new) approaches.

### Complete File List (42 files)

#### **Root Directory Structure**
```
project_root/
├── config/                    # Configuration management
├── data/                      # Data generation & management
├── features/                  # Feature engineering (classical ML)
├── models/                    # Model architectures
├── training/                  # Training infrastructure
├── evaluation/               # Evaluation & metrics
├── deployment/               # Production deployment
├── visualization/            # Plotting & dashboards
├── utils/                    # Shared utilities
├── experiments/              # Experiment tracking
└── tests/                    # Unit & integration tests
```

#### **1. Configuration System (5 files)**

**`config/base_config.py`**
- **Purpose**: Master configuration base class with validation
- **Key Classes**: 
  - `BaseConfig`: Abstract base with validation logic
  - `ConfigValidator`: JSON schema validation
- **Key Functions**:
  - `load_from_yaml(path)`: Load config from YAML
  - `save_to_yaml(path)`: Save config state
  - `validate()`: Ensure config consistency
  - `merge_configs(configs)`: Combine multiple configs
- **Dependencies**: `yaml`, `jsonschema`, `dataclasses`

**`config/data_config.py`**
- **Purpose**: Data generation parameters from existing generator.m
- **Key Classes**:
  - `SignalConfig`: Sampling, duration, speed parameters
  - `FaultConfig`: Fault types, severity levels
  - `NoiseConfig`: 7-layer noise configuration
  - `AugmentationConfig`: Augmentation strategies
- **Key Functions**:
  - `from_matlab_struct()`: Import existing MATLAB configs
  - `to_dict()`: Export for JSON serialization
- **Dependencies**: `base_config.py`, `scipy.io`

**`config/model_config.py`**
- **Purpose**: All model architectures configuration
- **Key Classes**:
  - `ClassicalMLConfig`: SVM, RF, NN hyperparameters
  - `CNNConfig`: 1D-CNN architecture parameters
  - `ResNetConfig`: ResNet-18 adapted for signals
  - `TransformerConfig`: Attention mechanism parameters
  - `HybridConfig`: Physics-informed neural network settings
- **Key Functions**:
  - `get_model_config(model_name)`: Factory pattern
  - `validate_hyperparams()`: Check parameter ranges
- **Dependencies**: `base_config.py`

**`config/training_config.py`**
- **Purpose**: Training loop parameters, optimization settings
- **Key Classes**:
  - `OptimizerConfig`: Adam, SGD, learning rate schedules
  - `LossConfig`: Loss functions, class weights
  - `CallbackConfig`: Early stopping, checkpointing
  - `DataLoaderConfig`: Batch size, num_workers
- **Key Functions**:
  - `create_optimizer(optimizer_type, model_params, **kwargs)`: Initialize optimizer
  - `create_scheduler(optimizer)`: Learning rate scheduler
- **Dependencies**: `base_config.py`, `torch.optim`

**`config/experiment_config.py`**
- **Purpose**: MLflow experiment tracking configuration
- **Key Classes**:
  - `ExperimentConfig`: Experiment name, tags, tracking URI
  - `LoggingConfig`: What metrics/artifacts to log
- **Key Functions**:
  - `init_mlflow_tracking()`: Set up MLflow
  - `log_config_to_mlflow(config)`: Store config as artifact
- **Dependencies**: `base_config.py`, `mlflow`

#### **2. Data Infrastructure (8 files)**

**`data/signal_generator.py`**
- **Purpose**: Python port of generator.m with enhancements
- **Key Classes**:
  - `SignalGenerator`: Main generation orchestrator
  - `FaultModeler`: Physics-based fault equations (from Section 7.3)
  - `NoiseGenerator`: 7-layer noise implementation
- **Key Functions**:
  - `generate_fault_signal(fault_type, severity, params)`: Core generation
  - `apply_temporal_evolution(signal, progression_params)`: Severity growth
  - `calculate_sommerfeld_number(load, speed, temp)`: Operating conditions
- **Dependencies**: `numpy`, `scipy.signal`, `data_config.py`

**`data/augmentation.py`**
- **Purpose**: Advanced augmentation beyond time-shift/scale
- **Key Classes**:
  - `SignalAugmenter`: Augmentation pipeline manager
  - `TimeWarping`: Non-linear time stretching
  - `MixUp`: Sample mixing for robustness
- **Key Functions**:
  - `time_shift(signal, shift_pct)`: Circular shift
  - `amplitude_scale(signal, scale_factor)`: Multiply amplitude
  - `add_noise(signal, snr_db)`: Inject Gaussian noise
  - `mixup(signal1, signal2, alpha)`: Convex combination
  - `time_warp(signal, warp_params)`: Dynamic time warping
- **Dependencies**: `numpy`, `scipy.interpolate`

**`data/dataset.py`**
- **Purpose**: PyTorch Dataset classes for efficient loading
- **Key Classes**:
  - `BearingFaultDataset(torch.utils.data.Dataset)`: Main dataset
  - `CachedDataset`: In-memory caching for speed
  - `StreamingDataset`: On-the-fly generation for large-scale
- **Key Functions**:
  - `__getitem__(idx)`: Return (signal, label) pair
  - `__len__()`: Dataset size
  - `get_class_weights()`: For imbalanced handling
  - `stratified_split(train_ratio, val_ratio, test_ratio)`: Data splitting
- **Dependencies**: `torch.utils.data`, `signal_generator.py`

**`data/dataloader.py`**
- **Purpose**: PyTorch DataLoader wrappers with optimizations
- **Key Classes**:
  - `FastDataLoader`: Optimized loading with prefetching
  - `BalancedBatchSampler`: Ensure class balance per batch
- **Key Functions**:
  - `create_dataloaders(dataset, config)`: Factory for train/val/test loaders
  - `collate_fn(batch)`: Custom batching logic
- **Dependencies**: `torch.utils.data`, `dataset.py`

**`data/transforms.py`**
- **Purpose**: Signal preprocessing transformations
- **Key Classes**:
  - `Compose`: Chain multiple transforms
  - `Normalize`: Z-score normalization
  - `Resample`: Change sampling rate
  - `BandpassFilter`: Frequency filtering
- **Key Functions**:
  - `__call__(signal)`: Apply transformation
- **Dependencies**: `numpy`, `scipy.signal`

**`data/matlab_importer.py`**
- **Purpose**: Import existing .mat files from generator.m
- **Key Functions**:
  - `load_mat_signals(directory)`: Batch load .mat files
  - `convert_matlab_to_pytorch(mat_data)`: Format conversion
  - `extract_metadata(mat_struct)`: Parse CONFIG structure
- **Dependencies**: `scipy.io`, `numpy`

**`data/data_validator.py`**
- **Purpose**: Validate generated data quality
- **Key Classes**:
  - `SignalValidator`: Check signal properties
  - `DistributionValidator`: Verify class balance
- **Key Functions**:
  - `validate_signal_quality(signal, expected_properties)`: Check SNR, duration
  - `validate_fault_characteristics(signal, fault_type)`: Verify dominant frequencies
  - `check_class_balance(dataset)`: Ensure stratification
- **Dependencies**: `numpy`, `scipy.stats`

**`data/cache_manager.py`**
- **Purpose**: Efficient caching for repeated experiments
- **Key Classes**:
  - `CacheManager`: Disk/memory cache orchestrator
- **Key Functions**:
  - `cache_dataset(dataset, cache_path)`: Save to HDF5
  - `load_cached_dataset(cache_path)`: Fast loading
  - `invalidate_cache(cache_path)`: Clear cache
- **Dependencies**: `h5py`, `pickle`

#### **3. Deep Learning Model Zoo (7 files)**

**`models/base_model.py`**
- **Purpose**: Abstract base for all models
- **Key Classes**:
  - `BaseModel(nn.Module)`: Abstract base with common methods
  - `ModelRegistry`: Track available models
- **Key Functions**:
  - `forward(x)`: Abstract method
  - `get_num_params()`: Count parameters
  - `get_feature_extractor()`: Return backbone
  - `freeze_backbone()`: Transfer learning utility
- **Dependencies**: `torch.nn`

**`models/cnn_1d.py`**
- **Purpose**: 1D CNN for raw signal classification
- **Key Classes**:
  - `CNN1D(BaseModel)`: Main CNN architecture
  - `ConvBlock`: Conv-BN-ReLU-Dropout block
- **Key Functions**:
  - `forward(x)`: Input [B, 1, T] → Output [B, 11]
  - `_make_conv_layer(in_ch, out_ch, kernel)`: Layer factory
- **Architecture**: 6 conv layers, adaptive pooling, 2 FC layers
- **Parameters**: ~500K trainable parameters
- **Dependencies**: `torch.nn`, `base_model.py`

**`models/resnet_1d.py`**
- **Purpose**: ResNet-18 adapted for 1D signals
- **Key Classes**:
  - `ResNet1D(BaseModel)`: Main ResNet architecture
  - `BasicBlock1D`: Residual block with skip connections
- **Key Functions**:
  - `forward(x)`: Input [B, 1, T] → Output [B, 11]
  - `_make_layer(block, channels, num_blocks)`: Build residual layers
- **Architecture**: 4 residual stages, global average pooling
- **Parameters**: ~1M trainable parameters
- **Dependencies**: `torch.nn`, `base_model.py`

**`models/transformer.py`**
- **Purpose**: Transformer encoder for time-series
- **Key Classes**:
  - `SignalTransformer(BaseModel)`: Transformer for signals
  - `PositionalEncoding`: Learnable position embeddings
  - `TransformerEncoderBlock`: Multi-head attention + FFN
- **Key Functions**:
  - `forward(x)`: Input [B, 1, T] → Output [B, 11]
  - `_create_attention_mask()`: Causal masking (if needed)
- **Architecture**: Patch embedding, 6 transformer layers, classification head
- **Parameters**: ~800K trainable parameters
- **Dependencies**: `torch.nn`, `base_model.py`

**`models/hybrid_pinn.py`**
- **Purpose**: Physics-Informed Neural Network combining data + physics
- **Key Classes**:
  - `HybridPINN(BaseModel)`: PINN architecture
  - `PhysicsConstraint`: Encodes bearing dynamics equations
  - `FeatureFusion`: Merge learned + engineered features
- **Key Functions**:
  - `forward(x, physics_features)`: Dual input
  - `compute_physics_loss(predictions, physics_params)`: Physics constraint
- **Architecture**: CNN backbone + physics branch, fusion layer
- **Parameters**: ~1.2M trainable parameters
- **Dependencies**: `torch.nn`, `base_model.py`, `cnn_1d.py`

**`models/ensemble.py`**
- **Purpose**: Ensemble combining multiple models
- **Key Classes**:
  - `EnsembleModel(BaseModel)`: Aggregates multiple models
  - `VotingEnsemble`: Hard/soft voting
  - `StackedEnsemble`: Meta-learner on top
- **Key Functions**:
  - `forward(x)`: Run all base models, aggregate predictions
  - `add_model(model, weight)`: Register ensemble member
- **Dependencies**: `torch.nn`, `base_model.py`

**`models/model_factory.py`**
- **Purpose**: Factory pattern for model instantiation
- **Key Functions**:
  - `create_model(model_name, config)`: Instantiate model
  - `load_pretrained(model_name, checkpoint_path)`: Load weights
  - `list_available_models()`: Return registered models
- **Dependencies**: All model classes, `model_config.py`

#### **4. Training Infrastructure (6 files)**

**`training/trainer.py`**
- **Purpose**: Main training loop orchestrator
- **Key Classes**:
  - `Trainer`: Manages training/validation loops
  - `TrainingState`: Tracks epoch, best_loss, etc.
- **Key Functions**:
  - `train_epoch(dataloader)`: Single epoch training
  - `validate_epoch(dataloader)`: Validation pass
  - `fit(num_epochs)`: Main training loop
  - `_backward_pass(loss)`: Gradient computation + clipping
- **Dependencies**: `torch`, `training_config.py`, `callbacks.py`

**`training/callbacks.py`**
- **Purpose**: Callback system for extensibility
- **Key Classes**:
  - `Callback`: Abstract base
  - `EarlyStopping`: Stop on plateau
  - `ModelCheckpoint`: Save best models
  - `LearningRateScheduler`: Adjust LR
  - `TensorBoardLogger`: Log to TensorBoard
  - `MLflowLogger`: Log to MLflow
- **Key Functions**:
  - `on_epoch_end(epoch, logs)`: Hook for callbacks
  - `on_train_begin()`: Setup hook
- **Dependencies**: `torch`, `mlflow`, `tensorboard`

**`training/losses.py`**
- **Purpose**: Loss functions for fault diagnosis
- **Key Classes**:
  - `FocalLoss`: Address class imbalance
  - `LabelSmoothingCrossEntropy`: Regularization
  - `PhysicsInformedLoss`: PINN constraint loss
- **Key Functions**:
  - `forward(predictions, targets)`: Compute loss
  - `compute_class_weights(dataset)`: For weighted loss
- **Dependencies**: `torch.nn`

**`training/cnn_optimizer.py`**
- **Purpose**: Optimizer configurations for CNN training (RECOMMENDED)
- **Key Functions**:
  - `create_optimizer(optimizer_type, model_params, lr, weight_decay, **kwargs)`: Factory for optimizers
  - `create_adamw_optimizer(model_params, lr, weight_decay, ...)`: Create AdamW optimizer
  - `create_sgd_optimizer(model_params, lr, momentum, ...)`: Create SGD with Nesterov momentum
  - `create_rmsprop_optimizer(model_params, lr, ...)`: Create RMSprop optimizer
  - `get_parameter_groups(model, lr, weight_decay, no_decay_bias)`: Separate weight decay for biases/norms
- **Key Classes**:
  - `OptimizerConfig`: Predefined configurations (default, fast_convergence, strong_regularization, sgd_baseline)
- **Dependencies**: `torch.optim`
- **Note**: This is the primary optimizer module; use instead of training.optimizers.create_optimizer()

**`training/optimizers.py`**
- **Purpose**: Optimizer wrappers with learning rate schedules (DEPRECATED: use cnn_optimizer.py)
- **Key Functions**:
  - `create_optimizer(model_params, optimizer_name, **kwargs)`: DEPRECATED, delegates to cnn_optimizer
  - `create_scheduler(optimizer, scheduler_name, **kwargs)`: Cosine annealing, step decay
  - `get_lr(optimizer)`: Current learning rate
  - `set_lr(optimizer, lr)`: Set learning rate
- **Note**: Use `training.cnn_optimizer.create_optimizer(optimizer_type, model_params, ...)` instead
- **Dependencies**: `torch.optim`, `training.cnn_optimizer`

**`training/metrics.py`**
- **Purpose**: Training metrics computation
- **Key Classes**:
  - `MetricsTracker`: Aggregate metrics across batches
- **Key Functions**:
  - `compute_accuracy(preds, targets)`: Classification accuracy
  - `compute_f1_score(preds, targets, average)`: Macro/micro F1
  - `compute_confusion_matrix(preds, targets)`: Confusion matrix
- **Dependencies**: `sklearn.metrics`, `numpy`

**`training/mixed_precision.py`**
- **Purpose**: Mixed precision training for speed
- **Key Classes**:
  - `MixedPrecisionTrainer(Trainer)`: FP16 training
- **Key Functions**:
  - `_backward_pass(loss)`: Scaled gradient computation
- **Dependencies**: `torch.cuda.amp`

#### **5. Evaluation Suite (5 files)**

**`evaluation/evaluator.py`**
- **Purpose**: Comprehensive model evaluation
- **Key Classes**:
  - `ModelEvaluator`: Evaluate on test set
- **Key Functions**:
  - `evaluate(model, dataloader)`: Full evaluation pipeline
  - `compute_per_class_metrics(preds, targets)`: Precision/recall/F1 per class
  - `generate_classification_report()`: Detailed report
- **Dependencies**: `sklearn.metrics`, `metrics.py`

**`evaluation/robustness_tester.py`**
- **Purpose**: Adversarial robustness testing (from Section 10.2)
- **Key Classes**:
  - `RobustnessTester`: Orchestrate robustness tests
  - `NoiseInjector`: Test 1 - Sensor noise
  - `FeatureDropout`: Test 2 - Missing features
  - `TemporalDrift`: Test 3 - Drift simulation
- **Key Functions**:
  - `test_sensor_noise(model, test_loader, noise_levels)`: Inject noise
  - `test_missing_features(model, test_loader, dropout_rates)`: Feature dropout
  - `test_temporal_drift(model, test_loader, drift_params)`: Systematic bias
- **Dependencies**: `numpy`, `evaluator.py`

**`evaluation/confusion_analyzer.py`**
- **Purpose**: Deep confusion matrix analysis (Section 11.5)
- **Key Classes**:
  - `ConfusionAnalyzer`: Analyze misclassification patterns
- **Key Functions**:
  - `analyze_confusion_matrix(cm, class_names)`: Identify error patterns
  - `find_most_confused_pairs()`: Top confusion pairs
  - `compute_error_concentration(cm)`: Mixed fault error percentage
- **Dependencies**: `numpy`, `pandas`

**`evaluation/roc_analyzer.py`**
- **Purpose**: ROC curve analysis (Section 11.6)
- **Key Classes**:
  - `ROCAnalyzer`: One-vs-rest ROC computation
- **Key Functions**:
  - `compute_roc_curves(probs, targets)`: ROC for each class
  - `compute_auc_scores(roc_curves)`: AUC per class
  - `compute_calibration_error(probs, targets)`: ECE
- **Dependencies**: `sklearn.metrics`, `numpy`

**`evaluation/benchmark.py`**
- **Purpose**: Compare against classical ML baselines
- **Key Functions**:
  - `benchmark_against_classical(dl_model, classical_models, test_loader)`: Comparison
  - `generate_comparison_table(results)`: Summary table
- **Dependencies**: `evaluator.py`, `sklearn`

#### **6. Utilities (6 files)**

**`utils/logging.py`**
- **Purpose**: Structured logging setup
- **Key Functions**:
  - `get_logger(name)`: Create logger with formatting
  - `log_system_info()`: Log GPU, CUDA, system specs
- **Dependencies**: `logging`, `torch`

**`utils/reproducibility.py`**
- **Purpose**: Ensure reproducibility
- **Key Functions**:
  - `set_seed(seed)`: Set all random seeds (NumPy, PyTorch, CUDA)
  - `make_deterministic()`: Disable non-deterministic algorithms
- **Dependencies**: `torch`, `numpy`, `random`

**`utils/device_manager.py`**
- **Purpose**: GPU/CPU device management
- **Key Functions**:
  - `get_device(prefer_gpu)`: Return torch.device
  - `move_to_device(data, device)`: Recursive device transfer
  - `get_gpu_memory_usage()`: Monitor GPU memory
- **Dependencies**: `torch`

**`utils/file_io.py`**
- **Purpose**: File I/O utilities
- **Key Functions**:
  - `save_pickle(obj, path)`: Serialize objects
  - `load_pickle(path)`: Deserialize objects
  - `save_json(data, path)`: JSON export
  - `load_json(path)`: JSON import
- **Dependencies**: `pickle`, `json`, `pathlib`

**`utils/timer.py`**
- **Purpose**: Timing utilities for profiling
- **Key Classes**:
  - `Timer`: Context manager for timing
- **Usage**: `with Timer("data_loading"): ...`
- **Dependencies**: `time`

**`utils/visualization_utils.py`**
- **Purpose**: Helper functions for plotting
- **Key Functions**:
  - `set_plot_style()`: Matplotlib style setup
  - `save_figure(fig, path, dpi)`: Save with consistent settings
- **Dependencies**: `matplotlib`

#### **7. Experiment Tracking (3 files)**

**`experiments/experiment_manager.py`**
- **Purpose**: MLflow experiment orchestration
- **Key Classes**:
  - `ExperimentManager`: Manage MLflow runs
- **Key Functions**:
  - `create_experiment(name, description)`: Initialize experiment
  - `start_run(run_name, tags)`: Start MLflow run
  - `log_params(params)`: Log hyperparameters
  - `log_metrics(metrics, step)`: Log training metrics
  - `log_artifacts(artifact_dir)`: Log files/models
- **Dependencies**: `mlflow`

**`experiments/hyperparameter_tuner.py`**
- **Purpose**: Bayesian hyperparameter optimization
- **Key Classes**:
  - `HyperparameterTuner`: Optuna-based tuning
- **Key Functions**:
  - `objective(trial)`: Define optimization objective
  - `tune(n_trials)`: Run optimization
  - `get_best_params()`: Retrieve best hyperparameters
- **Dependencies**: `optuna`, `mlflow`

**`experiments/compare_experiments.py`**
- **Purpose**: Compare multiple experiment runs
- **Key Functions**:
  - `load_experiments(experiment_ids)`: Retrieve from MLflow
  - `compare_metrics(experiments, metric_names)`: Statistical comparison
  - `generate_comparison_report()`: Summary report
- **Dependencies**: `mlflow`, `pandas`

#### **8. Testing Suite (2 files)**

**`tests/test_data_generation.py`**
- **Purpose**: Unit tests for data generation
- **Test Cases**:
  - `test_signal_generator_output_shape()`: Verify dimensions
  - `test_fault_characteristics()`: Check dominant frequencies
  - `test_noise_injection()`: Validate SNR levels
  - `test_augmentation_preserves_label()`: Ensure correctness
- **Dependencies**: `pytest`, `data/`

**`tests/test_models.py`**
- **Purpose**: Unit tests for model architectures
- **Test Cases**:
  - `test_model_forward_pass()`: Check output shape
  - `test_model_gradient_flow()`: Verify backprop
  - `test_model_serialization()`: Save/load consistency
- **Dependencies**: `pytest`, `torch`, `models/`

### Architecture Decisions

**1. Hybrid Classical ML + Deep Learning**
- **Decision**: Keep existing classical ML pipeline (`pipeline.m`) intact, add deep learning alongside
- **Rationale**: 
  - Existing system is production-proven (95.33% accuracy)
  - Provides baseline for deep learning comparison
  - Some users may prefer interpretable models (Random Forest)
- **Implementation**: Separate `models/classical/` directory for ported MATLAB code

**2. PyTorch as Deep Learning Framework**
- **Decision**: Use PyTorch over TensorFlow
- **Rationale**:
  - More pythonic, easier debugging
  - Better support for custom physics-informed losses
  - Strong ecosystem (torchvision, ignite)
  - Preferred in research community

**3. Modular Configuration System**
- **Decision**: YAML-based configs with dataclass validation
- **Rationale**:
  - Reproduce experiments by saving config files
  - Easy hyperparameter sweeps
  - Version control friendly (plain text)
- **Alternative Rejected**: Python dicts (no validation), JSON (no comments)

**4. MLflow for Experiment Tracking**
- **Decision**: MLflow over Weights & Biases, TensorBoard
- **Rationale**:
  - Open-source, self-hosted (no vendor lock-in)
  - Integrated hyperparameter tracking + artifact storage
  - Good comparison UI
- **Fallback**: TensorBoard logs also generated for real-time monitoring

**5. HDF5 for Data Caching**
- **Decision**: HDF5 over pickle, NPZ
- **Rationale**:
  - Efficient random access for large datasets
  - Cross-platform, language-agnostic
  - Supports compression
- **Usage**: Cache generated signals to avoid regeneration

**6. Test-Driven Development**
- **Decision**: Write tests alongside implementation
- **Rationale**:
  - Catch bugs early in complex DL systems
  - Refactoring safety net
  - Documentation through examples

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA GENERATION PHASE                     │
│                                                              │
│  config/data_config.yaml                                     │
│          ↓                                                   │
│  data/signal_generator.py                                    │
│   ├─ Generate fault signals (physics models)                │
│   ├─ Apply 7-layer noise                                    │
│   └─ Temporal evolution (30% of signals)                    │
│          ↓                                                   │
│  data/augmentation.py                                        │
│   ├─ Time shift, amplitude scale                            │
│   └─ MixUp (optional)                                       │
│          ↓                                                   │
│  data/cache_manager.py → signals.h5 (cached dataset)        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    DATASET PREPARATION                       │
│                                                              │
│  data/dataset.py (BearingFaultDataset)                      │
│   ├─ Load cached signals                                    │
│   ├─ Stratified split (70/15/15)                           │
│   └─ Apply transforms                                       │
│          ↓                                                   │
│  data/dataloader.py → DataLoaders (train/val/test)          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                          │
│                                                              │
│  models/model_factory.py → Instantiate model                │
│          ↓                                                   │
│  training/trainer.py                                         │
│   ├─ Training loop (forward/backward)                       │
│   ├─ Validation loop                                        │
│   └─ Callbacks (checkpointing, early stopping)             │
│          ↓                                                   │
│  experiments/experiment_manager.py                           │
│   └─ Log metrics/artifacts to MLflow                        │
│          ↓                                                   │
│  checkpoints/best_model.pth                                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                       EVALUATION                             │
│                                                              │
│  evaluation/evaluator.py                                     │
│   ├─ Test set accuracy                                      │
│   ├─ Per-class metrics                                      │
│   └─ Confusion matrix                                       │
│          ↓                                                   │
│  evaluation/robustness_tester.py                            │
│   ├─ Sensor noise test                                      │
│   ├─ Missing features test                                  │
│   └─ Temporal drift test                                    │
│          ↓                                                   │
│  reports/model_evaluation_report.pdf                        │
└─────────────────────────────────────────────────────────────┘
```

### Integration Points

**1. With Existing MATLAB Code**
- **Challenge**: Bridge MATLAB → Python
- **Solution**:
  - `data/matlab_importer.py` reads existing `.mat` files
  - `data/signal_generator.py` is a faithful Python port of `generator.m`
  - Validation: Generate 100 signals in both, verify < 1% numerical difference
- **Testing**: Unit tests compare MATLAB vs. Python outputs

**2. With Classical ML Pipeline**
- **Current**: `pipeline.m` (3500 lines) trains SVM/RF/NN
- **Integration**: 
  - Keep existing pipeline for classical ML baseline
  - Add `models/classical/` directory with ported sklearn models
  - Unified evaluation in `evaluation/benchmark.py` compares DL vs. classical
- **Benefit**: Direct performance comparison (Section 11.9 in report)

**3. With MLflow UI**
- **Setup**: MLflow tracking server runs locally or on server
- **Access**: Web UI at `http://localhost:5000`
- **Usage**: 
  - Compare experiments (model architectures, hyperparameters)
  - Download artifacts (trained models, plots)
  - Restore old experiments for reproducibility

**4. With Deployment System (Future Phases)**
- **Export**: Trained model → ONNX format
- **Inference**: Load ONNX in `deployment/inference_engine.py`
- **API**: REST API wrapper for production use

### Testing Strategy

**1. Unit Tests (tests/)**
- **Coverage Target**: >80% for core modules
- **Framework**: `pytest`
- **Key Test Suites**:
  - `test_data_generation.py`: Verify signal generation correctness
  - `test_models.py`: Check forward pass, gradient flow
  - `test_training.py`: Validate training loop logic
- **Continuous Integration**: Run on every commit (GitHub Actions)

**2. Integration Tests**
- **End-to-End Pipeline Test**: Generate data → Train model → Evaluate
- **Expected Runtime**: ~5 minutes (small dataset)
- **Success Criteria**: Pipeline completes without errors

**3. Validation Against MATLAB**
- **Approach**: Generate identical signals in MATLAB and Python
- **Comparison**: Assert numerical difference < 0.01%
- **Files**: 
  - `tests/test_matlab_parity.py`
  - `validation/compare_with_matlab.m` (MATLAB script)

**4. Regression Tests**
- **Purpose**: Ensure updates don't degrade performance
- **Baseline**: Current 95.33% test accuracy
- **Trigger**: Run after major refactors
- **Alert**: If accuracy drops > 2%, investigate

### Acceptance Criteria

**Phase 0 Complete When:**

✅ **All 42 files created with documented interfaces**
- Each file has docstring describing purpose
- Key functions have type hints
- README.md in each directory

✅ **Configuration system functional**
- Can load/save YAML configs
- Validation catches invalid parameters
- Merge multiple configs correctly

✅ **Data generation working**
- Python `SignalGenerator` produces same output as MATLAB `generator.m` (< 1% difference)
- All 11 fault types generate correctly
- Noise injection achieves target SNR levels

✅ **PyTorch infrastructure operational**
- Can create DataLoaders with batching
- Models instantiate without errors
- Training loop runs for 1 epoch (even if untrained)

✅ **MLflow tracking functional**
- Experiments log to MLflow
- Can view runs in MLflow UI
- Artifacts (configs, plots) saved correctly

✅ **All tests passing**
- Unit tests: 100% pass
- Integration test: End-to-end pipeline runs
- MATLAB parity test: < 1% numerical difference

✅ **Documentation complete**
- README files in each directory
- Architecture diagram (like above) documented
- Installation guide for dependencies

### Estimated Effort

**Time Breakdown:**
- Configuration system: 2 days
- Data infrastructure (8 files): 5 days
- Model zoo (7 files): 4 days
- Training infrastructure (6 files): 4 days
- Evaluation suite (5 files): 3 days
- Utilities (6 files): 2 days
- Experiment tracking (3 files): 2 days
- Testing (2 files + validation): 3 days
- Documentation: 2 days
- Buffer for debugging: 3 days

**Total: ~30 days (1.5 months) for Phase 0**

**Complexity**: ⭐⭐⭐☆☆ (Moderate-High)
- Mostly infrastructure, not research
- MATLAB→Python porting is tedious but straightforward
- PyTorch boilerplate is well-documented

**Dependencies**: None (first phase)

**Risk**: Low - Building on mature frameworks (PyTorch, MLflow)

---
