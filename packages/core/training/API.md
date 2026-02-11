# Training API Reference

> Complete API reference for `packages/core/training/`.

---

## Trainers

### `Trainer` — `trainer.py`

Base trainer with gradient accumulation, gradient clipping, mixed precision, and callback support.

```python
Trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    device: str = 'cuda',
    callbacks: Optional[List[Callable]] = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    mixed_precision: bool = False
)
```

| Method                    | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `fit(num_epochs: int)`    | Full training loop for `num_epochs` epochs           |
| `train_epoch() → Dict`    | One training epoch; returns `{'loss', 'accuracy'}`   |
| `validate_epoch() → Dict` | One validation epoch; returns `{'loss', 'accuracy'}` |

**`TrainingState`** — Tracks epoch, global step, best val loss, history, and early stopping flag.

---

### `CNNTrainer` — `cnn_trainer.py`

Extends `Trainer` with FP16 mixed precision, LR scheduling, and checkpoint management.

```python
CNNTrainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    device: str = 'cuda',
    callbacks: Optional[List[Callable]] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = True,
    checkpoint_dir: Optional[Path] = None
)
```

| Method                                         | Description                                          |
| ---------------------------------------------- | ---------------------------------------------------- |
| `train_epoch() → Dict`                         | Training with mixed precision; returns loss/accuracy |
| `validate_epoch() → Dict`                      | Validation; returns loss/accuracy                    |
| `save_checkpoint(epoch, val_metrics, is_best)` | Save model + optimizer + scheduler state             |
| `load_checkpoint(checkpoint_path) → Dict`      | Load checkpoint and return metadata                  |

---

### `PINNTrainer` — `pinn_trainer.py`

Extends `Trainer` for Physics-Informed Neural Networks with combined loss computation and adaptive lambda scheduling.

```python
PINNTrainer(
    model, train_loader,
    val_loader=None, optimizer=None, criterion=None,
    device='cuda', callbacks=None,
    gradient_accumulation_steps=1, max_grad_norm=1.0, mixed_precision=False,
    lambda_freq: float = 1.0,
    lambda_sommerfeld: float = 0.5,
    lambda_temporal: float = 0.1,
    adaptive_lambda: bool = True,
    lambda_schedule: str = 'linear',
    sample_rate: int = 20480,
    metadata_keys: Optional[List[str]] = None
)
```

| Method                                                                   | Description                               |
| ------------------------------------------------------------------------ | ----------------------------------------- |
| `compute_loss(outputs, targets, signal, metadata) → Tuple[Tensor, Dict]` | Combined classification + physics loss    |
| `train_epoch() → Dict`                                                   | PINN training with physics constraints    |
| `validate_epoch() → Dict`                                                | PINN validation                           |
| `train(num_epochs) → Dict`                                               | Full training loop with lambda scheduling |

**`PINNTrainingState`** — Extends `TrainingState` with physics-specific tracking (lambda history, loss components).

---

### `SpectrogramTrainer` — `spectrogram_trainer.py`

Extends `Trainer` with SpecAugment and spectrogram-specific logging/checkpointing.

```python
SpectrogramTrainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    device: str = 'cuda',
    use_specaugment: bool = True,
    time_mask_param: int = 40,
    freq_mask_param: int = 20,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
    **kwargs
)
```

| Method                                             | Description                       |
| -------------------------------------------------- | --------------------------------- |
| `apply_specaugment(spectrograms: Tensor) → Tensor` | Apply time/frequency masking      |
| `train_epoch() → Dict`                             | Training with SpecAugment         |
| `save_checkpoint(...)`                             | Save spectrogram model checkpoint |

---

### `MultiTFRTrainer` — `spectrogram_trainer.py`

Extends `SpectrogramTrainer` for training on multiple Time-Frequency Representations simultaneously.

```python
MultiTFRTrainer(
    model: nn.Module,
    train_loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    tfr_weights: Optional[Dict[str, float]] = None,
    **kwargs
)
```

| Method                    | Description                        |
| ------------------------- | ---------------------------------- |
| `train_epoch() → Dict`    | Weighted training across TFR types |
| `validate_epoch() → Dict` | Validation across all TFR types    |

---

### `MixedPrecisionTrainer` — `mixed_precision.py`

Dedicated FP16 training with explicit `GradScaler` control.

```python
MixedPrecisionTrainer(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cuda'
)
```

| Method                 | Description         |
| ---------------------- | ------------------- |
| `train_epoch() → Dict` | FP16 training epoch |

**Utility functions:**

| Function                                 | Description                   |
| ---------------------------------------- | ----------------------------- |
| `enable_mixed_precision()`               | Enable TF32 for Ampere GPUs   |
| `check_mixed_precision_support() → bool` | Check if device supports FP16 |

---

### `DistillationTrainer` — `knowledge_distillation.py`

Teacher → student knowledge transfer trainer.

```python
DistillationTrainer(
    teacher_model: nn.Module,
    student_model: nn.Module,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu'
)
```

| Method                                                      | Description                 |
| ----------------------------------------------------------- | --------------------------- |
| `train_epoch(train_loader, epoch) → Dict`                   | Distillation training epoch |
| `evaluate(val_loader) → Dict`                               | Student model evaluation    |
| `train(train_loader, val_loader, epochs, scheduler) → Dict` | Full training loop          |

**`compare_teacher_student(teacher, student, test_loader, device) → Dict`** — Compare teacher/student accuracy, compression ratio.

---

### `ProgressiveResizingTrainer` — `progressive_resizing.py`

Trains with progressively longer signals for faster convergence.

```python
ProgressiveResizingTrainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cpu',
    schedule: Optional[List[Tuple[int, int]]] = None
    # Default: [(25600,20), (51200,15), (76800,10), (102400,5)]
)
```

| Method                                                                                               | Description               |
| ---------------------------------------------------------------------------------------------------- | ------------------------- |
| `train_epoch(train_loader, epoch) → Dict`                                                            | Single epoch training     |
| `evaluate(val_loader) → Dict`                                                                        | Evaluation                |
| `train_progressive(base_train_dataset, base_val_dataset, batch_size, num_workers, scheduler) → Dict` | Full progressive training |

**`ResizableSignalDataset`** — Dataset wrapper that resizes signals via interpolation, truncation, or padding.

---

## Loss Functions

### `losses.py`

| Class                        | Constructor                                                               | Description                             |
| ---------------------------- | ------------------------------------------------------------------------- | --------------------------------------- |
| `FocalLoss`                  | `(alpha: Optional[Tensor]=None, gamma: float=2.0, reduction: str='mean')` | Hard-example mining for class imbalance |
| `LabelSmoothingCrossEntropy` | `(smoothing: float=0.1, reduction: str='mean')`                           | Prevents overconfident predictions      |
| `PhysicsInformedLoss`        | `(data_weight: float=1.0, physics_weight: float=0.1)`                     | Combined data + physics constraint loss |

**`compute_class_weights(labels: ndarray) → Tensor`** — Compute inverse-frequency class weights.

### `cnn_losses.py`

| Class                        | Constructor                                                    | Description                                 |
| ---------------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| `LabelSmoothingCrossEntropy` | `(smoothing: float=0.1, weight: Optional[Tensor]=None)`        | Label smoothing with optional class weights |
| `FocalLoss`                  | `(alpha: float=0.25, gamma: float=2.0, reduction: str='mean')` | Focal loss for CNNs                         |
| `SupConLoss`                 | `(temperature: float=0.1)`                                     | Supervised contrastive loss                 |

**`create_criterion(name: str, **kwargs) → nn.Module`** — Factory: `'ce'`, `'label_smoothing'`, `'focal'`, `'supcon'`.

### `physics_loss_functions.py`

| Class                       | Constructor                                                            | Description                             |
| --------------------------- | ---------------------------------------------------------------------- | --------------------------------------- |
| `FrequencyConsistencyLoss`  | `(expected_freqs: Dict, sample_rate: int=20480, tolerance: float=0.1)` | Penalizes fault-frequency inconsistency |
| `SommerfeldConsistencyLoss` | `(fault_severity_order: Optional[List]=None)`                          | Operating-condition severity ordering   |
| `TemporalSmoothnessLoss`    | `(lambda_smooth: float=0.1)`                                           | Penalizes erratic temporal changes      |
| `PhysicalConstraintLoss`    | `(lambda_freq=1.0, lambda_sommerfeld=0.5, lambda_temporal=0.1, ...)`   | Combined physics constraint             |

### `knowledge_distillation.py` — `DistillationLoss`

```python
DistillationLoss(
    temperature: float = 4.0,
    alpha: float = 0.7,
    hard_loss_type: str = 'cross_entropy'
)
```

Loss = α × KL(softened student ‖ softened teacher) + (1−α) × HardLoss(student, labels)

---

## Callbacks

### `callbacks.py`

Base `Callback` class with hooks: `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`.

| Class                   | Constructor                                                       | Description                  |
| ----------------------- | ----------------------------------------------------------------- | ---------------------------- |
| `EarlyStopping`         | `(monitor='val_loss', patience=10, mode='min', min_delta=0.0)`    | Stop on metric plateau       |
| `ModelCheckpoint`       | `(filepath, monitor='val_loss', save_best_only=True, mode='min')` | Save best model              |
| `LearningRateScheduler` | `(scheduler)`                                                     | Step LR scheduler each epoch |
| `TensorBoardLogger`     | `(log_dir='runs')`                                                | Log to TensorBoard           |
| `MLflowLogger`          | `(experiment_name, run_name=None)`                                | Log to MLflow                |

### `cnn_callbacks.py`

Extended callback classes with batch-level hooks (`on_batch_begin`, `on_batch_end`).

| Class                     | Constructor                                                                        | Description                               |
| ------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| `Callback`                | Base class                                                                         | Defines all hook methods                  |
| `LearningRateMonitor`     | `(optimizer, log_interval=1)`                                                      | Track LR per epoch                        |
| `GradientMonitor`         | `(model, log_interval=100, alert_threshold=10.0)`                                  | Detect vanishing/exploding gradients      |
| `ModelCheckpointCallback` | `(checkpoint_dir, model, optimizer, monitor='val_loss', mode='min', save_top_k=3)` | Save top-k checkpoints                    |
| `TimingCallback`          | `(verbose=True)`                                                                   | Profile epoch & batch timing              |
| `MetricLogger`            | `(log_file: Path, log_interval=1)`                                                 | Log metrics to JSON                       |
| `EarlyStoppingCallback`   | `(monitor='val_loss', patience=10, mode='min', min_delta=0.0, verbose=True)`       | Early stopping                            |
| `CallbackList`            | `(callbacks: List[Callback])`                                                      | Container for managing multiple callbacks |

---

## Optimizers

### `cnn_optimizer.py`

| Function                   | Signature                                                                                       | Description               |
| -------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------- |
| `create_optimizer`         | `(optimizer_type='adamw', model_params=None, lr=1e-3, weight_decay=1e-4, **kwargs) → Optimizer` | Factory function          |
| `create_adamw_optimizer`   | `(model_params, lr=1e-3, weight_decay=1e-4, betas=(0.9,0.999)) → AdamW`                         | AdamW optimizer           |
| `create_sgd_optimizer`     | `(model_params, lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True) → SGD`                 | SGD with Nesterov         |
| `create_rmsprop_optimizer` | `(model_params, lr=1e-3, weight_decay=1e-5, alpha=0.99) → RMSprop`                              | RMSprop optimizer         |
| `get_parameter_groups`     | `(model, lr, weight_decay, no_decay_keywords=['bias','norm']) → List[Dict]`                     | Differential weight decay |

**`OptimizerConfig`** — Class methods: `default()`, `fast_convergence()`, `strong_regularization()`, `sgd_baseline()`. Each returns a config dict.

### `optimizers.py`

| Function           | Signature                                                                       | Description                                                                   |
| ------------------ | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `create_optimizer` | `(model, optimizer_type='adam', lr=1e-3, weight_decay=0, **kwargs) → Optimizer` | **Deprecated** — delegates to `cnn_optimizer`                                 |
| `create_scheduler` | `(optimizer, scheduler_type='step', **kwargs) → _LRScheduler`                   | Factory: `'step'`, `'cosine'`, `'plateau'`, `'onecycle'`, `'cosine_restarts'` |

---

## Learning Rate Schedulers

### `cnn_schedulers.py`

| Function / Class                       | Signature                                                                   | Description                    |
| -------------------------------------- | --------------------------------------------------------------------------- | ------------------------------ |
| `create_cosine_scheduler`              | `(optimizer, num_epochs, eta_min=1e-6) → CosineAnnealingLR`                 | Cosine annealing               |
| `create_cosine_warmrestarts_scheduler` | `(optimizer, T_0=10, T_mult=2, eta_min=1e-6) → CosineAnnealingWarmRestarts` | Cosine warm restarts           |
| `create_onecycle_scheduler`            | `(optimizer, max_lr, total_steps, pct_start=0.3) → OneCycleLR`              | One-cycle policy               |
| `create_step_scheduler`                | `(optimizer, step_size=10, gamma=0.1) → StepLR`                             | Step decay                     |
| `create_exponential_scheduler`         | `(optimizer, gamma=0.95) → ExponentialLR`                                   | Exponential decay              |
| `create_plateau_scheduler`             | `(optimizer, mode='max', factor=0.1, patience=10) → ReduceLROnPlateau`      | Reduce on plateau              |
| `WarmupScheduler`                      | `(optimizer, warmup_epochs, base_scheduler)`                                | Linear warmup wrapper          |
| `PolynomialLRScheduler`                | `(optimizer, total_epochs, power=2.0, min_lr=1e-7)`                         | Polynomial decay               |
| `get_scheduler`                        | `(scheduler_type, optimizer, **kwargs)`                                     | Factory for all CNN schedulers |

### `transformer_schedulers.py`

| Function / Class                | Signature                                                              | Description                                             |
| ------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------- |
| `create_warmup_cosine_schedule` | `(optimizer, warmup_epochs, total_epochs, min_lr=1e-7) → LambdaLR`     | Warmup + cosine                                         |
| `create_warmup_linear_schedule` | `(optimizer, warmup_epochs, total_epochs) → LambdaLR`                  | Warmup + linear decay                                   |
| `create_noam_schedule`          | `(optimizer, d_model=256, warmup_steps=4000) → LambdaLR`               | Noam from "Attention Is All You Need"                   |
| `WarmupCosineScheduler`         | `(optimizer, warmup_epochs, total_epochs, min_lr=1e-7, last_epoch=-1)` | Stateful warmup-cosine                                  |
| `get_scheduler`                 | `(optimizer, scheduler_type, **kwargs)`                                | Factory: `'warmup_cosine'`, `'warmup_linear'`, `'noam'` |

---

## Metrics — `metrics.py`

| Component                          | Signature                                         | Description                   |
| ---------------------------------- | ------------------------------------------------- | ----------------------------- |
| `MetricsTracker`                   | `(num_classes: int = None)`                       | Batch-aggregating tracker     |
| `.update(predictions, targets)`    |                                                   | Add batch results             |
| `.compute(average='macro') → Dict` |                                                   | Compute aggregated metrics    |
| `.reset()`                         |                                                   | Reset tracked state           |
| `compute_accuracy`                 | `(predictions, targets) → float`                  | Classification accuracy       |
| `compute_f1_score`                 | `(predictions, targets, average='macro') → float` | F1 score                      |
| `compute_confusion_matrix`         | `(predictions, targets, num_classes) → ndarray`   | Confusion matrix              |
| `compute_top_k_accuracy`           | `(logits, targets, k=5) → float`                  | Top-k accuracy                |
| `compute_per_class_metrics`        | `(predictions, targets, num_classes) → Dict`      | Per-class precision/recall/F1 |

---

## Hyperparameter Search

### `GridSearchOptimizer` — `grid_search.py`

```python
GridSearchOptimizer(random_state: int = 42, n_jobs: int = -1, verbose: int = 2)
```

| Method                                                | Description                   |
| ----------------------------------------------------- | ----------------------------- |
| `search(model, X, y, param_grid, cv, scoring) → Dict` | Exhaustive grid search        |
| `get_results() → DataFrame`                           | Search results as DataFrame   |
| `plot_grid_results(param1, param2)`                   | Heatmap of param combinations |

**`get_default_param_grid(model_name) → Dict`** — Presets for `'RandomForest'`, `'SVM'`, `'GradientBoosting'`.

### `RandomSearchOptimizer` — `random_search.py`

```python
RandomSearchOptimizer(random_state: int = 42, n_jobs: int = -1, verbose: int = 2)
```

| Method                                                        | Description          |
| ------------------------------------------------------------- | -------------------- |
| `search(model, X, y, param_dist, n_iter, cv, scoring) → Dict` | Random search        |
| `get_results() → DataFrame`                                   | Search results       |
| `plot_search_results(param_name)`                             | Param vs. score plot |

**`get_default_param_distributions(model_name) → Dict`** — Distribution presets.

### `BayesianOptimizer` — `bayesian_optimizer.py`

```python
BayesianOptimizer(random_state: int = 42)
```

| Method                                                                                | Description           |
| ------------------------------------------------------------------------------------- | --------------------- |
| `optimize(model_class, X_train, y_train, X_val, y_val, n_trials, param_space) → Dict` | Bayesian optimization |
| `plot_optimization_history()`                                                         | Convergence plot      |
| `plot_param_importances()`                                                            | Parameter importance  |

**`get_model_search_space(model_name) → Dict`** — Optuna search spaces.

---

## Data Augmentation

### `advanced_augmentation.py` — Signal-Level

| Function / Class              | Signature                                                       | Description                   |
| ----------------------------- | --------------------------------------------------------------- | ----------------------------- |
| `cutmix`                      | `(signal1, signal2, label1, label2, alpha=1.0) → Tuple`         | Cut-and-paste between signals |
| `cutmix_batch`                | `(signals, labels, alpha=1.0, prob=0.5) → Tuple`                | Batch CutMix with probability |
| `adversarial_augmentation`    | `(model, signal, label, epsilon=0.01, criterion=None) → Tensor` | FGSM adversarial perturbation |
| `time_masking`                | `(signal, mask_param=1000, num_masks=1) → Tensor`               | Mask random time segments     |
| `gaussian_noise_augmentation` | `(signal, noise_std=0.01) → Tensor`                             | Add Gaussian noise            |
| `amplitude_scaling`           | `(signal, scale_range=(0.8, 1.2)) → Tensor`                     | Random scaling                |
| `time_shift`                  | `(signal, shift_max=1000) → Tensor`                             | Circular time shift           |
| `AutoAugment`                 | `(policy: Optional[List] = None)`                               | Learned augmentation policy   |
| `MixupAugmentation`           | `(alpha: float = 1.0)`                                          | Beta-distribution mixup       |

### `transformer_augmentation.py` — Patch-Level

| Function / Class            | Signature                                                   | Description                   |
| --------------------------- | ----------------------------------------------------------- | ----------------------------- |
| `patch_dropout`             | `(patches, drop_prob=0.1, training=True) → Tensor`          | Random patch dropout          |
| `patch_mixup`               | `(patches1, patches2, labels1, labels2, alpha=0.4) → Tuple` | Patch-level mixup             |
| `temporal_shift_patches`    | `(patches, max_shift=5, circular=True) → Tensor`            | Shift patches in time         |
| `patch_cutout`              | `(patches, n_holes=10, hole_size=5) → Tensor`               | Mask contiguous regions       |
| `patch_permutation`         | `(patches, permute_prob=0.1) → Tensor`                      | Swap random patches           |
| `patch_jitter`              | `(patches, noise_std=0.01) → Tensor`                        | Add noise to patch embeddings |
| `PatchAugmentation`         | `(use_dropout=True, dropout_prob=0.1, ...)`                 | Composable pipeline           |
| `get_light_augmentation()`  | → `PatchAugmentation`                                       | Light preset                  |
| `get_medium_augmentation()` | → `PatchAugmentation`                                       | Medium preset (recommended)   |
| `get_heavy_augmentation()`  | → `PatchAugmentation`                                       | Heavy preset                  |

### `progressive_resizing.py` — `ResizableSignalDataset`

```python
ResizableSignalDataset(
    base_dataset: Dataset,
    target_length: int,
    resize_method: str = 'interpolate'  # 'interpolate', 'truncate', 'pad'
)
```
