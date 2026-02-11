# Training Guide

> Practical guide for training bearing fault diagnosis models using the training infrastructure.

## 1. Basic Training Workflow

The simplest training workflow uses the base `Trainer` class:

```python
import torch
from training.trainer import Trainer

# 1. Prepare model, data, criterion, optimizer
model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# 2. Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda',
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    mixed_precision=False
)

# 3. Train
trainer.fit(num_epochs=100)

# 4. Access history
history = trainer.get_history()
# history keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
```

## 2. Choosing the Right Trainer

| Scenario                      | Trainer                      | Why                                             |
| ----------------------------- | ---------------------------- | ----------------------------------------------- |
| Standard LSTM/MLP training    | `Trainer`                    | Simple, supports all basic features             |
| CNN-based fault diagnosis     | `CNNTrainer`                 | Built-in mixed precision, checkpoint management |
| Physics-informed models       | `PINNTrainer`                | Handles combined classification + physics loss  |
| Spectrogram input models      | `SpectrogramTrainer`         | SpecAugment augmentation built-in               |
| Multi-TFR ensemble comparison | `MultiTFRTrainer`            | Trains on STFT, CWT, WVD simultaneously         |
| Dedicated FP16 training       | `MixedPrecisionTrainer`      | Explicit GradScaler control                     |
| Model compression             | `DistillationTrainer`        | Teacher → student knowledge transfer            |
| Large signal models           | `ProgressiveResizingTrainer` | Faster convergence with progressive lengths     |

## 3. Configuring Callbacks

### Base Callbacks (`callbacks.py`) — for `Trainer`

```python
from training.callbacks import (
    EarlyStopping, ModelCheckpoint,
    LearningRateScheduler, TensorBoardLogger, MLflowLogger
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0),
    ModelCheckpoint(filepath='checkpoints/best.pt', monitor='val_loss', save_best_only=True),
    LearningRateScheduler(scheduler=my_scheduler),
    TensorBoardLogger(log_dir='runs/experiment_1'),
]

trainer = Trainer(model=model, ..., callbacks=callbacks)
```

### Extended Callbacks (`cnn_callbacks.py`) — for `CNNTrainer`

```python
from training.cnn_callbacks import (
    LearningRateMonitor, GradientMonitor,
    ModelCheckpointCallback, TimingCallback,
    MetricLogger, EarlyStoppingCallback, CallbackList
)

callback_list = CallbackList([
    LearningRateMonitor(optimizer, log_interval=1),
    GradientMonitor(model, log_interval=100, alert_threshold=10.0),
    ModelCheckpointCallback(
        checkpoint_dir=Path('./checkpoints'),
        model=model, optimizer=optimizer,
        monitor='val_acc', mode='max'
    ),
    TimingCallback(verbose=True),
    MetricLogger(log_file=Path('./logs/metrics.json')),
    EarlyStoppingCallback(monitor='val_loss', patience=10),
])
```

## 4. Loss Function Selection

| Situation                     | Loss                                                       | Rationale                         |
| ----------------------------- | ---------------------------------------------------------- | --------------------------------- |
| Balanced classes              | `nn.CrossEntropyLoss`                                      | Standard baseline                 |
| Class imbalance (rare faults) | `FocalLoss(gamma=2.0)`                                     | Down-weights easy examples        |
| Overconfident predictions     | `LabelSmoothingCrossEntropy(smoothing=0.1)`                | Softens hard labels               |
| PINN models                   | `PhysicsInformedLoss(data_weight=1.0, physics_weight=0.1)` | Combined data + physics           |
| PINN with full physics        | `PhysicalConstraintLoss(lambda_freq=1.0, ...)`             | Frequency + Sommerfeld + temporal |
| Contrastive learning          | `SupConLoss(temperature=0.1)`                              | Embedding-space structure         |
| Knowledge distillation        | `DistillationLoss(temperature=4.0, alpha=0.7)`             | Soft + hard label combination     |

### Using the CNN Loss Factory

```python
from training.cnn_losses import create_criterion

# Options: 'ce', 'label_smoothing', 'focal', 'supcon'
criterion = create_criterion('focal', gamma=2.0)
```

## 5. Optimizer Configuration

### Using Factory Function

```python
from training.cnn_optimizer import create_optimizer

# Options: 'adamw' (default), 'sgd', 'rmsprop'
optimizer = create_optimizer('adamw', model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### Using Presets

```python
from training.cnn_optimizer import OptimizerConfig, create_optimizer

# Available presets: default, fast_convergence, strong_regularization, sgd_baseline
config = OptimizerConfig.fast_convergence()
# Returns: {'optimizer_type': 'adamw', 'lr': 3e-3, 'weight_decay': 1e-4, 'betas': (0.9, 0.999)}

optimizer = create_optimizer(model_params=model.parameters(), **config)
```

### Differential Weight Decay

```python
from training.cnn_optimizer import get_parameter_groups

# Skips weight decay for bias and normalization layers
param_groups = get_parameter_groups(model, lr=1e-3, weight_decay=1e-4)
optimizer = torch.optim.AdamW(param_groups)
```

> **Note:** `training.optimizers.create_optimizer()` is **deprecated**. Use `training.cnn_optimizer.create_optimizer()` instead.

## 6. Learning Rate Scheduling

### CNN Schedulers

```python
from training.cnn_schedulers import (
    create_cosine_scheduler,
    create_onecycle_scheduler,
    create_plateau_scheduler,
    WarmupScheduler
)

# Cosine annealing (smooth decay, stable convergence)
scheduler = create_cosine_scheduler(optimizer, num_epochs=100, eta_min=1e-6)

# One-cycle (super-convergence, fast training)
scheduler = create_onecycle_scheduler(
    optimizer, max_lr=1e-2, total_steps=num_epochs * len(train_loader)
)

# Plateau (adaptive, reduce when stuck)
scheduler = create_plateau_scheduler(optimizer, mode='max', patience=10)

# Warmup + cosine
base_sched = create_cosine_scheduler(optimizer, num_epochs=100)
scheduler = WarmupScheduler(optimizer, warmup_epochs=5, base_scheduler=base_sched)
```

### Transformer Schedulers

```python
from training.transformer_schedulers import get_scheduler

# Options: 'warmup_cosine', 'warmup_linear', 'noam'
scheduler = get_scheduler(
    optimizer,
    scheduler_type='warmup_cosine',
    warmup_epochs=10,
    total_epochs=100
)
```

### Scheduling Decision Guide

| Strategy             | Best For                    | Key Parameter             |
| -------------------- | --------------------------- | ------------------------- |
| Cosine annealing     | General training            | `eta_min`                 |
| One-cycle            | Fast convergence            | `max_lr`                  |
| Cosine warm restarts | Escaping local minima       | `T_0`, `T_mult`           |
| ReduceLROnPlateau    | Unknown training duration   | `patience`                |
| Warmup + cosine      | Transformers                | `warmup_epochs`           |
| Noam                 | Original Transformer recipe | `d_model`, `warmup_steps` |

## 7. Hyperparameter Optimization

### Grid Search (small, discrete spaces)

```python
from training.grid_search import GridSearchOptimizer, get_default_param_grid

optimizer = GridSearchOptimizer(random_state=42)
param_grid = get_default_param_grid('RandomForest')
# {'n_estimators': [50,100,150,200], 'max_depth': [10,20,30,None], ...}

best_params = optimizer.search(model, X_train, y_train, param_grid, cv=5)
optimizer.plot_grid_results('n_estimators', 'max_depth')
```

### Random Search (large spaces)

```python
from training.random_search import RandomSearchOptimizer, get_default_param_distributions

optimizer = RandomSearchOptimizer(random_state=42)
param_dist = get_default_param_distributions('RandomForest')
best_params = optimizer.search(model, X_train, y_train, param_dist, n_iter=50)
```

### Bayesian Optimization (most efficient)

```python
from training.bayesian_optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(random_state=42)
best_params = optimizer.optimize(
    model_class=RandomForestClassifier,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    n_trials=50
)
optimizer.plot_optimization_history()
optimizer.plot_param_importances()
```

> **When to use which:** Grid search for ≤4 parameters × ≤5 values each. Random search for larger spaces. Bayesian optimization when evaluations are expensive.

## 8. Data Augmentation

### Signal-Level Augmentation

```python
from training.advanced_augmentation import (
    cutmix_batch, adversarial_augmentation,
    gaussian_noise_augmentation, time_masking,
    amplitude_scaling, time_shift,
    AutoAugment, MixupAugmentation
)

# CutMix (probability-based)
signals, labels = cutmix_batch(signals, labels, alpha=1.0, prob=0.5)

# Gaussian noise
signals = gaussian_noise_augmentation(signals, noise_std=0.01)

# AutoAugment policy
augmentor = AutoAugment()
signals = augmentor(signals)

# Mixup
mixup = MixupAugmentation(alpha=1.0)
mixed_signal, mixed_label = mixup(signal1, signal2, label1, label2)
```

### Patch-Level Augmentation (Transformers)

```python
from training.transformer_augmentation import (
    PatchAugmentation,
    get_light_augmentation, get_medium_augmentation, get_heavy_augmentation
)

# Use a preset
augmentor = get_medium_augmentation()
# Applies: dropout (0.1) + cutout (5 holes, size 3) + shift (max 3) + jitter (0.01)

# During training
patches = augmentor(patches, training=True)
```

### SpecAugment (Spectrograms)

Built into `SpectrogramTrainer` — enabled by default:

```python
from training.spectrogram_trainer import SpectrogramTrainer

trainer = SpectrogramTrainer(
    model=model, train_loader=train_loader,
    use_specaugment=True,
    time_mask_param=40,
    freq_mask_param=20,
    num_time_masks=2,
    num_freq_masks=2
)
```

## 9. Knowledge Distillation

Transfer knowledge from a large teacher model to a smaller student:

```python
from training.knowledge_distillation import (
    DistillationLoss, DistillationTrainer, compare_teacher_student
)

# Setup
criterion = DistillationLoss(temperature=4.0, alpha=0.7)
trainer = DistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    criterion=criterion,
    optimizer=student_optimizer,
    device='cuda'
)

# Train
history = trainer.train(train_loader, val_loader, epochs=50)

# Compare
comparison = compare_teacher_student(teacher, student, test_loader)
```

## 10. Progressive Resizing

Train with progressively longer signals for faster convergence:

```python
from training.progressive_resizing import ProgressiveResizingTrainer

# Default schedule: (25600, 20 epochs) → (51200, 15) → (76800, 10) → (102400, 5)
trainer = ProgressiveResizingTrainer(
    model=model, optimizer=optimizer, criterion=criterion,
    device='cuda',
    schedule=[(25600, 20), (51200, 15), (76800, 10), (102400, 5)]
)

history = trainer.train_progressive(
    base_train_dataset=train_dataset,
    base_val_dataset=val_dataset,
    batch_size=32
)
```

## 11. Mixed Precision Training

### Option A: Built into Trainer/CNNTrainer

```python
trainer = Trainer(model=model, ..., mixed_precision=True)
# or
trainer = CNNTrainer(model=model, ..., mixed_precision=True)
```

### Option B: Dedicated MixedPrecisionTrainer

```python
from training.mixed_precision import MixedPrecisionTrainer, check_mixed_precision_support

if check_mixed_precision_support():
    trainer = MixedPrecisionTrainer(
        model=model, train_loader=train_loader,
        optimizer=optimizer, criterion=criterion
    )
```

### Option C: Global enable

```python
from training.mixed_precision import enable_mixed_precision
enable_mixed_precision()  # Enables TF32 for Ampere GPUs
```

## 12. Training Metrics

```python
from training.metrics import MetricsTracker, compute_per_class_metrics

# Track metrics across batches
tracker = MetricsTracker()
for batch in dataloader:
    preds = model(batch.inputs)
    tracker.update(preds, batch.labels)

metrics = tracker.compute(average='macro')
# {'accuracy': ..., 'f1_score': ..., 'precision': ..., 'recall': ...}

# Per-class breakdown
per_class = compute_per_class_metrics(all_preds, all_targets, num_classes=10)
```

## Performance

> ⚠️ **Results pending.** Performance metrics below will be populated after experiments are run on the current codebase.

| Metric                                               | Value                                |
| ---------------------------------------------------- | ------------------------------------ |
| Convergence speed improvement (progressive resizing) | `[PENDING — run experiment to fill]` |
| Mixed precision memory savings                       | `[PENDING — run experiment to fill]` |
| SpecAugment accuracy improvement                     | `[PENDING — run experiment to fill]` |
| Knowledge distillation compression                   | `[PENDING — run experiment to fill]` |
| Bayesian vs Grid search efficiency                   | `[PENDING — run experiment to fill]` |
