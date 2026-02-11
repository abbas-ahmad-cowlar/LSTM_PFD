# Configuration Guide — Complete Parameter Reference

> Every parameter, type, default, and valid range for all 23 configuration classes in the `config/` module.

All defaults and types below are verified directly from the source code. Constants like `SAMPLING_RATE`, `SIGNAL_LENGTH`, and `NUM_CLASSES` are imported from `utils.constants`.

---

## Base Layer

### `BaseConfig` (Abstract)

> File: `base_config.py` — Abstract base class for all configuration objects.

All config classes inherit from `BaseConfig` and must implement `get_schema()`.

**Inherited Methods:**

| Method            | Signature             | Description                                                                |
| ----------------- | --------------------- | -------------------------------------------------------------------------- |
| `validate()`      | `() → bool`           | Validates instance against its JSON schema; raises `ValueError` on failure |
| `from_yaml()`     | `(path: Path) → T`    | Class method — loads config from YAML, auto-validates                      |
| `to_yaml()`       | `(path: Path) → None` | Validates then saves to YAML                                               |
| `to_dict()`       | `() → Dict[str, Any]` | Converts to dictionary via `dataclasses.asdict`                            |
| `merge_configs()` | `(*configs: T) → T`   | Class method — merges configs (later overrides earlier)                    |
| `get_schema()`    | `() → Dict[str, Any]` | **Abstract** — must return JSON schema dict                                |

### `ConfigValidator`

> File: `base_config.py` — Static utility class for common validation patterns.

| Method                   | Signature                                                          | Description                                         |
| ------------------------ | ------------------------------------------------------------------ | --------------------------------------------------- |
| `validate_positive()`    | `(value: float, name: str) → None`                                 | Raises `ValueError` if `value ≤ 0`                  |
| `validate_range()`       | `(value: float, min_val: float, max_val: float, name: str) → None` | Raises `ValueError` if outside `[min_val, max_val]` |
| `validate_probability()` | `(value: float, name: str) → None`                                 | Calls `validate_range(value, 0.0, 1.0, name)`       |

---

## Data Generation — `data_config.py`

### `SignalConfig`

> Signal generation parameters.

| Parameter    | Type    | Default                        | Valid Range (Schema) | Description                             |
| ------------ | ------- | ------------------------------ | -------------------- | --------------------------------------- |
| `fs`         | `int`   | `20480` (from `SAMPLING_RATE`) | 1000–100000          | Sampling frequency in Hz                |
| `T`          | `float` | `5.0` (from `SIGNAL_DURATION`) | 0.1–60.0             | Signal duration in seconds              |
| `Omega_base` | `float` | `60.0`                         | 1.0–1000.0           | Nominal rotation speed in Hz (3600 RPM) |

**Computed Properties:**

| Property     | Type    | Description                                   |
| ------------ | ------- | --------------------------------------------- |
| `N`          | `int`   | Total samples per signal (`fs × T` = 102400)  |
| `omega_base` | `float` | Angular velocity in rad/s (`2π × Omega_base`) |

---

### `FaultConfig`

> Fault type selection configuration.

| Parameter         | Type              | Default    | Description                                                                                                               |
| ----------------- | ----------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| `include_single`  | `bool`            | `True`     | Include 7 single fault types                                                                                              |
| `include_mixed`   | `bool`            | `True`     | Include 3 mixed fault combinations                                                                                        |
| `include_healthy` | `bool`            | `True`     | Include healthy baseline                                                                                                  |
| `single_faults`   | `Dict[str, bool]` | All `True` | Individual enable/disable for: `desalignement`, `desequilibre`, `jeu`, `lubrification`, `cavitation`, `usure`, `oilwhirl` |
| `mixed_faults`    | `Dict[str, bool]` | All `True` | Individual enable/disable for: `misalign_imbalance`, `wear_lube`, `cavit_jeu`                                             |

**Methods:**

| Method             | Returns     | Description                                                                  |
| ------------------ | ----------- | ---------------------------------------------------------------------------- |
| `get_fault_list()` | `List[str]` | Builds list of active fault type names (prefixes mixed faults with `mixed_`) |

---

### `SeverityConfig`

> Multi-severity fault progression configuration.

| Parameter            | Type                             | Default                                                                                           | Valid Range (Schema) | Description                                    |
| -------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------------- |
| `enabled`            | `bool`                           | `True`                                                                                            | —                    | Enable severity levels                         |
| `levels`             | `List[str]`                      | `['incipient', 'mild', 'moderate', 'severe']`                                                     | —                    | Severity level names                           |
| `ranges`             | `Dict[str, Tuple[float, float]]` | `incipient: (0.20, 0.45)`, `mild: (0.45, 0.70)`, `moderate: (0.70, 0.90)`, `severe: (0.90, 1.00)` | —                    | Non-overlapping severity factor ranges         |
| `temporal_evolution` | `float`                          | `0.30`                                                                                            | 0–1                  | Fraction of signals showing progressive growth |

---

### `NoiseConfig`

> 7-layer independent noise model configuration.

**Enable/Disable Flags:**

| Parameter      | Type   | Default | Description                             |
| -------------- | ------ | ------- | --------------------------------------- |
| `measurement`  | `bool` | `True`  | Sensor electronics thermal noise        |
| `emi`          | `bool` | `True`  | Electromagnetic interference (50/60 Hz) |
| `pink`         | `bool` | `True`  | 1/f environmental noise                 |
| `drift`        | `bool` | `True`  | Low-frequency thermal drift             |
| `quantization` | `bool` | `True`  | ADC resolution limits                   |
| `sensor_drift` | `bool` | `True`  | Sensor calibration decay                |
| `impulse`      | `bool` | `True`  | Sporadic mechanical impacts             |

**Noise Level Parameters:**

| Key (in `levels` dict) | Type    | Default | Description         |
| ---------------------- | ------- | ------- | ------------------- |
| `measurement`          | `float` | `0.03`  | Gaussian std        |
| `emi`                  | `float` | `0.01`  | EMI amplitude       |
| `pink`                 | `float` | `0.02`  | Pink noise std      |
| `drift`                | `float` | `0.015` | Drift amplitude     |
| `quantization_step`    | `float` | `0.001` | ADC step size       |
| `sensor_drift_rate`    | `float` | `0.001` | Drift per second    |
| `impulse_rate`         | `float` | `2.0`   | Impulses per second |

| Parameter  | Type    | Default | Valid Range (Schema) | Description                                 |
| ---------- | ------- | ------- | -------------------- | ------------------------------------------- |
| `aliasing` | `float` | `0.10`  | 0–1                  | Fraction of signals with aliasing artifacts |

---

### `OperatingConfig`

> Variable operating conditions configuration.

| Parameter         | Type                  | Default        | Valid Range (Schema) | Description                               |
| ----------------- | --------------------- | -------------- | -------------------- | ----------------------------------------- |
| `speed_variation` | `float`               | `0.10`         | 0–0.5                | ± fractional deviation from nominal speed |
| `load_range`      | `Tuple[float, float]` | `(0.30, 1.00)` | —                    | Load as fraction of rated (30–100%)       |
| `temp_range`      | `Tuple[float, float]` | `(40.0, 80.0)` | —                    | Operating temperature range in °C         |

---

### `PhysicsConfig`

> Physics-based parameter calculation configuration.

| Parameter               | Type                  | Default           | Valid Range (Schema) | Description                                    |
| ----------------------- | --------------------- | ----------------- | -------------------- | ---------------------------------------------- |
| `enabled`               | `bool`                | `True`            | —                    | Enable physics calculations                    |
| `calculate_sommerfeld`  | `bool`                | `True`            | —                    | Calculate Sommerfeld from operating conditions |
| `sommerfeld_base`       | `float`               | `0.15`            | 0.01–1.0             | Base Sommerfeld number                         |
| `reynolds_range`        | `Tuple[float, float]` | `(500.0, 5000.0)` | —                    | Reynolds number range                          |
| `clearance_ratio_range` | `Tuple[float, float]` | `(0.001, 0.003)`  | —                    | Clearance ratio range                          |

---

### `TransientConfig`

> Non-stationary behavior (transients) configuration.

| Parameter     | Type        | Default                                            | Valid Range (Schema) | Description                         |
| ------------- | ----------- | -------------------------------------------------- | -------------------- | ----------------------------------- |
| `enabled`     | `bool`      | `True`                                             | —                    | Enable transient injection          |
| `probability` | `float`     | `0.25`                                             | 0–1                  | Fraction of signals with transients |
| `types`       | `List[str]` | `['speed_ramp', 'load_step', 'thermal_expansion']` | —                    | Available transient types           |

---

### `AugmentationConfig`

> Data augmentation configuration.

| Parameter               | Type                  | Default                                                | Valid Range (Schema) | Description                                 |
| ----------------------- | --------------------- | ------------------------------------------------------ | -------------------- | ------------------------------------------- |
| `enabled`               | `bool`                | `True`                                                 | —                    | Enable augmentation                         |
| `ratio`                 | `float`               | `0.30`                                                 | 0–1.0                | Fraction of additional augmented samples    |
| `methods`               | `List[str]`           | `['time_shift', 'amplitude_scale', 'noise_injection']` | —                    | Augmentation methods                        |
| `time_shift_max`        | `float`               | `0.02`                                                 | 0–0.5                | Max time shift as fraction of signal length |
| `amplitude_scale_range` | `Tuple[float, float]` | `(0.85, 1.15)`                                         | —                    | Amplitude scaling range                     |
| `extra_noise_range`     | `Tuple[float, float]` | `(0.02, 0.05)`                                         | —                    | Extra noise injection range                 |

---

### `DataConfig` (Master)

> Master data generation configuration aggregating all sub-configs.

| Parameter                   | Type                 | Default                         | Valid Range (Schema) | Description                         |
| --------------------------- | -------------------- | ------------------------------- | -------------------- | ----------------------------------- |
| `num_signals_per_fault`     | `int`                | `100`                           | 1–10000              | Number of signals per fault type    |
| `output_dir`                | `str`                | `'data_signaux_sep_production'` | —                    | Output directory for generated data |
| `signal`                    | `SignalConfig`       | `SignalConfig()`                | —                    | Signal generation sub-config        |
| `fault`                     | `FaultConfig`        | `FaultConfig()`                 | —                    | Fault selection sub-config          |
| `severity`                  | `SeverityConfig`     | `SeverityConfig()`              | —                    | Severity progression sub-config     |
| `noise`                     | `NoiseConfig`        | `NoiseConfig()`                 | —                    | 7-layer noise model sub-config      |
| `operating`                 | `OperatingConfig`    | `OperatingConfig()`             | —                    | Operating conditions sub-config     |
| `physics`                   | `PhysicsConfig`      | `PhysicsConfig()`               | —                    | Physics parameters sub-config       |
| `transient`                 | `TransientConfig`    | `TransientConfig()`             | —                    | Transient behavior sub-config       |
| `augmentation`              | `AugmentationConfig` | `AugmentationConfig()`          | —                    | Data augmentation sub-config        |
| `rng_seed`                  | `int`                | `42`                            | —                    | Random seed for reproducibility     |
| `per_signal_seed_variation` | `bool`               | `True`                          | —                    | Vary seed per signal                |
| `save_metadata`             | `bool`               | `True`                          | —                    | Save generation metadata            |
| `verbose`                   | `bool`               | `True`                          | —                    | Enable verbose output               |

**Methods:**

| Method                 | Returns | Description                                                                                       |
| ---------------------- | ------- | ------------------------------------------------------------------------------------------------- |
| `get_total_signals()`  | `int`   | Calculates total signals (faults × per-fault + augmented)                                         |
| `from_matlab_struct()` | —       | **DEPRECATED** — raises `NotImplementedError`. Use `data.matlab_importer.MatlabImporter` instead. |

---

## Model Architecture — `model_config.py`

### `CNN1DConfig`

> 1D Convolutional Neural Network configuration.

| Parameter        | Type        | Default                         | Valid Range (Schema) | Description                                             |
| ---------------- | ----------- | ------------------------------- | -------------------- | ------------------------------------------------------- |
| `input_length`   | `int`       | `102400` (from `SIGNAL_LENGTH`) | ≥ 1                  | Input signal length                                     |
| `num_classes`    | `int`       | `11` (from `NUM_CLASSES`)       | ≥ 2                  | Number of output classes                                |
| `conv_channels`  | `List[int]` | `[32, 64, 128, 256]`            | each ≥ 1             | Channels per conv layer                                 |
| `kernel_sizes`   | `List[int]` | `[15, 11, 7, 5]`                | —                    | Kernel sizes per layer                                  |
| `strides`        | `List[int]` | `[2, 2, 2, 2]`                  | —                    | Strides per layer                                       |
| `pool_sizes`     | `List[int]` | `[4, 4, 4, 4]`                  | —                    | Max pool sizes per layer                                |
| `fc_hidden_dims` | `List[int]` | `[512, 256]`                    | —                    | Fully connected layer sizes                             |
| `dropout_prob`   | `float`     | `0.5`                           | 0.0–1.0              | Dropout probability                                     |
| `batch_norm`     | `bool`      | `True`                          | —                    | Use batch normalization                                 |
| `activation`     | `str`       | `'relu'`                        | —                    | Activation: `'relu'`, `'leaky_relu'`, `'elu'`, `'gelu'` |

---

### `ResNet1DConfig`

> 1D ResNet configuration.

| Parameter              | Type        | Default               | Valid Range (Schema) | Description                                 |
| ---------------------- | ----------- | --------------------- | -------------------- | ------------------------------------------- |
| `input_length`         | `int`       | `102400`              | ≥ 1                  | Input signal length                         |
| `num_classes`          | `int`       | `11`                  | ≥ 2                  | Number of output classes                    |
| `blocks`               | `List[int]` | `[2, 2, 2, 2]`        | each ≥ 1             | Residual blocks per stage (ResNet-18 style) |
| `channels`             | `List[int]` | `[64, 128, 256, 512]` | —                    | Channels per stage                          |
| `initial_kernel_size`  | `int`       | `15`                  | —                    | First conv kernel size                      |
| `initial_stride`       | `int`       | `2`                   | —                    | First conv stride                           |
| `residual_kernel_size` | `int`       | `7`                   | —                    | Residual block kernel size                  |
| `use_bottleneck`       | `bool`      | `False`               | —                    | Use 1×1 convs for dimensionality reduction  |
| `dropout_prob`         | `float`     | `0.3`                 | 0.0–1.0              | Dropout probability                         |
| `batch_norm`           | `bool`      | `True`                | —                    | Use batch normalization                     |
| `use_global_avg_pool`  | `bool`      | `True`                | —                    | Use global average pooling                  |

---

### `TransformerConfig`

> Transformer model for time series classification.

| Parameter                 | Type    | Default  | Valid Range (Schema) | Description                                 |
| ------------------------- | ------- | -------- | -------------------- | ------------------------------------------- |
| `input_length`            | `int`   | `102400` | ≥ 1                  | Input signal length                         |
| `num_classes`             | `int`   | `11`     | ≥ 2                  | Number of output classes                    |
| `d_model`                 | `int`   | `256`    | ≥ 1                  | Model embedding dimension                   |
| `nhead`                   | `int`   | `8`      | ≥ 1                  | Number of attention heads                   |
| `num_layers`              | `int`   | `6`      | ≥ 1                  | Number of encoder layers                    |
| `dim_feedforward`         | `int`   | `1024`   | —                    | FFN hidden dimension                        |
| `use_positional_encoding` | `bool`  | `True`   | —                    | Enable positional encoding                  |
| `max_seq_length`          | `int`   | `10000`  | —                    | Max sequence length for positional encoding |
| `dropout_prob`            | `float` | `0.1`    | —                    | General dropout                             |
| `attention_dropout`       | `float` | `0.1`    | —                    | Attention-specific dropout                  |
| `use_patch_embedding`     | `bool`  | `True`   | —                    | Use patch/segment embedding                 |
| `patch_size`              | `int`   | `256`    | —                    | Patch length for embedding                  |
| `patch_stride`            | `int`   | `128`    | —                    | Stride for patching                         |
| `activation`              | `str`   | `'gelu'` | —                    | Activation function                         |

---

### `LSTMConfig`

> LSTM model configuration.

| Parameter        | Type        | Default      | Valid Range (Schema) | Description                     |
| ---------------- | ----------- | ------------ | -------------------- | ------------------------------- |
| `input_length`   | `int`       | `102400`     | ≥ 1                  | Input signal length             |
| `num_classes`    | `int`       | `11`         | ≥ 2                  | Number of output classes        |
| `input_size`     | `int`       | `1`          | —                    | Number of features per timestep |
| `hidden_size`    | `int`       | `256`        | ≥ 1                  | LSTM hidden state size          |
| `num_layers`     | `int`       | `3`          | ≥ 1                  | Number of stacked LSTM layers   |
| `bidirectional`  | `bool`      | `True`       | —                    | Use bidirectional LSTM          |
| `dropout_prob`   | `float`     | `0.3`        | —                    | Dropout probability             |
| `fc_hidden_dims` | `List[int]` | `[256, 128]` | —                    | FC layers after LSTM            |

---

### `HybridPINNConfig`

> Hybrid Physics-Informed Neural Network configuration.

| Parameter             | Type                       | Default                                                      | Valid Range (Schema)                     | Description                                   |
| --------------------- | -------------------------- | ------------------------------------------------------------ | ---------------------------------------- | --------------------------------------------- |
| `input_length`        | `int`                      | `102400`                                                     | ≥ 1                                      | Input signal length                           |
| `num_classes`         | `int`                      | `11`                                                         | ≥ 2                                      | Number of output classes                      |
| `backbone`            | `str`                      | `'resnet1d'`                                                 | `'cnn1d'`, `'resnet1d'`, `'transformer'` | Backbone architecture                         |
| `backbone_config`     | `Optional[Dict[str, Any]]` | `None`                                                       | —                                        | Custom backbone parameters                    |
| `use_physics_branch`  | `bool`                     | `True`                                                       | —                                        | Enable physics feature branch                 |
| `physics_features`    | `List[str]`                | `['sommerfeld', 'reynolds', 'speed', 'load', 'temperature']` | —                                        | Physics features to use                       |
| `physics_hidden_dims` | `List[int]`                | `[64, 32]`                                                   | —                                        | Physics branch hidden layers                  |
| `fusion_method`       | `str`                      | `'concat'`                                                   | —                                        | Fusion: `'concat'`, `'attention'`, `'gating'` |
| `use_physics_loss`    | `bool`                     | `True`                                                       | —                                        | Enable physics loss term                      |
| `physics_loss_weight` | `float`                    | `0.1`                                                        | 0.0–1.0                                  | Weight of physics loss                        |
| `dropout_prob`        | `float`                    | `0.3`                                                        | —                                        | Dropout probability                           |

---

### `EnsembleConfig`

> Ensemble model configuration.

| Parameter         | Type                    | Default                                | Valid Range (Schema)                                         | Description                                            |
| ----------------- | ----------------------- | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| `num_classes`     | `int`                   | `11`                                   | ≥ 2                                                          | Number of output classes                               |
| `model_configs`   | `List[Dict[str, Any]]`  | `[]`                                   | —                                                            | Per-model configuration dicts                          |
| `model_types`     | `List[str]`             | `['cnn1d', 'resnet1d', 'transformer']` | —                                                            | Model types in ensemble                                |
| `ensemble_method` | `str`                   | `'soft_voting'`                        | `'hard_voting'`, `'soft_voting'`, `'stacking'`, `'weighted'` | Combination method                                     |
| `model_weights`   | `Optional[List[float]]` | `None`                                 | —                                                            | Weights for weighted voting                            |
| `meta_learner`    | `str`                   | `'logistic'`                           | —                                                            | Meta-learner: `'logistic'`, `'random_forest'`, `'mlp'` |

---

### `ModelConfig` (Master)

> Master model configuration with factory pattern.

| Parameter         | Type                | Default               | Valid Range (Schema)                                                              | Description                |
| ----------------- | ------------------- | --------------------- | --------------------------------------------------------------------------------- | -------------------------- |
| `model_type`      | `str`               | `'cnn1d'`             | `'cnn1d'`, `'resnet1d'`, `'transformer'`, `'lstm'`, `'hybrid_pinn'`, `'ensemble'` | Active model selection     |
| `cnn1d`           | `CNN1DConfig`       | `CNN1DConfig()`       | —                                                                                 | CNN sub-config             |
| `resnet1d`        | `ResNet1DConfig`    | `ResNet1DConfig()`    | —                                                                                 | ResNet sub-config          |
| `transformer`     | `TransformerConfig` | `TransformerConfig()` | —                                                                                 | Transformer sub-config     |
| `lstm`            | `LSTMConfig`        | `LSTMConfig()`        | —                                                                                 | LSTM sub-config            |
| `hybrid_pinn`     | `HybridPINNConfig`  | `HybridPINNConfig()`  | —                                                                                 | PINN sub-config            |
| `ensemble`        | `EnsembleConfig`    | `EnsembleConfig()`    | —                                                                                 | Ensemble sub-config        |
| `pretrained`      | `bool`              | `False`               | —                                                                                 | Load pretrained weights    |
| `pretrained_path` | `Optional[str]`     | `None`                | —                                                                                 | Path to pretrained weights |

**Methods:**

| Method                | Returns      | Description                                  |
| --------------------- | ------------ | -------------------------------------------- |
| `get_active_config()` | `BaseConfig` | Returns the sub-config matching `model_type` |

---

## Training — `training_config.py`

### `OptimizerConfig`

> Optimizer configuration.

| Parameter         | Type              | Default        | Valid Range (Schema)                      | Description                          |
| ----------------- | ----------------- | -------------- | ----------------------------------------- | ------------------------------------ |
| `name`            | `str`             | `'adamw'`      | `'adam'`, `'adamw'`, `'sgd'`, `'rmsprop'` | Optimizer type                       |
| `lr`              | `float`           | `0.001`        | ≥ 0.0                                     | Learning rate                        |
| `weight_decay`    | `float`           | `0.01`         | ≥ 0.0                                     | Weight decay / L2 regularization     |
| `betas`           | `tuple`           | `(0.9, 0.999)` | —                                         | Adam/AdamW beta parameters           |
| `eps`             | `float`           | `1e-8`         | —                                         | Adam epsilon for numerical stability |
| `amsgrad`         | `bool`            | `False`        | —                                         | Use AMSGrad variant                  |
| `momentum`        | `float`           | `0.9`          | 0.0–1.0                                   | SGD momentum                         |
| `nesterov`        | `bool`            | `True`         | —                                         | Use Nesterov momentum                |
| `alpha`           | `float`           | `0.99`         | —                                         | RMSprop smoothing constant           |
| `clip_grad_norm`  | `Optional[float]` | `1.0`          | —                                         | Max gradient norm for clipping       |
| `clip_grad_value` | `Optional[float]` | `None`         | —                                         | Max absolute gradient value          |

---

### `SchedulerConfig`

> Learning rate scheduler configuration.

| Parameter          | Type    | Default    | Valid Range (Schema)                                      | Description                                          |
| ------------------ | ------- | ---------- | --------------------------------------------------------- | ---------------------------------------------------- |
| `name`             | `str`   | `'cosine'` | `'step'`, `'cosine'`, `'plateau'`, `'onecycle'`, `'none'` | Scheduler type                                       |
| `step_size`        | `int`   | `30`       | ≥ 1                                                       | StepLR: epochs between decays                        |
| `gamma`            | `float` | `0.1`      | 0.0–1.0                                                   | StepLR: decay factor                                 |
| `T_max`            | `int`   | `100`      | —                                                         | CosineAnnealing: total epochs                        |
| `eta_min`          | `float` | `1e-6`     | —                                                         | CosineAnnealing: minimum LR                          |
| `mode`             | `str`   | `'min'`    | —                                                         | Plateau: `'min'` or `'max'`                          |
| `factor`           | `float` | `0.1`      | —                                                         | Plateau: reduction factor                            |
| `patience`         | `int`   | `10`       | —                                                         | Plateau: epochs to wait                              |
| `threshold`        | `float` | `1e-4`     | —                                                         | Plateau: improvement threshold                       |
| `min_lr`           | `float` | `1e-6`     | —                                                         | Plateau: minimum LR                                  |
| `max_lr`           | `float` | `0.01`     | —                                                         | OneCycleLR: peak LR                                  |
| `pct_start`        | `float` | `0.3`      | —                                                         | OneCycleLR: fraction of cycle for warm-up            |
| `anneal_strategy`  | `str`   | `'cos'`    | —                                                         | OneCycleLR: annealing strategy                       |
| `div_factor`       | `float` | `25.0`     | —                                                         | OneCycleLR: initial LR = max_lr / div_factor         |
| `final_div_factor` | `float` | `1e4`      | —                                                         | OneCycleLR: final LR = initial_lr / final_div_factor |
| `warmup_epochs`    | `int`   | `0`        | —                                                         | Warmup epochs before scheduler                       |
| `warmup_start_lr`  | `float` | `1e-5`     | —                                                         | Starting LR during warmup                            |

---

### `CallbackConfig`

> Training callback configuration.

| Parameter                  | Type    | Default              | Description                    |
| -------------------------- | ------- | -------------------- | ------------------------------ |
| `use_early_stopping`       | `bool`  | `True`               | Enable early stopping          |
| `early_stopping_patience`  | `int`   | `20`                 | Epochs to wait before stopping |
| `early_stopping_metric`    | `str`   | `'val_loss'`         | Metric to monitor              |
| `early_stopping_mode`      | `str`   | `'min'`              | `'min'` or `'max'`             |
| `early_stopping_min_delta` | `float` | `1e-4`               | Minimum improvement threshold  |
| `use_checkpoint`           | `bool`  | `True`               | Enable model checkpointing     |
| `checkpoint_dir`           | `str`   | `'checkpoints'`      | Checkpoint save directory      |
| `save_best_only`           | `bool`  | `True`               | Only save best model           |
| `checkpoint_metric`        | `str`   | `'val_loss'`         | Checkpoint selection metric    |
| `checkpoint_mode`          | `str`   | `'min'`              | `'min'` or `'max'`             |
| `save_frequency`           | `int`   | `1`                  | Save every N epochs            |
| `use_tensorboard`          | `bool`  | `True`               | Enable TensorBoard logging     |
| `tensorboard_dir`          | `str`   | `'runs'`             | TensorBoard log directory      |
| `log_histograms`           | `bool`  | `False`              | Log weight histograms          |
| `use_csv_logger`           | `bool`  | `True`               | Enable CSV metric logging      |
| `csv_log_file`             | `str`   | `'training_log.csv'` | CSV log file path              |
| `log_lr`                   | `bool`  | `True`               | Log learning rate              |

---

### `MixedPrecisionConfig`

> FP16/BF16 mixed precision training configuration.

| Parameter         | Type    | Default         | Valid Range (Schema)      | Description                   |
| ----------------- | ------- | --------------- | ------------------------- | ----------------------------- |
| `enabled`         | `bool`  | `False`         | —                         | Enable mixed precision        |
| `dtype`           | `str`   | `'float16'`     | `'float16'`, `'bfloat16'` | Precision type                |
| `loss_scale`      | `str`   | `'dynamic'`     | —                         | Loss scaling mode             |
| `init_scale`      | `float` | `65536.0` (2¹⁶) | —                         | Initial loss scale factor     |
| `growth_interval` | `int`   | `2000`          | —                         | Steps between scale increases |

---

### `RegularizationConfig`

> Additional regularization techniques.

| Parameter               | Type    | Default | Valid Range (Schema) | Description                        |
| ----------------------- | ------- | ------- | -------------------- | ---------------------------------- |
| `label_smoothing`       | `float` | `0.0`   | 0.0–1.0              | Label smoothing factor             |
| `use_mixup`             | `bool`  | `False` | —                    | Enable Mixup augmentation          |
| `mixup_alpha`           | `float` | `0.2`   | ≥ 0.0                | Mixup beta distribution parameter  |
| `use_cutmix`            | `bool`  | `False` | —                    | Enable CutMix augmentation         |
| `cutmix_alpha`          | `float` | `1.0`   | —                    | CutMix beta distribution parameter |
| `stochastic_depth_prob` | `float` | `0.0`   | —                    | Stochastic depth drop probability  |
| `dropout_prob`          | `float` | `0.3`   | 0.0–1.0              | Model-level dropout                |
| `use_ema`               | `bool`  | `False` | —                    | Exponential Moving Average         |
| `ema_decay`             | `float` | `0.999` | —                    | EMA decay rate                     |

---

### `TrainingConfig` (Master)

> Master training configuration aggregating all training settings.

| Parameter                | Type                   | Default                                     | Valid Range (Schema)        | Description                                             |
| ------------------------ | ---------------------- | ------------------------------------------- | --------------------------- | ------------------------------------------------------- |
| `num_epochs`             | `int`                  | `100`                                       | ≥ 1                         | Total training epochs                                   |
| `batch_size`             | `int`                  | `64`                                        | ≥ 1                         | Batch size                                              |
| `num_workers`            | `int`                  | `4`                                         | ≥ 0                         | DataLoader worker threads                               |
| `pin_memory`             | `bool`                 | `True`                                      | —                           | Pin memory for GPU transfer                             |
| `optimizer`              | `OptimizerConfig`      | `OptimizerConfig()`                         | —                           | Optimizer sub-config                                    |
| `scheduler`              | `SchedulerConfig`      | `SchedulerConfig()`                         | —                           | Scheduler sub-config                                    |
| `callbacks`              | `CallbackConfig`       | `CallbackConfig()`                          | —                           | Callbacks sub-config                                    |
| `mixed_precision`        | `MixedPrecisionConfig` | `MixedPrecisionConfig()`                    | —                           | Mixed precision sub-config                              |
| `regularization`         | `RegularizationConfig` | `RegularizationConfig()`                    | —                           | Regularization sub-config                               |
| `loss_function`          | `str`                  | `'cross_entropy'`                           | —                           | Loss: `'cross_entropy'`, `'focal'`, `'label_smoothing'` |
| `metrics`                | `List[str]`            | `['accuracy', 'f1', 'precision', 'recall']` | —                           | Metrics to track                                        |
| `device`                 | `str`                  | `'cuda'`                                    | `'cuda'`, `'cpu'`, `'auto'` | Computation device                                      |
| `multi_gpu`              | `bool`                 | `False`                                     | —                           | Enable multi-GPU                                        |
| `gpu_ids`                | `Optional[List[int]]`  | `None`                                      | —                           | Specific GPU IDs                                        |
| `seed`                   | `int`                  | `42`                                        | ≥ 0                         | Random seed                                             |
| `deterministic`          | `bool`                 | `True`                                      | —                           | Deterministic mode                                      |
| `log_interval`           | `int`                  | `10`                                        | —                           | Log every N batches                                     |
| `val_interval`           | `int`                  | `1`                                         | —                           | Validate every N epochs                                 |
| `resume_from_checkpoint` | `Optional[str]`        | `None`                                      | —                           | Path to resume checkpoint                               |
| `accumulation_steps`     | `int`                  | `1`                                         | —                           | Gradient accumulation steps                             |
| `distributed`            | `bool`                 | `False`                                     | —                           | Enable distributed training                             |
| `world_size`             | `int`                  | `1`                                         | —                           | Number of distributed processes                         |
| `rank`                   | `int`                  | `0`                                         | —                           | Process rank                                            |
| `dist_backend`           | `str`                  | `'nccl'`                                    | —                           | Distributed backend                                     |

**Methods:**

| Method                       | Signature                   | Description                                                                                             |
| ---------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------- |
| `get_total_training_steps()` | `(dataset_size: int) → int` | Calculates total optimization steps: `(dataset_size // (batch_size × accumulation_steps)) × num_epochs` |

---

## Experiment Tracking — `experiment_config.py`

### `MLflowConfig`

> MLflow tracking server configuration.

| Parameter           | Type             | Default                     | Valid Range (Schema) | Description                       |
| ------------------- | ---------------- | --------------------------- | -------------------- | --------------------------------- |
| `tracking_uri`      | `str`            | `'./mlruns'`                | —                    | MLflow tracking server URI        |
| `registry_uri`      | `Optional[str]`  | `None`                      | —                    | Model registry URI                |
| `experiment_name`   | `str`            | `'bearing_fault_diagnosis'` | min 1 char           | MLflow experiment name            |
| `run_name`          | `Optional[str]`  | `None`                      | —                    | Run name (auto-generated if None) |
| `tags`              | `Dict[str, str]` | `{}`                        | —                    | MLflow tags                       |
| `autolog`           | `bool`           | `True`                      | —                    | Enable auto-logging               |
| `log_models`        | `bool`           | `True`                      | —                    | Log model artifacts               |
| `log_every_n_steps` | `int`            | `10`                        | ≥ 1                  | Metric logging frequency          |
| `artifact_location` | `Optional[str]`  | `None`                      | —                    | Custom artifact storage path      |

---

### `LoggingConfig`

> Experiment logging configuration.

| Parameter              | Type        | Default                                                   | Description                      |
| ---------------------- | ----------- | --------------------------------------------------------- | -------------------------------- |
| `log_params`           | `bool`      | `True`                                                    | Log hyperparameters              |
| `log_metrics`          | `bool`      | `True`                                                    | Log training metrics             |
| `log_artifacts`        | `bool`      | `True`                                                    | Log artifacts                    |
| `log_models`           | `bool`      | `True`                                                    | Log model files                  |
| `log_system_info`      | `bool`      | `True`                                                    | Log system information           |
| `metrics_to_log`       | `List[str]` | `['loss', 'accuracy', 'f1_score', 'precision', 'recall']` | Metrics list                     |
| `log_confusion_matrix` | `bool`      | `True`                                                    | Log confusion matrix             |
| `log_roc_curves`       | `bool`      | `True`                                                    | Log ROC curves                   |
| `artifacts_to_log`     | `List[str]` | `['config', 'model', 'plots', 'checkpoints']`             | Artifact types                   |
| `save_best_model`      | `bool`      | `True`                                                    | Save best model during training  |
| `save_final_model`     | `bool`      | `True`                                                    | Save model at training end       |
| `log_training_plots`   | `bool`      | `True`                                                    | Log training visualization plots |
| `log_evaluation_plots` | `bool`      | `True`                                                    | Log evaluation plots             |
| `plot_frequency`       | `int`       | `5`                                                       | Generate plots every N epochs    |

---

### `ExperimentConfig` (Master)

> Master experiment configuration.

| Parameter                 | Type             | Default                | Valid Range (Schema) | Description                  |
| ------------------------- | ---------------- | ---------------------- | -------------------- | ---------------------------- |
| `experiment_name`         | `str`            | `'default_experiment'` | min 1 char           | Experiment name              |
| `description`             | `str`            | `''`                   | —                    | Experiment description       |
| `tags`                    | `Dict[str, str]` | `{}`                   | —                    | Experiment tags              |
| `mlflow`                  | `MLflowConfig`   | `MLflowConfig()`       | —                    | MLflow sub-config            |
| `logging`                 | `LoggingConfig`  | `LoggingConfig()`      | —                    | Logging sub-config           |
| `seed`                    | `int`            | `42`                   | ≥ 0                  | Random seed                  |
| `deterministic`           | `bool`           | `True`                 | —                    | Deterministic mode           |
| `checkpoint_dir`          | `str`            | `'checkpoints'`        | —                    | Checkpoint directory         |
| `checkpoint_frequency`    | `int`            | `5`                    | ≥ 1                  | Save every N epochs          |
| `keep_n_checkpoints`      | `int`            | `3`                    | —                    | Keep only last N checkpoints |
| `early_stopping_patience` | `int`            | `20`                   | ≥ 1                  | Patience for early stopping  |
| `early_stopping_metric`   | `str`            | `'val_loss'`           | —                    | Metric to monitor            |
| `early_stopping_mode`     | `str`            | `'min'`                | —                    | `'min'` or `'max'`           |

**Methods:**

| Method                     | Signature                                           | Description                                                                                                  |
| -------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `get_run_name()`           | `(model_name: str, timestamp: Optional[str]) → str` | Generates `{model_name}_{timestamp}` (timestamp auto-generated if None)                                      |
| `get_tags_with_defaults()` | `(**additional_tags) → Dict[str, str]`              | Merges default system tags (experiment, seed, platform, python_version) with config tags and additional tags |
