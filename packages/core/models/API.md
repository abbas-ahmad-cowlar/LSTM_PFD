# Models API Reference

> Complete API documentation for all model classes in `packages/core/models/`.

---

## Base Class

### `BaseModel` — `base_model.py`

> Abstract base class for all fault diagnosis models. Inherits from `nn.Module` and `ABC`.

**Constructor:**

```python
BaseModel()
```

All subclasses must implement `forward()` and `get_config()`.

**Abstract Methods:**

#### `forward(x: torch.Tensor) -> torch.Tensor`

Forward pass. Input: `[B, C, T]` or `[B, T]`. Returns: `[B, num_classes]`.

#### `get_config() -> dict`

Return model hyperparameters as a dictionary.

**Utility Methods:**

#### `count_parameters() -> dict`

Returns `{'total': int, 'trainable': int, 'non_trainable': int}`.

#### `get_num_params() -> int`

Returns total parameter count (convenience wrapper).

#### `get_model_size_mb() -> float`

Returns estimated model size in megabytes.

#### `summary(input_shape: Optional[Tuple[int, ...]] = None) -> str`

Generates Keras-style model summary string.

#### `save_checkpoint(path, epoch, optimizer_state=None, metrics=None, **kwargs)`

Save model checkpoint to disk.

#### `load_checkpoint(cls, path, device=None) -> Tuple[model, checkpoint_dict]`

Class method. Load model from checkpoint file.

#### `freeze_backbone()` / `unfreeze_backbone()`

Freeze/unfreeze all parameters for transfer learning.

#### `get_layer_names() -> List[str]`

Returns names of all layers/modules.

#### `get_activation(layer_name: str) -> torch.Tensor`

Placeholder for layer activation extraction (requires forward hooks).

#### `to_device(device: torch.device) -> self`

Move model to device and return self for chaining.

---

## CNN Models

### `CNN1D` — `cnn_1d.py`

> 6-layer 1D CNN with adaptive pooling for variable-length signals.

**Constructor:**

```python
CNN1D(
    num_classes: int = 11,
    input_channels: int = 1,
    dropout: float = 0.3,
    use_bn: bool = True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | `int` | `11` | Number of output classes |
| `input_channels` | `int` | `1` | Input channels (1 for raw signal) |
| `dropout` | `float` | `0.3` | Dropout probability |
| `use_bn` | `bool` | `True` | Use batch normalization |

**Architecture:** Conv(1→32, k=7) → Conv(32→64, k=5) → Pool → Conv(64→128) → Conv(128→128) → Pool → Conv(128→256) → Conv(256→256) → AdaptiveAvgPool → FC(256→128) → FC(128→num_classes)

**Methods:**

- `forward(x: Tensor) -> Tensor` — `[B, C, T]` → `[B, num_classes]`
- `get_feature_extractor() -> nn.Module` — Returns conv layers as sequential
- `freeze_backbone()` / `unfreeze_backbone()` — Transfer learning
- `get_config() -> dict`

**Factory:** `create_cnn1d(num_classes=11, **kwargs) -> CNN1D`

**Example:**

```python
from packages.core.models import CNN1D
model = CNN1D(num_classes=11, dropout=0.3)
x = torch.randn(4, 1, 102400)
logits = model(x)  # [4, 11]
```

---

### `ResNet1D` — `resnet_1d.py`

> ResNet-18/34 adapted for 1D signal classification with residual connections.

**Constructor:**

```python
ResNet1D(
    num_classes: int = 11,
    input_channels: int = 1,
    dropout: float = 0.2,
    layers: List[int] = None  # Default: [2, 2, 2, 2] for ResNet-18
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | `int` | `11` | Number of output classes |
| `input_channels` | `int` | `1` | Input channels |
| `dropout` | `float` | `0.2` | Dropout probability |
| `layers` | `List[int]` | `[2,2,2,2]` | Blocks per layer (ResNet-18: [2,2,2,2], ResNet-34: [3,4,6,3]) |

**Architecture:** Conv(1→64, k=7, s=2) → BN → ReLU → MaxPool → Layer1(64, ×2) → Layer2(128, ×2, s=2) → Layer3(256, ×2, s=2) → Layer4(512, ×2, s=2) → AdaptiveAvgPool → FC(512→num_classes)

**Methods:** Same as `CNN1D` (forward, get_feature_extractor, freeze/unfreeze, get_config).

**Factories:**

- `create_resnet18_1d(num_classes=11, **kwargs) -> ResNet1D` — layers=[2,2,2,2]
- `create_resnet34_1d(num_classes=11, **kwargs) -> ResNet1D` — layers=[3,4,6,3]

---

## Transformer Models

### `SignalTransformer` — `transformer/signal_transformer.py`

> Transformer encoder for time-series classification using patch embeddings.

**Constructor:**

```python
SignalTransformer(
    num_classes: int = 11,
    input_channels: int = 1,
    patch_size: int = 512,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 1024,
    dropout: float = 0.1,
    max_len: int = 5000,
    learnable_pe: bool = True
)
```

**Methods:**

- `forward(x: Tensor) -> Tensor`
- `get_attention_weights(x, layer_idx=-1) -> Tensor` — `[B, num_heads, L, L]`
- `get_all_attention_weights(x) -> List[Tensor]`
- `freeze_backbone()` / `unfreeze_backbone()`
- `get_config() -> dict`

**Factory:** `create_transformer(num_classes=11, **kwargs)`

---

### `VisionTransformer1D` — `transformer/vision_transformer_1d.py`

> ViT-style architecture with learnable [CLS] token for 1D signals.

**Constructor:**

```python
VisionTransformer1D(
    num_classes: int = 11,
    input_channels: int = 1,
    patch_size: int = 512,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 1024,
    dropout: float = 0.1,
    max_len: int = 5000
)
```

**Additional Methods:**

- `get_cls_token_attention(x, layer_idx=-1) -> Tensor` — `[B, num_heads, L]` showing which patches [CLS] attends to

**Presets:**

- `vit_tiny_1d(num_classes=11)` — Lightweight variant
- `vit_small_1d(num_classes=11)`
- `vit_base_1d(num_classes=11)`

**Factory:** `create_vit_1d(num_classes=11, patch_size=512, **kwargs)`

---

## Hybrid Models

### `CNNTransformerHybrid` — `hybrid/cnn_transformer.py`

> CNN backbone for feature extraction + Transformer for sequence reasoning.

**Constructor:**

```python
CNNTransformerHybrid(
    num_classes: int = 11,
    cnn_backbone: str = 'resnet18',  # 'resnet18', 'resnet34', 'efficientnet'
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 4,
    d_ff: int = 2048,
    dropout: float = 0.1,
    freeze_cnn: bool = False
)
```

**Additional Methods:**

- `freeze_cnn_backbone()` / `unfreeze_cnn_backbone()`
- `freeze_transformer()` / `unfreeze_transformer()`

**Presets:**

- `cnn_transformer_small(num_classes=11)` — ~10M params
- `cnn_transformer_base(num_classes=11)` — ~15M params (recommended)
- `cnn_transformer_large(num_classes=11)` — ~25M params

**Factory:** `create_cnn_transformer_hybrid(num_classes=11, cnn_backbone='resnet18', **kwargs)`

---

### `CNNLSTM` — `hybrid/cnn_lstm.py`

> CNN feature extraction + Bidirectional LSTM with attention pooling.

**Constructor:**

```python
CNNLSTM(
    num_classes: int = 11,
    input_channels: int = 1,
    cnn_backbone: str = 'resnet18',  # 'resnet18', 'resnet34', 'simple'
    lstm_hidden: int = 256,
    lstm_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = True,
    use_attention: bool = True
)
```

**Additional Methods:**

- `get_attention_weights() -> Optional[Tensor]` — From last forward pass
- `get_cnn_features(x) -> Tensor` — Extract CNN features only
- `get_lstm_features(x) -> Tensor` — Extract CNN + LSTM features
- `freeze_cnn()` / `unfreeze_cnn()`

**Factory:** `create_cnn_lstm(num_classes=11, backbone='resnet18', **kwargs)`

---

### `CNNTCN` — `hybrid/cnn_tcn.py`

> CNN backbone + Temporal Convolutional Network with dilated causal convolutions.

**Constructor:**

```python
CNNTCN(
    num_classes: int = 11,
    input_channels: int = 1,
    cnn_backbone: str = 'simple',
    tcn_channels: Optional[List[int]] = None,  # Default: [128, 128, 128, 128]
    tcn_kernel_size: int = 3,
    dropout: float = 0.2
)
```

**Factory:** `create_cnn_tcn(num_classes=11, **kwargs)`

---

## Physics-Informed Models

### `HybridPINN` — `hybrid_pinn.py`

> Dual-branch architecture fusing CNN features with physics-based features.

**Constructor:**

```python
HybridPINN(
    num_classes: int = 11,
    input_channels: int = 1,
    physics_dim: int = 32,
    fusion_dim: int = 256,
    fusion_type: str = 'concat',  # 'concat', 'attention', 'gated'
    physics_weight: float = 0.1,
    dropout: float = 0.3
)
```

**Additional Methods:**

- `extract_cnn_features(x) -> Tensor` — `[B, 256]`
- `compute_total_loss(predictions, targets, data_loss_fn, physics_params=None) -> Tensor`

**Factory:** `create_hybrid_pinn(num_classes=11, **kwargs)`

---

### `PhysicsConstrainedCNN` — `pinn/physics_constrained_cnn.py`

> Standard CNN with physics-based loss constraints (no physics input branch).

**Constructor:**

```python
PhysicsConstrainedCNN(
    num_classes: int = 11,
    input_length: int = 102400,
    backbone: str = 'resnet18',
    dropout: float = 0.3,
    sample_rate: int = 20480,
    bearing_params: Optional[Dict[str, float]] = None
)
```

**Additional Methods:**

- `compute_physics_loss(signal, predictions, metadata=None, ...) -> Tensor`
- `forward_with_physics_loss(signal, metadata=None, lambda_physics=0.5) -> Tuple[Tensor, dict]`
- `get_model_info() -> dict`

---

### `MultitaskPINN` — `pinn/multitask_pinn.py`

> Multi-task learning: fault classification + speed/load/severity regression.

**Constructor:**

```python
MultitaskPINN(
    num_fault_classes: int = 11,
    num_severity_levels: int = 4,
    input_length: int = 102400,
    backbone: str = 'resnet18',
    shared_feature_dim: int = 512,
    task_hidden_dim: int = 128,
    dropout: float = 0.3
)
```

**Additional Methods:**

- `forward(signal, return_all_tasks=False)` — Returns logits or dict of all task outputs
- `compute_multitask_loss(signal, fault_labels, speed_labels=None, load_labels=None, severity_labels=None, task_weights=None) -> Tuple[Tensor, dict]`

**Factory:** `create_multitask_pinn(num_fault_classes=11, backbone='resnet18', adaptive=False, **kwargs)`

---

### `KnowledgeGraphPINN` — `pinn/knowledge_graph_pinn.py`

> Graph neural network leveraging fault relationship knowledge graph.

**Constructor:**

```python
KnowledgeGraphPINN(
    num_classes: int = 11,
    input_length: int = 102400,
    backbone: str = 'resnet18',
    node_feature_dim: int = 64,
    gcn_hidden_dim: int = 128,
    num_gcn_layers: int = 2,
    dropout: float = 0.3
)
```

**Additional Methods:**

- `forward_with_attention(signal) -> Tuple[Tensor, Tensor]` — Returns logits + attention over graph edges `[B, 11, 11]`
- `get_model_info() -> dict`

**Support Classes:**

- `FaultKnowledgeGraph` — Encodes fault relationships as adjacency matrix
- `GraphConvolutionLayer(in_features, out_features, bias=True)` — GCN layer implementing H' = σ(A·H·W)

---

## Ensemble Models

### `VotingEnsemble` (v2) — `ensemble/voting_ensemble.py`

> Voting ensemble with optimizable weights via grid search.

**Constructor:**

```python
VotingEnsemble(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
    voting_type: str = 'soft',  # 'soft' or 'hard'
    num_classes: int = 11
)
```

**Additional Methods:**

- `predict_proba(dataloader, device='cuda') -> np.ndarray` — `[N, num_classes]`

**Standalone Functions:**

- `soft_voting(predictions_list, weights=None) -> Tuple[np.ndarray, np.ndarray]`
- `hard_voting(predictions_list, weights=None) -> np.ndarray`
- `optimize_ensemble_weights(models, val_loader, device, search_resolution=10) -> np.ndarray`

---

### `BoostingEnsemble` — `ensemble/boosting_ensemble.py`

> AdaBoost-style sequential training for neural networks.

**Constructor:**

```python
BoostingEnsemble(
    models: List[nn.Module],
    model_weights: List[float],
    num_classes: int = 11
)
```

**Training wrapper:**

```python
AdaptiveBoosting(
    base_model_fn: Callable,
    n_estimators: int = 5,
    learning_rate: float = 1.0,
    num_classes: int = 11
)
```

- `fit(train_loader, val_loader=None, num_epochs_per_model=20, lr=0.001, ...)`
- `predict(dataloader, device) -> np.ndarray`
- `evaluate(dataloader, device) -> float`

**Factory:** `train_boosting(base_model_class, train_loader, ...)`

---

### `MixtureOfExperts` — `ensemble/mixture_of_experts.py`

> Gating network dynamically selects which expert to use per sample.

**Constructor:**

```python
MixtureOfExperts(
    experts: List[nn.Module],
    gating_network: Optional[GatingNetwork] = None,
    num_classes: int = 11,
    input_length: int = 102400,
    top_k: Optional[int] = None
)
```

**Additional Methods:**

- `forward(x, return_gates=False)` — Optionally returns gating weights
- `train_gating_network(dataloader, optimizer, criterion, num_epochs=20, ...)`
- `get_expert_usage(dataloader, device) -> dict`

**Support Classes:**

- `GatingNetwork(input_length, n_experts=3, hidden_dim=128)` — 1D CNN that outputs expert selection weights
- `ExpertModel(model, specialization=None)` — Wrapper adding specialization metadata

**Factory:** `create_specialized_experts(base_model_fn, specializations, train_loader, ...)`

---

## Fusion Models

### `EarlyFusion` — `fusion/early_fusion.py`

> Feature-level fusion — concatenates features from multiple extractors before classification.

### `LateFusion` — `fusion/late_fusion.py`

> Decision-level fusion — combines final predictions from multiple models.

**Standalone Functions:**

- `create_late_fusion(models, ...) -> LateFusion`
- `train_late_fusion_weights(models, val_loader, ...)`
- `late_fusion_weighted_average(predictions, weights)`
- `late_fusion_max(predictions)`
- `late_fusion_product(predictions)`
- `late_fusion_borda_count(predictions)`

---

## Factory Module

### `model_factory.py`

**Functions:**

| Function | Description |
|----------|-------------|
| `create_model(model_name, num_classes=11, **kwargs)` | Create model by registry key |
| `create_model_from_config(config: dict)` | Create from config dict with `model_name` key |
| `load_pretrained(model_name, checkpoint_path, num_classes=11, device='cpu', strict=True)` | Load pretrained model |
| `save_checkpoint(model, checkpoint_path, optimizer=None, epoch=None, metrics=None)` | Save checkpoint |
| `create_ensemble(model_names, checkpoint_paths=None, ensemble_type='voting', weights=None, ...)` | Create ensemble from model names |
| `list_available_models() -> List[str]` | List all registered model keys |
| `register_model(name, creation_fn)` | Register new model type |
| `get_model_info(model_name) -> dict` | Get model metadata |
| `print_model_summary(model, input_shape=(1, 1, 5000))` | Print architecture summary |
