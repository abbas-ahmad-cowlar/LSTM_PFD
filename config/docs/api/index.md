# API Reference

Complete API documentation for programmatic access to LSTM PFD.

## Core Modules

<div class="grid cards" markdown>

- :material-cube:{ .lg .middle } **Models**

  ***

  Model architectures from Classical ML to PINN.

  [:octicons-arrow-right-24: Models API](core/models.md)

- :material-school:{ .lg .middle } **Training**

  ***

  Training loops, optimizers, and callbacks.

  [:octicons-arrow-right-24: Training API](core/training.md)

- :material-database:{ .lg .middle } **Data**

  ***

  Dataset classes, loaders, and augmentation.

  [:octicons-arrow-right-24: Data API](core/data.md)

- :material-chart-scatter-plot:{ .lg .middle } **Explainability**

  ***

  SHAP, LIME, and attribution methods.

  [:octicons-arrow-right-24: XAI API](core/explainability.md)

</div>

## Dashboard API

- :material-api:{ .lg .middle } **REST API**

  ***

  HTTP endpoints for inference and management.

  [:octicons-arrow-right-24: REST API](dashboard/rest-api.md)

---

## Quick Examples

### Model Creation

```python
from packages.core.models import create_model

# Create any registered model
model = create_model('resnet34', num_classes=11, dropout=0.3)

# PINN with physics constraints
from packages.core.models.pinn import HybridPINN
pinn = HybridPINN(base_model=model, physics_weight=0.1)
```

### Training

```python
from packages.core.training import Trainer

trainer = Trainer(
    model=model,
    optimizer='adamw',
    lr=1e-3,
    epochs=100
)
history = trainer.fit(train_loader, val_loader)
```

### Inference

```python
import requests

response = requests.post(
    'http://localhost:8050/api/predict',
    json={'signal': signal.tolist()},
    headers={'Authorization': 'Bearer <token>'}
)
prediction = response.json()
```

---

## Auto-Generated Documentation

!!! info "API Generation"
Detailed API documentation is auto-generated using pdoc.

    Build locally with:
    ```bash
    pdoc packages/core -o docs/api/generated
    ```
