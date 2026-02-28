# Architecture Overview

> System architecture of the LSTM-PFD bearing fault diagnosis platform.

## System Context

LSTM-PFD ingests raw vibration signals (MATLAB `.mat` files), processes them through a multi-model ML/DL pipeline, and serves predictions via either a web dashboard or a REST API.

```mermaid
graph LR
    SENSOR["Vibration<br/>Sensors"] -->|".mat files"| SYSTEM["LSTM-PFD<br/>System"]
    SYSTEM -->|"Fault diagnosis"| OPERATOR["Maintenance<br/>Operator"]
    SYSTEM -->|"REST API"| EXTERNAL["External<br/>Systems"]
    OPERATOR -->|"Web UI"| SYSTEM
```

## Five-Domain Architecture

The project is organized into five independent development domains:

```mermaid
graph TB
    subgraph D1["Domain 1: Core ML Engine"]
        M["Models<br/>(IDB 1.1)"]
        T["Training<br/>(IDB 1.2)"]
        EV["Evaluation<br/>(IDB 1.3)"]
        F["Features<br/>(IDB 1.4)"]
        X["Explainability<br/>(IDB 1.5)"]
    end

    subgraph D2["Domain 2: Dashboard Platform"]
        UI["Frontend / UI<br/>(IDB 2.1)"]
        SVC["Services<br/>(IDB 2.2)"]
        CB["Callbacks<br/>(IDB 2.3)"]
        AT["Async Tasks<br/>(IDB 2.4)"]
    end

    subgraph D3["Domain 3: Data Engineering"]
        SG["Signal Generation<br/>(IDB 3.1)"]
        DL["Data Loading<br/>(IDB 3.2)"]
        ST["Storage Layer<br/>(IDB 3.3)"]
    end

    subgraph D4["Domain 4: Infrastructure"]
        DB["Database<br/>(IDB 4.1)"]
        DP["Deployment<br/>(IDB 4.2)"]
        TS["Testing<br/>(IDB 4.3)"]
        CF["Configuration<br/>(IDB 4.4)"]
    end

    subgraph D5["Domain 5: Research & Science"]
        RS["Research Scripts<br/>(IDB 5.1)"]
        VZ["Visualization<br/>(IDB 5.2)"]
    end

    D3 --> D1
    D1 --> D2
    D4 --> D1
    D4 --> D2
    D1 --> D5
```

## Data Flow

```mermaid
flowchart LR
    MAT[".mat Files"] --> HDF5["HDF5 Cache"]
    HDF5 --> FE["Feature<br/>Extraction"]
    HDF5 --> TF["Time-Frequency<br/>Transforms"]
    FE --> ML["Classical ML<br/>(SVM, RF, XGBoost)"]
    TF --> DL["Deep Learning<br/>(CNN, ResNet, Transformer)"]
    HDF5 --> PINN["PINN<br/>(Physics-Informed)"]
    ML & DL & PINN --> ENS["Ensemble"]
    ENS --> QUANT["Quantization<br/>(INT8/FP16)"]
    ENS --> XAI["XAI<br/>(SHAP, LIME, IG)"]
    QUANT --> ONNX["ONNX Export"]
    ONNX --> API["REST API"]
    XAI --> DASH["Dashboard"]
```

## Technology Stack

| Layer                 | Technologies                              |
| --------------------- | ----------------------------------------- |
| **ML/DL Framework**   | PyTorch 2.0+, scikit-learn, XGBoost       |
| **Signal Processing** | SciPy, PyWavelets                         |
| **XAI**               | SHAP, LIME, Captum (Integrated Gradients) |
| **Dashboard**         | Dash/Plotly, Flask                        |
| **Database**          | PostgreSQL, SQLAlchemy                    |
| **Task Queue**        | Celery, Redis                             |
| **Data Storage**      | HDF5 (h5py), MATLAB (.mat)                |
| **Deployment**        | Docker, Kubernetes, ONNX Runtime          |
| **Testing**           | pytest, coverage                          |
| **Optimization**      | Optuna (HPO)                              |
| **Visualization**     | Matplotlib, Seaborn, Plotly, TensorBoard  |

## Key Design Decisions

### 1. Monorepo with `packages/` Layout

All Python modules live under `packages/core/` and `packages/dashboard/` to enforce clear domain boundaries while sharing a single virtual environment and test suite.

### 2. HDF5 Caching Layer

Raw MATLAB `.mat` files are converted to HDF5 once, then reused. This avoids repeated deserialization and enables efficient random access for large datasets.

### 3. Model Factory Pattern

All model architectures implement a common interface and are instantiated through a factory pattern (`packages/core/models/`), enabling uniform training, evaluation, and ensemble composition.

### 4. Physics-Informed Constraints

PINN models incorporate bearing dynamics equations (energy conservation, momentum conservation) as additional loss terms, regularizing the network with domain knowledge.

### 5. Independent Development Blocks (IDB)

The project uses an IDB decomposition (18 blocks across 5 domains + integration layer) to enable parallel development and isolated documentation ownership.

## Related Documentation

- [Project README](../README.md)
- [IDB Decomposition Reference](INDEPENDENT_DEVELOPMENT_BLOCKS.md)
- [Documentation Hub](index.md)
