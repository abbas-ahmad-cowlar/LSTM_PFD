# Getting Started

Welcome to LSTM PFD! This section will help you install, configure, and run your first bearing fault diagnosis experiment.

## What You'll Learn

1. **Installation** - Set up Python, PyTorch, and all dependencies
2. **Quick Start** - Run a pre-trained model on sample data
3. **Configuration** - Customize your environment and settings
4. **First Experiment** - Train your own model from scratch

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** (3.10 recommended)
- **CUDA 11.8+** (for GPU acceleration, optional)
- **8GB+ RAM** (16GB recommended)
- **GPU with 6GB+ VRAM** (optional but recommended)

## Quick Navigation

<div class="grid cards" markdown>

- :material-download:{ .lg .middle } **Installation**

  ***

  Complete installation guide for Windows, Linux, and macOS.

  [:octicons-arrow-right-24: Install](installation.md)

- :material-play:{ .lg .middle } **Quick Start**

  ***

  Get up and running in 5 minutes.

  [:octicons-arrow-right-24: Quick Start](quickstart.md)

- :material-cog:{ .lg .middle } **Configuration**

  ***

  Environment variables and settings.

  [:octicons-arrow-right-24: Configure](configuration.md)

- :material-flask:{ .lg .middle } **First Experiment**

  ***

  Train your first fault detection model.

  [:octicons-arrow-right-24: Train](first-experiment.md)

</div>

## Two Paths to Success

Choose the path that fits your workflow:

=== "No-Code Dashboard"

    Perfect for researchers and engineers who prefer visual interfaces.

    ```bash
    # Install and launch
    pip install -r requirements.txt
    cd packages/dashboard
    python app.py
    ```

    Open http://localhost:8050 and follow the wizard!

=== "Command Line"

    For power users who prefer scripts and automation.

    ```bash
    # Generate data
    python scripts/run_phase0.py

    # Train model
    python scripts/train_cnn.py --model resnet18 --epochs 100

    # Evaluate
    python scripts/evaluate_model.py --checkpoint checkpoints/best.pth
    ```

## Expected Outcomes

After completing the Getting Started guide, you will:

- ✅ Have a working LSTM PFD installation
- ✅ Understand the project structure
- ✅ Trained a model with 95%+ accuracy
- ✅ Generated explainability visualizations
