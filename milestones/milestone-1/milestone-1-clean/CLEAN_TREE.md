# Clean Repository Tree

This file lists the files and directories that constitute the clean, client-ready repository.

## Root Directory Files

- `README.md`: Main documentation (Updated with recent evaluation results)
- `requirements.txt`: Python dependencies
- `train_all_models.bat`: Batch script for training all models
- `.gitignore`: Git ignore patterns
- `CLEAN_TREE.md`: This file

## Directory Structure

### `data/`

Core data handling modules:

- `__init__.py`
- `matlab_importer.py`: Load .MAT files
- `cnn_dataset.py`: PyTorch Dataset classes
- `cnn_dataloader.py`: DataLoader creation
- `cnn_transforms.py`: Preprocessing transforms
- `augmentation.py`: Data augmentation
- `signal_augmentation.py`: Signal-specific augmentation
- `data_validator.py`: Data quality validation

Subdirectories:

- `data/raw/bearing_data/`: Directory for .mat files (11 fault type subdirectories)
- `data/processed/`: Directory for processed data

### `models/`

Model architecture definitions:

- `__init__.py`: Model factory
- `base_model.py`: Base model class

`models/cnn/`:

- `__init__.py`
- `cnn_1d.py`: Baseline CNN1D model
- `attention_cnn.py`: AttentionCNN & LightweightAttentionCNN
- `multi_scale_cnn.py`: MultiScaleCNN & DilatedMultiScaleCNN
- `conv_blocks.py`: Reusable convolutional blocks

### `training/`

Training utilities and logic:

- `__init__.py`
- `cnn_trainer.py`: Main training loop
- `cnn_optimizer.py`: Optimizer creation
- `cnn_losses.py`: Loss functions (cross-entropy, focal, label smoothing)
- `cnn_schedulers.py`: Learning rate schedulers
- `cnn_callbacks.py`: Training callbacks
- `metrics.py`: Evaluation metrics
- `losses.py`: Additional loss functions
- `optimizers.py`: Optimizer utilities
- `mixed_precision.py`: Mixed precision training support
- `advanced_augmentation.py`: Advanced augmentation techniques
- `callbacks.py`: General callbacks

### `evaluation/`

Model evaluation tools:

- `__init__.py`
- `cnn_evaluator.py`: Main model evaluator
- `evaluator.py`: Base evaluator class
- `benchmark.py`: Benchmarking utilities
- `confusion_analyzer.py`: Confusion matrix analysis
- `roc_analyzer.py`: ROC curve analysis
- `error_analysis.py`: Error analysis tools
- `robustness_tester.py`: Model robustness testing
- `ensemble_evaluator.py`: Ensemble model evaluation
- `ensemble_voting.py`: Ensemble voting strategies
- `attention_visualization.py`: Attention weight visualization
- `architecture_comparison.py`: Compare different architectures
- `time_vs_frequency_comparison.py`: Time vs frequency domain comparison
- `spectrogram_evaluator.py`: Spectrogram-based evaluation
- `physics_interpretability.py`: Physics-based interpretability
- `pinn_evaluator.py`: Physics-informed neural network evaluation

### `visualization/`

Visualization and plotting tools:

- `__init__.py`
- `plot_training.py`: Training curves
- `plot_confusion.py`: Confusion matrices
- `plot_signals.py`: Signal visualization
- `activation_maps.py`: CNN activation maps
- `plot_roc.py`: ROC curves
- `signal_plots.py`: Signal analysis plots

### `utils/`

Utility functions:

- `__init__.py`
- `constants.py`: Project constants (fault types, signal params)
- `device_manager.py`: GPU/CPU management
- `logging.py`: Logging utilities
- `reproducibility.py`: Random seed setting
- `file_io.py`: File I/O utilities
- `checkpoint_manager.py`: Model checkpoint management
- `early_stopping.py`: Early stopping logic
- `timer.py`: Timing utilities
- `visualization_utils.py`: Visualization helpers

### `scripts/`

Executable scripts:

- `train_cnn.py`: Main training script
- `evaluate_cnn.py`: Evaluation script
- `inference_cnn.py`: Inference script
- `generate_dataset.py`: Synthetic dataset generation
- `import_mat_dataset.py`: Data import utility

### `config/`

Configuration files (if present):

- `default_config.yaml`: Default settings
- `training_config.yaml`: Training configurations
- `model_config.yaml`: Model configurations

### `results/`

Output directories (create if not exist):

- `results/checkpoints_full/`: Trained model checkpoints (CNN1D, Attention, MultiScale)
- `results/final_eval/`: Recent evaluation results with confusion matrices and ROC curves
- `results/evaluation/`: Additional evaluation outputs
- `results/logs/`: Training logs

### `data_generation/`

Data generation utilities:

- `physics_based_generator.py`: Physics-based signal generation
- `fault_simulator.py`: Fault simulation

## Excluded Files (Do Not Copy to Client)

**Personal Development Scripts:**

- `check_models.py`
- `check_training_results.py`
- `inspect_checkpoint.py`
- `inspect_all_checkpoints.py`
- `quick_check.py`
- `example_usage.py`

**Temporary/Internal Files:**

- `attention_eval_error.txt`
- `attention_metrics.txt`
- `cnn1d_failure_analysis.txt`
- `cnn1d_metrics.txt`
- `checkpoint_inventory.txt`
- `checkpoint_keys.txt`

**Internal Documentation:**

- `DELIVERY_NOTES.md`
- `EVALUATION_REPORT.md`
- `QUICKSTART.md` (content merged into README)

**Build/Cache Directories:**

- `venv/` (client should create their own)
- `__pycache__/` (all instances)
- `.pytest_cache/`
- `.mypy_cache/`

## Notes for Client Delivery

1. **Virtual Environment**: The client should create their own `venv` following the installation instructions in README.md.
2. **Data Directory**: Include the structure but the client should generate or provide their own .MAT files using `scripts/generate_dataset.py`.
3. **Results Directory**: Can be empty or include sample evaluation results from `results/final_eval/` to demonstrate expected outputs.
4. **Checkpoints**: Optionally include the trained model checkpoints from `results/checkpoints_full/` (CNN1D and Attention models are production-ready).

## Quick Copy Command (PowerShell)

To copy only the clean files to a new directory:

```powershell
# Create clean directory
$cleanDir = "milestone-1-clean"
New-Item -ItemType Directory -Path $cleanDir -Force

# Copy essential files
Copy-Item README.md, requirements.txt, train_all_models.bat, .gitignore, CLEAN_TREE.md -Destination $cleanDir

# Copy directories (excluding __pycache__)
robocopy data "$cleanDir\data" /E /XD __pycache__ .pytest_cache venv
robocopy models "$cleanDir\models" /E /XD __pycache__
robocopy training "$cleanDir\training" /E /XD __pycache__
robocopy evaluation "$cleanDir\evaluation" /E /XD __pycache__
robocopy visualization "$cleanDir\visualization" /E /XD __pycache__
robocopy utils "$cleanDir\utils" /E /XD __pycache__
robocopy scripts "$cleanDir\scripts" /E /XD __pycache__
robocopy config "$cleanDir\config" /E /XD __pycache__
robocopy data_generation "$cleanDir\data_generation" /E /XD __pycache__

# Optionally copy trained models and evaluation results
robocopy results\checkpoints_full "$cleanDir\results\checkpoints_full" /E
robocopy results\final_eval "$cleanDir\results\final_eval" /E
```
