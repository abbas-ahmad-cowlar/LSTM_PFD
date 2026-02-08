# Phase 1: Classical ML Pipeline - Usage Guide

This guide explains how to merge the Phase 1 implementation into main and how to run the classical ML pipeline.

---

## 📋 What Was Implemented

Phase 1 adds **31 new files** implementing the complete classical ML pipeline:

- **Feature Engineering** (12 files): Extract 36 features from vibration signals
- **Classical ML Models** (7 files): SVM, Random Forest, Neural Network, Gradient Boosting
- **Hyperparameter Optimization** (3 files): Bayesian, Grid, and Random search
- **Pipeline Integration** (5 files): End-to-end pipeline orchestration
- **Visualization** (4 files): Feature analysis and performance plots

---

## 🔀 How to Merge Phase 1 into Main

### Option 1: Merge via GitHub (Recommended)

```bash
# 1. Push to your repository (already done!)
# The branch 'claude/fix-response-clarity-013f6J8Gj5K4TeYmzyjLwzZx' is already pushed

# 2. Create a Pull Request on GitHub:
#    - Go to your repository on GitHub
#    - Click "Pull requests" → "New pull request"
#    - Base: main
#    - Compare: claude/fix-response-clarity-013f6J8Gj5K4TeYmzyjLwzZx
#    - Click "Create pull request"
#    - Review changes and merge

# 3. After merging on GitHub, update your local main:
git checkout main
git pull origin main
```

### Option 2: Direct Local Merge

```bash
# 1. Checkout main branch
git checkout main

# 2. Merge the Phase 1 branch
git merge claude/fix-response-clarity-013f6J8Gj5K4TeYmzyjLwzZx

# 3. Push to remote main
git push origin main

# 4. (Optional) Delete the feature branch
git branch -d claude/fix-response-clarity-013f6J8Gj5K4TeYmzyjLwzZx
git push origin --delete claude/fix-response-clarity-013f6J8Gj5K4TeYmzyjLwzZx
```

---

## 🚀 How to Run Phase 1

### Step 1: Install Dependencies

```bash
# Install required Python packages
pip install numpy scipy scikit-learn optuna matplotlib seaborn pywt h5py joblib
```

### Step 2: Generate Synthetic Data (Using Phase 0)

```python
"""
generate_data.py - Generate synthetic bearing fault signals
"""
import numpy as np
from data.signal_generator import SignalGenerator
from config.data_config import DataConfig

# Create config
config = DataConfig(
    num_signals_per_fault=130,  # 130 signals × 11 classes = 1,430 total
    rng_seed=42
)

# Generate dataset
generator = SignalGenerator(config)
dataset = generator.generate_dataset()

# Access signals and labels
signals = dataset['signals']  # Shape: (1430, 102400)
labels = dataset['labels']    # Shape: (1430,)
metadata = dataset['metadata']

print(f"Generated {len(signals)} signals")
print(f"Signal shape: {signals.shape}")
print(f"Unique classes: {np.unique(labels)}")
```

### Step 3: Run Classical ML Pipeline

```python
"""
run_classical_ml.py - Run complete classical ML pipeline
"""
import numpy as np
from pathlib import Path
from pipelines.classical_ml_pipeline import ClassicalMLPipeline

# Load or generate signals (from Step 2)
# signals: (n_samples, signal_length)
# labels: (n_samples,)

# Initialize pipeline
pipeline = ClassicalMLPipeline(random_state=42)

# Run complete pipeline
results = pipeline.run(
    signals=signals,
    labels=labels,
    fs=20480,  # Sampling frequency
    optimize_hyperparams=True,  # Use Bayesian optimization
    n_trials=50,  # Number of optimization trials
    save_dir=Path('results/classical_ml')  # Save results here
)

# Print results
print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Best Model: {results['best_model']}")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
print(f"Selected Features: {len(results['selected_features'])}")
print(f"Elapsed Time: {results['elapsed_time_seconds']:.1f}s")
```

### Step 4: Make Predictions on New Data

```python
"""
predict.py - Predict fault classes for new signals
"""
import numpy as np
from pipelines.classical_ml_pipeline import ClassicalMLPipeline

# Load trained pipeline (after running Step 3)
pipeline = ClassicalMLPipeline()
# ... run pipeline first or load from saved state ...

# New signals to classify
new_signals = np.random.randn(10, 102400)  # 10 new signals

# Predict
predictions = pipeline.predict(new_signals)

print(f"Predictions: {predictions}")
```

---

## 📊 Visualization Examples

### Visualize Features

```python
"""
visualize_features.py - Visualize extracted features
"""
from features.feature_extractor import FeatureExtractor
from visualization.feature_visualization import FeatureVisualizer
import numpy as np

# Extract features
extractor = FeatureExtractor(fs=20480)
features = extractor.extract_batch(signals)
feature_names = extractor.get_feature_names()

# Visualize
visualizer = FeatureVisualizer()

# Correlation matrix (Figure 4)
visualizer.plot_correlation_matrix(
    features, feature_names,
    save_path='figures/feature_correlation.png'
)

# Feature distributions by class (Figure 5)
visualizer.plot_feature_distributions(
    features, labels, feature_names,
    save_path='figures/feature_distributions.png'
)

# t-SNE clustering (Figure 6)
visualizer.plot_tsne_clusters(
    features, labels,
    save_path='figures/tsne_clusters.png'
)
```

### Visualize Performance

```python
"""
visualize_performance.py - Visualize model performance
"""
from visualization.performance_plots import PerformancePlotter
import numpy as np

plotter = PerformancePlotter()

# Confusion matrix (Figure 8)
plotter.plot_confusion_matrix(
    cm=results['confusion_matrix'],
    class_names=['Healthy', 'Misalign', 'Imbalance', 'Clearance',
                'Lube', 'Cavitation', 'Wear', 'OilWhirl',
                'Misalign+Imb', 'Wear+Lube', 'Cavit+Clearance'],
    normalize=True,
    save_path='figures/confusion_matrix.png'
)

# Model comparison (Figure 7)
plotter.plot_model_comparison(
    results['model_comparison'],
    save_path='figures/model_comparison.png'
)
```

---

## 🧪 Testing the Implementation

### Quick Sanity Test

```python
"""
test_phase1.py - Quick sanity test
"""
import numpy as np
from features.feature_extractor import FeatureExtractor
from models.classical import RandomForestClassifier

# Generate dummy data
signals = np.random.randn(100, 102400)  # 100 signals
labels = np.random.randint(0, 11, 100)   # 11 classes

# Extract features
extractor = FeatureExtractor(fs=20480)
features = extractor.extract_batch(signals)
print(f"✓ Feature extraction: {features.shape}")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.train(X_train, y_train)
accuracy = rf.score(X_test, y_test)
print(f"✓ Random Forest trained: {accuracy:.4f} accuracy")

print("\n✓ Phase 1 implementation is working!")
```

---

## 📁 Directory Structure After Phase 1

```
LSTM_PFD/
├── features/              # NEW: Feature extraction
│   ├── __init__.py
│   ├── feature_extractor.py      # Main orchestrator (36 features)
│   ├── time_domain.py             # 7 time-domain features
│   ├── frequency_domain.py        # 12 frequency features
│   ├── envelope_analysis.py       # 4 envelope features
│   ├── wavelet_features.py        # 7 wavelet features
│   ├── bispectrum.py              # 6 bispectrum features
│   ├── feature_selector.py        # MRMR selection (36→15)
│   ├── feature_normalization.py   # Z-score normalization
│   ├── feature_validator.py       # Validation utilities
│   ├── feature_importance.py      # Importance analysis
│   └── advanced_features.py       # 16 advanced features
│
├── models/
│   └── classical/         # NEW: Classical ML models
│       ├── __init__.py
│       ├── svm_classifier.py      # SVM with ECOC
│       ├── random_forest.py       # Random Forest
│       ├── neural_network.py      # MLP (36→20→10→11)
│       ├── gradient_boosting.py   # Gradient Boosting
│       ├── stacked_ensemble.py    # Stacking
│       └── model_selector.py      # Model selection
│
├── training/              # UPDATED: Add hyperparam optimization
│   ├── bayesian_optimizer.py      # NEW: Bayesian optimization
│   ├── grid_search.py             # NEW: Grid search
│   └── random_search.py           # NEW: Random search
│
├── pipelines/             # NEW: Pipeline orchestration
│   ├── __init__.py
│   ├── classical_ml_pipeline.py   # Main end-to-end pipeline
│   ├── feature_pipeline.py        # Feature extraction pipeline
│   ├── matlab_compat.py           # MATLAB compatibility
│   └── pipeline_validator.py      # Validation utilities
│
├── visualization/         # NEW: Plotting utilities
│   ├── __init__.py
│   ├── feature_visualization.py   # Feature plots (Figs 4,5,6)
│   ├── performance_plots.py       # Performance plots (Figs 7,8,9)
│   └── signal_plots.py            # Signal plots (Figs 2,3)
│
├── config/                # FROM PHASE 0
├── data/                  # FROM PHASE 0
├── evaluation/            # FROM PHASE 0
├── experiments/           # FROM PHASE 0
├── tests/                 # FROM PHASE 0
├── utils/                 # FROM PHASE 0
└── PHASE_1_USAGE_GUIDE.md # This guide
```

---

## 🎯 Expected Performance

Based on phase_1.md specifications:

- **Random Forest**: ~95% validation accuracy (best performer)
- **SVM**: ~92-94% validation accuracy
- **Neural Network**: ~90-93% validation accuracy
- **Gradient Boosting**: ~92-94% validation accuracy

**Note**: Actual performance depends on:
1. Quality of synthetic signals from Phase 0
2. Number of signals per class
3. Hyperparameter optimization settings
4. Random seed

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Solution: Install missing dependencies
pip install <missing_package>

# Or install all at once:
pip install numpy scipy scikit-learn optuna matplotlib seaborn pywt h5py joblib
```

### Issue: Low Accuracy (<80%)

Possible causes:
1. **Too few samples**: Generate more signals (increase `num_signals_per_fault`)
2. **Poor hyperparameters**: Increase `n_trials` in Bayesian optimization
3. **Data quality**: Check Phase 0 signal generation

### Issue: Out of Memory

```python
# Solution: Extract features in batches
from features.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(fs=20480)

# Process in batches
batch_size = 100
features_list = []
for i in range(0, len(signals), batch_size):
    batch = signals[i:i+batch_size]
    features_batch = extractor.extract_batch(batch)
    features_list.append(features_batch)

features = np.vstack(features_list)
```

---

## 📚 Next Steps

After verifying Phase 1 works:

1. **Phase 2**: 1D CNN implementation
2. **Phase 3**: Advanced CNNs (ResNet, EfficientNet)
3. **Phase 4**: Transformer models
4. **Phase 5**: Time-frequency (spectrograms + 2D CNNs)

---

## 💡 Tips

1. **Save extracted features**: Feature extraction is slow (~3 min for 1,430 signals). Use `FeaturePipeline` to cache features.

2. **Start with small datasets**: Test with 100-200 signals first, then scale up.

3. **Use Bayesian optimization**: Much faster than grid search (50 trials vs 1000s).

4. **Check feature importances**: After training, use `feature_importance.py` to see which features matter most.

5. **Monitor with MLflow**: The pipeline integrates with MLflow for experiment tracking.

---

## 📞 Support

If you encounter issues:
1. Check that Phase 0 is working correctly
2. Verify all dependencies are installed
3. Check the technical report (Sections 8-9) for algorithm details
4. Review test files in `tests/` for examples

---

**Happy Fault Diagnosing! 🔧⚙️**
