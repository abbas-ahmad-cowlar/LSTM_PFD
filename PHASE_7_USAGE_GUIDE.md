# Phase 7: Explainable AI (XAI) - Usage Guide

This guide explains how to use explainability tools to interpret model predictions and build trust in the bearing fault diagnosis system. Understand **why** the model makes specific predictions using SHAP, LIME, Integrated Gradients, and more.

---

## 📋 What Was Implemented

Phase 7 implements comprehensive **Explainable AI (XAI)** tools for model interpretability:

- **SHAP (SHapley Additive exPlanations)**: Feature importance based on game theory
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local approximations
- **Integrated Gradients**: Attribution method for neural networks
- **Concept Activation Vectors (CAVs)**: Concept-based explanations
- **Partial Dependence Plots (PDP)**: Feature effect visualization
- **Anchors**: Rule-based explanations
- **Uncertainty Quantification**: Confidence estimation
- **Interactive Dashboard**: Streamlit-based visualization tool

**Goal**: Provide interpretable explanations for every prediction, enabling trust and debugging.

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install XAI libraries
pip install shap lime captum
pip install streamlit plotly  # For interactive dashboard
pip install scikit-learn matplotlib seaborn
```

### Step 2: Basic SHAP Explanations

```python
"""
explain_with_shap.py - Generate SHAP explanations for predictions
"""
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from explainability.shap_explainer import SHAPExplainer

# Load trained model
model = torch.load('checkpoints/phase6/best_model.pth')
model.eval()

# Load test data
import h5py
with h5py.File('data/processed/signals_cache.h5', 'r') as f:
    X_test = f['test/signals'][:]
    y_test = f['test/labels'][:]

# Select background data for SHAP (used as reference)
background_data = torch.FloatTensor(X_test[:100])  # 100 samples

# Create SHAP explainer
explainer = SHAPExplainer(
    model=model,
    background_data=background_data,
    algorithm='gradient'  # Options: 'gradient', 'deep', 'kernel'
)

# Explain a specific prediction
test_signal = torch.FloatTensor(X_test[0:1])  # First test sample
true_label = y_test[0]

print("Generating SHAP explanation...")
shap_values = explainer.explain(
    test_signal,
    target_class=None  # Explain predicted class (or specify a class)
)

# Visualize SHAP values
fig = explainer.plot_signal_attribution(
    signal=test_signal[0, 0].numpy(),
    shap_values=shap_values[0],
    true_label=true_label,
    save_path='results/phase7/shap_explanation.png'
)
plt.show()

print(f"\nExplanation saved to: results/phase7/shap_explanation.png")
print(f"Most important time regions: {explainer.get_top_time_regions(shap_values, top_k=5)}")
```

### Step 3: LIME Explanations

```python
"""
explain_with_lime.py - Generate LIME explanations
"""
from explainability.lime_explainer import LIMEExplainer

# Create LIME explainer
lime_explainer = LIMEExplainer(
    model=model,
    num_features=20,  # Number of features in explanation
    num_samples=5000  # Number of perturbed samples
)

# Explain a prediction
test_signal = X_test[0:1]
explanation = lime_explainer.explain(
    signal=test_signal,
    target_class=None,
    save_path='results/phase7/lime_explanation.html'
)

# Print explanation
print("\nLIME Explanation:")
print(f"Top contributing segments:")
for feature, weight in explanation.as_list()[:10]:
    print(f"  {feature}: {weight:.4f}")

# The explanation shows which time segments contribute most to the prediction
print(f"\nInteractive explanation saved to: results/phase7/lime_explanation.html")
```

---

## 🎯 Advanced Usage

### Option 1: Integrated Gradients

Attribute prediction to input features using gradients:

```python
"""
integrated_gradients.py - Gradient-based attribution
"""
from explainability.integrated_gradients import IntegratedGradientsExplainer
import torch

# Create IG explainer
ig_explainer = IntegratedGradientsExplainer(
    model=model,
    n_steps=50,  # Number of integration steps
    internal_batch_size=8
)

# Generate attribution
test_signal = torch.FloatTensor(X_test[0:1])
attribution = ig_explainer.explain(
    signal=test_signal,
    target_class=1,  # Explain prediction for class 1 (ball fault)
    baseline='zeros'  # Options: 'zeros', 'random', 'mean'
)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# Plot 1: Original signal
axes[0].plot(test_signal[0, 0].numpy())
axes[0].set_title('Original Signal')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Amplitude')

# Plot 2: Attribution
axes[1].plot(attribution[0, 0].numpy())
axes[1].set_title('Integrated Gradients Attribution')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('Attribution Score')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('results/phase7/integrated_gradients.png', dpi=300)
plt.show()

# Find most important regions
attribution_magnitude = torch.abs(attribution[0, 0])
top_k = 10
top_indices = torch.topk(attribution_magnitude, k=top_k).indices
print(f"\nMost important time samples: {top_indices.numpy()}")
```

### Option 2: Concept Activation Vectors (CAVs)

Test model's understanding of human-interpretable concepts:

```python
"""
concept_activation_vectors.py - Concept-based explanations
"""
from explainability.concept_activation_vectors import CAVAnalyzer
import numpy as np

# Define concepts (human-interpretable signal characteristics)
concepts = {
    'high_frequency': {
        'description': 'Signals with dominant high-frequency components',
        'signals': []  # Collect examples
    },
    'low_frequency': {
        'description': 'Signals with dominant low-frequency components',
        'signals': []
    },
    'high_amplitude': {
        'description': 'Signals with high amplitude',
        'signals': []
    },
    'periodic': {
        'description': 'Signals with strong periodic structure',
        'signals': []
    }
}

# Populate concept examples
from scipy import signal as scipy_signal

for i, sig in enumerate(X_test[:500]):
    # Compute signal characteristics
    freqs, psd = scipy_signal.welch(sig[0], fs=20480, nperseg=2048)
    dominant_freq = freqs[np.argmax(psd)]
    rms = np.sqrt(np.mean(sig**2))

    # Assign to concepts
    if dominant_freq > 1000:  # High frequency
        concepts['high_frequency']['signals'].append(sig)
    elif dominant_freq < 200:  # Low frequency
        concepts['low_frequency']['signals'].append(sig)

    if rms > 0.5:  # High amplitude
        concepts['high_amplitude']['signals'].append(sig)

    # Check periodicity via autocorrelation
    autocorr = np.correlate(sig[0], sig[0], mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if np.std(autocorr) > 100:  # Strong periodic structure
        concepts['periodic']['signals'].append(sig)

# Train CAVs
cav_analyzer = CAVAnalyzer(model=model)

print("Training Concept Activation Vectors...")
cavs = cav_analyzer.train_cavs(
    concepts=concepts,
    layer_name='layer4',  # Which layer to extract activations from
    num_random_counterexamples=200
)

# Test concept importance for each fault class
print("\nConcept Importance (TCAV Scores):")
print("="*70)

for class_idx, class_name in enumerate(['normal', 'ball_fault', 'inner_race',
                                         'outer_race', 'combined', 'imbalance',
                                         'misalignment', 'oil_whirl', 'cavitation',
                                         'looseness', 'oil_deficiency']):

    # Get test samples for this class
    class_samples = X_test[y_test == class_idx][:50]

    # Compute TCAV scores
    tcav_scores = cav_analyzer.tcav(
        test_samples=class_samples,
        cavs=cavs,
        target_class=class_idx
    )

    print(f"\n{class_name.upper()}:")
    for concept, score in tcav_scores.items():
        print(f"  {concept:20s}: {score:.3f}")

# Example interpretation:
# - If "ball_fault" has high score for "high_frequency" concept (>0.7),
#   it means high-frequency components are important for detecting ball faults
# - If "misalignment" has high score for "low_frequency" concept (>0.7),
#   it means low-frequency (2X harmonics) drive misalignment detection
```

### Option 3: Partial Dependence Plots

Visualize how predictions change with feature values:

```python
"""
partial_dependence.py - Feature effect visualization
"""
from explainability.partial_dependence import PartialDependencePlotter

# For classical ML models (Phase 1), PDP shows feature effects
from pipelines.classical_ml_pipeline import ClassicalMLPipeline

# Load trained classical ML model
pipeline = torch.load('checkpoints/phase1/best_model.pkl')
model_classic = pipeline.best_model
feature_names = pipeline.selected_feature_names

# Create PDP plotter
pdp_plotter = PartialDependencePlotter(
    model=model_classic,
    feature_names=feature_names
)

# Generate PDP for all features
print("Generating Partial Dependence Plots...")
fig = pdp_plotter.plot_all_features(
    X=X_test_features,  # Extracted features
    y=y_test,
    num_classes=11,
    save_dir='results/phase7/pdp'
)

print(f"PDPs saved to: results/phase7/pdp/")

# For deep learning models, visualize effect of signal modifications
from explainability.signal_perturbation import SignalPerturbationAnalyzer

perturb_analyzer = SignalPerturbationAnalyzer(model=model)

# How does prediction change with amplitude scaling?
amplitude_effects = perturb_analyzer.vary_amplitude(
    signal=X_test[0],
    scale_range=(0.5, 1.5),
    num_steps=20
)

# How does prediction change with added noise?
noise_effects = perturb_analyzer.vary_noise(
    signal=X_test[0],
    noise_std_range=(0.0, 0.3),
    num_steps=20
)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(amplitude_effects['scales'], amplitude_effects['probabilities'])
axes[0].set_xlabel('Amplitude Scale Factor')
axes[0].set_ylabel('Prediction Probability')
axes[0].set_title('Effect of Amplitude Scaling on Predictions')
axes[0].legend([f'Class {i}' for i in range(11)])

axes[1].plot(noise_effects['noise_stds'], noise_effects['probabilities'])
axes[1].set_xlabel('Noise Standard Deviation')
axes[1].set_ylabel('Prediction Probability')
axes[1].set_title('Effect of Noise on Predictions')

plt.tight_layout()
plt.savefig('results/phase7/perturbation_analysis.png', dpi=300)
plt.show()
```

### Option 4: Uncertainty Quantification

Estimate prediction confidence using multiple methods:

```python
"""
uncertainty_quantification.py - Measure prediction confidence
"""
from explainability.uncertainty_quantification import UncertaintyEstimator

# Create uncertainty estimator
uncertainty_estimator = UncertaintyEstimator(
    model=model,
    methods=['mc_dropout', 'ensemble', 'temperature_scaling']
)

# Estimate uncertainty for test samples
test_signal = torch.FloatTensor(X_test[0:1])

# Method 1: MC Dropout (enable dropout at inference, sample multiple times)
mc_uncertainty = uncertainty_estimator.mc_dropout_uncertainty(
    signal=test_signal,
    num_samples=100,
    dropout_rate=0.1
)

print(f"MC Dropout Uncertainty: {mc_uncertainty['uncertainty']:.4f}")
print(f"Predicted class: {mc_uncertainty['predicted_class']}")
print(f"Confidence: {mc_uncertainty['confidence']:.4f}")

# Method 2: Ensemble uncertainty (if you have multiple models)
ensemble_models = [
    torch.load('checkpoints/phase3/resnet18.pth'),
    torch.load('checkpoints/phase4/transformer.pth'),
    torch.load('checkpoints/phase6/pinn.pth')
]

ensemble_uncertainty = uncertainty_estimator.ensemble_uncertainty(
    signal=test_signal,
    models=ensemble_models
)

print(f"\nEnsemble Uncertainty: {ensemble_uncertainty['uncertainty']:.4f}")
print(f"Agreement: {ensemble_uncertainty['agreement']:.4f}")

# Method 3: Temperature scaling (calibrate confidence)
calibrated_probs = uncertainty_estimator.temperature_scaling(
    signal=test_signal,
    temperature=1.5  # Learned from validation set
)

print(f"\nCalibrated probabilities:")
for i, prob in enumerate(calibrated_probs[0]):
    print(f"  Class {i}: {prob:.4f}")

# Visualize uncertainty
fig = uncertainty_estimator.plot_uncertainty_distribution(
    signals=X_test[:100],
    labels=y_test[:100],
    save_path='results/phase7/uncertainty_distribution.png'
)
plt.show()
```

---

## 📊 Interactive Dashboard

Launch a comprehensive XAI dashboard:

```python
"""
xai_dashboard.py - Interactive explainability dashboard

Run with: streamlit run explainability/xai_dashboard.py
"""
import streamlit as st
import torch
import numpy as np
from explainability import (
    SHAPExplainer, LIMEExplainer,
    IntegratedGradientsExplainer,
    UncertaintyEstimator
)

st.set_page_config(page_title="Bearing Fault Diagnosis - XAI Dashboard", layout="wide")

st.title("🔍 Explainable AI Dashboard")
st.markdown("Understand model predictions with multiple explanation methods")

# Sidebar: Load model and data
st.sidebar.header("Configuration")
model_path = st.sidebar.selectbox(
    "Select Model",
    ["checkpoints/phase3/resnet18.pth",
     "checkpoints/phase4/transformer.pth",
     "checkpoints/phase6/pinn.pth"]
)

model = torch.load(model_path)
model.eval()

# Load test data
@st.cache_data
def load_data():
    with h5py.File('data/processed/signals_cache.h5', 'r') as f:
        return f['test/signals'][:], f['test/labels'][:]

X_test, y_test = load_data()

# Main panel: Select test sample
st.header("1️⃣ Select Test Sample")
sample_idx = st.slider("Sample Index", 0, len(X_test)-1, 0)
signal = X_test[sample_idx]
true_label = y_test[sample_idx]

# Display signal
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Signal Visualization")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal[0])
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Signal {sample_idx} (True Label: {true_label})")
    st.pyplot(fig)

with col2:
    st.subheader("Prediction")
    with torch.no_grad():
        output = model(torch.FloatTensor(signal).unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)[0].numpy()
        predicted_class = probabilities.argmax()

    st.metric("Predicted Class", predicted_class)
    st.metric("Confidence", f"{probabilities[predicted_class]:.2%}")
    st.metric("True Class", true_label)

    # Show all probabilities
    st.bar_chart(probabilities)

# Explanation methods
st.header("2️⃣ Explanation Methods")

tabs = st.tabs(["SHAP", "LIME", "Integrated Gradients", "Uncertainty", "Concepts"])

# Tab 1: SHAP
with tabs[0]:
    st.subheader("SHAP Explanation")
    if st.button("Generate SHAP Explanation"):
        with st.spinner("Computing SHAP values..."):
            background = torch.FloatTensor(X_test[:100])
            explainer = SHAPExplainer(model, background)
            shap_values = explainer.explain(
                torch.FloatTensor(signal).unsqueeze(0)
            )

            fig = explainer.plot_signal_attribution(signal[0], shap_values[0], true_label)
            st.pyplot(fig)

            st.info("**Interpretation**: Red regions increase prediction confidence, blue regions decrease it.")

# Tab 2: LIME
with tabs[1]:
    st.subheader("LIME Explanation")
    num_features = st.slider("Number of Features", 5, 50, 20)

    if st.button("Generate LIME Explanation"):
        with st.spinner("Computing LIME explanation..."):
            lime_explainer = LIMEExplainer(model, num_features=num_features)
            explanation = lime_explainer.explain(signal)

            # Display explanation
            st.write("**Top Contributing Segments:**")
            for feature, weight in explanation.as_list()[:10]:
                st.write(f"- {feature}: {weight:.4f}")

# Tab 3: Integrated Gradients
with tabs[2]:
    st.subheader("Integrated Gradients")
    target_class = st.selectbox("Target Class", range(11))

    if st.button("Generate IG Attribution"):
        with st.spinner("Computing attribution..."):
            ig_explainer = IntegratedGradientsExplainer(model)
            attribution = ig_explainer.explain(
                torch.FloatTensor(signal).unsqueeze(0),
                target_class=target_class
            )

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(attribution[0, 0].numpy())
            ax.set_xlabel("Sample")
            ax.set_ylabel("Attribution Score")
            ax.set_title(f"Attribution for Class {target_class}")
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            st.pyplot(fig)

# Tab 4: Uncertainty
with tabs[3]:
    st.subheader("Uncertainty Quantification")

    if st.button("Estimate Uncertainty"):
        with st.spinner("Computing uncertainty..."):
            uncertainty_estimator = UncertaintyEstimator(model)

            mc_uncertainty = uncertainty_estimator.mc_dropout_uncertainty(
                torch.FloatTensor(signal).unsqueeze(0),
                num_samples=100
            )

            st.metric("Uncertainty Score", f"{mc_uncertainty['uncertainty']:.4f}")
            st.metric("Confidence", f"{mc_uncertainty['confidence']:.2%}")

            if mc_uncertainty['uncertainty'] > 0.5:
                st.warning("⚠️ High uncertainty - model is not confident about this prediction")
            else:
                st.success("✓ Low uncertainty - model is confident")

# Tab 5: Concepts
with tabs[4]:
    st.subheader("Concept Analysis")
    st.write("Analyze which human-interpretable concepts drive predictions")

    concepts_to_test = st.multiselect(
        "Select Concepts",
        ['High Frequency', 'Low Frequency', 'High Amplitude', 'Periodic'],
        default=['High Frequency', 'Low Frequency']
    )

    if st.button("Analyze Concepts"):
        st.info("Concept analysis requires pre-trained CAVs. See PHASE_7_USAGE_GUIDE.md for details.")

st.markdown("---")
st.markdown("**Dashboard created with Phase 7: Explainable AI**")
```

---

## 🎛️ Best Practices

### 1. Choose the Right Explanation Method

| Method | Use Case | Speed | Accuracy |
|--------|----------|-------|----------|
| **SHAP** | Global feature importance, model debugging | Slow | High |
| **LIME** | Local explanations, simple interpretations | Medium | Medium |
| **Integrated Gradients** | Neural network attributions | Fast | High |
| **CAVs** | Concept-based explanations for domain experts | Medium | High |
| **Attention** | Transformer models, built-in interpretability | Fast | Medium |

### 2. Validate Explanations

```python
# Check if explanations are consistent
def validate_explanation_consistency(model, signal, num_trials=10):
    """Generate explanation multiple times, check consistency."""
    explainer = SHAPExplainer(model, background_data)

    explanations = []
    for _ in range(num_trials):
        shap_values = explainer.explain(signal)
        explanations.append(shap_values)

    # Compute variance across explanations
    explanations_stacked = np.stack(explanations)
    variance = np.var(explanations_stacked, axis=0).mean()

    print(f"Explanation variance: {variance:.6f}")
    if variance < 0.01:
        print("✓ Explanations are consistent")
    else:
        print("✗ Explanations are inconsistent - increase num_samples")

    return variance
```

### 3. Combine Multiple Methods

```python
def comprehensive_explanation(model, signal, true_label):
    """Generate explanations using multiple methods."""
    results = {}

    # SHAP
    shap_explainer = SHAPExplainer(model, background_data)
    results['shap'] = shap_explainer.explain(signal)

    # LIME
    lime_explainer = LIMEExplainer(model, num_features=20)
    results['lime'] = lime_explainer.explain(signal)

    # Integrated Gradients
    ig_explainer = IntegratedGradientsExplainer(model)
    results['ig'] = ig_explainer.explain(signal, target_class=true_label)

    # Compute agreement between methods
    # (Simplified - compare top-k important regions)
    shap_top_k = get_top_k_regions(results['shap'], k=10)
    lime_top_k = get_top_k_regions(results['lime'], k=10)
    ig_top_k = get_top_k_regions(results['ig'], k=10)

    agreement = len(set(shap_top_k) & set(lime_top_k) & set(ig_top_k)) / 10

    print(f"Agreement between methods: {agreement:.2%}")

    return results, agreement
```

---

## 🐛 Troubleshooting

### Issue 1: SHAP Takes Too Long

**Solution**: Use DeepExplainer instead of GradientExplainer

```python
# Slow
explainer = shap.GradientExplainer(model, background_data)

# Faster
explainer = shap.DeepExplainer(model, background_data)

# Fastest (but less accurate)
explainer = shap.KernelExplainer(model.predict, background_data[:50])
```

### Issue 2: LIME Explanations Unstable

**Solution**: Increase number of samples

```python
explainer = LIMEExplainer(
    model,
    num_features=20,
    num_samples=10000  # Increase from 5000 to 10000
)
```

### Issue 3: Out of Memory with IG

**Solution**: Reduce internal batch size

```python
ig_explainer = IntegratedGradientsExplainer(
    model,
    internal_batch_size=4  # Reduce from 8 to 4
)
```

---

## 📈 Expected Results

- **Explanation Generation Time**:
  - SHAP: 5-30 seconds per sample
  - LIME: 10-60 seconds per sample
  - Integrated Gradients: 1-5 seconds per sample
  - CAVs: 5-15 minutes (one-time training)

- **Explanation Quality**: High correlation (>0.7) between different methods for the same prediction

- **Dashboard**: Interactive, <2s response time per query

---

## 🚀 Next Steps

After Phase 7, you can:

1. **Phase 8**: Use XAI insights to build better ensemble models
2. **Phase 9**: Deploy models with explainability in production
3. **Research**: Publish findings on interpretable fault diagnosis
4. **Certification**: Use explanations for regulatory compliance

---

## 📚 Additional Resources

- **Paper**: ["A Unified Approach to Interpreting Model Predictions" (SHAP)](https://arxiv.org/abs/1705.07874)
- **Paper**: ["Why Should I Trust You?" (LIME)](https://arxiv.org/abs/1602.04938)
- **Paper**: ["Axiomatic Attribution for Deep Networks" (IG)](https://arxiv.org/abs/1703.01365)
- **Tutorial**: `notebooks/phase7_xai_tutorial.ipynb`
- **Code**: `explainability/` directory

---

**Phase 7 Complete!** You now have comprehensive explainability tools to understand and trust model predictions. Build confidence in your AI system! 🎉
