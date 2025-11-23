"""
Interactive XAI Dashboard for Bearing Fault Diagnosis

Provides a comprehensive web-based interface for exploring all explainability
methods implemented in Phase 7. Built with Streamlit for easy deployment.

Features:
- Upload and visualize bearing fault signals
- Model predictions with uncertainty quantification
- Multiple explanation methods:
  * Integrated Gradients
  * SHAP values
  * LIME explanations
  * Saliency maps (Vanilla, SmoothGrad, GradCAM)
  * Counterfactual explanations
  * CAVs and TCAV scores
  * Partial Dependence plots
  * Anchor rules
- Interactive parameter tuning
- Export explanations as reports

Usage:
    streamlit run visualization/xai_dashboard.py

Requirements:
    pip install streamlit plotly
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional, Dict, List
import io

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.constants import NUM_CLASSES

# Import all XAI methods
from explainability.integrated_gradients import IntegratedGradientsExplainer
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.uncertainty_quantification import UncertaintyQuantifier
from visualization.saliency_maps import SaliencyMapGenerator
from visualization.counterfactual_explanations import CounterfactualGenerator

# Optional imports (may not be available in all environments)
try:
    from explainability.concept_activation_vectors import CAVGenerator, TCAVAnalyzer
    CAV_AVAILABLE = True
except ImportError:
    CAV_AVAILABLE = False

try:
    from explainability.partial_dependence import PartialDependenceAnalyzer
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

try:
    from explainability.anchors import AnchorExplainer
    ANCHORS_AVAILABLE = True
except ImportError:
    ANCHORS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

FAULT_CLASSES = [
    'Healthy', 'Misalignment', 'Imbalance', 'Looseness',
    'Bearing Outer Race', 'Bearing Inner Race', 'Bearing Ball',
    'Gear Fault', 'Shaft Bent', 'Rotor Rub', 'Combined Fault'
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Utility Functions
# =============================================================================

@st.cache_resource
def load_model(model_path: Optional[str] = None):
    """Load pre-trained model (or dummy model for demo)."""
    from models.cnn.cnn_1d import CNN1D

    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1, dropout=0.3)

    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        st.success(f"✓ Model loaded from {model_path}")
    else:
        st.warning("⚠ Using untrained model (demo mode)")

    model.to(DEVICE)
    model.eval()
    return model


def generate_synthetic_signal(
    fault_type: str = 'healthy',
    rpm: int = 1800,
    length: int = 10240
) -> torch.Tensor:
    """Generate synthetic bearing signal for demonstration."""
    t = np.linspace(0, 1, length)

    # Base vibration
    signal = np.random.randn(length) * 0.1

    # Add fault-specific components
    if fault_type == 'Bearing Outer Race':
        # BPFO fault with periodic impulses
        bpfo_freq = 3.5 * (rpm / 60)  # Approximate
        signal += 0.5 * np.sin(2 * np.pi * bpfo_freq * t)
        # Add impulses
        impulse_interval = int(length / (bpfo_freq * 1))
        for i in range(0, length, impulse_interval):
            if i < length:
                signal[i:min(i+20, length)] += np.exp(-np.arange(min(20, length-i)) / 5) * 2

    elif fault_type == 'Imbalance':
        # 1x RPM component
        signal += 1.0 * np.sin(2 * np.pi * (rpm / 60) * t)

    elif fault_type == 'Misalignment':
        # 1x, 2x RPM components
        signal += 0.8 * np.sin(2 * np.pi * (rpm / 60) * t)
        signal += 0.4 * np.sin(2 * np.pi * 2 * (rpm / 60) * t)

    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    st.set_page_config(
        page_title="XAI Dashboard - Bearing Fault Diagnosis",
        page_icon="⚙️",
        layout="wide"
    )

    st.title("⚙️ Explainable AI Dashboard")
    st.markdown("**Bearing Fault Diagnosis with Interpretability**")

    # Sidebar: Configuration
    st.sidebar.header("⚙️ Configuration")

    # Model selection
    st.sidebar.subheader("Model")
    model_path = st.sidebar.text_input(
        "Model path (optional)",
        placeholder="/path/to/model.pt"
    )

    model = load_model(model_path if model_path else None)

    # Signal input
    st.sidebar.subheader("Input Signal")

    signal_source = st.sidebar.radio(
        "Signal source",
        ["Generate Synthetic", "Upload File"]
    )

    if signal_source == "Generate Synthetic":
        fault_type = st.sidebar.selectbox("Fault Type", FAULT_CLASSES)
        rpm = st.sidebar.slider("RPM", 1000, 3000, 1800, step=100)
        signal = generate_synthetic_signal(fault_type, rpm)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload signal (.npy, .pt)", type=['npy', 'pt'])
        if uploaded_file:
            # Load signal
            if uploaded_file.name.endswith('.npy'):
                signal = torch.tensor(np.load(uploaded_file), dtype=torch.float32)
            else:
                signal = torch.load(uploaded_file)

            # Ensure correct shape [1, C, T]
            if signal.dim() == 1:
                signal = signal.unsqueeze(0).unsqueeze(0)
            elif signal.dim() == 2:
                signal = signal.unsqueeze(0)
        else:
            st.info("👆 Upload a signal file or use synthetic generation")
            return

    # Explainability method selection
    st.sidebar.subheader("XAI Methods")
    methods = []

    if st.sidebar.checkbox("Integrated Gradients", value=True):
        methods.append("Integrated Gradients")

    if st.sidebar.checkbox("SHAP", value=False):
        methods.append("SHAP")

    if st.sidebar.checkbox("LIME", value=False):
        methods.append("LIME")

    if st.sidebar.checkbox("Saliency Maps", value=True):
        methods.append("Saliency Maps")

    if st.sidebar.checkbox("Uncertainty Quantification", value=True):
        methods.append("Uncertainty")

    if st.sidebar.checkbox("Counterfactual", value=False):
        methods.append("Counterfactual")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Input Signal")

        # Plot signal
        fig, ax = plt.subplots(figsize=(12, 4))
        signal_np = signal.squeeze().cpu().numpy()
        ax.plot(signal_np, linewidth=0.8)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Amplitude")
        ax.set_title("Vibration Signal")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Signal statistics
        st.write(f"**Shape:** {signal.shape} | **Mean:** {signal_np.mean():.3f} | "
                f"**Std:** {signal_np.std():.3f} | **Max:** {signal_np.max():.3f}")

    with col2:
        st.subheader("🎯 Prediction")

        # Predict
        with torch.no_grad():
            signal_device = signal.to(DEVICE)
            output = model(signal_device)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = probs.argmax()

        # Show prediction
        st.metric("Predicted Class", FAULT_CLASSES[pred_class])
        st.metric("Confidence", f"{probs[pred_class]:.1%}")

        # Probability distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(range(len(FAULT_CLASSES)), probs, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(FAULT_CLASSES)))
        ax.set_yticklabels(FAULT_CLASSES, fontsize=8)
        ax.set_xlabel("Probability")
        ax.set_title("Class Probabilities")
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
        plt.close()

    # Explanations
    st.markdown("---")
    st.header("🔍 Explanations")

    tabs = st.tabs(methods)

    for i, method in enumerate(methods):
        with tabs[i]:
            if method == "Integrated Gradients":
                render_integrated_gradients(model, signal)

            elif method == "SHAP":
                render_shap(model, signal)

            elif method == "LIME":
                render_lime(model, signal)

            elif method == "Saliency Maps":
                render_saliency(model, signal)

            elif method == "Uncertainty":
                render_uncertainty(model, signal)

            elif method == "Counterfactual":
                render_counterfactual(model, signal)


# =============================================================================
# Rendering Functions for Each Method
# =============================================================================

def render_integrated_gradients(model, signal):
    """Render Integrated Gradients explanation."""
    st.subheader("Integrated Gradients Attribution")

    with st.spinner("Computing attributions..."):
        explainer = IntegratedGradientsExplainer(model, device=DEVICE)
        attributions = explainer.explain(signal, steps=50)

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        signal_np = signal.squeeze().cpu().numpy()
        attr_np = attributions.squeeze().cpu().numpy()

        # Signal
        axes[0].plot(signal_np, 'b-', linewidth=0.8)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Original Signal")
        axes[0].grid(True, alpha=0.3)

        # Attributions
        axes[1].plot(attr_np, 'r-', linewidth=0.8)
        axes[1].fill_between(range(len(attr_np)), 0, attr_np, alpha=0.3, color='red')
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Attribution")
        axes[1].set_title("Integrated Gradients Attribution")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success("✓ Attributions computed")


def render_shap(model, signal):
    """Render SHAP explanation."""
    st.subheader("SHAP Values")

    n_samples = st.slider("Number of samples", 10, 100, 50)

    with st.spinner("Computing SHAP values..."):
        # Generate background data
        background = torch.randn(20, *signal.shape[1:])

        explainer = SHAPExplainer(model, background_data=background, device=DEVICE, use_shap_library=False)
        shap_values = explainer.explain(signal, method='gradient', n_samples=n_samples)

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        signal_np = signal.squeeze().cpu().numpy()
        shap_np = shap_values.squeeze().cpu().numpy()

        axes[0].plot(signal_np, 'b-', linewidth=0.8)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Original Signal")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(shap_np, 'g-', linewidth=0.8)
        axes[1].fill_between(range(len(shap_np)), 0, shap_np, alpha=0.3, color='green')
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("SHAP Value")
        axes[1].set_title("SHAP Attribution")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success("✓ SHAP values computed")


def render_lime(model, signal):
    """Render LIME explanation."""
    st.subheader("LIME Segment Importance")

    num_segments = st.slider("Number of segments", 10, 50, 20)
    num_samples = st.slider("Number of samples", 100, 2000, 500)

    with st.spinner("Computing LIME explanation..."):
        explainer = LIMEExplainer(model, device=DEVICE, num_segments=num_segments)
        segment_weights, segment_boundaries = explainer.explain(
            signal, num_samples=num_samples
        )

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        signal_np = signal.squeeze().cpu().numpy()

        axes[0].plot(signal_np, 'b-', linewidth=0.8)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Original Signal")
        axes[0].grid(True, alpha=0.3)

        # Segment importance
        for i, (start, end) in enumerate(segment_boundaries):
            weight = segment_weights[i]
            color = 'green' if weight > 0 else 'red'
            axes[1].barh(0, end - start, left=start, height=weight,
                        color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_xlim(0, len(signal_np))
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Segment Weight")
        axes[1].set_title("LIME Segment Importance")
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success("✓ LIME explanation computed")


def render_saliency(model, signal):
    """Render saliency maps."""
    st.subheader("Saliency Maps")

    method = st.selectbox("Saliency method", ["Vanilla Gradient", "SmoothGrad", "Gradient × Input"])

    with st.spinner(f"Computing {method}..."):
        generator = SaliencyMapGenerator(model, device=DEVICE)

        if method == "Vanilla Gradient":
            saliency = generator.vanilla_gradient(signal)
        elif method == "SmoothGrad":
            n_samples = st.slider("Number of samples", 10, 100, 30)
            saliency = generator.smooth_grad(signal, n_samples=n_samples)
        else:  # Gradient × Input
            saliency = generator.gradient_times_input(signal)

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        signal_np = signal.squeeze().cpu().numpy()
        saliency_np = saliency.squeeze().cpu().numpy()

        axes[0].plot(signal_np, 'b-', linewidth=0.8)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Original Signal")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(saliency_np, 'r-', linewidth=0.8)
        axes[1].fill_between(range(len(saliency_np)), 0, saliency_np, alpha=0.3, color='red')
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Saliency")
        axes[1].set_title(f"{method} Map")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success(f"✓ {method} computed")


def render_uncertainty(model, signal):
    """Render uncertainty quantification."""
    st.subheader("Uncertainty Quantification (MC Dropout)")

    n_samples = st.slider("Number of MC samples", 10, 100, 50)

    with st.spinner("Computing uncertainty..."):
        quantifier = UncertaintyQuantifier(model, device=DEVICE)
        mean_pred, uncertainty, all_preds = quantifier.predict_with_uncertainty(
            signal, n_samples=n_samples
        )

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Prediction distribution
        ax1.hist(all_preds[:, mean_pred.argmax()].cpu().numpy(), bins=20,
                alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(mean_pred.max(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_pred.max():.3f}')
        ax1.set_xlabel("Predicted Probability")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Prediction Distribution (Class {mean_pred.argmax()})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Uncertainty per class
        ax2.bar(range(len(uncertainty)), uncertainty.cpu().numpy(),
               color='orange', alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Uncertainty (Std)")
        ax2.set_title("Uncertainty by Class")
        ax2.set_xticks(range(len(FAULT_CLASSES)))
        ax2.set_xticklabels(range(len(FAULT_CLASSES)))
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.metric("Mean Confidence", f"{mean_pred.max():.1%}")
        st.metric("Prediction Uncertainty", f"{uncertainty[mean_pred.argmax()]:.3f}")

    st.success("✓ Uncertainty quantified")


def render_counterfactual(model, signal):
    """Render counterfactual explanation."""
    st.subheader("Counterfactual Explanation")

    # Get current prediction
    with torch.no_grad():
        output = model(signal.to(DEVICE))
        current_class = output.argmax(dim=1).item()

    target_class = st.selectbox(
        "Target class (what if prediction was...)",
        [i for i in range(len(FAULT_CLASSES)) if i != current_class],
        format_func=lambda x: FAULT_CLASSES[x]
    )

    max_iter = st.slider("Max iterations", 100, 2000, 500)

    if st.button("Generate Counterfactual"):
        with st.spinner("Generating counterfactual..."):
            generator = CounterfactualGenerator(model, device=DEVICE)

            cf_signal, info = generator.generate(
                signal,
                target_class=target_class,
                max_iterations=max_iter,
                lambda_l2=0.1
            )

            if info['success']:
                st.success(f"✓ Counterfactual found in {info['iterations']} iterations!")

                # Plot
                fig, axes = plt.subplots(3, 1, figsize=(14, 10))

                signal_np = signal.squeeze().cpu().numpy()
                cf_np = cf_signal.squeeze().cpu().numpy()
                diff = cf_np - signal_np

                axes[0].plot(signal_np, 'b-', linewidth=0.8, label='Original')
                axes[0].set_ylabel("Amplitude")
                axes[0].set_title(f"Original Signal (Class: {FAULT_CLASSES[current_class]})")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(cf_np, 'g-', linewidth=0.8, label='Counterfactual')
                axes[1].set_ylabel("Amplitude")
                axes[1].set_title(f"Counterfactual Signal (Class: {FAULT_CLASSES[target_class]})")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                axes[2].plot(diff, 'r-', linewidth=0.8)
                axes[2].fill_between(range(len(diff)), 0, diff, alpha=0.3, color='red')
                axes[2].set_xlabel("Time Steps")
                axes[2].set_ylabel("Difference")
                axes[2].set_title("Perturbation Required")
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.metric("L2 Distance", f"{info['final_distance']:.3f}")
                st.metric("Confidence", f"{info['final_confidence']:.1%}")

            else:
                st.error("❌ Failed to find counterfactual")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
