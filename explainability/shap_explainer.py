"""
SHAP (SHapley Additive exPlanations) Explainer

Implements SHAP values for deep learning models. SHAP is based on Shapley values
from coalitional game theory, providing a unified measure of feature importance.

Key Properties:
- Local accuracy: f(x) = φ₀ + Σφᵢ
- Missingness: features with no contribution have φᵢ = 0
- Consistency: if model changes so feature contributes more, Shapley value increases

Methods Implemented:
- DeepSHAP: Fast approximation for deep learning models
- Kernel SHAP: Model-agnostic method (slower but works with any model)
- Gradient SHAP: Uses gradients as approximation

Reference:
Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.
NeurIPS.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


class SHAPExplainer:
    """
    SHAP explainer with multiple backends.

    Supports:
    - Native PyTorch implementation (GradientSHAP)
    - SHAP library integration (DeepSHAP, KernelSHAP) if available
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        use_shap_library: bool = True
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: PyTorch model to explain
            background_data: Background dataset for SHAP [N, C, T]
                           (used to approximate expectations)
            device: Device to run on
            use_shap_library: If True, try to use official SHAP library
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Store background data
        if background_data is not None:
            self.background_data = background_data.to(device)
        else:
            self.background_data = None

        # Try to import SHAP library
        self.shap_available = False
        if use_shap_library:
            try:
                import shap
                self.shap = shap
                self.shap_available = True
                print("✓ SHAP library available")
            except ImportError:
                warnings.warn(
                    "SHAP library not installed. Using native PyTorch implementation. "
                    "Install with: pip install shap"
                )
                self.shap_available = False

    def explain(
        self,
        input_signal: torch.Tensor,
        method: str = 'gradient',
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Compute SHAP values for input signal.

        Args:
            input_signal: Input signal to explain [1, C, T] or [C, T]
            method: SHAP method ('gradient', 'deep', 'kernel')
            n_samples: Number of samples for approximation

        Returns:
            SHAP values with same shape as input
        """
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)

        # Choose method
        if method == 'gradient':
            return self._gradient_shap(input_signal, n_samples)
        elif method == 'deep' and self.shap_available:
            return self._deep_shap(input_signal)
        elif method == 'kernel':
            return self._kernel_shap(input_signal, n_samples)
        else:
            warnings.warn(f"Method {method} not available. Using gradient SHAP.")
            return self._gradient_shap(input_signal, n_samples)

    def _gradient_shap(
        self,
        input_signal: torch.Tensor,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        GradientSHAP: Uses gradients and random baselines.

        This is a native PyTorch implementation that doesn't require SHAP library.

        Algorithm:
        1. Sample random baselines from background data
        2. Interpolate between baseline and input
        3. Compute gradients at interpolated points
        4. Average gradients × (input - baseline)
        """
        if self.background_data is None:
            # Use zero baseline if no background data
            baselines = torch.zeros(n_samples, *input_signal.shape[1:], device=self.device)
        else:
            # Sample from background
            indices = torch.randint(0, len(self.background_data), (n_samples,))
            baselines = self.background_data[indices]

        # Accumulate gradients
        total_gradients = torch.zeros_like(input_signal)

        for baseline in baselines:
            baseline = baseline.unsqueeze(0)

            # Random interpolation coefficient
            alpha = torch.rand(1, device=self.device)

            # Interpolated input
            interpolated = baseline + alpha * (input_signal - baseline)
            interpolated.requires_grad = True

            # Forward pass
            output = self.model(interpolated)

            # Get predicted class
            target_class = output.argmax(dim=1)

            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()

            # Accumulate gradients
            total_gradients += interpolated.grad

        # Average gradients
        avg_gradients = total_gradients / n_samples

        # SHAP values: gradient × (input - mean_baseline)
        mean_baseline = baselines.mean(dim=0, keepdim=True)
        shap_values = avg_gradients * (input_signal - mean_baseline)

        return shap_values.detach()

    def _deep_shap(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        DeepSHAP using official SHAP library.

        Requires: pip install shap
        """
        if not self.shap_available:
            raise ImportError("SHAP library not available. Use gradient method instead.")

        if self.background_data is None:
            raise ValueError("Background data required for DeepSHAP")

        # Create DeepExplainer
        explainer = self.shap.DeepExplainer(
            self.model,
            self.background_data[:min(100, len(self.background_data))]
        )

        # Compute SHAP values
        shap_values = explainer.shap_values(input_signal)

        # Convert to torch tensor
        if isinstance(shap_values, list):
            # Multi-class output
            target_class = self.model(input_signal).argmax(dim=1).item()
            shap_values = torch.tensor(shap_values[target_class], device=self.device)
        else:
            shap_values = torch.tensor(shap_values, device=self.device)

        return shap_values

    def _kernel_shap(
        self,
        input_signal: torch.Tensor,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        KernelSHAP: Model-agnostic SHAP using linear regression.

        This is a simplified implementation that works on signal segments.
        """
        # Segment the signal into patches
        signal_np = input_signal.squeeze().cpu().numpy()
        n_segments = 20
        segment_size = len(signal_np) // n_segments

        # Create mask for each segment
        def predict_with_mask(mask):
            """Predict with masked input (masked regions replaced with baseline)."""
            masked_signal = signal_np.copy()

            for i, keep in enumerate(mask):
                if not keep:
                    start = i * segment_size
                    end = (i + 1) * segment_size
                    masked_signal[start:end] = 0  # Replace with baseline (zeros)

            # Reshape and predict
            input_tensor = torch.tensor(masked_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                return output.softmax(dim=1).cpu().numpy()[0]

        # Generate samples
        masks = np.random.randint(0, 2, size=(n_samples, n_segments))
        predictions = np.array([predict_with_mask(mask) for mask in masks])

        # Get target class
        target_class = self.model(input_signal).argmax(dim=1).item()
        target_predictions = predictions[:, target_class]

        # Fit linear model (simplified SHAP)
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(masks, target_predictions)

        # Segment-level SHAP values
        segment_shap = lr.coef_

        # Expand to full signal
        shap_values = np.zeros_like(signal_np)
        for i, shap_val in enumerate(segment_shap):
            start = i * segment_size
            end = (i + 1) * segment_size
            shap_values[start:end] = shap_val

        # Convert to torch
        shap_tensor = torch.tensor(shap_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        shap_tensor = shap_tensor.to(self.device)

        return shap_tensor


def plot_shap_waterfall(
    shap_values: Union[torch.Tensor, np.ndarray],
    base_value: float,
    predicted_value: float,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    save_path: Optional[str] = None
):
    """
    Create SHAP waterfall plot showing how features contribute to prediction.

    Args:
        shap_values: SHAP values for features
        base_value: Base value (expected model output)
        predicted_value: Actual prediction
        feature_names: Names of features
        max_display: Maximum features to display
        save_path: Path to save figure
    """
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()

    shap_values = shap_values.flatten()

    # Get top features by absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-max_display:][::-1]

    top_shap = shap_values[top_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in top_indices]
    else:
        feature_names = [feature_names[i] for i in top_indices]

    # Create waterfall plot
    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))

    # Starting point
    y_pos = np.arange(len(top_shap))
    cumulative = base_value

    colors = ['red' if val < 0 else 'green' for val in top_shap]

    for i, (name, val) in enumerate(zip(feature_names, top_shap)):
        ax.barh(i, val, left=cumulative, color=colors[i], alpha=0.7)
        cumulative += val

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=11)
    ax.set_title(f'SHAP Waterfall Plot\nBase: {base_value:.3f} → Prediction: {predicted_value:.3f}',
                fontsize=12, fontweight='bold')
    ax.axvline(x=base_value, color='black', linestyle='--', linewidth=1, label='Base value')
    ax.axvline(x=predicted_value, color='blue', linestyle='--', linewidth=1, label='Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_shap_summary(
    shap_values_batch: Union[torch.Tensor, np.ndarray],
    signals_batch: Union[torch.Tensor, np.ndarray],
    max_display: int = 20,
    save_path: Optional[str] = None
):
    """
    Create SHAP summary plot showing feature importance across multiple samples.

    Args:
        shap_values_batch: SHAP values for batch [B, C, T]
        signals_batch: Original signals [B, C, T]
        max_display: Max features to display
        save_path: Save path
    """
    if isinstance(shap_values_batch, torch.Tensor):
        shap_values_batch = shap_values_batch.cpu().numpy()
    if isinstance(signals_batch, torch.Tensor):
        signals_batch = signals_batch.cpu().numpy()

    # Flatten to [B, Features]
    shap_flat = shap_values_batch.reshape(shap_values_batch.shape[0], -1)
    signals_flat = signals_batch.reshape(signals_batch.shape[0], -1)

    # Get mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_flat).mean(axis=0)
    top_features = np.argsort(mean_abs_shap)[-max_display:][::-1]

    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))

    feature_names = [f"Feature {i}" for i in top_features]
    y_pos = np.arange(len(top_features))

    # Swarm-like plot
    for i, feat_idx in enumerate(top_features):
        shap_values = shap_flat[:, feat_idx]
        feature_values = signals_flat[:, feat_idx]

        # Normalize feature values for coloring
        if feature_values.std() > 0:
            feature_values_norm = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
        else:
            feature_values_norm = np.ones_like(feature_values) * 0.5

        # Plot points
        scatter = ax.scatter(
            shap_values,
            np.ones(len(shap_values)) * i + np.random.randn(len(shap_values)) * 0.1,
            c=feature_values_norm,
            cmap='coolwarm',
            alpha=0.6,
            s=20
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=11)
    ax.set_title('SHAP Summary Plot', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Feature Value', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Test SHAP explainer
    print("=" * 60)
    print("SHAP Explainer - Validation")
    print("=" * 60)

    from models.cnn.cnn_1d import CNN1D

    # Create model and data
    model = CNN1D(num_classes=11, input_channels=1, dropout=0.3)
    signal = torch.randn(1, 1, 10240)
    background = torch.randn(50, 1, 10240)

    # Create explainer
    explainer = SHAPExplainer(model, background_data=background, device='cpu', use_shap_library=False)

    print("\nComputing GradientSHAP...")
    shap_values = explainer.explain(signal, method='gradient', n_samples=50)
    print(f"  Input shape: {signal.shape}")
    print(f"  SHAP values shape: {shap_values.shape}")
    print(f"  SHAP range: [{shap_values.min().item():.4f}, {shap_values.max().item():.4f}]")
    print(f"  SHAP sum: {shap_values.sum().item():.4f}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
