"""
LIME (Local Interpretable Model-agnostic Explanations)

Implements LIME for explaining neural network predictions on time-series data.
LIME explains predictions by fitting an interpretable linear model locally
around the prediction.

Key Idea:
- Perturb input by masking segments
- Observe how predictions change
- Fit linear model: g(z) = w₀ + Σwᵢzᵢ where zᵢ indicates segment presence
- Interpret linear weights as feature importance

Properties:
- Model-agnostic (works with any black-box model)
- Local fidelity (accurate explanations locally)
- Interpretable (produces human-understandable explanations)

Reference:
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?":
Explaining the Predictions of Any Classifier. KDD.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List, Tuple, Callable
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import pairwise_distances

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


class LIMEExplainer:
    """
    LIME explainer for time-series neural network models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        num_segments: int = 20,
        kernel_width: float = 0.25
    ):
        """
        Initialize LIME explainer.

        Args:
            model: PyTorch model to explain
            device: Device to run on
            num_segments: Number of segments to divide signal into
            kernel_width: Kernel width for sample weighting
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.num_segments = num_segments
        self.kernel_width = kernel_width

    def explain(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None,
        num_samples: int = 1000,
        distance_metric: str = 'cosine',
        model_regressor: Optional[any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Explain prediction using LIME.

        Args:
            input_signal: Input signal to explain [1, C, T] or [C, T]
            target_class: Target class to explain (if None, uses predicted)
            num_samples: Number of perturbed samples
            distance_metric: Distance metric for sample weighting
            model_regressor: Linear model to use (default: Ridge)

        Returns:
            segment_weights: Importance weights for each segment [num_segments]
            segment_boundaries: Boundaries of segments [(start, end), ...]
        """
        # Ensure batch dimension
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)

        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_signal)
                target_class = output.argmax(dim=1).item()

        # Convert to numpy
        signal_np = input_signal.squeeze().cpu().numpy()
        signal_length = len(signal_np)

        # Create segments
        segment_boundaries = self._create_segments(signal_length)

        # Generate perturbed samples
        perturbed_data, perturbed_signals = self._generate_perturbed_samples(
            signal_np, segment_boundaries, num_samples
        )

        # Get predictions for perturbed samples
        predictions = self._get_predictions(perturbed_signals, target_class)

        # Compute distances and weights
        distances = pairwise_distances(
            perturbed_data,
            perturbed_data[0:1],  # Original (all segments present)
            metric=distance_metric
        ).ravel()

        weights = self._kernel_function(distances)

        # Fit linear model
        if model_regressor is None:
            model_regressor = Ridge(alpha=1.0, fit_intercept=True)

        model_regressor.fit(perturbed_data, predictions, sample_weight=weights)

        # Get coefficients (segment importance)
        segment_weights = model_regressor.coef_

        return segment_weights, segment_boundaries

    def _create_segments(self, signal_length: int) -> List[Tuple[int, int]]:
        """
        Divide signal into segments.

        Args:
            signal_length: Length of signal

        Returns:
            List of (start, end) tuples for each segment
        """
        segment_size = signal_length // self.num_segments
        boundaries = []

        for i in range(self.num_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < self.num_segments - 1 else signal_length
            boundaries.append((start, end))

        return boundaries

    def _generate_perturbed_samples(
        self,
        signal: np.ndarray,
        segment_boundaries: List[Tuple[int, int]],
        num_samples: int
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate perturbed versions of signal by masking segments.

        Args:
            signal: Original signal
            segment_boundaries: Segment boundaries
            num_samples: Number of samples to generate

        Returns:
            perturbed_data: Binary indicators of which segments are present [num_samples, num_segments]
            perturbed_signals: List of perturbed signals
        """
        # First sample is original (all segments present)
        perturbed_data = np.ones((num_samples, len(segment_boundaries)))

        # Random binary masks for remaining samples
        perturbed_data[1:] = np.random.binomial(1, 0.5, size=(num_samples - 1, len(segment_boundaries)))

        # Create perturbed signals
        perturbed_signals = []

        for mask in perturbed_data:
            perturbed_signal = signal.copy()

            # Mask out segments where mask=0
            for i, (start, end) in enumerate(segment_boundaries):
                if mask[i] == 0:
                    # Replace with zeros or noise
                    perturbed_signal[start:end] = 0  # or np.mean(signal)

            perturbed_signals.append(perturbed_signal)

        return perturbed_data, perturbed_signals

    def _get_predictions(
        self,
        perturbed_signals: List[np.ndarray],
        target_class: int
    ) -> np.ndarray:
        """
        Get model predictions for perturbed signals.

        Args:
            perturbed_signals: List of perturbed signals
            target_class: Target class

        Returns:
            Predictions for target class [num_samples]
        """
        predictions = []

        with torch.no_grad():
            for signal in perturbed_signals:
                # Convert to tensor
                signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                signal_tensor = signal_tensor.to(self.device)

                # Predict
                output = self.model(signal_tensor)
                prob = torch.softmax(output, dim=1)[0, target_class].item()
                predictions.append(prob)

        return np.array(predictions)

    def _kernel_function(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute sample weights using exponential kernel.

        Args:
            distances: Distances from original sample

        Returns:
            Weights for each sample
        """
        kernel_width = self.kernel_width
        weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))
        return weights


def plot_lime_explanation(
    signal: Union[torch.Tensor, np.ndarray],
    segment_weights: np.ndarray,
    segment_boundaries: List[Tuple[int, int]],
    predicted_class: int,
    true_class: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize LIME explanation.

    Args:
        signal: Original signal
        segment_weights: Importance weights for segments
        segment_boundaries: Segment boundaries
        predicted_class: Predicted class
        true_class: True class (optional)
        class_names: Class names
        save_path: Save path
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()

    signal = signal.squeeze()

    # Normalize weights for visualization
    weights_normalized = segment_weights / (np.abs(segment_weights).max() + 1e-8)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    time = np.arange(len(signal))

    # Plot 1: Original Signal
    axes[0].plot(time, signal, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Segment Importance
    for i, (start, end) in enumerate(segment_boundaries):
        weight = segment_weights[i]
        color = 'green' if weight > 0 else 'red'
        alpha = min(abs(weights_normalized[i]), 1.0)

        axes[1].barh(
            0, end - start, left=start,
            height=weight, color=color, alpha=alpha, edgecolor='black', linewidth=0.5
        )

    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlim(0, len(signal))
    axes[1].set_ylabel('Segment Weight', fontsize=11)
    axes[1].set_title('LIME Segment Importance', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    # Plot 3: Signal with Overlay
    axes[2].plot(time, signal, 'b-', linewidth=0.8, alpha=0.5, label='Signal')

    # Color segments by importance
    for i, (start, end) in enumerate(segment_boundaries):
        weight_norm = weights_normalized[i]
        if weight_norm > 0:
            color = plt.cm.Greens(abs(weight_norm))
        else:
            color = plt.cm.Reds(abs(weight_norm))

        axes[2].axvspan(start, end, alpha=abs(weight_norm) * 0.5, color=color, linewidth=0)

    axes[2].set_xlabel('Time Steps', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=11)

    # Title
    if class_names:
        title = f'Predicted: {class_names[predicted_class]}'
        if true_class is not None:
            title += f' | True: {class_names[true_class]}'
    else:
        title = f'Predicted: {predicted_class}'
        if true_class is not None:
            title += f' | True: {true_class}'

    axes[2].set_title(f'Signal with LIME Overlay\n{title}', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"LIME explanation saved to {save_path}")

    plt.show()


def plot_lime_bar_chart(
    segment_weights: np.ndarray,
    top_k: int = 10,
    class_name: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Create bar chart of top segment importances.

    Args:
        segment_weights: Segment importance weights
        top_k: Number of top segments to show
        class_name: Predicted class name
        save_path: Save path
    """
    # Get top-k by absolute value
    abs_weights = np.abs(segment_weights)
    top_indices = np.argsort(abs_weights)[-top_k:][::-1]

    top_weights = segment_weights[top_indices]
    labels = [f"Segment {i+1}" for i in top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))

    colors = ['green' if w > 0 else 'red' for w in top_weights]

    y_pos = np.arange(len(top_weights))
    ax.barh(y_pos, top_weights, color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('LIME Weight (Contribution to Prediction)', fontsize=11)

    title = f'Top {top_k} Segments by LIME Importance'
    if class_name:
        title += f'\n(Predicted Class: {class_name})'
    ax.set_title(title, fontsize=12, fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Test LIME explainer
    print("=" * 60)
    print("LIME Explainer - Validation")
    print("=" * 60)

    from models.cnn.cnn_1d import CNN1D
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

    # Create model and data
    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1, dropout=0.3)
    signal = torch.randn(1, 1, 10240)

    # Create explainer
    explainer = LIMEExplainer(model, device='cpu', num_segments=20)

    print("\nComputing LIME explanation...")
    segment_weights, segment_boundaries = explainer.explain(
        signal,
        target_class=3,
        num_samples=500
    )

    print(f"  Input shape: {signal.shape}")
    print(f"  Number of segments: {len(segment_weights)}")
    print(f"  Weight range: [{segment_weights.min():.4f}, {segment_weights.max():.4f}]")
    print(f"  Top positive segment: {np.argmax(segment_weights)} (weight: {segment_weights.max():.4f})")
    print(f"  Top negative segment: {np.argmin(segment_weights)} (weight: {segment_weights.min():.4f})")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
