"""
Uncertainty Quantification for Neural Networks

Implements methods to quantify model confidence and uncertainty:
- Monte Carlo Dropout (Gal & Ghahramani, 2016)
- Deep Ensembles
- Calibration analysis
- Prediction rejection based on uncertainty

Critical for production deployment in safety-critical applications:
- Flag low-confidence predictions for manual review
- Measure model calibration (predicted confidence vs actual accuracy)
- Identify out-of-distribution samples

Reference:
Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning. ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.calibration import calibration_curve

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


class UncertaintyQuantifier:
    """
    Quantifies prediction uncertainty using Monte Carlo Dropout.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize uncertainty quantifier.

        Args:
            model: PyTorch model with dropout layers
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device

    def predict_with_uncertainty(
        self,
        input_signal: torch.Tensor,
        n_samples: int = 50,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict with uncertainty using Monte Carlo Dropout.

        Args:
            input_signal: Input signal [B, C, T]
            n_samples: Number of MC samples
            return_all: If True, return all MC predictions

        Returns:
            mean_prediction: Mean prediction [B, num_classes]
            uncertainty: Predictive uncertainty (std) [B, num_classes]
            all_predictions: All MC predictions [n_samples, B, num_classes] (if return_all)
        """
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)

        # Enable dropout during inference
        self.model.train()

        # Collect predictions
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(input_signal)
                probs = torch.softmax(output, dim=1)
                predictions.append(probs)

        # Stack predictions
        predictions = torch.stack(predictions)  # [n_samples, B, num_classes]

        # Compute mean and std
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        # Return to eval mode
        self.model.eval()

        if return_all:
            return mean_prediction, uncertainty, predictions
        else:
            return mean_prediction, uncertainty, None

    def entropy_based_uncertainty(
        self,
        mean_prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy-based uncertainty.

        High entropy = high uncertainty (uniform distribution)
        Low entropy = low uncertainty (peaked distribution)

        Args:
            mean_prediction: Mean prediction [B, num_classes]

        Returns:
            Entropy for each sample [B]
        """
        # H(p) = -Σ p_i log(p_i)
        entropy = -(mean_prediction * torch.log(mean_prediction + 1e-10)).sum(dim=1)
        return entropy

    def mutual_information(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mutual information between predictions and model parameters.

        MI = H(E[p]) - E[H(p)]
        High MI = epistemic uncertainty (model uncertainty)

        Args:
            predictions: All MC predictions [n_samples, B, num_classes]

        Returns:
            Mutual information [B]
        """
        # Mean prediction
        mean_pred = predictions.mean(dim=0)  # [B, num_classes]

        # Entropy of mean
        H_mean = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)  # [B]

        # Mean of entropy
        H_samples = -(predictions * torch.log(predictions + 1e-10)).sum(dim=2)  # [n_samples, B]
        mean_H = H_samples.mean(dim=0)  # [B]

        # Mutual information
        MI = H_mean - mean_H

        return MI

    def reject_uncertain_predictions(
        self,
        mean_prediction: torch.Tensor,
        uncertainty: torch.Tensor,
        threshold: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flag predictions with high uncertainty for manual review.

        Args:
            mean_prediction: Mean predictions [B, num_classes]
            uncertainty: Uncertainty [B, num_classes]
            threshold: Uncertainty threshold

        Returns:
            accepted_mask: Boolean mask of accepted predictions [B]
            rejected_indices: Indices of rejected predictions
        """
        # Max uncertainty across classes
        max_uncertainty = uncertainty.max(dim=1)[0]  # [B]

        # Accept predictions with low uncertainty
        accepted_mask = max_uncertainty < threshold

        # Rejected indices
        rejected_indices = torch.where(~accepted_mask)[0]

        return accepted_mask, rejected_indices


def calibrate_model(
    model: nn.Module,
    dataloader,
    device: str = 'cuda',
    num_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute calibration of model predictions.

    Calibration measures if predicted probabilities match empirical frequencies.
    Well-calibrated: P(correct | confidence=0.8) = 0.8

    Args:
        model: PyTorch model
        dataloader: Data loader for calibration dataset
        device: Device
        num_bins: Number of bins for calibration curve

    Returns:
        prob_true: Empirical probabilities
        prob_pred: Mean predicted probabilities
        ece: Expected Calibration Error
    """
    model.eval()
    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # Get predicted class and confidence
            confidences, predictions = probs.max(dim=1)

            # Check correctness
            correct = (predictions == targets)

            all_confidences.extend(confidences.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        all_correct, all_confidences, n_bins=num_bins, strategy='uniform'
    )

    # Compute Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_mask = (all_confidences >= bin_edges[i]) & (all_confidences < bin_edges[i+1])
        if bin_mask.sum() > 0:
            bin_acc = all_correct[bin_mask].mean()
            bin_conf = all_confidences[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(all_confidences)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return prob_true, prob_pred, ece


def plot_calibration_curve(
    prob_true: np.ndarray,
    prob_pred: np.ndarray,
    ece: float,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Plot calibration curve (reliability diagram).

    Args:
        prob_true: Empirical probabilities
        prob_pred: Predicted probabilities
        ece: Expected Calibration Error
        model_name: Name of model
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

    # Actual calibration
    ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label=f'{model_name}')

    # Fill gap
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2)

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Empirical Probability', fontsize=12)
    ax.set_title(f'Calibration Curve\nECE = {ece:.4f}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_uncertainty_distribution(
    uncertainties: np.ndarray,
    correct_mask: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot distribution of uncertainty for correct vs incorrect predictions.

    Args:
        uncertainties: Prediction uncertainties [N]
        correct_mask: Boolean mask of correct predictions [N]
        save_path: Save path
    """
    correct_uncertainties = uncertainties[correct_mask]
    incorrect_uncertainties = uncertainties[~correct_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(correct_uncertainties, bins=30, alpha=0.6, label='Correct', color='green', density=True)
    axes[0].hist(incorrect_uncertainties, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
    axes[0].set_xlabel('Uncertainty', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data = [correct_uncertainties, incorrect_uncertainties]
    axes[1].boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Uncertainty', fontsize=11)
    axes[1].set_title('Uncertainty by Correctness', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_prediction_with_uncertainty(
    signal: torch.Tensor,
    mean_prediction: torch.Tensor,
    uncertainty: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize prediction with uncertainty bars.

    Args:
        signal: Input signal
        mean_prediction: Mean prediction [num_classes]
        uncertainty: Uncertainty [num_classes]
        class_names: Class names
        save_path: Save path
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy().squeeze()
    if isinstance(mean_prediction, torch.Tensor):
        mean_prediction = mean_prediction.cpu().numpy().squeeze()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.cpu().numpy().squeeze()

    num_classes = len(mean_prediction)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Signal
    axes[0].plot(signal, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Input Signal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prediction with uncertainty
    y_pos = np.arange(num_classes)
    axes[1].barh(y_pos, mean_prediction, xerr=uncertainty, alpha=0.7,
                color='skyblue', edgecolor='black', capsize=5)

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(class_names, fontsize=9)
    axes[1].set_xlabel('Probability ± Uncertainty', fontsize=11)
    axes[1].set_title('Prediction with Uncertainty (Monte Carlo Dropout)',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='x')

    # Highlight predicted class
    predicted_class = np.argmax(mean_prediction)
    axes[1].get_children()[predicted_class].set_color('green')
    axes[1].get_children()[predicted_class].set_alpha(0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Test uncertainty quantification
    print("=" * 60)
    print("Uncertainty Quantification - Validation")
    print("=" * 60)

    from models.cnn.cnn_1d import CNN1D

    # Create model with dropout
    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1, dropout=0.3)
    signal = torch.randn(1, 1, 10240)

    # Create quantifier
    quantifier = UncertaintyQuantifier(model, device='cpu')

    print("\nComputing prediction with uncertainty...")
    mean_pred, uncertainty, all_preds = quantifier.predict_with_uncertainty(
        signal, n_samples=50, return_all=True
    )

    print(f"  Input shape: {signal.shape}")
    print(f"  Mean prediction shape: {mean_pred.shape}")
    print(f"  Uncertainty shape: {uncertainty.shape}")
    print(f"  All predictions shape: {all_preds.shape}")

    print(f"\n  Predicted class: {mean_pred.argmax(dim=1).item()}")
    print(f"  Confidence: {mean_pred.max(dim=1)[0].item():.4f}")
    print(f"  Max uncertainty: {uncertainty.max(dim=1)[0].item():.4f}")

    # Entropy
    entropy = quantifier.entropy_based_uncertainty(mean_pred)
    print(f"  Entropy: {entropy.item():.4f}")

    # Mutual information
    MI = quantifier.mutual_information(all_preds)
    print(f"  Mutual Information: {MI.item():.4f}")

    # Rejection
    accepted, rejected = quantifier.reject_uncertain_predictions(
        mean_pred, uncertainty, threshold=0.2
    )
    print(f"  Accepted: {accepted.sum().item()}/{len(accepted)}")
    print(f"  Rejected: {len(rejected)}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
