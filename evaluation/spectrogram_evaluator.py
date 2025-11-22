"""
Spectrogram Model Evaluator

Extends base ModelEvaluator with spectrogram-specific evaluation:
- Visualization of predictions on spectrograms
- Analysis of which frequency ranges contribute to predictions
- Grad-CAM heatmaps on spectrograms
- Comparison between different TFR types (STFT vs CWT vs WVD)

Usage:
    from evaluation.spectrogram_evaluator import SpectrogramEvaluator

    evaluator = SpectrogramEvaluator(model, device='cuda')
    results = evaluator.evaluate(test_loader, class_names=fault_types)
    evaluator.visualize_predictions(test_spectrograms[:4], save_path='predictions.png')
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluator import ModelEvaluator


class SpectrogramEvaluator(ModelEvaluator):
    """
    Evaluator for spectrogram-based models.

    Extends base evaluator with:
    - Spectrogram visualization
    - Frequency contribution analysis
    - Grad-CAM for attention maps

    Args:
        model: Trained 2D CNN model
        device: Device to run evaluation on
        class_names: Optional list of class names
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        class_names: Optional[List[str]] = None
    ):
        super().__init__(model, device)
        self.class_names = class_names or [
            'Normal', 'Ball Fault', 'Inner Race', 'Outer Race',
            'Combined', 'Imbalance', 'Misalignment', 'Oil Whirl',
            'Cavitation', 'Looseness', 'Oil Deficiency'
        ]

    def visualize_predictions(
        self,
        spectrograms: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Visualize spectrograms with predictions.

        Args:
            spectrograms: Input spectrograms [B, 1, H, W]
            targets: Optional ground truth labels [B]
            save_path: Optional path to save figure
            figsize: Figure size
        """
        self.model.eval()

        with torch.no_grad():
            spectrograms = spectrograms.to(self.device)
            outputs = self.model(spectrograms)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)

        spectrograms = spectrograms.cpu().numpy()
        predictions = predictions.cpu().numpy()
        probs = probs.cpu().numpy()

        if targets is not None:
            targets = targets.cpu().numpy()

        batch_size = min(spectrograms.shape[0], 8)  # Max 8 samples

        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()

        for i in range(batch_size):
            ax = axes[i]

            # Plot spectrogram
            spec = spectrograms[i, 0, :, :]  # [H, W]
            im = ax.imshow(
                spec,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='nearest'
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Title with prediction and ground truth
            pred_class = self.class_names[predictions[i]]
            confidence = probs[i, predictions[i]] * 100

            if targets is not None:
                true_class = self.class_names[targets[i]]
                color = 'green' if predictions[i] == targets[i] else 'red'
                title = f"True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)"
            else:
                color = 'blue'
                title = f"Pred: {pred_class} ({confidence:.1f}%)"

            ax.set_title(title, color=color, fontsize=10, weight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def compute_grad_cam(
        self,
        spectrogram: torch.Tensor,
        target_layer: str = 'layer4',
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Grad-CAM heatmap for a spectrogram.

        Grad-CAM visualizes which regions of the spectrogram are important
        for the model's prediction.

        Args:
            spectrogram: Input spectrogram [1, 1, H, W]
            target_layer: Name of layer to compute gradients for
            target_class: Target class (None = predicted class)

        Returns:
            Tuple of (heatmap [H, W], predicted_class)
        """
        self.model.eval()

        spectrogram = spectrogram.to(self.device)
        spectrogram.requires_grad = True

        # Forward pass
        outputs = self.model(spectrogram)
        _, predicted = outputs.max(1)
        pred_class = predicted.item()

        if target_class is None:
            target_class = pred_class

        # Get target layer activations
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations['value'] = output

        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]

        # Register hooks
        target_module = dict(self.model.named_modules())[target_layer]
        forward_handle = target_module.register_forward_hook(forward_hook)
        backward_handle = target_module.register_full_backward_hook(backward_hook)

        # Backward pass
        self.model.zero_grad()
        outputs[0, target_class].backward()

        # Compute Grad-CAM
        pooled_gradients = torch.mean(gradients['value'], dim=[0, 2, 3])
        activation = activations['value'][0]

        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()

        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize to input size
        from scipy.ndimage import zoom
        H, W = spectrogram.shape[2:]
        heatmap = zoom(heatmap, (H / heatmap.shape[0], W / heatmap.shape[1]))

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        return heatmap, pred_class

    def visualize_grad_cam(
        self,
        spectrograms: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Visualize spectrograms with Grad-CAM overlays.

        Args:
            spectrograms: Input spectrograms [B, 1, H, W]
            targets: Optional ground truth labels
            save_path: Optional path to save figure
            figsize: Figure size
        """
        batch_size = min(spectrograms.shape[0], 8)

        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()

        for i in range(batch_size):
            ax = axes[i]

            # Get spectrogram
            spec = spectrograms[i:i+1].to(self.device)
            spec_np = spec[0, 0].cpu().numpy()

            # Compute Grad-CAM
            heatmap, pred_class = self.compute_grad_cam(spec)

            # Overlay heatmap on spectrogram
            ax.imshow(spec_np, aspect='auto', origin='lower', cmap='gray', alpha=0.7)
            im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Title
            pred_name = self.class_names[pred_class]
            if targets is not None:
                true_name = self.class_names[targets[i]]
                color = 'green' if pred_class == targets[i] else 'red'
                title = f"True: {true_name}\nPred: {pred_name}"
            else:
                color = 'blue'
                title = f"Pred: {pred_name}"

            ax.set_title(title, color=color, fontsize=10, weight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved Grad-CAM visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def analyze_frequency_contributions(
        self,
        spectrograms: torch.Tensor,
        num_freq_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze which frequency bins contribute most to predictions.

        Splits spectrogram into frequency bands and measures importance.

        Args:
            spectrograms: Input spectrograms [B, 1, H, W]
            num_freq_bins: Number of frequency bins to analyze

        Returns:
            Dictionary with frequency contributions per class
        """
        self.model.eval()

        spectrograms = spectrograms.to(self.device)
        H = spectrograms.shape[2]
        bin_size = H // num_freq_bins

        # Baseline prediction
        with torch.no_grad():
            baseline_outputs = self.model(spectrograms)
            baseline_probs = torch.softmax(baseline_outputs, dim=1)

        # Measure importance by masking each frequency bin
        freq_importance = torch.zeros(spectrograms.shape[0], num_freq_bins, self.model.fc.out_features)

        for bin_idx in range(num_freq_bins):
            # Mask frequency bin
            masked_spec = spectrograms.clone()
            start_freq = bin_idx * bin_size
            end_freq = min((bin_idx + 1) * bin_size, H)
            masked_spec[:, :, start_freq:end_freq, :] = 0

            # Predict with masked spectrogram
            with torch.no_grad():
                masked_outputs = self.model(masked_spec)
                masked_probs = torch.softmax(masked_outputs, dim=1)

            # Importance = drop in probability
            importance = baseline_probs - masked_probs
            freq_importance[:, bin_idx, :] = importance.cpu()

        # Average over batch
        freq_importance = freq_importance.mean(dim=0).numpy()

        return {
            'frequency_importance': freq_importance,
            'num_bins': num_freq_bins,
            'bin_size': bin_size
        }

    def compare_tfr_types(
        self,
        dataloaders: Dict[str, DataLoader],
        tfr_names: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Compare model performance on different TFR types.

        Args:
            dataloaders: Dictionary of DataLoaders for different TFR types
                e.g., {'STFT': loader1, 'CWT': loader2, 'WVD': loader3}
            tfr_names: Optional custom names for TFR types

        Returns:
            Dictionary of results per TFR type
        """
        results = {}

        for tfr_type, loader in dataloaders.items():
            print(f"\nEvaluating on {tfr_type}...")
            tfr_results = self.evaluate(loader, self.class_names)
            results[tfr_type] = tfr_results

        # Create comparison table
        print("\n" + "="*70)
        print(f"{'TFR Type':<15} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
        print("="*70)

        for tfr_type, result in results.items():
            acc = result['accuracy']
            per_class = result['per_class_metrics']

            # Compute macro-averaged metrics
            f1_scores = [m['f1_score'] for m in per_class.values()]
            precisions = [m['precision'] for m in per_class.values()]
            recalls = [m['recall'] for m in per_class.values()]

            f1_avg = np.mean(f1_scores)
            prec_avg = np.mean(precisions)
            rec_avg = np.mean(recalls)

            print(f"{tfr_type:<15} {acc:>10.2f}% {f1_avg:>10.4f}  {prec_avg:>10.4f}  {rec_avg:>10.4f}")

        print("="*70)

        return results


if __name__ == '__main__':
    # Test evaluator
    from models.spectrogram_cnn import resnet18_2d
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

    model = resnet18_2d(num_classes=NUM_CLASSES)
    evaluator = SpectrogramEvaluator(model, device='cpu')

    # Test with dummy data
    spectrograms = torch.randn(4, 1, 129, 400)
    targets = torch.tensor([0, 1, 2, 3])

    # Test visualization
    evaluator.visualize_predictions(spectrograms, targets)

    # Test Grad-CAM
    evaluator.visualize_grad_cam(spectrograms, targets)

    # Test frequency analysis
    freq_contrib = evaluator.analyze_frequency_contributions(spectrograms)
    print(f"Frequency importance shape: {freq_contrib['frequency_importance'].shape}")
