"""
Anchors: High-Precision Model-Agnostic Explanations

Implements the Anchors algorithm for generating rule-based explanations.
Anchors finds a set of simple IF-THEN rules that "anchor" a prediction,
meaning the prediction remains constant even if other features change.

Key Properties:
- Anchors are sufficient conditions: IF anchor THEN prediction (with high probability)
- High precision: The rule holds for a large fraction of perturbations
- Model-agnostic: Works with any black-box model
- Human-interpretable: Produces simple rules (e.g., "IF freq_100Hz > 0.5 AND amp < 2.0 THEN fault_X")

Algorithm:
1. Start with empty anchor
2. Iteratively add predicates (feature conditions) that increase coverage
3. Use multi-armed bandit (MAB) to efficiently search predicate space
4. Stop when precision threshold reached

Reference:
Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). Anchors: High-Precision
Model-Agnostic Explanations. AAAI.

For time-series signals, predicates are based on:
- Time-domain statistics (mean, std, peaks, etc.)
- Frequency-domain features (dominant frequencies, spectral energy)
- Temporal segments (signal value in specific windows)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Set
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from dataclasses import dataclass
from scipy import stats as scipy_stats

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class Predicate:
    """
    Represents a single condition (predicate) in an anchor rule.

    Examples:
    - "mean_amplitude > 1.5"
    - "dominant_frequency in [80, 120]"
    - "segment_10_energy < 0.3"
    """
    feature_name: str
    feature_idx: int
    operator: str  # '>', '<', '>=', '<=', 'in_range'
    threshold: float
    threshold_max: Optional[float] = None  # For 'in_range'

    def evaluate(self, feature_value: float) -> bool:
        """Check if feature value satisfies predicate."""
        if self.operator == '>':
            return feature_value > self.threshold
        elif self.operator == '<':
            return feature_value < self.threshold
        elif self.operator == '>=':
            return feature_value >= self.threshold
        elif self.operator == '<=':
            return feature_value <= self.threshold
        elif self.operator == 'in_range':
            return self.threshold <= feature_value <= self.threshold_max
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def __str__(self):
        if self.operator == 'in_range':
            return f"{self.feature_name} in [{self.threshold:.3f}, {self.threshold_max:.3f}]"
        else:
            return f"{self.feature_name} {self.operator} {self.threshold:.3f}"

    def __hash__(self):
        return hash((self.feature_name, self.feature_idx, self.operator,
                    self.threshold, self.threshold_max))

    def __eq__(self, other):
        return (self.feature_name == other.feature_name and
                self.feature_idx == other.feature_idx and
                self.operator == other.operator and
                self.threshold == other.threshold and
                self.threshold_max == other.threshold_max)


@dataclass
class Anchor:
    """
    Represents an anchor: a set of predicates that anchor a prediction.
    """
    predicates: List[Predicate]
    precision: float
    coverage: float
    target_class: int

    def evaluate(self, features: np.ndarray) -> bool:
        """Check if all predicates are satisfied."""
        return all(pred.evaluate(features[pred.feature_idx]) for pred in self.predicates)

    def __str__(self):
        if not self.predicates:
            return "Empty anchor"

        rule_str = "IF " + " AND ".join(str(pred) for pred in self.predicates)
        rule_str += f" THEN class={self.target_class}"
        rule_str += f" (precision={self.precision:.3f}, coverage={self.coverage:.3f})"
        return rule_str


class AnchorExplainer:
    """
    Generates anchor explanations for neural network predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Callable,
        feature_names: List[str],
        device: str = 'cuda',
        precision_threshold: float = 0.95,
        beam_size: int = 5,
        max_predicates: int = 5
    ):
        """
        Initialize Anchor explainer.

        Args:
            model: PyTorch model
            feature_extractor: Function to extract features from signals (signal -> features)
            feature_names: Names of features
            device: Device
            precision_threshold: Minimum precision for anchor (default: 0.95)
            beam_size: Beam search width
            max_predicates: Maximum predicates in anchor
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.feature_names = feature_names
        self.precision_threshold = precision_threshold
        self.beam_size = beam_size
        self.max_predicates = max_predicates

    def explain(
        self,
        input_signal: torch.Tensor,
        n_samples: int = 1000,
        verbose: bool = True
    ) -> Anchor:
        """
        Generate anchor explanation for input signal.

        Args:
            input_signal: Input signal to explain [1, C, T] or [C, T]
            n_samples: Number of perturbation samples for precision estimation
            verbose: Print progress

        Returns:
            Anchor object
        """
        # Ensure batch dimension
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)

        # Get prediction for original signal
        with torch.no_grad():
            output = self.model(input_signal)
            target_class = output.argmax(dim=1).item()

        if verbose:
            print(f"  Target class: {target_class}")

        # Extract features
        original_features = self.feature_extractor(input_signal.squeeze().cpu())

        # Generate candidate predicates
        candidate_predicates = self._generate_candidate_predicates(original_features)

        if verbose:
            print(f"  Generated {len(candidate_predicates)} candidate predicates")

        # Beam search to build anchor
        current_anchors = [Anchor(predicates=[], precision=0.0, coverage=1.0, target_class=target_class)]

        for iteration in range(self.max_predicates):
            if verbose:
                print(f"\n  Iteration {iteration + 1}/{self.max_predicates}")

            # Generate candidate anchors by adding one predicate
            candidate_anchors = []

            for anchor in current_anchors:
                # Get unused predicates
                used_features = {pred.feature_idx for pred in anchor.predicates}
                available_predicates = [p for p in candidate_predicates
                                       if p.feature_idx not in used_features]

                # Try adding each predicate
                for pred in available_predicates:
                    new_anchor = Anchor(
                        predicates=anchor.predicates + [pred],
                        precision=0.0,
                        coverage=0.0,
                        target_class=target_class
                    )
                    candidate_anchors.append(new_anchor)

            if not candidate_anchors:
                break

            # Evaluate precision and coverage
            for anchor in candidate_anchors:
                precision, coverage = self._evaluate_anchor(
                    anchor, input_signal, target_class, n_samples
                )
                anchor.precision = precision
                anchor.coverage = coverage

            # Keep top beam_size anchors by precision
            candidate_anchors.sort(key=lambda a: a.precision, reverse=True)
            current_anchors = candidate_anchors[:self.beam_size]

            if verbose:
                best = current_anchors[0]
                print(f"    Best anchor: {len(best.predicates)} predicates, "
                      f"precision={best.precision:.3f}, coverage={best.coverage:.3f}")

            # Stop if best anchor meets threshold
            if current_anchors[0].precision >= self.precision_threshold:
                if verbose:
                    print(f"  ✓ Precision threshold reached!")
                break

        # Return best anchor
        best_anchor = max(current_anchors, key=lambda a: a.precision)

        if verbose:
            print(f"\n  Final anchor:")
            print(f"    {best_anchor}")

        return best_anchor

    def _generate_candidate_predicates(
        self,
        features: np.ndarray,
        n_splits: int = 3
    ) -> List[Predicate]:
        """
        Generate candidate predicates based on feature quartiles.

        For each feature, create predicates:
        - feature < q1
        - feature > q3
        - q1 <= feature <= q3

        Args:
            features: Feature vector
            n_splits: Number of quantile splits

        Returns:
            List of candidate predicates
        """
        predicates = []

        for i, (feat_val, feat_name) in enumerate(zip(features, self.feature_names)):
            # Create range predicate around current value
            # This anchors the feature to be "similar" to current value

            # Simple strategy: create predicates at ±10%, ±25% of feature value
            for factor in [0.1, 0.25, 0.5]:
                if abs(feat_val) < 1e-6:
                    # Near-zero features
                    predicates.append(Predicate(
                        feature_name=feat_name,
                        feature_idx=i,
                        operator='in_range',
                        threshold=-factor,
                        threshold_max=factor
                    ))
                else:
                    delta = abs(feat_val) * factor
                    predicates.append(Predicate(
                        feature_name=feat_name,
                        feature_idx=i,
                        operator='in_range',
                        threshold=feat_val - delta,
                        threshold_max=feat_val + delta
                    ))

            # Also add simple threshold predicates
            predicates.append(Predicate(
                feature_name=feat_name,
                feature_idx=i,
                operator='>',
                threshold=feat_val * 0.9
            ))
            predicates.append(Predicate(
                feature_name=feat_name,
                feature_idx=i,
                operator='<',
                threshold=feat_val * 1.1
            ))

        return predicates

    def _evaluate_anchor(
        self,
        anchor: Anchor,
        original_signal: torch.Tensor,
        target_class: int,
        n_samples: int
    ) -> Tuple[float, float]:
        """
        Evaluate precision and coverage of anchor.

        Precision: P(f(x') = target_class | anchor(x') = True)
        Coverage: P(anchor(x') = True)

        Args:
            anchor: Anchor to evaluate
            original_signal: Original signal
            target_class: Target class
            n_samples: Number of perturbation samples

        Returns:
            (precision, coverage)
        """
        # Generate perturbed samples
        perturbed_signals = self._generate_perturbations(original_signal, n_samples)

        # Extract features for perturbed samples
        perturbed_features = []
        for signal in perturbed_signals:
            features = self.feature_extractor(signal.squeeze().cpu())
            perturbed_features.append(features)
        perturbed_features = np.array(perturbed_features)

        # Check which samples satisfy anchor
        satisfies_anchor = np.array([anchor.evaluate(feat) for feat in perturbed_features])

        # Coverage
        coverage = satisfies_anchor.mean()

        if coverage == 0:
            return 0.0, 0.0

        # Predict for samples satisfying anchor
        satisfying_signals = perturbed_signals[satisfies_anchor]

        with torch.no_grad():
            # Batch prediction
            batch_size = 32
            predictions = []

            for i in range(0, len(satisfying_signals), batch_size):
                batch = satisfying_signals[i:i+batch_size].to(self.device)
                outputs = self.model(batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)

        predictions = np.array(predictions)

        # Precision
        precision = (predictions == target_class).mean()

        return precision, coverage

    def _generate_perturbations(
        self,
        signal: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """
        Generate perturbed versions of signal.

        Strategy: Add Gaussian noise with varying levels.

        Args:
            signal: Original signal [1, C, T]
            n_samples: Number of perturbations

        Returns:
            Perturbed signals [n_samples, C, T]
        """
        signal = signal.squeeze()
        std = signal.std()

        perturbed = []

        for _ in range(n_samples):
            # Random noise level
            noise_level = np.random.uniform(0.05, 0.3) * std
            noise = torch.randn_like(signal) * noise_level
            perturbed_signal = signal + noise
            perturbed.append(perturbed_signal)

        return torch.stack(perturbed)


def plot_anchor_explanation(
    anchor: Anchor,
    original_features: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize anchor as bar chart showing which features are anchored.

    Args:
        anchor: Anchor object
        original_features: Original feature values
        feature_names: Feature names
        save_path: Save path
    """
    # Determine which features are in anchor
    anchored_indices = {pred.feature_idx for pred in anchor.predicates}

    colors = ['green' if i in anchored_indices else 'gray' for i in range(len(feature_names))]
    alphas = [0.8 if i in anchored_indices else 0.3 for i in range(len(feature_names))]

    # Plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(feature_names) * 0.3)))

    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, original_features, color=colors, alpha=alphas, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel('Feature Value', fontsize=11)
    ax.set_title(f'Anchor Explanation\n{anchor}', fontsize=12, fontweight='bold')

    # Add predicate annotations
    for pred in anchor.predicates:
        idx = pred.feature_idx
        ax.text(
            original_features[idx], idx,
            f' {pred.operator} {pred.threshold:.2f}',
            va='center', fontsize=8, fontweight='bold'
        )

    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Anchor plot saved to {save_path}")

    plt.show()


def compare_anchors(
    anchors: List[Anchor],
    save_path: Optional[str] = None
):
    """
    Compare multiple anchors (e.g., for different classes).

    Args:
        anchors: List of Anchor objects
        save_path: Save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Precision and Coverage
    classes = [a.target_class for a in anchors]
    precisions = [a.precision for a in anchors]
    coverages = [a.coverage for a in anchors]
    n_predicates = [len(a.predicates) for a in anchors]

    x = np.arange(len(anchors))
    width = 0.35

    ax1.bar(x - width/2, precisions, width, label='Precision', color='green', alpha=0.7)
    ax1.bar(x + width/2, coverages, width, label='Coverage', color='blue', alpha=0.7)

    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Anchor Quality Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Class {c}" for c in classes])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Anchor Complexity
    ax2.bar(x, n_predicates, color='purple', alpha=0.7)
    ax2.set_xlabel('Class', fontsize=11)
    ax2.set_ylabel('Number of Predicates', fontsize=11)
    ax2.set_title('Anchor Complexity', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Class {c}" for c in classes])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Test anchor generation
    print("=" * 60)
    print("Anchors - Validation")
    print("=" * 60)

    from models.cnn.cnn_1d import CNN1D

    # Create model
    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1, dropout=0.3)

    # Simple feature extractor (time-domain statistics)
    def extract_features(signal):
        """Extract basic statistical features from signal."""
        if isinstance(signal, torch.Tensor):
            signal = signal.numpy()

        features = [
            signal.mean(),
            signal.std(),
            signal.max(),
            signal.min(),
            np.median(signal),
            scipy_stats.skew(signal.flatten()),
            scipy_stats.kurtosis(signal.flatten()),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            (signal > 0).mean()  # Fraction positive
        ]
        return np.array(features)

    feature_names = [
        'mean', 'std', 'max', 'min', 'median',
        'skewness', 'kurtosis', 'q25', 'q75', 'frac_positive'
    ]

    # Generate test signal
    signal = torch.randn(1, 1, 10240)

    print("\n1. Extracting features...")
    features = extract_features(signal.squeeze())
    print(f"  Features: {features}")

    print("\n2. Generating anchor...")
    explainer = AnchorExplainer(
        model=model,
        feature_extractor=extract_features,
        feature_names=feature_names,
        device='cpu',
        precision_threshold=0.90,
        beam_size=3,
        max_predicates=3
    )

    anchor = explainer.explain(signal, n_samples=100, verbose=True)

    print("\n3. Final Anchor:")
    print(f"  {anchor}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
