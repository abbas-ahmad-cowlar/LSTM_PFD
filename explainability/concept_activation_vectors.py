"""
Concept Activation Vectors (CAVs) and Testing with CAVs (TCAV)

Implements concept-based explanations for neural networks. CAVs provide
human-interpretable explanations by measuring model sensitivity to
user-defined concepts rather than individual features.

Key Idea:
- Define concepts using example sets (e.g., "high frequency vibration")
- Learn CAV as normal vector to hyperplane separating concept from random examples
- Compute TCAV score: proportion of samples where concept positively influences prediction
- Provides global explanations: "X% of fault predictions are positively influenced by concept C"

Reference:
Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018).
Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation
Vectors (TCAV). ICML.

Implementation Notes:
- Trains linear classifiers (SVM/Logistic Regression) on layer activations
- Supports multiple layers for hierarchical concept analysis
- Includes statistical significance testing
- Provides both directional derivatives and TCAV scores
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


class ConceptActivationVector:
    """
    Represents a Concept Activation Vector (CAV) for a specific concept.

    A CAV is the normal vector to the hyperplane that separates concept
    examples from random examples in the activation space of a layer.
    """

    def __init__(
        self,
        concept_name: str,
        layer_name: str,
        vector: np.ndarray,
        accuracy: float,
        classifier: any
    ):
        """
        Initialize CAV.

        Args:
            concept_name: Human-readable name of concept
            layer_name: Name of layer where CAV was computed
            vector: CAV vector (normal to decision boundary)
            accuracy: Classifier accuracy on test set
            classifier: Trained classifier
        """
        self.concept_name = concept_name
        self.layer_name = layer_name
        self.vector = vector
        self.accuracy = accuracy
        self.classifier = classifier

    def __repr__(self):
        return (f"CAV(concept='{self.concept_name}', layer='{self.layer_name}', "
                f"accuracy={self.accuracy:.3f})")


class CAVGenerator:
    """
    Generates Concept Activation Vectors for neural networks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        classifier_type: str = 'linear_svm'
    ):
        """
        Initialize CAV generator.

        Args:
            model: PyTorch model to analyze
            device: Device to run on
            classifier_type: Type of classifier ('linear_svm' or 'logistic')
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.classifier_type = classifier_type

        # Store layer names
        self.layer_names = self._get_layer_names()

    def _get_layer_names(self) -> List[str]:
        """Get names of all intermediate layers."""
        names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LSTM, nn.GRU)):
                if name:  # Skip unnamed modules
                    names.append(name)
        return names

    def generate_cav(
        self,
        concept_examples: torch.Tensor,
        random_examples: torch.Tensor,
        layer_name: str,
        concept_name: str = "concept",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> ConceptActivationVector:
        """
        Generate CAV for a concept at a specific layer.

        Args:
            concept_examples: Examples of concept [N_concept, C, T]
            random_examples: Random/negative examples [N_random, C, T]
            layer_name: Name of layer to compute CAV at
            concept_name: Human-readable concept name
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            ConceptActivationVector object
        """
        # Get activations for both sets
        print(f"  Computing activations at layer '{layer_name}'...")
        concept_acts = self._get_activations(concept_examples, layer_name)
        random_acts = self._get_activations(random_examples, layer_name)

        # Flatten activations to [N, Features]
        concept_acts = concept_acts.reshape(concept_acts.shape[0], -1)
        random_acts = random_acts.reshape(random_acts.shape[0], -1)

        # Create labels (1 for concept, 0 for random)
        X = np.vstack([concept_acts, random_acts])
        y = np.hstack([np.ones(len(concept_acts)), np.zeros(len(random_acts))])

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train classifier
        print(f"  Training {self.classifier_type} classifier...")
        if self.classifier_type == 'linear_svm':
            clf = LinearSVC(random_state=random_state, max_iter=5000, dual='auto')
        elif self.classifier_type == 'logistic':
            clf = LogisticRegression(random_state=random_state, max_iter=5000)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Get CAV (normal vector to decision boundary)
        cav_vector = clf.coef_[0]
        cav_vector = cav_vector / (np.linalg.norm(cav_vector) + 1e-8)  # Normalize

        print(f"  ✓ CAV generated (accuracy: {accuracy:.3f})")

        return ConceptActivationVector(
            concept_name=concept_name,
            layer_name=layer_name,
            vector=cav_vector,
            accuracy=accuracy,
            classifier=clf
        )

    def _get_activations(
        self,
        inputs: torch.Tensor,
        layer_name: str
    ) -> np.ndarray:
        """
        Get activations at a specific layer.

        Args:
            inputs: Input batch [B, C, T]
            layer_name: Layer name

        Returns:
            Activations as numpy array
        """
        activations = []

        # Hook to capture activations
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())

        # Register hook
        target_layer = dict(self.model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)

        try:
            # Forward pass in batches
            batch_size = 32
            with torch.no_grad():
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i+batch_size].to(self.device)
                    _ = self.model(batch)

            # Concatenate activations
            all_acts = torch.cat(activations, dim=0).numpy()

        finally:
            handle.remove()

        return all_acts


class TCAVAnalyzer:
    """
    Performs TCAV (Testing with Concept Activation Vectors) analysis.

    TCAV measures the proportion of examples in a class where a concept
    positively influences the prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        cav: ConceptActivationVector,
        device: str = 'cuda'
    ):
        """
        Initialize TCAV analyzer.

        Args:
            model: PyTorch model
            cav: Concept Activation Vector
            device: Device
        """
        self.model = model.to(device)
        self.cav = cav
        self.device = device
        self.model.eval()

    def compute_tcav_score(
        self,
        test_examples: torch.Tensor,
        target_class: int,
        n_random_runs: int = 10
    ) -> Dict[str, float]:
        """
        Compute TCAV score for a class.

        TCAV score = fraction of examples where directional derivative
        (gradient in CAV direction) is positive.

        Args:
            test_examples: Test examples [N, C, T]
            target_class: Target class to analyze
            n_random_runs: Number of random CAVs for statistical testing

        Returns:
            Dictionary with TCAV score and p-value
        """
        # Compute directional derivatives for concept CAV
        print(f"  Computing directional derivatives for concept '{self.cav.concept_name}'...")
        concept_derivatives = self._compute_directional_derivatives(
            test_examples, target_class, self.cav
        )

        # TCAV score: fraction of positive derivatives
        tcav_score = (concept_derivatives > 0).mean()

        # Statistical significance testing with random CAVs
        print(f"  Running statistical test with {n_random_runs} random CAVs...")
        random_scores = []

        for i in range(n_random_runs):
            # Create random CAV
            random_vector = np.random.randn(len(self.cav.vector))
            random_vector = random_vector / (np.linalg.norm(random_vector) + 1e-8)

            random_cav = ConceptActivationVector(
                concept_name=f"random_{i}",
                layer_name=self.cav.layer_name,
                vector=random_vector,
                accuracy=0.5,
                classifier=None
            )

            # Compute derivatives for random CAV
            random_derivatives = self._compute_directional_derivatives(
                test_examples, target_class, random_cav
            )
            random_score = (random_derivatives > 0).mean()
            random_scores.append(random_score)

        # Two-tailed p-value
        random_scores = np.array(random_scores)
        p_value = (np.abs(random_scores - 0.5) >= np.abs(tcav_score - 0.5)).mean()

        print(f"  ✓ TCAV score: {tcav_score:.3f} (p={p_value:.3f})")

        return {
            'tcav_score': tcav_score,
            'p_value': p_value,
            'concept_name': self.cav.concept_name,
            'layer_name': self.cav.layer_name,
            'target_class': target_class,
            'n_examples': len(test_examples),
            'random_mean': random_scores.mean(),
            'random_std': random_scores.std()
        }

    def _compute_directional_derivatives(
        self,
        inputs: torch.Tensor,
        target_class: int,
        cav: ConceptActivationVector
    ) -> np.ndarray:
        """
        Compute directional derivatives: ∂y_c/∂h · v_CAV

        where y_c is logit for target class, h is activation, v_CAV is CAV vector.

        Args:
            inputs: Input examples [N, C, T]
            target_class: Target class
            cav: CAV to use

        Returns:
            Directional derivatives [N]
        """
        derivatives = []

        # Get target layer
        target_layer = dict(self.model.named_modules())[cav.layer_name]

        for input_sample in inputs:
            input_sample = input_sample.unsqueeze(0).to(self.device)
            input_sample.requires_grad = False

            # Hook to get gradients w.r.t. activations
            activation_grads = []

            def backward_hook(module, grad_input, grad_output):
                activation_grads.append(grad_output[0].detach().cpu())

            handle = target_layer.register_full_backward_hook(backward_hook)

            try:
                # Forward pass
                output = self.model(input_sample)

                # Backward pass
                self.model.zero_grad()
                output[0, target_class].backward()

                # Get gradient w.r.t. activations
                grad = activation_grads[0].numpy().flatten()

                # Directional derivative: gradient · CAV
                directional_deriv = np.dot(grad, cav.vector)
                derivatives.append(directional_deriv)

            finally:
                handle.remove()

        return np.array(derivatives)


def plot_tcav_results(
    tcav_results: List[Dict],
    save_path: Optional[str] = None
):
    """
    Visualize TCAV results for multiple concepts.

    Args:
        tcav_results: List of TCAV result dictionaries
        save_path: Path to save figure
    """
    n_concepts = len(tcav_results)

    fig, ax = plt.subplots(figsize=(10, max(6, n_concepts * 0.5)))

    concept_names = [r['concept_name'] for r in tcav_results]
    tcav_scores = [r['tcav_score'] for r in tcav_results]
    p_values = [r['p_value'] for r in tcav_results]

    # Color by significance
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]

    y_pos = np.arange(n_concepts)
    bars = ax.barh(y_pos, tcav_scores, color=colors, alpha=0.7, edgecolor='black')

    # Add significance markers
    for i, (score, p) in enumerate(zip(tcav_scores, p_values)):
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(score + 0.02, i, f'{marker} (p={p:.3f})', va='center', fontsize=9)

    # Add baseline (random)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Random baseline', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlabel('TCAV Score (Fraction of Positive Influence)', fontsize=11)
    ax.set_title('Concept Influence on Model Predictions\n(TCAV Scores)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"TCAV results saved to {save_path}")

    plt.show()


def plot_cav_comparison(
    cavs: List[ConceptActivationVector],
    save_path: Optional[str] = None
):
    """
    Compare CAV accuracies across concepts and layers.

    Args:
        cavs: List of CAV objects
        save_path: Save path
    """
    # Group by layer
    layers = list(set(cav.layer_name for cav in cavs))
    concepts = list(set(cav.concept_name for cav in cavs))

    # Create accuracy matrix
    accuracy_matrix = np.zeros((len(concepts), len(layers)))

    for i, concept in enumerate(concepts):
        for j, layer in enumerate(layers):
            matching_cavs = [cav for cav in cavs
                           if cav.concept_name == concept and cav.layer_name == layer]
            if matching_cavs:
                accuracy_matrix[i, j] = matching_cavs[0].accuracy
            else:
                accuracy_matrix[i, j] = np.nan

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.8), max(6, len(concepts) * 0.5)))

    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(concepts)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_yticklabels(concepts)

    # Add text annotations
    for i in range(len(concepts)):
        for j in range(len(layers)):
            if not np.isnan(accuracy_matrix[i, j]):
                text = ax.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black', fontsize=9)

    ax.set_title('CAV Classifier Accuracy by Concept and Layer', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Concept', fontsize=11)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Test CAV generation
    print("=" * 60)
    print("Concept Activation Vectors (CAVs) - Validation")
    print("=" * 60)

    from models.cnn.cnn_1d import CNN1D
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

    # Create model
    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1, dropout=0.3)

    # Generate synthetic concept examples
    # Concept: "high frequency content"
    print("\n1. Generating synthetic concept examples...")
    n_concept = 100
    n_random = 100

    # High frequency signals (concept)
    t = np.linspace(0, 1, 10240)
    concept_examples = []
    for _ in range(n_concept):
        freq = np.random.uniform(80, 120)  # High frequency
        signal = np.sin(2 * np.pi * freq * t) + np.random.randn(10240) * 0.1
        concept_examples.append(signal)
    concept_examples = torch.tensor(np.array(concept_examples), dtype=torch.float32).unsqueeze(1)

    # Random/low frequency signals
    random_examples = []
    for _ in range(n_random):
        freq = np.random.uniform(5, 20)  # Low frequency
        signal = np.sin(2 * np.pi * freq * t) + np.random.randn(10240) * 0.1
        random_examples.append(signal)
    random_examples = torch.tensor(np.array(random_examples), dtype=torch.float32).unsqueeze(1)

    print(f"  Concept examples: {concept_examples.shape}")
    print(f"  Random examples: {random_examples.shape}")

    # Generate CAV
    print("\n2. Generating CAV...")
    generator = CAVGenerator(model, device='cpu', classifier_type='linear_svm')

    # Get first conv layer name
    layer_name = generator.layer_names[0] if generator.layer_names else 'conv1'
    print(f"  Using layer: {layer_name}")

    cav = generator.generate_cav(
        concept_examples=concept_examples,
        random_examples=random_examples,
        layer_name=layer_name,
        concept_name="high_frequency"
    )

    print(f"\n  {cav}")
    print(f"  CAV vector shape: {cav.vector.shape}")
    print(f"  CAV vector norm: {np.linalg.norm(cav.vector):.4f}")

    # TCAV analysis
    print("\n3. Computing TCAV score...")
    analyzer = TCAVAnalyzer(model, cav, device='cpu')

    # Generate test examples (more high frequency signals)
    test_examples = torch.randn(50, 1, 10240)

    tcav_result = analyzer.compute_tcav_score(
        test_examples=test_examples,
        target_class=3,
        n_random_runs=5  # Reduced for speed
    )

    print(f"\n  TCAV Results:")
    print(f"    Score: {tcav_result['tcav_score']:.3f}")
    print(f"    P-value: {tcav_result['p_value']:.3f}")
    print(f"    Significant: {tcav_result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
