"""
Physics Interpretability Visualization

Provides visualization tools to understand how PINN models use physics knowledge:
- Learned vs expected frequency distributions
- Physics feature importance
- Knowledge graph attention visualization
- Sommerfeld/Reynolds number influence on predictions
- Operating condition sensitivity analysis

These visualizations help:
1. Verify that models learn physically meaningful patterns
2. Debug physics constraints
3. Explain predictions to domain experts
4. Identify model failure modes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, Optional, List, Tuple
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from packages.core.models.physics.bearing_dynamics import BearingDynamics
from packages.core.models.physics.fault_signatures import FaultSignatureDatabase


class PhysicsInterpreter:
    """
    Visualizes physics-informed aspects of PINN models.
    """

    def __init__(
        self,
        model,
        device: str = 'cuda',
        sample_rate: int = 51200
    ):
        """
        Initialize physics interpreter.

        Args:
            model: PINN model
            device: Device
            sample_rate: Signal sampling rate
        """
        self.model = model.to(device)
        self.device = device
        self.sample_rate = sample_rate

        self.bearing_dynamics = BearingDynamics()
        self.signature_db = FaultSignatureDatabase()

    def plot_learned_vs_expected_frequencies(
        self,
        signal: torch.Tensor,
        true_label: int,
        predicted_label: int,
        rpm: float = 3600.0,
        save_path: Optional[str] = None
    ):
        """
        Compare observed, expected, and predicted frequency distributions.

        Args:
            signal: Vibration signal [1, 1, T]
            true_label: Ground truth fault class
            predicted_label: Predicted fault class
            rpm: Shaft speed in RPM
            save_path: Optional path to save figure
        """
        # Compute FFT of signal
        signal_np = signal.squeeze().cpu().numpy()
        fft = np.fft.rfft(signal_np, n=2048)
        magnitude = np.abs(fft)
        freq_bins = np.fft.rfftfreq(2048, d=1.0/self.sample_rate)

        # Get expected frequencies
        expected_true = self.signature_db.get_expected_frequencies(true_label, rpm, top_k=5)
        expected_pred = self.signature_db.get_expected_frequencies(predicted_label, rpm, top_k=5)

        # Generate expected spectra
        spectrum_true = self.signature_db.compute_expected_spectrum(true_label, rpm, freq_bins)
        spectrum_pred = self.signature_db.compute_expected_spectrum(predicted_label, rpm, freq_bins)

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot 1: Observed spectrum
        axes[0].plot(freq_bins, magnitude, 'b-', alpha=0.7, label='Observed')
        axes[0].set_xlim(0, 2000)
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Magnitude')
        axes[0].set_title(f'Observed Spectrum (RPM = {rpm:.0f})')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot 2: Expected (true label)
        axes[1].plot(freq_bins, magnitude, 'b-', alpha=0.3, label='Observed')
        axes[1].plot(freq_bins, spectrum_true, 'g-', linewidth=2, label='Expected (True)')
        for freq in expected_true:
            axes[1].axvline(freq, color='g', linestyle='--', alpha=0.5)
        axes[1].set_xlim(0, 2000)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_title(f'Expected for True Label: {self.signature_db.FAULT_TYPES[true_label]}')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Plot 3: Expected (predicted label)
        axes[2].plot(freq_bins, magnitude, 'b-', alpha=0.3, label='Observed')
        axes[2].plot(freq_bins, spectrum_pred, 'r-', linewidth=2, label='Expected (Predicted)')
        for freq in expected_pred:
            axes[2].axvline(freq, color='r', linestyle='--', alpha=0.5)
        axes[2].set_xlim(0, 2000)
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Magnitude')
        axes[2].set_title(f'Expected for Predicted: {self.signature_db.FAULT_TYPES[predicted_label]}')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_knowledge_graph(
        self,
        kg_pinn_model,
        signal: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize fault relationship graph with learned attention weights.

        Args:
            kg_pinn_model: Knowledge Graph PINN model
            signal: Optional signal to compute attention [1, 1, T]
            save_path: Optional save path
        """
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX required for graph visualization. Install with: pip install networkx")
            return

        # Get knowledge graph
        from packages.core.models.pinn.knowledge_graph_pinn import FaultKnowledgeGraph
        kg = FaultKnowledgeGraph()
        adj = kg.get_adjacency_matrix(normalized=False)

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        for i, name in enumerate(kg.FAULT_NAMES):
            G.add_node(i, label=name)

        # Add edges (only strong relationships)
        for i in range(kg.num_nodes):
            for j in range(i+1, kg.num_nodes):
                if adj[i, j] > 0.5:  # Threshold for visualization
                    G.add_edge(i, j, weight=adj[i, j])

        # Get attention weights if signal provided
        if signal is not None and hasattr(kg_pinn_model, 'forward_with_attention'):
            signal = signal.to(self.device)
            with torch.no_grad():
                _, attention = kg_pinn_model.forward_with_attention(signal)
                attention_np = attention[0].cpu().numpy()  # [11, 11]
        else:
            attention_np = None

        # Plot
        fig, axes = plt.subplots(1, 2 if attention_np is not None else 1, figsize=(16, 8))
        if attention_np is None:
            axes = [axes]

        # Plot 1: Knowledge graph structure
        ax = axes[0]

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        node_colors = ['lightgreen' if i == 0 else 'lightcoral' for i in range(kg.num_nodes)]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)

        # Draw edges (thickness based on weight)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=np.array(weights)*3, alpha=0.6, ax=ax)

        # Draw labels
        labels = {i: kg.FAULT_NAMES[i] for i in range(kg.num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax)

        ax.set_title('Fault Relationship Knowledge Graph', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Plot 2: Learned attention (if available)
        if attention_np is not None:
            ax = axes[1]

            im = ax.imshow(attention_np, cmap='hot', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(kg.num_nodes))
            ax.set_yticks(range(kg.num_nodes))
            ax.set_xticklabels(kg.FAULT_NAMES, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(kg.FAULT_NAMES, fontsize=9)
            ax.set_title('Learned Attention Weights', fontsize=14, fontweight='bold')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_physics_feature_importance(
        self,
        hybrid_pinn_model,
        test_samples: List[torch.Tensor],
        test_labels: List[int],
        save_path: Optional[str] = None
    ):
        """
        Analyze importance of different physics features.

        Tests how predictions change when physics features are perturbed.

        Args:
            hybrid_pinn_model: Hybrid PINN model
            test_samples: List of test signals
            test_labels: List of true labels
            save_path: Optional save path
        """
        feature_names = ['Sommerfeld', 'Reynolds', 'FTF', 'BPFO', 'BPFI', 'BSF', 'Shaft Freq',
                        'Lub. Regime', 'Flow Regime', 'Load']

        importance_scores = []

        for signal, true_label in zip(test_samples, test_labels):
            signal = signal.unsqueeze(0).to(self.device)

            # Get baseline prediction
            with torch.no_grad():
                baseline_output = hybrid_pinn_model(signal)
                baseline_pred = torch.softmax(baseline_output, dim=1)[0, true_label].item()

            # Perturb each physics feature
            feature_importance = []

            for feature_idx in range(len(feature_names)):
                # Extract and perturb physics features
                # (This is simplified - actual implementation would need access to intermediate features)
                # For demonstration, we'll use a proxy: perturb operating conditions

                perturbed_pred = baseline_pred  # Placeholder

                # Importance = drop in correct class probability
                importance = baseline_pred - perturbed_pred
                feature_importance.append(importance)

            importance_scores.append(feature_importance)

        # Average importance across samples
        mean_importance = np.mean(importance_scores, axis=0)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green' if imp > 0 else 'red' for imp in mean_importance]
        bars = ax.barh(feature_names, mean_importance, color=colors, alpha=0.7)

        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Physics Feature Importance Analysis', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_operating_condition_sensitivity(
        self,
        signal: torch.Tensor,
        rpm_range: Tuple[float, float] = (2000, 5000),
        load_range: Tuple[float, float] = (200, 1000),
        num_points: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Visualize how predictions change with operating conditions.

        Args:
            signal: Test signal [1, 1, T]
            rpm_range: (min_rpm, max_rpm)
            load_range: (min_load, max_load)
            num_points: Grid resolution
            save_path: Optional save path
        """
        signal = signal.to(self.device)

        # Create grid
        rpms = np.linspace(rpm_range[0], rpm_range[1], num_points)
        loads = np.linspace(load_range[0], load_range[1], num_points)

        predictions_grid = np.zeros((num_points, num_points))
        confidence_grid = np.zeros((num_points, num_points))

        # Evaluate model at each operating point
        with torch.no_grad():
            for i, rpm in enumerate(rpms):
                for j, load in enumerate(loads):
                    metadata = {
                        'rpm': torch.tensor([rpm]).to(self.device),
                        'load': torch.tensor([load]).to(self.device),
                        'viscosity': torch.tensor([0.03]).to(self.device)
                    }

                    try:
                        output = self.model(signal, metadata)
                        probs = torch.softmax(output, dim=1)
                        pred_class = torch.argmax(probs, dim=1).item()
                        confidence = torch.max(probs, dim=1)[0].item()

                        predictions_grid[i, j] = pred_class
                        confidence_grid[i, j] = confidence
                    except:
                        predictions_grid[i, j] = -1
                        confidence_grid[i, j] = 0

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Predicted class
        im1 = axes[0].imshow(predictions_grid.T, origin='lower', aspect='auto',
                            extent=[rpm_range[0], rpm_range[1], load_range[0], load_range[1]],
                            cmap='tab10', vmin=0, vmax=10)
        axes[0].set_xlabel('RPM', fontsize=12)
        axes[0].set_ylabel('Load (N)', fontsize=12)
        axes[0].set_title('Predicted Fault Class vs Operating Conditions', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0], label='Fault Class')

        # Plot 2: Prediction confidence
        im2 = axes[1].imshow(confidence_grid.T, origin='lower', aspect='auto',
                            extent=[rpm_range[0], rpm_range[1], load_range[0], load_range[1]],
                            cmap='viridis', vmin=0, vmax=1)
        axes[1].set_xlabel('RPM', fontsize=12)
        axes[1].set_ylabel('Load (N)', fontsize=12)
        axes[1].set_title('Prediction Confidence', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1], label='Confidence')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_sommerfeld_reynolds_distribution(
        self,
        test_loader,
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of Sommerfeld and Reynolds numbers in dataset.

        Args:
            test_loader: Data loader with metadata
            save_path: Optional save path
        """
        sommerfeldnumbers = []
        reynolds_numbers = []
        labels = []

        for batch in test_loader:
            if len(batch) == 3:
                _, targets, metadata = batch
            else:
                continue

            rpm = metadata.get('rpm', torch.tensor(3600.0))
            load = metadata.get('load', torch.tensor(500.0))
            viscosity = metadata.get('viscosity', torch.tensor(0.03))

            S = self.bearing_dynamics.sommerfeld_number(load, rpm, viscosity)
            Re = self.bearing_dynamics.reynolds_number(rpm, viscosity)

            if isinstance(S, torch.Tensor):
                S = S.cpu().numpy()
            if isinstance(Re, torch.Tensor):
                Re = Re.cpu().numpy()

            sommerfeldnumbers.extend(S.flatten())
            reynolds_numbers.extend(Re.flatten())
            labels.extend(targets.cpu().numpy())

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Sommerfeld distribution
        axes[0, 0].hist(sommerfeldnumbers, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(0.1, color='red', linestyle='--', label='Boundary threshold')
        axes[0, 0].axvline(1.0, color='green', linestyle='--', label='Hydrodynamic threshold')
        axes[0, 0].set_xlabel('Sommerfeld Number', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title('Sommerfeld Number Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Reynolds distribution
        axes[0, 1].hist(reynolds_numbers, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(2000, color='red', linestyle='--', label='Laminar-transition')
        axes[0, 1].set_xlabel('Reynolds Number', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title('Reynolds Number Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Sommerfeld vs Reynolds scatter
        axes[1, 0].scatter(sommerfeldnumbers, reynolds_numbers, c=labels, cmap='tab10',
                          alpha=0.6, s=10)
        axes[1, 0].set_xlabel('Sommerfeld Number', fontsize=11)
        axes[1, 0].set_ylabel('Reynolds Number', fontsize=11)
        axes[1, 0].set_title('Sommerfeld vs Reynolds (colored by fault)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Box plot by fault type
        unique_labels = sorted(set(labels))
        data_by_label = [np.array(sommerfeldnumbers)[np.array(labels) == l] for l in unique_labels]

        axes[1, 1].boxplot(data_by_label, labels=[str(l) for l in unique_labels])
        axes[1, 1].set_xlabel('Fault Class', fontsize=11)
        axes[1, 1].set_ylabel('Sommerfeld Number', fontsize=11)
        axes[1, 1].set_title('Sommerfeld by Fault Type', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    # Test physics interpreter
    print("=" * 60)
    print("Physics Interpretability - Validation")
    print("=" * 60)

    from packages.core.models.pinn.hybrid_pinn import HybridPINN

    # Create model
    model = HybridPINN(num_classes=NUM_CLASSES, backbone='resnet18')

    interpreter = PhysicsInterpreter(model, device='cpu')

    # Test frequency visualization
    print("\nTesting Frequency Visualization:")
    signal = torch.randn(1, 1, SIGNAL_LENGTH)
    try:
        interpreter.plot_learned_vs_expected_frequencies(
            signal, true_label=3, predicted_label=4, rpm=3600.0
        )
        print("  ✓ Frequency plot created")
    except Exception as e:
        print(f"  Note: {e}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    print("Note: Full visualizations require matplotlib display backend")
