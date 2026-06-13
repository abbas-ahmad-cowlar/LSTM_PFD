"""
Physics-Constrained CNN

Standard CNN architecture with physics-based loss constraints.
Unlike Hybrid PINN, this doesn't use physics features as input, but enforces
physics constraints through the loss function during training.

This is simpler than Hybrid PINN but still benefits from physics knowledge
through the training objective.

Architecture:
    Input: Signal [B, 1, 102400]
    ├─ Standard CNN/ResNet
    └─ Output: [B, 11]

    Loss = CrossEntropy + λ_physics * PhysicsLoss
           where PhysicsLoss checks frequency consistency

Expected Performance: 96-97% accuracy
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


from packages.core.models.base_model import BaseModel
from packages.core.models.resnet.resnet_1d import ResNet1D
from packages.core.models.cnn.cnn_1d import CNN1D
from packages.core.models.physics.bearing_dynamics import BearingDynamics
from packages.core.models.physics.fault_signatures import FaultSignatureDatabase


class PhysicsConstrainedCNN(BaseModel):
    """
    CNN with physics-based loss constraints.

    This model uses a standard CNN architecture but is trained with additional
    physics-based loss terms that penalize predictions inconsistent with
    expected fault frequencies and bearing dynamics.

    Key difference from standard CNN:
    - Training: Includes physics loss (frequency consistency, etc.)
    - Inference: Same as standard CNN (no metadata required)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_length: int = SIGNAL_LENGTH,
        backbone: str = 'resnet18',
        dropout: float = 0.3,
        sample_rate: int = 20480,
        bearing_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Physics-Constrained CNN.

        Args:
            num_classes: Number of fault classes (default: 11)
            input_length: Signal length (default: 102400)
            backbone: CNN backbone ('resnet18', 'resnet34', 'cnn1d')
            dropout: Dropout probability
            sample_rate: Sampling rate for physics loss computation
            bearing_params: Optional bearing parameters
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_length = input_length
        self.backbone_name = backbone
        self.dropout_prob = dropout
        self.sample_rate = sample_rate

        # Initialize physics models (for loss computation)
        self.bearing_dynamics = BearingDynamics(bearing_params)
        self.signature_db = FaultSignatureDatabase()

        # ===== CNN BACKBONE =====
        if backbone == 'resnet18':
            self.backbone = ResNet1D(
                num_classes=num_classes,
                input_channels=1,
                layers=[2, 2, 2, 2],
                dropout=dropout,
                input_length=input_length
            )

        elif backbone == 'resnet34':
            self.backbone = ResNet1D(
                num_classes=num_classes,
                input_channels=1,
                layers=[3, 4, 6, 3],
                dropout=dropout,
                input_length=input_length
            )

        elif backbone == 'cnn1d':
            self.backbone = CNN1D(
                num_classes=num_classes,
                input_channels=1,
                dropout=dropout
            )

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            signal: Input vibration signal [B, 1, T]

        Returns:
            Class logits [B, num_classes]
        """
        return self.backbone(signal)

    def compute_physics_loss(
        self,
        signal: torch.Tensor,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
        n_fft: int = 2048,
        top_k: int = 5,
        tolerance: float = 0.15
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-based loss term.

        This function checks if the predicted fault class is consistent with
        the dominant frequencies observed in the signal FFT.

        Args:
            signal: Input signal [B, 1, T] or [B, T]
            predictions: Predicted class logits [B, num_classes]
            metadata: Optional dict with 'rpm' (default: 3600)
            n_fft: FFT size
            top_k: Number of frequencies to check
            tolerance: Frequency matching tolerance (fraction)

        Returns:
            physics_loss: Physics constraint loss (scalar)
            loss_dict: Dictionary with loss components
        """
        batch_size = signal.shape[0]
        device = signal.device

        # --------------------------------------------------------------
        # DIFFERENTIABLE physics-consistency loss (see
        # experiments/PHYSICS_LOSS_DIAGNOSIS.md §8.0-bis).
        #
        # The old implementation selected the target class with
        # torch.argmax(predictions) -- a non-differentiable op. Every term
        # downstream was then a function of the (constant) input-signal FFT
        # and a frequency-table lookup, so the loss had no gradient to the
        # model parameters (requires_grad=False) and the physics weight was
        # inert. We instead weight a per-CLASS frequency-mismatch penalty by
        # the model's softmax probabilities: the gradient flows
        #   loss -> probs -> logits -> params,
        # penalizing probability mass on classes whose characteristic
        # frequencies are absent from the signal's spectrum.
        # --------------------------------------------------------------

        # softmax class probabilities -- the ONLY parameter-dependent factor.
        probs = F.softmax(predictions, dim=1)  # [B, C], differentiable

        # Extract RPM (scalar, as in the original)
        rpm = 3600.0
        if metadata is not None and 'rpm' in metadata:
            rpm_val = metadata['rpm']
            if isinstance(rpm_val, torch.Tensor):
                rpm_val = rpm_val.detach().cpu().numpy()
            if hasattr(rpm_val, 'mean'):
                rpm = float(rpm_val.mean())
            else:
                rpm = float(rpm_val)

        # Observed spectral peaks per sample (constant w.r.t. params)
        if signal.dim() == 3:
            signal = signal.squeeze(1)
        magnitude = torch.abs(torch.fft.rfft(signal, n=n_fft, dim=-1))  # [B, F]
        freq_bins = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate).to(device)
        k = min(top_k * 2, magnitude.shape[-1])
        _, peak_indices = torch.topk(magnitude, k=k, dim=-1)  # [B, k]
        peak_freqs = freq_bins[peak_indices]                  # [B, k]

        # Per-class mismatch penalty pen[B, C] (no params -> constant tensor).
        num_classes = predictions.shape[1]
        pen = torch.zeros(batch_size, num_classes, device=device)
        for c in range(num_classes):
            try:
                exp = self.signature_db.get_expected_frequencies(c, rpm, top_k=top_k)
            except Exception:
                continue  # leave pen[:, c] = 0 if the lookup fails
            exp = torch.as_tensor(
                [float(f) for f in exp if f > 0], device=device, dtype=peak_freqs.dtype
            )
            if exp.numel() == 0:
                continue  # e.g. healthy class: no characteristic freqs -> no penalty
            # distance of each expected freq to the nearest observed peak, per sample
            rel_dist = torch.abs(
                peak_freqs.unsqueeze(1) - exp.view(1, -1, 1)
            ) / (exp.view(1, -1, 1) + 1e-6)          # [B, E, k]
            min_dist = rel_dist.min(dim=-1).values    # [B, E]
            pen[:, c] = F.relu(min_dist - tolerance).mean(dim=1)  # [B]

        # Differentiable: penalize probability on spectrally-inconsistent classes.
        freq_loss = (probs * pen).sum(dim=1).mean()

        loss_dict = {
            'frequency_consistency': float(freq_loss.detach()),
            'mean_penalty': float(pen.mean()),
        }

        return freq_loss, loss_dict

    def forward_with_physics_loss(
        self,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
        lambda_physics: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Forward pass with physics loss computation.

        Useful during training to compute both predictions and physics loss.

        Args:
            signal: Input signal [B, 1, T]
            metadata: Operating conditions
            lambda_physics: Weight for physics loss

        Returns:
            predictions: Class logits [B, num_classes]
            physics_loss: Physics constraint loss (scalar)
            loss_dict: Dictionary with loss components
        """
        # Standard forward pass
        predictions = self.forward(signal)

        # Compute physics loss
        physics_loss, loss_dict = self.compute_physics_loss(
            signal, predictions, metadata
        )

        # Weight physics loss
        physics_loss = lambda_physics * physics_loss

        return predictions, physics_loss, loss_dict

    def get_config(self) -> Dict[str, any]:
        """Get model configuration dictionary (satisfies BaseModel ABC)."""
        return {
            'model_type': 'PhysicsConstrainedCNN',
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'input_length': self.input_length,
            'sample_rate': self.sample_rate,
        }

    def get_model_info(self) -> Dict[str, any]:
        """Get model configuration and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            **self.get_config(),
            'total_params': total_params,
            'trainable_params': trainable_params
        }


class AdaptivePhysicsConstrainedCNN(PhysicsConstrainedCNN):
    """
    Physics-Constrained CNN with adaptive physics loss weighting.

    This variant automatically adjusts the physics loss weight during training
    based on model performance. Early training focuses on classification,
    later training emphasizes physics consistency.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_length: int = SIGNAL_LENGTH,
        backbone: str = 'resnet18',
        dropout: float = 0.3,
        sample_rate: int = 20480,
        bearing_params: Optional[Dict[str, float]] = None,
        lambda_physics_schedule: str = 'linear',  # 'linear', 'exponential', 'step'
        lambda_physics_max: float = 1.0
    ):
        """
        Initialize Adaptive Physics-Constrained CNN.

        Args:
            num_classes: Number of fault classes
            input_length: Signal length
            backbone: CNN backbone
            dropout: Dropout probability
            sample_rate: Sampling rate
            bearing_params: Bearing parameters
            lambda_physics_schedule: Schedule type for physics loss weight
            lambda_physics_max: Maximum physics loss weight
        """
        super().__init__(
            num_classes=num_classes,
            input_length=input_length,
            backbone=backbone,
            dropout=dropout,
            sample_rate=sample_rate,
            bearing_params=bearing_params
        )

        self.lambda_physics_schedule = lambda_physics_schedule
        self.lambda_physics_max = lambda_physics_max
        self.current_lambda_physics = 0.0

    def update_lambda_physics(self, epoch: int, max_epochs: int):
        """
        Update physics loss weight based on training progress.

        Args:
            epoch: Current epoch (0-indexed)
            max_epochs: Total number of epochs
        """
        progress = epoch / max_epochs

        if self.lambda_physics_schedule == 'linear':
            # Linearly increase from 0 to max
            self.current_lambda_physics = self.lambda_physics_max * progress

        elif self.lambda_physics_schedule == 'exponential':
            # Exponentially increase
            self.current_lambda_physics = self.lambda_physics_max * (1 - np.exp(-5 * progress))

        elif self.lambda_physics_schedule == 'step':
            # Step increase at 50% and 75%
            if progress < 0.5:
                self.current_lambda_physics = 0.0
            elif progress < 0.75:
                self.current_lambda_physics = self.lambda_physics_max * 0.5
            else:
                self.current_lambda_physics = self.lambda_physics_max

        else:
            # Default: constant
            self.current_lambda_physics = self.lambda_physics_max

    def forward_with_adaptive_physics_loss(
        self,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Forward pass with adaptive physics loss.

        Uses the current lambda_physics value (updated via update_lambda_physics).

        Args:
            signal: Input signal [B, 1, T]
            metadata: Operating conditions

        Returns:
            predictions: Class logits
            physics_loss: Weighted physics loss
            loss_dict: Loss components
        """
        return self.forward_with_physics_loss(
            signal, metadata, lambda_physics=self.current_lambda_physics
        )


def create_physics_constrained_cnn(
    num_classes: int = NUM_CLASSES,
    backbone: str = 'resnet18',
    adaptive: bool = False,
    **kwargs
) -> PhysicsConstrainedCNN:
    """
    Factory function to create Physics-Constrained CNN.

    Args:
        num_classes: Number of fault classes
        backbone: CNN backbone
        adaptive: If True, use adaptive physics loss weighting
        **kwargs: Additional arguments

    Returns:
        Configured PhysicsConstrainedCNN model
    """
    if adaptive:
        return AdaptivePhysicsConstrainedCNN(
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )
    else:
        return PhysicsConstrainedCNN(
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )


if __name__ == "__main__":
    # Test Physics-Constrained CNN
    print("=" * 60)
    print("Physics-Constrained CNN - Validation")
    print("=" * 60)

    # Create model
    model = PhysicsConstrainedCNN(
        num_classes=NUM_CLASSES,
        backbone='resnet18',
        dropout=0.3,
        sample_rate=20480
    )

    print("\nModel Architecture:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    print("\nTesting Forward Pass:")
    batch_size = 4
    signal = torch.randn(batch_size, 1, SIGNAL_LENGTH)

    output = model(signal)
    print(f"  Input shape: {signal.shape}")
    print(f"  Output shape: {output.shape}")

    # Test physics loss
    print("\nTesting Physics Loss Computation:")
    metadata = {
        'rpm': torch.tensor([3600.0] * batch_size)
    }

    physics_loss, loss_dict = model.compute_physics_loss(signal, output, metadata)
    print(f"  Physics loss: {physics_loss.item():.4f}")
    print(f"  Loss components: {loss_dict}")

    # Test forward with physics loss
    print("\nTesting Forward with Physics Loss:")
    predictions, phys_loss, loss_dict = model.forward_with_physics_loss(
        signal, metadata, lambda_physics=0.5
    )
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Weighted physics loss: {phys_loss.item():.4f}")

    # Test adaptive variant
    print("\nTesting Adaptive Physics-Constrained CNN:")
    adaptive_model = AdaptivePhysicsConstrainedCNN(
        num_classes=NUM_CLASSES,
        backbone='resnet18',
        lambda_physics_schedule='linear',
        lambda_physics_max=1.0
    )

    print(f"  Initial lambda: {adaptive_model.current_lambda_physics:.3f}")

    adaptive_model.update_lambda_physics(epoch=10, max_epochs=50)
    print(f"  Lambda at epoch 10/50: {adaptive_model.current_lambda_physics:.3f}")

    adaptive_model.update_lambda_physics(epoch=40, max_epochs=50)
    print(f"  Lambda at epoch 40/50: {adaptive_model.current_lambda_physics:.3f}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
