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
        eps: float = 1e-8,
        **_legacy,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Band-energy physics-consistency loss (ratified PROTOCOL §7, 2026-06-14).

        Penalizes softmax probability on a class whose corrected expected spectral
        bands (`FaultSignatureDatabase`, journal-bearing physics — tonal harmonics
        AND absolute broadband bands) are NOT elevated above what a flat/"healthy"
        spectrum would carry, i.e. the class's characteristic energy is ABSENT.
        This supersedes the earlier tonal-only peak-distance loss, which left the
        broadband (cavitation / lubrication / wear) and mixed classes unconstrained
        (audit Findings 6; PHYSICS_LOSS_AUDIT §3).

        Per class c, with band-bin mask M_c (tonal harmonics built from PER-SAMPLE
        rpm, plus the class's absolute Hz bands):

            concentration_c = (E_c / E_total) * (F_bins / n_bins_c)

        — the band set's share of total spectral energy relative to its share of
        bandwidth. concentration_c > 1 means the class's characteristic bands carry
        MORE energy than a flat (healthy) spectrum would (signature present);
        pen_c = relu(1 - concentration_c) rises toward 1 as the signature is absent.
        Healthy ('sain') has no expected bands -> pen = 0.

        Differentiability: pen[B, C] is constant w.r.t. model parameters (a function
        of the input spectrum and per-sample rpm only); the loss
        `(probs * pen).sum(1).mean()` carries gradient through the softmax
        probabilities only —  loss -> probs -> logits -> params. No tunable knob
        (the consistency threshold is concentration = 1, the flat-spectrum level).

        Args:
            signal: Input signal [B, 1, T] or [B, T]
            predictions: Predicted class logits [B, num_classes]
            metadata: Optional dict with PER-SAMPLE 'rpm' [B] (default 3600);
                      tonal band centers scale with shaft speed = rpm/60.
            n_fft: FFT size
            eps: numerical floor

        Returns:
            physics_loss: scalar band-energy consistency loss
            loss_dict: components (band_energy_consistency, mean_penalty)
        """
        if signal.dim() == 3:
            signal = signal.squeeze(1)
        device = signal.device
        B = signal.shape[0]
        C = predictions.shape[1]

        # softmax probabilities -- the ONLY parameter-dependent factor.
        probs = F.softmax(predictions, dim=1)  # [B, C], differentiable

        # per-sample shaft frequency omega = rpm / 60  [B]
        rpm = torch.full((B,), 3600.0, device=device)
        if metadata is not None and metadata.get('rpm', None) is not None:
            r = metadata['rpm']
            r = r if isinstance(r, torch.Tensor) else torch.as_tensor(r)
            r = r.to(device=device, dtype=torch.float32).reshape(-1)
            rpm = r.expand(B).clone() if r.numel() == 1 else r[:B]
        omega = (rpm / 60.0).clamp(min=1e-3)  # [B]

        # Band-energy penalty pen[B, C] -- constant w.r.t. params (no grad needed).
        with torch.no_grad():
            x = signal.detach().float()
            psd = torch.abs(torch.fft.rfft(x, n=n_fft, dim=-1)) ** 2  # [B, F]
            freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate).to(device)  # [F]
            F_bins = freqs.shape[0]
            e_total = psd.sum(dim=-1) + eps          # [B]
            f_row = freqs.unsqueeze(0)               # [1, F]

            pen = torch.zeros(B, C, device=device)
            for c in range(C):
                try:
                    sig = self.signature_db.get_signature(c)
                except Exception:
                    continue
                mask = torch.zeros(B, F_bins, device=device, dtype=torch.bool)
                for m, hw in sig.tonal:  # harmonics scale with PER-SAMPLE rpm
                    lo = (m * omega * (1.0 - hw)).unsqueeze(1)   # [B,1]
                    hi = (m * omega * (1.0 + hw)).unsqueeze(1)
                    mask |= (f_row >= lo) & (f_row <= hi)
                for lo_hz, hi_hz in sig.bands_hz:  # absolute broadband bands
                    mask |= (f_row >= lo_hz) & (f_row <= hi_hz)
                n_bins = mask.sum(dim=-1)            # [B]
                if int(n_bins.max()) == 0:
                    continue  # e.g. healthy: no expected bands -> no constraint
                e_band = (psd * mask.float()).sum(dim=-1)  # [B]
                concentration = (e_band * F_bins) / (e_total * n_bins.clamp(min=1).float())
                penc = F.relu(1.0 - concentration)  # 0 if signature present, ->1 if absent
                pen[:, c] = torch.where(n_bins > 0, penc, torch.zeros_like(penc))

        # Differentiable: penalize probability on spectrally-inconsistent classes.
        pen = pen.to(probs.dtype)
        band_loss = (probs * pen).sum(dim=1).mean()
        loss_dict = {
            'band_energy_consistency': float(band_loss.detach()),
            'mean_penalty': float(pen.mean()),
        }
        return band_loss, loss_dict

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
