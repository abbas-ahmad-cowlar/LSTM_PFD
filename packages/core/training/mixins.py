"""
Trainer Mixins — reusable cross-cutting behaviors for trainers.

Mixins provide composable functionality that any trainer can use:
- SpecAugmentMixin: SpecAugment data augmentation for spectrograms
- PhysicsLossMixin: Physics-informed loss computation and adaptive scheduling

Usage:
    class MyCustomTrainer(SpecAugmentMixin, BaseTrainer):
        def _forward_pass(self, inputs, targets, **kwargs):
            inputs = self.apply_specaugment(inputs)
            return self.model(inputs)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple

from utils.logging import get_logger

logger = get_logger(__name__)


class SpecAugmentMixin:
    """
    Mixin providing SpecAugment data augmentation.

    Call ``apply_specaugment(spectrograms)`` in your ``_forward_pass``.

    Attributes to set (or pass to __init__):
        use_specaugment: bool (default True)
        time_mask_param: int (default 40)
        freq_mask_param: int (default 20)
        num_time_masks: int (default 2)
        num_freq_masks: int (default 2)
    """

    use_specaugment: bool = True
    time_mask_param: int = 40
    freq_mask_param: int = 20
    num_time_masks: int = 2
    num_freq_masks: int = 2

    def apply_specaugment(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrograms (Park et al., 2019).

        Args:
            spectrograms: [B, C, H, W] spectrogram batch.

        Returns:
            Augmented spectrograms (same shape). No-op during eval or if disabled.
        """
        if not self.use_specaugment or not getattr(self, "model", None) or not self.model.training:
            return spectrograms

        B, C, H, W = spectrograms.shape
        spec_aug = spectrograms.clone()

        for i in range(B):
            for _ in range(self.num_freq_masks):
                f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
                f0 = torch.randint(0, max(1, H - f), (1,)).item()
                spec_aug[i, :, f0 : f0 + f, :] = 0

            for _ in range(self.num_time_masks):
                t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
                t0 = torch.randint(0, max(1, W - t), (1,)).item()
                spec_aug[i, :, :, t0 : t0 + t] = 0

        return spec_aug


class PhysicsLossMixin:
    """
    Mixin providing physics-informed loss computation.

    Provides:
        - ``compute_physics_loss()`` — combined CE + physics loss
        - ``_update_lambda_physics()`` — adaptive scheduling
        - ``_extract_metadata()`` — 2-tuple / 3-tuple batch parsing

    Attributes to set:
        criterion: nn.Module
        lambda_physics: float
        current_lambda_physics: float
        adaptive_lambda: bool
        lambda_schedule: str
        physics_loss_fn: PhysicalConstraintLoss instance
        metadata_keys: list of str
    """

    def _extract_metadata(
        self, batch: tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Extract (inputs, targets, metadata?) from batch."""
        metadata_keys = getattr(self, "metadata_keys", ["rpm", "load", "viscosity"])

        if len(batch) == 2:
            return batch[0], batch[1], None
        elif len(batch) == 3:
            inputs, targets, metadata = batch
            if isinstance(metadata, dict):
                metadata = {
                    k: v for k, v in metadata.items()
                    if k in metadata_keys and v is not None
                }
                if not metadata:
                    metadata = None
            return inputs, targets, metadata
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

    def compute_physics_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined classification + physics loss."""
        ce_loss = self.criterion(outputs, targets)

        current_lambda = getattr(self, "current_lambda_physics", 0.0)
        physics_loss_fn = getattr(self, "physics_loss_fn", None)

        if current_lambda > 0 and physics_loss_fn is not None:
            physics_loss, physics_dict = physics_loss_fn(
                signal=signal,
                predictions=outputs,
                metadata=metadata,
                severity_predictions=None,
                predictions_sequence=None,
            )
            total_loss = ce_loss + current_lambda * physics_loss
            loss_dict = {
                "ce_loss": ce_loss.item(),
                "physics_loss": physics_loss.item(),
                **physics_dict,
                "total_loss": total_loss.item(),
            }
        else:
            total_loss = ce_loss
            loss_dict = {
                "ce_loss": ce_loss.item(),
                "physics_loss": 0.0,
                "total_loss": total_loss.item(),
            }

        return total_loss, loss_dict

    def _update_lambda_physics(self, epoch: int, max_epochs: int) -> None:
        """Update physics loss weight based on training progress."""
        adaptive = getattr(self, "adaptive_lambda", False)
        target_lambda = getattr(self, "lambda_physics", 0.0)
        schedule = getattr(self, "lambda_schedule", "linear")

        if not adaptive:
            self.current_lambda_physics = target_lambda
            return

        progress = epoch / max(max_epochs, 1)

        if schedule == "linear":
            self.current_lambda_physics = target_lambda * progress
        elif schedule == "exponential":
            self.current_lambda_physics = target_lambda * (1 - np.exp(-5 * progress))
        elif schedule == "step":
            if progress < 0.4:
                self.current_lambda_physics = 0.0
            elif progress < 0.7:
                self.current_lambda_physics = target_lambda * 0.5
            else:
                self.current_lambda_physics = target_lambda
        else:
            self.current_lambda_physics = target_lambda
