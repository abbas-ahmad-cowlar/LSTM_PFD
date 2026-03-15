"""
Spectrogram-Specific Trainer

Extends BaseTrainer with spectrogram-specific features:
- SpecAugment data augmentation
- Support for on-the-fly TFR generation
- Multi-TFR training (STFT, CWT, WVD simultaneously)

Usage:
    from training.spectrogram_trainer import SpectrogramTrainer

    trainer = SpectrogramTrainer(
        model=resnet2d,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device='cuda',
        use_specaugment=True
    )
    history = trainer.fit(train_loader, val_loader, num_epochs=100)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any
from pathlib import Path

from utils.logging import get_logger
from .base_trainer import BaseTrainer

logger = get_logger(__name__)


class SpectrogramTrainer(BaseTrainer):
    """
    Trainer for spectrogram-based models.

    Extends BaseTrainer with SpecAugment for data augmentation.

    Args:
        model: 2D CNN model for spectrograms
        optimizer: Optimizer
        criterion: Loss function (default: CrossEntropyLoss)
        device: Device to train on
        lr_scheduler: Optional LR scheduler
        max_grad_norm: Gradient clipping max norm
        gradient_accumulation_steps: Gradient accumulation steps
        mixed_precision: Use AMP FP16
        checkpoint_dir: Where to save checkpoints
        callbacks: Optional list of callback objects
        use_specaugment: Whether to apply SpecAugment (default: True)
        time_mask_param: Time masking parameter (default: 40)
        freq_mask_param: Frequency masking parameter (default: 20)
        num_time_masks: Number of time masks (default: 2)
        num_freq_masks: Number of frequency masks (default: 2)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        device: str = "cpu",
        lr_scheduler=None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        checkpoint_dir: Optional[Path] = None,
        callbacks: Optional[List] = None,
        use_specaugment: bool = True,
        time_mask_param: int = 40,
        freq_mask_param: int = 20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion or nn.CrossEntropyLoss(),
            device=device,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            checkpoint_dir=checkpoint_dir,
            callbacks=callbacks,
        )

        self.use_specaugment = use_specaugment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

        logger.info(
            f"SpectrogramTrainer initialized — "
            f"specaugment={use_specaugment}, "
            f"time_mask={time_mask_param}, freq_mask={freq_mask_param}"
        )

    # -- SpecAugment -----------------------------------------------------

    def apply_specaugment(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrograms.

        SpecAugment (Park et al., 2019) masks parts of the spectrogram
        along time and frequency axes for regularization.

        Args:
            spectrograms: Input spectrograms [B, C, H, W]

        Returns:
            Augmented spectrograms [B, C, H, W]
        """
        if not self.use_specaugment or not self.model.training:
            return spectrograms

        B, C, H, W = spectrograms.shape
        spec_aug = spectrograms.clone()

        for i in range(B):
            # Frequency masking
            for _ in range(self.num_freq_masks):
                f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
                f0 = torch.randint(0, max(1, H - f), (1,)).item()
                spec_aug[i, :, f0 : f0 + f, :] = 0

            # Time masking
            for _ in range(self.num_time_masks):
                t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
                t0 = torch.randint(0, max(1, W - t), (1,)).item()
                spec_aug[i, :, :, t0 : t0 + t] = 0

        return spec_aug

    # -- Template hooks --------------------------------------------------

    def _forward_pass(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass with SpecAugment applied during training."""
        inputs = self.apply_specaugment(inputs)
        return self.model(inputs)

    # -- Checkpoint with specaugment config ------------------------------

    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint with SpecAugment configuration."""
        if self.checkpoint_dir is None:
            logger.warning("No checkpoint_dir specified, skipping save")
            return

        # Use parent's logic to build checkpoint
        super().save_checkpoint(filename)

        # Re-open and add specaugment config
        filepath = self.checkpoint_dir / filename
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        checkpoint["specaugment_config"] = {
            "use_specaugment": self.use_specaugment,
            "time_mask_param": self.time_mask_param,
            "freq_mask_param": self.freq_mask_param,
            "num_time_masks": self.num_time_masks,
            "num_freq_masks": self.num_freq_masks,
        }
        torch.save(checkpoint, filepath)

    @staticmethod
    def load_pretrained(
        filepath: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict:
        """
        Load model checkpoint (static convenience method).

        Args:
            filepath: Path to checkpoint
            model: Model to load weights into
            optimizer: Optional optimizer to load state into

        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    # Backward-compat alias
    def train(self, num_epochs: int = 50, **kwargs):
        """Backward-compatible alias. Delegates to inherited ``fit()``."""
        train_loader = kwargs.pop("train_loader", None)
        val_loader = kwargs.pop("val_loader", None)
        if train_loader is None:
            raise ValueError(
                "SpectrogramTrainer.train() requires train_loader keyword argument. "
                "Preferred API is trainer.fit(train_loader, val_loader, num_epochs)."
            )
        return self.fit(
            train_loader, val_loader, num_epochs=num_epochs, **kwargs
        )


class MultiTFRTrainer(SpectrogramTrainer):
    """
    Trainer for models that use multiple TFR types simultaneously.

    Useful for ensemble models or comparison studies where you want
    to train on STFT, CWT, and WVD in the same run.

    Args:
        model: Model that accepts multi-channel input
        train_loaders: Dictionary of DataLoaders for different TFR types
        tfr_weights: Weights for each TFR type in loss (default: equal)
        **kwargs: Additional arguments passed to SpectrogramTrainer
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Optional[Dict[str, DataLoader]] = None,
        tfr_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            **kwargs,
        )

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders or {}

        # TFR type weights (for weighted loss)
        self.tfr_weights = tfr_weights or {
            tfr_type: 1.0 for tfr_type in train_loaders.keys()
        }

        # Normalize weights
        total_weight = sum(self.tfr_weights.values())
        self.tfr_weights = {
            k: v / total_weight for k, v in self.tfr_weights.items()
        }

    def train_epoch(
        self,
        train_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Train on all TFR types with weighted loss.

        The ``train_loader`` arg is ignored; uses ``self.train_loaders``.
        """
        from tqdm import tqdm

        self.model.train()

        metrics: Dict[str, float] = {
            f"{t}_loss": 0.0 for t in self.train_loaders
        }
        metrics.update({f"{t}_acc": 0.0 for t in self.train_loaders})
        counts: Dict[str, int] = {t: 0 for t in self.train_loaders}

        for tfr_type, loader in self.train_loaders.items():
            pbar = tqdm(
                loader,
                desc=f"Epoch {self.current_epoch + 1} [{tfr_type.upper()}]",
                leave=False,
            )

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                inputs = self.apply_specaugment(inputs)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss * self.tfr_weights[tfr_type]

                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                metrics[f"{tfr_type}_loss"] += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                metrics[f"{tfr_type}_acc"] += correct
                counts[tfr_type] += targets.size(0)

        # Average metrics
        for tfr_type in self.train_loaders:
            if counts[tfr_type] > 0:
                metrics[f"{tfr_type}_loss"] /= counts[tfr_type]
                metrics[f"{tfr_type}_acc"] = (
                    100.0 * metrics[f"{tfr_type}_acc"] / counts[tfr_type]
                )

        # Overall metrics (weighted average)
        metrics["loss"] = sum(
            metrics[f"{t}_loss"] * self.tfr_weights[t]
            for t in self.train_loaders
        )
        metrics["accuracy"] = sum(
            metrics[f"{t}_acc"] * self.tfr_weights[t]
            for t in self.train_loaders
        )

        return metrics
