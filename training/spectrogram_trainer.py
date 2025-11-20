"""
Spectrogram-Specific Trainer

Extends the base Trainer class with spectrogram-specific features:
- SpecAugment data augmentation
- Spectrogram-specific logging and visualization
- Support for on-the-fly TFR generation
- Multi-TFR training (STFT, CWT, WVD simultaneously)

Usage:
    from training.spectrogram_trainer import SpectrogramTrainer

    trainer = SpectrogramTrainer(
        model=resnet2d,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        use_specaugment=True
    )

    trainer.train(num_epochs=100)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trainer import Trainer, TrainingState


class SpectrogramTrainer(Trainer):
    """
    Trainer for spectrogram-based models.

    Extends base Trainer with:
    - SpecAugment for data augmentation
    - Spectrogram visualization in logs
    - Support for different TFR types (STFT, CWT, WVD)

    Args:
        model: 2D CNN model for spectrograms
        train_loader: DataLoader for training spectrograms
        val_loader: Optional DataLoader for validation
        optimizer: Optimizer
        criterion: Loss function (default: CrossEntropyLoss)
        device: Device to train on
        use_specaugment: Whether to apply SpecAugment (default: True)
        time_mask_param: Time masking parameter (default: 40)
        freq_mask_param: Frequency masking parameter (default: 20)
        num_time_masks: Number of time masks (default: 2)
        num_freq_masks: Number of frequency masks (default: 2)
        **kwargs: Additional arguments passed to base Trainer
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        use_specaugment: bool = True,
        time_mask_param: int = 40,
        freq_mask_param: int = 20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        **kwargs
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            **kwargs
        )

        self.use_specaugment = use_specaugment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

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
                spec_aug[i, :, f0:f0+f, :] = 0

            # Time masking
            for _ in range(self.num_time_masks):
                t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
                t0 = torch.randint(0, max(1, W - t), (1,)).item()
                spec_aug[i, :, :, t0:t0+t] = 0

        return spec_aug

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch with SpecAugment.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.state.epoch+1} [Train]",
            leave=False
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Apply SpecAugment
            inputs = self.apply_specaugment(inputs)

            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )

                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Statistics
            total_loss += loss.item() * self.gradient_accumulation_steps * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / total,
                'acc': 100. * correct / total
            })

        metrics = {
            'train_loss': total_loss / total,
            'train_acc': 100. * correct / total
        }

        return metrics

    def save_checkpoint(
        self,
        filepath: Path,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """
        Save model checkpoint with additional spectrogram-specific info.

        Args:
            filepath: Path to save checkpoint
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.state.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.state.best_metric,
            'history': self.state.history,
            'specaugment_config': {
                'use_specaugment': self.use_specaugment,
                'time_mask_param': self.time_mask_param,
                'freq_mask_param': self.freq_mask_param,
                'num_time_masks': self.num_time_masks,
                'num_freq_masks': self.num_freq_masks,
            }
        }

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = filepath.parent / 'best_model.pth'
            torch.save(checkpoint, best_path)

    @staticmethod
    def load_checkpoint(
        filepath: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint
            model: Model to load weights into
            optimizer: Optional optimizer to load state into

        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint


class MultiTFRTrainer(SpectrogramTrainer):
    """
    Trainer for models that use multiple TFR types simultaneously.

    Useful for ensemble models or comparison studies where you want
    to train on STFT, CWT, and WVD in the same run.

    Args:
        model: Model that accepts multi-channel input
        train_loaders: Dictionary of DataLoaders for different TFR types
            e.g., {'stft': loader1, 'cwt': loader2}
        tfr_weights: Weights for each TFR type in loss (default: equal)
        **kwargs: Additional arguments passed to SpectrogramTrainer
    """

    def __init__(
        self,
        model: nn.Module,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Optional[Dict[str, DataLoader]] = None,
        tfr_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        # Use first loader as primary
        primary_loader = list(train_loaders.values())[0]
        primary_val = list(val_loaders.values())[0] if val_loaders else None

        super().__init__(
            model=model,
            train_loader=primary_loader,
            val_loader=primary_val,
            **kwargs
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

    def train_epoch(self) -> Dict[str, float]:
        """
        Train on all TFR types with weighted loss.

        Returns:
            Dictionary of training metrics per TFR type
        """
        self.model.train()

        metrics = {f'{tfr_type}_loss': 0.0 for tfr_type in self.train_loaders}
        metrics.update({f'{tfr_type}_acc': 0.0 for tfr_type in self.train_loaders})
        counts = {tfr_type: 0 for tfr_type in self.train_loaders}

        # Train on each TFR type
        for tfr_type, loader in self.train_loaders.items():
            pbar = tqdm(
                loader,
                desc=f"Epoch {self.state.epoch+1} [{tfr_type.upper()}]",
                leave=False
            )

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Apply SpecAugment
                inputs = self.apply_specaugment(inputs)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Weight loss by TFR type
                loss = loss * self.tfr_weights[tfr_type]

                # Backward pass
                loss.backward()

                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Statistics
                metrics[f'{tfr_type}_loss'] += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                metrics[f'{tfr_type}_acc'] += correct
                counts[tfr_type] += targets.size(0)

        # Average metrics
        for tfr_type in self.train_loaders:
            if counts[tfr_type] > 0:
                metrics[f'{tfr_type}_loss'] /= counts[tfr_type]
                metrics[f'{tfr_type}_acc'] = 100.0 * metrics[f'{tfr_type}_acc'] / counts[tfr_type]

        # Overall metrics (weighted average)
        metrics['train_loss'] = sum(
            metrics[f'{t}_loss'] * self.tfr_weights[t]
            for t in self.train_loaders
        )
        metrics['train_acc'] = sum(
            metrics[f'{t}_acc'] * self.tfr_weights[t]
            for t in self.train_loaders
        )

        return metrics


if __name__ == '__main__':
    # Test SpecAugment
    trainer = SpectrogramTrainer(
        model=nn.Identity(),
        train_loader=None,
        use_specaugment=True,
        time_mask_param=40,
        freq_mask_param=20
    )

    # Test spectrogram
    spec = torch.randn(4, 1, 129, 400)
    spec_aug = trainer.apply_specaugment(spec)

    print(f"Original spectrogram shape: {spec.shape}")
    print(f"Augmented spectrogram shape: {spec_aug.shape}")
    print(f"Percentage masked: {((spec_aug == 0).sum() / spec_aug.numel() * 100):.2f}%")
