"""
Knowledge Distillation Trainer.

Inherits from BaseTrainer; overrides _forward_pass and _compute_loss
to implement teacher–student distillation.

Author: Phase 2 - CNN Implementation
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any
from pathlib import Path

from utils.logging import get_logger
from .base_trainer import BaseTrainer

logger = get_logger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined distillation + cross-entropy loss.

    L = α · T² · KL(soft_student ‖ soft_teacher) + (1 - α) · CE(student, labels)

    Args:
        temperature: Softmax temperature for soft targets
        alpha: Weight for distillation loss (1 - alpha → hard-label loss)
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (B, C) raw logits from student
            teacher_logits: (B, C) raw logits from teacher
            targets: (B,) ground-truth class indices
        """
        T = self.temperature

        # Soft targets (KL Divergence)
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

        # Hard targets
        ce = self.ce_loss(student_logits, targets)

        return self.alpha * kl_loss + (1 - self.alpha) * ce


class DistillationTrainer(BaseTrainer):
    """
    Knowledge distillation trainer (teacher → student).

    Freezes the teacher model and trains the student via a
    combination of soft-target KL loss and hard-target CE loss.

    Args:
        student: Student model (trainable)
        teacher: Teacher model (frozen)
        optimizer: Optimizer for student parameters
        device: 'cuda' or 'cpu'
        lr_scheduler: Optional learning-rate scheduler
        max_grad_norm: Gradient clipping max norm
        gradient_accumulation_steps: Gradient accumulation steps
        mixed_precision: Use AMP FP16
        checkpoint_dir: Where to save checkpoints
        temperature: Distillation temperature (default 4.0)
        alpha: Weight for distillation loss (default 0.7)

    Examples:
        >>> teacher = load_pretrained_model()
        >>> student = SmallCNN(num_classes=11)
        >>> optimizer = torch.optim.Adam(student.parameters())
        >>> trainer = DistillationTrainer(student, teacher, optimizer, device='cuda')
        >>> history = trainer.fit(train_loader, val_loader, num_epochs=50)
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        lr_scheduler=None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        checkpoint_dir: Optional[Path] = None,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        distillation_loss = DistillationLoss(temperature=temperature, alpha=alpha)

        super().__init__(
            model=student,
            optimizer=optimizer,
            criterion=distillation_loss,
            device=device,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            checkpoint_dir=checkpoint_dir,
        )

        # Freeze teacher
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.temperature = temperature
        self.alpha = alpha

        logger.info(
            f"DistillationTrainer initialized — "
            f"temperature={temperature}, alpha={alpha}"
        )

    # -- Template hooks -------------------------------------------------

    def _forward_pass(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass through the student model."""
        return self.model(inputs)

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Override: compute combined distillation + CE loss.

        Re-runs teacher in no-grad to get teacher logits.
        """
        with torch.no_grad():
            teacher_logits = self.teacher(kwargs.get("inputs", targets))

        return self.criterion(outputs, teacher_logits, targets)

    # -- Override train_epoch to thread inputs through _compute_loss -----

    def train_epoch(
        self,
        train_loader: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Training epoch that passes inputs to _compute_loss for teacher."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    student_logits = self._forward_pass(inputs, targets)
                    with torch.no_grad():
                        teacher_logits = self.teacher(inputs)
                    loss = self.criterion(student_logits, teacher_logits, targets)
            else:
                student_logits = self._forward_pass(inputs, targets)
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                loss = self.criterion(student_logits, teacher_logits, targets)

            # Scale for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with grad accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Metrics
            batch_size = targets.size(0)
            running_loss += loss.item() * self.gradient_accumulation_steps * batch_size
            _, predicted = student_logits.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

        return {
            "loss": running_loss / max(total, 1),
            "accuracy": 100.0 * correct / max(total, 1),
        }

    # Backward-compat alias: train() → fit()
    def train(self, *args, **kwargs):
        """Backward-compatible alias. Use ``fit()`` instead."""
        return self.fit(*args, **kwargs)
