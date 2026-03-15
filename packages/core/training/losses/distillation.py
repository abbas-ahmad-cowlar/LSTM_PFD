"""
Distillation Loss Function

Combined distillation + cross-entropy loss for knowledge distillation.

Moved from: training/knowledge_distillation.py (DistillationLoss class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

        ce = self.ce_loss(student_logits, targets)

        return self.alpha * kl_loss + (1 - self.alpha) * ce
