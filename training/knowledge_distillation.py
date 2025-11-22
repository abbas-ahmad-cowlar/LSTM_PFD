"""
Knowledge Distillation Training Framework

Train smaller "student" models using larger "teacher" models to
transfer knowledge without directly copying weights.

Key benefits:
- Student models often match teacher accuracy with fewer parameters
- Combines soft targets (teacher) and hard targets (true labels)
- Enables model compression for deployment

Reference:
- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network"
- Teacher-student learning framework

Example:
    Train ResNet-50 (teacher) → Distill to ResNet-18 (student)
    Student achieves 95-96% accuracy (within 1% of teacher's 97%)
    Student is 2-3× faster inference than teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import numpy as np
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    Loss = α × SoftLoss(student, teacher) + (1-α) × HardLoss(student, labels)

    Where:
    - SoftLoss: KL divergence between softened predictions
    - HardLoss: Standard cross-entropy with true labels
    - α: Weight balancing soft vs hard targets

    Args:
        temperature: Temperature for softening predictions (higher = softer)
        alpha: Weight for soft loss vs hard loss
        hard_loss_type: Type of hard loss ('cross_entropy' or 'mse')
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        hard_loss_type: str = 'cross_entropy'
    ):
        super().__init__()

        self.temperature = temperature
        self.alpha = alpha
        self.hard_loss_type = hard_loss_type

        if hard_loss_type == 'cross_entropy':
            self.hard_loss_fn = nn.CrossEntropyLoss()
        elif hard_loss_type == 'mse':
            self.hard_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown hard loss type: {hard_loss_type}")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model output [B, num_classes]
            teacher_logits: Teacher model output [B, num_classes]
            labels: True labels [B]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with loss components
        """
        T = self.temperature

        # Soft targets from teacher (KL divergence)
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (T * T)  # Scale by T^2 to match gradient magnitudes

        # Hard targets from true labels
        if self.hard_loss_type == 'cross_entropy':
            hard_loss = self.hard_loss_fn(student_logits, labels)
        else:  # MSE with one-hot labels
            num_classes = student_logits.size(1)
            labels_onehot = F.one_hot(labels, num_classes).float()
            hard_loss = self.hard_loss_fn(
                F.softmax(student_logits, dim=1),
                labels_onehot
            )

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        loss_dict = {
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


class DistillationTrainer:
    """
    Trainer for knowledge distillation.

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        criterion: Distillation loss function
        optimizer: Optimizer for student model
        device: Device to train on
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        criterion: DistillationLoss,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu'
    ):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with distillation.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.student_model.train()

        total_loss = 0.0
        total_soft_loss = 0.0
        total_hard_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)

            # Get student predictions
            student_logits = self.student_model(inputs)

            # Compute distillation loss
            loss, loss_dict = self.criterion(student_logits, teacher_logits, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss_dict['total_loss']
            total_soft_loss += loss_dict['soft_loss']
            total_hard_loss += loss_dict['hard_loss']

            # Accuracy
            _, predicted = torch.max(student_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Average metrics
        num_batches = len(train_loader)
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_soft_loss': total_soft_loss / num_batches,
            'train_hard_loss': total_hard_loss / num_batches,
            'train_accuracy': 100.0 * correct / total
        }

        return metrics

    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate student model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.student_model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Student predictions
                student_logits = self.student_model(inputs)

                # Teacher predictions
                teacher_logits = self.teacher_model(inputs)

                # Loss
                loss, _ = self.criterion(student_logits, teacher_logits, labels)
                total_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(student_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': 100.0 * correct / total
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, list]:
        """
        Full training loop with distillation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            scheduler: Optional learning rate scheduler

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        best_val_acc = 0.0

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Track history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_accuracy'].append(val_metrics['val_accuracy'])

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.2f}%")

            # Save best model
            if val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']
                print(f"  → New best validation accuracy: {best_val_acc:.2f}%")

        return history


def compare_teacher_student(
    teacher_model: nn.Module,
    student_model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compare teacher and student model performance.

    Args:
        teacher_model: Teacher model
        student_model: Student model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary with comparison metrics
    """
    teacher_model.eval()
    student_model.eval()

    teacher_correct = 0
    student_correct = 0
    both_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Teacher predictions
            teacher_logits = teacher_model(inputs)
            _, teacher_pred = torch.max(teacher_logits, 1)

            # Student predictions
            student_logits = student_model(inputs)
            _, student_pred = torch.max(student_logits, 1)

            # Count correct predictions
            teacher_mask = (teacher_pred == labels)
            student_mask = (student_pred == labels)

            teacher_correct += teacher_mask.sum().item()
            student_correct += student_mask.sum().item()
            both_correct += (teacher_mask & student_mask).sum().item()
            total += labels.size(0)

    # Count parameters
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    comparison = {
        'teacher_accuracy': 100.0 * teacher_correct / total,
        'student_accuracy': 100.0 * student_correct / total,
        'accuracy_gap': 100.0 * (teacher_correct - student_correct) / total,
        'agreement': 100.0 * both_correct / total,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': teacher_params / student_params
    }

    return comparison


# Example usage
if __name__ == "__main__":
    print("Knowledge Distillation Framework")
    print("\nExample usage:")
    print("""
    # Create teacher and student models
    teacher = create_resnet50_1d(num_classes=NUM_CLASSES)
    student = create_resnet18_1d(num_classes=NUM_CLASSES)

    # Load pre-trained teacher
    teacher.load_state_dict(torch.load('teacher_model.pth'))

    # Setup distillation
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        criterion=criterion,
        optimizer=optimizer,
        device='cuda'
    )

    # Train student with distillation
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50
    )

    # Compare teacher vs student
    comparison = compare_teacher_student(
        teacher, student, test_loader, device='cuda'
    )
    print(f"Teacher accuracy: {comparison['teacher_accuracy']:.2f}%")
    print(f"Student accuracy: {comparison['student_accuracy']:.2f}%")
    print(f"Compression ratio: {comparison['compression_ratio']:.1f}×")
    """)
