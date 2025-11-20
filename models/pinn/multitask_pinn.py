"""
Multi-Task Physics-Informed Neural Network

Learns multiple related tasks simultaneously for better feature representations.
Primary task: Fault classification (11 classes)
Auxiliary tasks:
- Speed regression (RPM prediction)
- Load regression (N prediction)
- Severity classification (4 levels: low, medium, high, critical)

Rationale:
Auxiliary tasks provide additional supervision signals that help the network
learn more robust and physically meaningful features. Predicting operating
conditions forces the network to extract information about speed and load
from vibration patterns, which improves fault diagnosis.

Architecture:
    Input: Signal [B, 1, 102400]
    ↓
    Shared CNN Encoder → [B, 512]
    ↓
    ├─ Task 1: Fault Classification → [B, 11]
    ├─ Task 2: Speed Regression → [B, 1]
    ├─ Task 3: Load Regression → [B, 1]
    └─ Task 4: Severity Classification → [B, 4]

Expected Benefit: +1-2% accuracy over single-task models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_model import BaseModel
from resnet.resnet_1d import ResNet1D
from cnn.cnn_1d import CNN1D


class MultitaskPINN(BaseModel):
    """
    Multi-task Physics-Informed Neural Network.

    Learns fault classification alongside auxiliary tasks (speed, load, severity)
    to improve feature learning through multi-task regularization.
    """

    def __init__(
        self,
        num_fault_classes: int = 11,
        num_severity_levels: int = 4,
        input_length: int = 102400,
        backbone: str = 'resnet18',
        shared_feature_dim: int = 512,
        task_hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize Multi-task PINN.

        Args:
            num_fault_classes: Number of fault classes (default: 11)
            num_severity_levels: Number of severity levels (default: 4)
            input_length: Signal length
            backbone: CNN backbone ('resnet18', 'resnet34', 'cnn1d')
            shared_feature_dim: Dimension of shared features
            task_hidden_dim: Hidden dimension for task-specific heads
            dropout: Dropout probability
        """
        super().__init__()

        self.num_fault_classes = num_fault_classes
        self.num_severity_levels = num_severity_levels
        self.input_length = input_length
        self.backbone_name = backbone
        self.shared_feature_dim = shared_feature_dim

        # ===== SHARED ENCODER =====
        # Remove final classification layer to get features
        if backbone == 'resnet18':
            self.encoder = ResNet1D(
                num_classes=num_fault_classes,
                input_channels=1,
                layers=[2, 2, 2, 2],
                dropout=dropout,
                input_length=input_length
            )
            # Remove final FC layer
            self.encoder.fc = nn.Identity()

        elif backbone == 'resnet34':
            self.encoder = ResNet1D(
                num_classes=num_fault_classes,
                input_channels=1,
                layers=[3, 4, 6, 3],
                dropout=dropout,
                input_length=input_length
            )
            self.encoder.fc = nn.Identity()

        elif backbone == 'cnn1d':
            self.encoder = CNN1D(
                num_classes=num_fault_classes,
                input_channels=1,
                dropout=dropout
            )
            # Assuming CNN1D has fc layer
            if hasattr(self.encoder, 'fc'):
                self.encoder.fc = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ===== TASK-SPECIFIC HEADS =====

        # Task 1: Fault Classification (Primary)
        self.fault_head = nn.Sequential(
            nn.Linear(shared_feature_dim, task_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, num_fault_classes)
        )

        # Task 2: Speed Regression
        self.speed_head = nn.Sequential(
            nn.Linear(shared_feature_dim, task_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Task 3: Load Regression
        self.load_head = nn.Sequential(
            nn.Linear(shared_feature_dim, task_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Task 4: Severity Classification
        self.severity_head = nn.Sequential(
            nn.Linear(shared_feature_dim, task_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, num_severity_levels)
        )

        # Initialize task heads
        self._initialize_task_heads()

    def _initialize_task_heads(self):
        """Initialize task head weights."""
        for head in [self.fault_head, self.speed_head, self.load_head, self.severity_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self,
        signal: torch.Tensor,
        return_all_tasks: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through multi-task PINN.

        Args:
            signal: Input vibration signal [B, 1, T]
            return_all_tasks: If True, return predictions for all tasks

        Returns:
            If return_all_tasks=False:
                Fault classification logits [B, num_fault_classes]
            If return_all_tasks=True:
                Dictionary with all task outputs
        """
        # Shared feature extraction
        shared_features = self.encoder(signal)  # [B, 512]

        if return_all_tasks:
            # Compute all task outputs
            fault_logits = self.fault_head(shared_features)
            speed_pred = self.speed_head(shared_features)
            load_pred = self.load_head(shared_features)
            severity_logits = self.severity_head(shared_features)

            return {
                'fault': fault_logits,
                'speed': speed_pred,
                'load': load_pred,
                'severity': severity_logits,
                'features': shared_features
            }
        else:
            # Only return primary task (fault classification)
            return self.fault_head(shared_features)

    def compute_multitask_loss(
        self,
        signal: torch.Tensor,
        fault_labels: torch.Tensor,
        speed_labels: Optional[torch.Tensor] = None,
        load_labels: Optional[torch.Tensor] = None,
        severity_labels: Optional[torch.Tensor] = None,
        task_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            signal: Input signal [B, 1, T]
            fault_labels: Fault class labels [B]
            speed_labels: Speed labels in RPM [B] (optional)
            load_labels: Load labels in Newtons [B] (optional)
            severity_labels: Severity class labels [B] (optional)
            task_weights: Dictionary of task weights (default: equal weighting)

        Returns:
            total_loss: Combined multi-task loss
            loss_dict: Dictionary of individual task losses
        """
        # Get all task predictions
        outputs = self.forward(signal, return_all_tasks=True)

        # Default task weights
        if task_weights is None:
            task_weights = {
                'fault': 1.0,
                'speed': 0.3,
                'load': 0.3,
                'severity': 0.5
            }

        loss_dict = {}
        total_loss = 0.0

        # Task 1: Fault classification (always computed)
        fault_loss = F.cross_entropy(outputs['fault'], fault_labels)
        loss_dict['fault_loss'] = fault_loss.item()
        total_loss += task_weights['fault'] * fault_loss

        # Task 2: Speed regression (if labels provided)
        if speed_labels is not None:
            # Normalize speed to [0, 1] range for better training
            speed_pred_normalized = outputs['speed'].squeeze(-1) / 10000.0  # Assume max 10000 RPM
            speed_target_normalized = speed_labels / 10000.0

            speed_loss = F.mse_loss(speed_pred_normalized, speed_target_normalized)
            loss_dict['speed_loss'] = speed_loss.item()
            total_loss += task_weights['speed'] * speed_loss

        # Task 3: Load regression (if labels provided)
        if load_labels is not None:
            # Normalize load to [0, 1] range
            load_pred_normalized = outputs['load'].squeeze(-1) / 2000.0  # Assume max 2000N
            load_target_normalized = load_labels / 2000.0

            load_loss = F.mse_loss(load_pred_normalized, load_target_normalized)
            loss_dict['load_loss'] = load_loss.item()
            total_loss += task_weights['load'] * load_loss

        # Task 4: Severity classification (if labels provided)
        if severity_labels is not None:
            severity_loss = F.cross_entropy(outputs['severity'], severity_labels)
            loss_dict['severity_loss'] = severity_loss.item()
            total_loss += task_weights['severity'] * severity_loss

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def get_model_info(self) -> Dict[str, any]:
        """Get model configuration and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        fault_head_params = sum(p.numel() for p in self.fault_head.parameters())
        speed_head_params = sum(p.numel() for p in self.speed_head.parameters())
        load_head_params = sum(p.numel() for p in self.load_head.parameters())
        severity_head_params = sum(p.numel() for p in self.severity_head.parameters())

        return {
            'model_name': 'MultitaskPINN',
            'backbone': self.backbone_name,
            'num_fault_classes': self.num_fault_classes,
            'num_severity_levels': self.num_severity_levels,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'encoder_params': encoder_params,
            'fault_head_params': fault_head_params,
            'speed_head_params': speed_head_params,
            'load_head_params': load_head_params,
            'severity_head_params': severity_head_params
        }


class AdaptiveMultitaskPINN(MultitaskPINN):
    """
    Multi-task PINN with adaptive task weighting.

    Automatically adjusts task weights during training based on the
    rate of loss decrease for each task (uncertainty weighting).

    Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """

    def __init__(
        self,
        num_fault_classes: int = 11,
        num_severity_levels: int = 4,
        input_length: int = 102400,
        backbone: str = 'resnet18',
        shared_feature_dim: int = 512,
        task_hidden_dim: int = 128,
        dropout: float = 0.3,
        learnable_weights: bool = True
    ):
        """
        Initialize Adaptive Multi-task PINN.

        Args:
            Same as MultitaskPINN, plus:
            learnable_weights: If True, learn task weights as parameters
        """
        super().__init__(
            num_fault_classes=num_fault_classes,
            num_severity_levels=num_severity_levels,
            input_length=input_length,
            backbone=backbone,
            shared_feature_dim=shared_feature_dim,
            task_hidden_dim=task_hidden_dim,
            dropout=dropout
        )

        # Learnable task weights (log variance)
        if learnable_weights:
            self.log_var_fault = nn.Parameter(torch.zeros(1))
            self.log_var_speed = nn.Parameter(torch.zeros(1))
            self.log_var_load = nn.Parameter(torch.zeros(1))
            self.log_var_severity = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('log_var_fault', torch.zeros(1))
            self.register_buffer('log_var_speed', torch.zeros(1))
            self.register_buffer('log_var_load', torch.zeros(1))
            self.register_buffer('log_var_severity', torch.zeros(1))

        self.learnable_weights = learnable_weights

    def compute_adaptive_multitask_loss(
        self,
        signal: torch.Tensor,
        fault_labels: torch.Tensor,
        speed_labels: Optional[torch.Tensor] = None,
        load_labels: Optional[torch.Tensor] = None,
        severity_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss with adaptive weighting.

        Task weights are computed as: w_i = 1 / (2 * exp(log_var_i))
        This automatically balances tasks based on their uncertainty.

        Args:
            signal: Input signal
            fault_labels: Fault labels
            speed_labels: Speed labels (optional)
            load_labels: Load labels (optional)
            severity_labels: Severity labels (optional)

        Returns:
            total_loss: Weighted multi-task loss
            loss_dict: Individual losses and weights
        """
        outputs = self.forward(signal, return_all_tasks=True)

        loss_dict = {}
        total_loss = 0.0

        # Task 1: Fault classification
        fault_loss = F.cross_entropy(outputs['fault'], fault_labels)
        precision_fault = torch.exp(-self.log_var_fault)
        weighted_fault_loss = precision_fault * fault_loss + self.log_var_fault
        loss_dict['fault_loss'] = fault_loss.item()
        loss_dict['fault_weight'] = precision_fault.item()
        total_loss += weighted_fault_loss

        # Task 2: Speed regression
        if speed_labels is not None:
            speed_pred = outputs['speed'].squeeze(-1) / 10000.0
            speed_target = speed_labels / 10000.0
            speed_loss = F.mse_loss(speed_pred, speed_target)

            precision_speed = torch.exp(-self.log_var_speed)
            weighted_speed_loss = precision_speed * speed_loss + self.log_var_speed
            loss_dict['speed_loss'] = speed_loss.item()
            loss_dict['speed_weight'] = precision_speed.item()
            total_loss += weighted_speed_loss

        # Task 3: Load regression
        if load_labels is not None:
            load_pred = outputs['load'].squeeze(-1) / 2000.0
            load_target = load_labels / 2000.0
            load_loss = F.mse_loss(load_pred, load_target)

            precision_load = torch.exp(-self.log_var_load)
            weighted_load_loss = precision_load * load_loss + self.log_var_load
            loss_dict['load_loss'] = load_loss.item()
            loss_dict['load_weight'] = precision_load.item()
            total_loss += weighted_load_loss

        # Task 4: Severity classification
        if severity_labels is not None:
            severity_loss = F.cross_entropy(outputs['severity'], severity_labels)

            precision_severity = torch.exp(-self.log_var_severity)
            weighted_severity_loss = precision_severity * severity_loss + self.log_var_severity
            loss_dict['severity_loss'] = severity_loss.item()
            loss_dict['severity_weight'] = precision_severity.item()
            total_loss += weighted_severity_loss

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


def create_multitask_pinn(
    num_fault_classes: int = 11,
    backbone: str = 'resnet18',
    adaptive: bool = False,
    **kwargs
) -> MultitaskPINN:
    """
    Factory function to create Multi-task PINN.

    Args:
        num_fault_classes: Number of fault classes
        backbone: CNN backbone
        adaptive: If True, use adaptive task weighting
        **kwargs: Additional arguments

    Returns:
        MultitaskPINN or AdaptiveMultitaskPINN
    """
    if adaptive:
        return AdaptiveMultitaskPINN(
            num_fault_classes=num_fault_classes,
            backbone=backbone,
            **kwargs
        )
    else:
        return MultitaskPINN(
            num_fault_classes=num_fault_classes,
            backbone=backbone,
            **kwargs
        )


if __name__ == "__main__":
    # Test Multi-task PINN
    print("=" * 60)
    print("Multi-task PINN - Validation")
    print("=" * 60)

    # Create model
    model = MultitaskPINN(
        num_fault_classes=11,
        num_severity_levels=4,
        backbone='resnet18',
        dropout=0.3
    )

    print("\nModel Architecture:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    print("\nTesting Forward Pass:")
    batch_size = 4
    signal = torch.randn(batch_size, 1, 102400)

    # Primary task only
    output_primary = model(signal, return_all_tasks=False)
    print(f"  Primary task output shape: {output_primary.shape}")

    # All tasks
    outputs_all = model(signal, return_all_tasks=True)
    print(f"\n  All task outputs:")
    for task_name, task_output in outputs_all.items():
        if task_name != 'features':
            print(f"    {task_name}: {task_output.shape}")

    # Test multi-task loss
    print("\nTesting Multi-task Loss:")
    fault_labels = torch.randint(0, 11, (batch_size,))
    speed_labels = torch.tensor([3000.0, 3600.0, 4000.0, 3800.0])
    load_labels = torch.tensor([400.0, 500.0, 600.0, 550.0])
    severity_labels = torch.randint(0, 4, (batch_size,))

    total_loss, loss_dict = model.compute_multitask_loss(
        signal, fault_labels, speed_labels, load_labels, severity_labels
    )
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Loss components:")
    for loss_name, loss_value in loss_dict.items():
        print(f"    {loss_name}: {loss_value:.4f}")

    # Test adaptive variant
    print("\nTesting Adaptive Multi-task PINN:")
    adaptive_model = AdaptiveMultitaskPINN(
        num_fault_classes=11,
        backbone='resnet18',
        learnable_weights=True
    )

    total_loss, loss_dict = adaptive_model.compute_adaptive_multitask_loss(
        signal, fault_labels, speed_labels, load_labels, severity_labels
    )
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Adaptive weights:")
    print(f"    fault: {loss_dict.get('fault_weight', 'N/A')}")
    print(f"    speed: {loss_dict.get('speed_weight', 'N/A')}")
    print(f"    load: {loss_dict.get('load_weight', 'N/A')}")
    print(f"    severity: {loss_dict.get('severity_weight', 'N/A')}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
