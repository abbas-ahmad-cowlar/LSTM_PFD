"""
Hybrid Physics-Informed Neural Network (PINN)

Combines data-driven learning with physics-based constraints from bearing
dynamics equations. Fuses learned features from raw signals with handcrafted
physics-based features.

Reference:
- Raissi et al. (2019). "Physics-informed neural networks"
- Bearing dynamics equations from existing pipeline

Input:
    - Raw signal: [B, 1, T]
    - Physics features: [B, P] where P is number of physics features
Output: [B, 11] for 11 fault classes
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from .base_model import BaseModel
from .cnn_1d import CNN1D


class PhysicsConstraint(nn.Module):
    """
    Encodes bearing dynamics equations as differentiable constraints.

    Based on:
    - Sommerfeld number equation
    - Fault characteristic frequencies (BPFO, BPFI, BSF, FTF)
    - Vibration amplitude relationships

    Args:
        num_classes: Number of fault classes
        physics_dim: Dimension of physics feature vector
    """
    def __init__(self, num_classes: int = NUM_CLASSES, physics_dim: int = 32):
        super().__init__()

        self.num_classes = num_classes
        self.physics_dim = physics_dim

        # Physics encoder: Map raw physics parameters to learned representation
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

    def forward(self, physics_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            physics_features: Physics-based features [B, P]

        Returns:
            Encoded physics features [B, 64]
        """
        return self.physics_encoder(physics_features)

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        physics_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute physics-informed loss term.

        Enforces consistency between predictions and known physics:
        - Fault frequencies should match bearing geometry
        - Severity should correlate with amplitude

        Args:
            predictions: Model predictions [B, num_classes]
            physics_params: Dictionary containing:
                - 'fault_frequencies': Expected fault frequencies [B, 4]
                - 'amplitude_ratios': Expected amplitude ratios [B, 1]
                - 'severity_levels': Ground truth severity [B, 1]

        Returns:
            Physics loss scalar
        """
        # Placeholder for physics loss computation
        # In practice, this would compute:
        # 1. Consistency between predicted class and expected fault frequency
        # 2. Correlation between prediction confidence and severity level

        physics_loss = torch.tensor(0.0, device=predictions.device)

        # Example: Penalize high confidence predictions for low severity faults
        if 'severity_levels' in physics_params:
            severity = physics_params['severity_levels']
            probs = F.softmax(predictions, dim=1)
            max_probs = probs.max(dim=1)[0]

            # Low severity should have lower confidence
            severity_penalty = torch.relu(max_probs - severity.squeeze())
            physics_loss += severity_penalty.mean()

        return physics_loss


class FeatureFusion(nn.Module):
    """
    Fuses learned features from CNN with physics-based features.

    Fusion strategies:
    - Concatenation + FC layers
    - Cross-attention between data and physics features
    - Gated fusion

    Args:
        data_dim: Dimension of data-driven features
        physics_dim: Dimension of physics features
        fusion_dim: Dimension of fused features
        fusion_type: Type of fusion ('concat', 'attention', 'gated')
    """
    def __init__(
        self,
        data_dim: int,
        physics_dim: int,
        fusion_dim: int = 256,
        fusion_type: str = 'concat'
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.data_dim = data_dim
        self.physics_dim = physics_dim

        if fusion_type == 'concat':
            # Simple concatenation + FC
            self.fusion = nn.Sequential(
                nn.Linear(data_dim + physics_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fusion_dim, fusion_dim)
            )

        elif fusion_type == 'attention':
            # Cross-attention fusion
            self.query = nn.Linear(data_dim, fusion_dim)
            self.key = nn.Linear(physics_dim, fusion_dim)
            self.value = nn.Linear(physics_dim, fusion_dim)
            self.output = nn.Linear(fusion_dim, fusion_dim)

        elif fusion_type == 'gated':
            # Gated fusion (learnable weights for data vs physics)
            self.data_transform = nn.Linear(data_dim, fusion_dim)
            self.physics_transform = nn.Linear(physics_dim, fusion_dim)
            self.gate = nn.Sequential(
                nn.Linear(data_dim + physics_dim, fusion_dim),
                nn.Sigmoid()
            )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        data_features: torch.Tensor,
        physics_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            data_features: Learned features from CNN [B, data_dim]
            physics_features: Physics-based features [B, physics_dim]

        Returns:
            Fused features [B, fusion_dim]
        """
        if self.fusion_type == 'concat':
            combined = torch.cat([data_features, physics_features], dim=1)
            return self.fusion(combined)

        elif self.fusion_type == 'attention':
            # Cross-attention: data attends to physics
            Q = self.query(data_features).unsqueeze(1)  # [B, 1, fusion_dim]
            K = self.key(physics_features).unsqueeze(1)  # [B, 1, fusion_dim]
            V = self.value(physics_features).unsqueeze(1)  # [B, 1, fusion_dim]

            attention_weights = F.softmax(
                torch.bmm(Q, K.transpose(1, 2)) / (self.data_dim ** 0.5),
                dim=-1
            )
            attended = torch.bmm(attention_weights, V).squeeze(1)

            return self.output(attended)

        elif self.fusion_type == 'gated':
            # Gated fusion
            data_transformed = self.data_transform(data_features)
            physics_transformed = self.physics_transform(physics_features)

            gate_input = torch.cat([data_features, physics_features], dim=1)
            gate_weights = self.gate(gate_input)

            fused = gate_weights * data_transformed + (1 - gate_weights) * physics_transformed
            return fused


class HybridPINN(BaseModel):
    """
    Hybrid Physics-Informed Neural Network.

    Architecture:
        - Data branch: 1D CNN for raw signal processing
        - Physics branch: FC network for physics features
        - Fusion layer: Combine both branches
        - Classification head

    Args:
        num_classes: Number of output classes (default: 11)
        input_channels: Number of input signal channels (default: 1)
        physics_dim: Dimension of physics feature vector (default: 32)
        fusion_dim: Dimension of fused features (default: 256)
        fusion_type: Type of fusion ('concat', 'attention', 'gated')
        physics_weight: Weight for physics loss (default: 0.1)
        dropout: Dropout probability (default: 0.3)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 1,
        physics_dim: int = 32,
        fusion_dim: int = 256,
        fusion_type: str = 'concat',
        physics_weight: float = 0.1,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.physics_dim = physics_dim
        self.physics_weight = physics_weight

        # Data-driven branch: CNN backbone
        self.cnn_backbone = CNN1D(
            num_classes=num_classes,
            input_channels=input_channels,
            dropout=dropout,
            use_bn=True
        )

        # Remove the final FC layer from CNN to get features
        # CNN1D outputs 256-dim features before final FC
        data_feature_dim = 256

        # Physics branch
        self.physics_constraint = PhysicsConstraint(
            num_classes=num_classes,
            physics_dim=physics_dim
        )

        # Fusion layer
        self.fusion = FeatureFusion(
            data_dim=data_feature_dim,
            physics_dim=64,  # Output of physics encoder
            fusion_dim=fusion_dim,
            fusion_type=fusion_type
        )

        # Classification head on fused features
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from CNN backbone (before final FC layer).

        Args:
            x: Input signal [B, C, T]

        Returns:
            Features [B, 256]
        """
        # Forward through CNN feature extractor
        features = self.cnn_backbone.get_feature_extractor()(x)
        features = features.squeeze(-1)  # Remove spatial dimension
        return features

    def forward(
        self,
        x: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input signal [B, C, T]
            physics_features: Optional physics features [B, P]
                             If None, uses zero vector

        Returns:
            logits: Output tensor [B, num_classes]
        """
        # Data-driven features from CNN
        data_features = self.extract_cnn_features(x)

        # Physics features
        if physics_features is None:
            # If no physics features provided, use zeros
            physics_features = torch.zeros(
                x.size(0),
                self.physics_dim,
                device=x.device
            )

        physics_encoded = self.physics_constraint(physics_features)

        # Fuse features
        fused_features = self.fusion(data_features, physics_encoded)

        # Classification
        logits = self.classifier(fused_features)

        return logits

    def compute_total_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        data_loss_fn: nn.Module,
        physics_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss = data loss + physics loss.

        Args:
            predictions: Model predictions [B, num_classes]
            targets: Ground truth labels [B]
            data_loss_fn: Loss function for data (e.g., CrossEntropyLoss)
            physics_params: Optional physics parameters for physics loss

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Data-driven loss
        data_loss = data_loss_fn(predictions, targets)

        # Physics-informed loss
        physics_loss = torch.tensor(0.0, device=predictions.device)
        if physics_params is not None:
            physics_loss = self.physics_constraint.compute_physics_loss(
                predictions,
                physics_params
            )

        # Total loss
        total_loss = data_loss + self.physics_weight * physics_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item()
        }

        return total_loss, loss_dict

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'HybridPINN',
            'num_classes': self.num_classes,
            'physics_dim': self.physics_dim,
            'physics_weight': self.physics_weight,
            'num_parameters': self.get_num_params()
        }


def create_hybrid_pinn(num_classes: int = NUM_CLASSES, **kwargs) -> HybridPINN:
    """
    Factory function to create Hybrid PINN model.

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to HybridPINN

    Returns:
        HybridPINN model instance
    """
    return HybridPINN(num_classes=num_classes, **kwargs)
