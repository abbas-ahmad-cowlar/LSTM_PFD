"""
Early Fusion for Multi-Modal Bearing Fault Diagnosis

Concatenate features from multiple domains/models before classification.

Architecture:
  Signal:
    ├─ Time-domain features (Phase 1): [B, 36]
    ├─ CNN features (Phase 2): [B, 512]
    ├─ Transformer features (Phase 4): [B, 512]
    └─ Physics features (Phase 6): [B, 64]
          ↓ Concatenate
        [B, 36+512+512+64] = [B, 1124]
          ↓ FC layers
        [B, 256] → [B, 11]

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
sys.path.append('/home/user/LSTM_PFD')
from models.base_model import BaseModel


class MultiModalFeatureExtractor(nn.Module):
    """
    Extract features from multiple models/domains.

    Args:
        feature_extractors: List of (model, layer_name) tuples
            Each model should have a method to extract features from specified layer
    """
    def __init__(self, feature_extractors: List[Tuple[nn.Module, Optional[str]]]):
        super().__init__()

        self.feature_extractors = nn.ModuleList([fe[0] for fe in feature_extractors])
        self.layer_names = [fe[1] for fe in feature_extractors]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from all models.

        Args:
            x: Input signal [B, C, T]

        Returns:
            List of feature tensors from each model
        """
        features = []

        for i, model in enumerate(self.feature_extractors):
            model.eval()
            with torch.no_grad():
                # If layer name is specified, extract features from that layer
                if self.layer_names[i] is not None:
                    feat = self._extract_layer_features(model, x, self.layer_names[i])
                else:
                    # Use the full forward pass
                    feat = model(x)

                # Flatten if needed
                if len(feat.shape) > 2:
                    feat = feat.view(feat.size(0), -1)

                features.append(feat)

        return features

    def _extract_layer_features(self, model: nn.Module, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Extract features from a specific layer."""
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        # Register hook
        layer = dict(model.named_modules())[layer_name]
        handle = layer.register_forward_hook(get_activation(layer_name))

        # Forward pass
        _ = model(x)

        # Remove hook
        handle.remove()

        return activation[layer_name]


class EarlyFusion(BaseModel):
    """
    Early fusion model that concatenates features from multiple sources.

    Benefit: Joint representation learning across modalities
    Challenge: High-dimensional feature space (overfitting risk)

    Args:
        feature_extractors: List of models to extract features from
        feature_dims: List of feature dimensions from each extractor
        num_classes: Number of output classes (default: 11)
        fusion_dim: Dimension of fusion layer (default: 256)
        dropout: Dropout rate (default: 0.3)

    Example:
        >>> # Create feature extractors
        >>> cnn_model = CNN1D(...)
        >>> transformer_model = Transformer(...)
        >>>
        >>> # Create early fusion
        >>> fusion = EarlyFusion(
        ...     feature_extractors=[(cnn_model, 'fc1'), (transformer_model, 'fc1')],
        ...     feature_dims=[512, 512],
        ...     num_classes=11
        ... )
        >>> predictions = fusion(x)
    """
    def __init__(
        self,
        feature_extractors: List[Tuple[nn.Module, Optional[str]]],
        feature_dims: List[int],
        num_classes: int = 11,
        fusion_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        if len(feature_extractors) != len(feature_dims):
            raise ValueError("Number of feature extractors must match number of feature dimensions")

        self.num_extractors = len(feature_extractors)
        self.feature_dims = feature_dims
        self.num_classes = num_classes

        # Multi-modal feature extractor
        self.feature_extractor = MultiModalFeatureExtractor(feature_extractors)

        # Fusion layers
        total_feature_dim = sum(feature_dims)

        self.fusion_layers = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_dim * 2),
            nn.BatchNorm1d(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with early fusion.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Predictions [B, num_classes]
        """
        # Extract features from all modalities
        features = self.feature_extractor(x)

        # Concatenate features
        fused_features = torch.cat(features, dim=1)  # [B, sum(feature_dims)]

        # Classification
        logits = self.fusion_layers(fused_features)

        return logits

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'EarlyFusion',
            'num_extractors': self.num_extractors,
            'feature_dims': self.feature_dims,
            'num_classes': self.num_classes,
            'total_feature_dim': sum(self.feature_dims),
            'num_parameters': self.get_num_params()
        }


class SimpleEarlyFusion(BaseModel):
    """
    Simplified early fusion for manual feature concatenation.

    Use this when you already have extracted features from different models.

    Args:
        input_dim: Total dimension of concatenated features
        num_classes: Number of output classes (default: 11)
        hidden_dims: List of hidden layer dimensions (default: [512, 256, 128])
        dropout: Dropout rate (default: 0.3)

    Example:
        >>> # Extract features manually
        >>> cnn_features = cnn_model.extract_features(x)  # [B, 512]
        >>> transformer_features = transformer_model.extract_features(x)  # [B, 512]
        >>>
        >>> # Concatenate
        >>> combined_features = torch.cat([cnn_features, transformer_features], dim=1)  # [B, 1024]
        >>>
        >>> # Create fusion model
        >>> fusion = SimpleEarlyFusion(input_dim=1024, num_classes=11)
        >>> predictions = fusion(combined_features)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 11,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build fusion network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.fusion_network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Concatenated features [B, input_dim]

        Returns:
            Predictions [B, num_classes]
        """
        return self.fusion_network(x)

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'SimpleEarlyFusion',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_params()
        }


def create_early_fusion(
    feature_extractors: List[Tuple[nn.Module, Optional[str]]],
    feature_dims: List[int],
    num_classes: int = 11,
    fusion_dim: int = 256,
    dropout: float = 0.3
) -> EarlyFusion:
    """
    Factory function to create early fusion model.

    Args:
        feature_extractors: List of (model, layer_name) tuples
        feature_dims: List of feature dimensions from each extractor
        num_classes: Number of output classes
        fusion_dim: Dimension of fusion layer
        dropout: Dropout rate

    Returns:
        EarlyFusion instance

    Example:
        >>> from models import create_cnn1d, create_transformer
        >>>
        >>> cnn = create_cnn1d(num_classes=11)
        >>> transformer = create_transformer(num_classes=11)
        >>>
        >>> fusion = create_early_fusion(
        ...     feature_extractors=[(cnn, 'fc'), (transformer, 'encoder')],
        ...     feature_dims=[512, 512],
        ...     num_classes=11
        ... )
    """
    return EarlyFusion(
        feature_extractors=feature_extractors,
        feature_dims=feature_dims,
        num_classes=num_classes,
        fusion_dim=fusion_dim,
        dropout=dropout
    )


def extract_and_concatenate_features(
    models: List[nn.Module],
    x: torch.Tensor,
    layer_names: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Helper function to extract and concatenate features from multiple models.

    Args:
        models: List of models
        x: Input signal [B, C, T]
        layer_names: Optional layer names to extract from each model

    Returns:
        Concatenated features [B, total_feature_dim]

    Example:
        >>> models = [cnn_model, resnet_model, transformer_model]
        >>> features = extract_and_concatenate_features(models, x)
        >>> # features: [32, 1536] for batch_size=32, 512 features per model
    """
    if layer_names is None:
        layer_names = [None] * len(models)

    all_features = []

    for model, layer_name in zip(models, layer_names):
        model.eval()
        with torch.no_grad():
            if layer_name is not None:
                # Extract from specific layer
                activation = {}

                def get_activation(name):
                    def hook(model, input, output):
                        activation[name] = output
                    return hook

                layer = dict(model.named_modules())[layer_name]
                handle = layer.register_forward_hook(get_activation(layer_name))
                _ = model(x)
                handle.remove()
                feat = activation[layer_name]
            else:
                # Use full forward pass
                feat = model(x)

            # Flatten if needed
            if len(feat.shape) > 2:
                feat = feat.view(feat.size(0), -1)

            all_features.append(feat)

    # Concatenate all features
    concatenated = torch.cat(all_features, dim=1)

    return concatenated
