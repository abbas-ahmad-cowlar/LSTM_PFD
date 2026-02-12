"""
Contrastive Classifier

Classifier that uses pretrained encoder features.
Encoder weights can be frozen or fine-tuned.

Extracted from: scripts/research/contrastive_physics.py
"""

import torch
import torch.nn as nn

from .signal_encoder import SignalEncoder


class ContrastiveClassifier(nn.Module):
    """
    Classifier that uses pretrained encoder features.
    Encoder weights can be frozen or fine-tuned.

    Args:
        encoder: Pretrained SignalEncoder instance
        num_classes: Number of output classes
        freeze_encoder: Whether to freeze encoder weights (default: False)
    """

    def __init__(self,
                 encoder: SignalEncoder = None,
                 num_classes: int = 11,
                 freeze_encoder: bool = False,
                 **kwargs):
        super().__init__()

        # Create default encoder if none provided
        if encoder is None:
            encoder = SignalEncoder()

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x, return_features=True)
        return self.classifier(features)

    def get_config(self):
        """Return model configuration."""
        return {
            'model_type': 'ContrastiveClassifier',
            'freeze_encoder': self.freeze_encoder,
        }

    def get_num_params(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
