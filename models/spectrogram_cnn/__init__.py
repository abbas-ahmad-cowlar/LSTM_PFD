"""
Spectrogram CNN Models Module

2D CNN architectures for time-frequency representation classification.
Processes spectrograms, scalograms, and Wigner-Ville distributions for
bearing fault diagnosis.

Models:
- ResNet-2D: ResNet architecture adapted for spectrograms
- EfficientNet-2D: Parameter-efficient CNN with compound scaling
- Dual-Stream: Combined time-domain + frequency-domain processing

Usage:
    from models.spectrogram_cnn import resnet18_2d, efficientnet_b0

    # ResNet-2D with ImageNet transfer learning
    model = resnet18_2d(num_classes=11, pretrained=True)

    # EfficientNet-B0 (parameter-efficient)
    model = efficientnet_b0(num_classes=11)

    # Forward pass
    spectrogram = torch.randn(4, 1, 129, 400)  # [B, C, H, W]
    logits = model(spectrogram)  # [4, 11]
"""

from .resnet2d_spectrogram import (
    ResNet2DSpectrogram,
    resnet18_2d,
    resnet34_2d,
    resnet50_2d,
)

from .efficientnet2d_spectrogram import (
    EfficientNet2DSpectrogram,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b3,
)

__all__ = [
    # ResNet models
    'ResNet2DSpectrogram',
    'resnet18_2d',
    'resnet34_2d',
    'resnet50_2d',

    # EfficientNet models
    'EfficientNet2DSpectrogram',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b3',
]


def get_model(model_name: str, num_classes: int = 11, **kwargs):
    """
    Factory function to create models by name.

    Args:
        model_name: Name of the model ('resnet18_2d', 'efficientnet_b0', etc.)
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Instantiated model

    Example:
        model = get_model('resnet18_2d', num_classes=11, pretrained=True)
    """
    models = {
        'resnet18_2d': resnet18_2d,
        'resnet34_2d': resnet34_2d,
        'resnet50_2d': resnet50_2d,
        'efficientnet_b0': efficientnet_b0,
        'efficientnet_b1': efficientnet_b1,
        'efficientnet_b3': efficientnet_b3,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(models.keys())}"
        )

    return models[model_name](num_classes=num_classes, **kwargs)


def list_models():
    """List all available spectrogram CNN models."""
    return [
        'resnet18_2d',
        'resnet34_2d',
        'resnet50_2d',
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b3',
    ]
