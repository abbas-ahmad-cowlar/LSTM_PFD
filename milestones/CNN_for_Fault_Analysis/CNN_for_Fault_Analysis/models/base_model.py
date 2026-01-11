"""
Abstract base model class for all neural network architectures.

Purpose:
    Provides common interface and utilities for all models:
    - Standard forward pass interface
    - Model summary and parameter counting
    - Save/load functionality
    - Device management

Author: Author Name
Date: 2025-11-19
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

from utils.logging import get_logger

logger = get_logger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all fault diagnosis models.

    All models should inherit from this class and implement:
    - forward()
    - get_config()

    Example:
        >>> class MyModel(BaseModel):
        ...     def __init__(self, input_length, num_classes):
        ...         super().__init__()
        ...         self.fc = nn.Linear(input_length, num_classes)
        ...
        ...     def forward(self, x):
        ...         return self.fc(x)
        ...
        ...     def get_config(self):
        ...         return {'input_length': self.input_length, 'num_classes': self.num_classes}
    """

    def __init__(self):
        """Initialize base model."""
        super().__init__()
        self._input_shape = None
        self._output_shape = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor (batch_size, input_length) or (batch_size, channels, length)

        Returns:
            Output tensor (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration dictionary.

        Returns:
            Dictionary with model hyperparameters
        """
        pass

    def count_parameters(self) -> Dict[str, int]:
        """
        Count total and trainable parameters.

        Returns:
            Dictionary with parameter counts

        Example:
            >>> model = MyModel(...)
            >>> params = model.count_parameters()
            >>> print(f"Total: {params['total']:,}, Trainable: {params['trainable']:,}")
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }

    def get_num_params(self) -> int:
        """
        Get total number of parameters (wrapper for backward compatibility).

        This method provides a simpler interface for getting just the total
        parameter count, which is commonly needed for logging and reporting.

        Returns:
            Total number of parameters

        Example:
            >>> model = MyModel(...)
            >>> print(f"Parameters: {model.get_num_params():,}")

        Note:
            For detailed parameter counts (trainable/non-trainable breakdown),
            use count_parameters() instead.
        """
        return self.count_parameters()['total']

    def get_model_size_mb(self) -> float:
        """
        Estimate model size in megabytes.

        Returns:
            Model size in MB

        Example:
            >>> size_mb = model.get_model_size_mb()
            >>> print(f"Model size: {size_mb:.2f} MB")
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None) -> str:
        """
        Generate model summary similar to Keras model.summary().

        Args:
            input_shape: Optional input shape (excluding batch dimension)

        Returns:
            Summary string

        Example:
            >>> model = MyModel(input_length=SIGNAL_LENGTH, num_classes=NUM_CLASSES)
            >>> print(model.summary(input_shape=(SIGNAL_LENGTH,)))
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Model: {self.__class__.__name__}")
        lines.append("=" * 80)

        # Parameters
        params = self.count_parameters()
        lines.append(f"Total parameters: {params['total']:,}")
        lines.append(f"Trainable parameters: {params['trainable']:,}")
        lines.append(f"Non-trainable parameters: {params['non_trainable']:,}")

        # Model size
        size_mb = self.get_model_size_mb()
        lines.append(f"Model size: {size_mb:.2f} MB")

        # Input/output shapes
        if input_shape:
            lines.append(f"Input shape: {input_shape}")
            try:
                # Try to infer output shape
                dummy_input = torch.randn(1, *input_shape)
                with torch.no_grad():
                    dummy_output = self.forward(dummy_input)
                lines.append(f"Output shape: {tuple(dummy_output.shape[1:])}")
            except Exception as e:
                lines.append(f"Output shape: <Could not infer: {e}>")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer_state: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optional optimizer state dict
            metrics: Optional dictionary of metrics
            **kwargs: Additional items to save

        Example:
            >>> model.save_checkpoint(
            ...     Path('checkpoints/model_epoch_50.pt'),
            ...     epoch=50,
            ...     optimizer_state=optimizer.state_dict(),
            ...     metrics={'val_acc': 0.95}
            ... )
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config(),
            'model_class': self.__class__.__name__
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if metrics is not None:
            checkpoint['metrics'] = metrics

        # Add any additional items
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        device: Optional[torch.device] = None
    ) -> Tuple['BaseModel', Dict[str, Any]]:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model onto

        Returns:
            (model, checkpoint_dict) tuple

        Example:
            >>> model, checkpoint = MyModel.load_checkpoint('checkpoints/best.pt')
            >>> epoch = checkpoint['epoch']
            >>> metrics = checkpoint['metrics']
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device)

        # Create model instance (subclass must handle config)
        # This is a placeholder - actual instantiation depends on subclass
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")

        return checkpoint

    def freeze_backbone(self) -> None:
        """
        Freeze all parameters (useful for transfer learning).

        Example:
            >>> model.freeze_backbone()
            >>> # Now only train final classifier
        """
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Froze all model parameters")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all parameters.

        Example:
            >>> model.unfreeze_backbone()
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all model parameters")

    def get_layer_names(self) -> list:
        """
        Get names of all layers/modules.

        Returns:
            List of layer names
        """
        return [name for name, _ in self.named_modules() if name]

    def get_activation(self, layer_name: str) -> torch.Tensor:
        """
        Get activation from a specific layer (requires forward hook).

        This is a placeholder - actual implementation requires registering hooks.

        Args:
            layer_name: Name of layer to get activation from

        Returns:
            Activation tensor
        """
        raise NotImplementedError("Activation extraction requires forward hooks")

    def to_device(self, device: torch.device) -> 'BaseModel':
        """
        Move model to device and return self for chaining.

        Args:
            device: Target device

        Returns:
            self

        Example:
            >>> model = MyModel(...).to_device(torch.device('cuda'))
        """
        self.to(device)
        logger.debug(f"Moved model to {device}")
        return self


def print_model_summary(model: BaseModel, input_shape: Optional[Tuple[int, ...]] = None):
    """
    Print formatted model summary.

    Args:
        model: Model to summarize
        input_shape: Optional input shape

    Example:
        >>> model = MyModel(input_length=SIGNAL_LENGTH, num_classes=NUM_CLASSES)
        >>> print_model_summary(model, input_shape=(SIGNAL_LENGTH,))
    """
    print(model.summary(input_shape=input_shape))


def get_model_info(model: BaseModel) -> Dict[str, Any]:
    """
    Get comprehensive model information.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with model info

    Example:
        >>> info = get_model_info(model)
        >>> print(json.dumps(info, indent=2))
    """
    params = model.count_parameters()

    info = {
        'class_name': model.__class__.__name__,
        'total_parameters': params['total'],
        'trainable_parameters': params['trainable'],
        'non_trainable_parameters': params['non_trainable'],
        'model_size_mb': model.get_model_size_mb(),
        'config': model.get_config()
    }

    return info
