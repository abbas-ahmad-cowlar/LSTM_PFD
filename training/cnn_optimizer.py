"""
Optimizer configurations for CNN training.

Purpose:
    Create optimizers with proper hyperparameters for CNN training:
    - AdamW: Adam with decoupled weight decay (recommended)
    - SGD: Stochastic Gradient Descent with momentum
    - RMSprop: Adaptive learning rate method

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import torch
import torch.optim as optim
from typing import Iterable, Dict, Any, Optional


def create_adamw_optimizer(
    model_params: Iterable,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8
) -> optim.AdamW:
    """
    Create AdamW optimizer (Adam with decoupled weight decay).

    AdamW is preferred over Adam for better generalization.
    Decouples weight decay from gradient updates for proper L2 regularization.

    Recommended hyperparameters for CNN:
    - lr: 1e-3 (initial learning rate, will be scheduled)
    - weight_decay: 1e-4 (L2 regularization)
    - betas: (0.9, 0.999) (momentum and RMSprop-like term)

    Args:
        model_params: Model parameters (model.parameters())
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Coefficients for gradient and squared gradient
        eps: Small constant for numerical stability

    Returns:
        AdamW optimizer

    Example:
        >>> model = CNN1D(num_classes=NUM_CLASSES)
        >>> optimizer = create_adamw_optimizer(model.parameters(), lr=1e-3)
    """
    optimizer = optim.AdamW(
        params=model_params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

    return optimizer


def create_sgd_optimizer(
    model_params: Iterable,
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    nesterov: bool = True
) -> optim.SGD:
    """
    Create SGD optimizer with Nesterov momentum.

    SGD with momentum often achieves better final accuracy than Adam,
    but may require more careful learning rate scheduling.

    Recommended hyperparameters:
    - lr: 1e-2 (higher than Adam, will be scheduled)
    - momentum: 0.9 (standard value)
    - weight_decay: 1e-4 (L2 regularization)
    - nesterov: True (lookahead momentum, often better)

    Args:
        model_params: Model parameters
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: Weight decay
        nesterov: Whether to use Nesterov momentum

    Returns:
        SGD optimizer

    Example:
        >>> optimizer = create_sgd_optimizer(model.parameters(), lr=1e-2)
    """
    optimizer = optim.SGD(
        params=model_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov
    )

    return optimizer


def create_rmsprop_optimizer(
    model_params: Iterable,
    lr: float = 1e-3,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    momentum: float = 0.0
) -> optim.RMSprop:
    """
    Create RMSprop optimizer.

    RMSprop adapts learning rate per parameter based on recent gradients.
    Less commonly used than Adam/SGD but can work well for CNNs.

    Args:
        model_params: Model parameters
        lr: Learning rate
        alpha: Smoothing constant
        eps: Small constant for numerical stability
        weight_decay: Weight decay
        momentum: Momentum factor

    Returns:
        RMSprop optimizer

    Example:
        >>> optimizer = create_rmsprop_optimizer(model.parameters())
    """
    optimizer = optim.RMSprop(
        params=model_params,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum
    )

    return optimizer


def create_optimizer(
    optimizer_type: str,
    model_params: Iterable,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """
    Factory function to create optimizer by name.

    Args:
        optimizer_type: Type of optimizer ('adamw', 'sgd', 'rmsprop')
        model_params: Model parameters
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance

    Example:
        >>> optimizer = create_optimizer('adamw', model.parameters(), lr=1e-3)
        >>> optimizer = create_optimizer('sgd', model.parameters(), lr=1e-2, momentum=0.9)
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'adamw':
        betas = kwargs.get('betas', (0.9, 0.999))
        return create_adamw_optimizer(
            model_params=model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )

    elif optimizer_type == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', True)
        return create_sgd_optimizer(
            model_params=model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )

    elif optimizer_type == 'rmsprop':
        alpha = kwargs.get('alpha', 0.99)
        return create_rmsprop_optimizer(
            model_params=model_params,
            lr=lr,
            alpha=alpha,
            weight_decay=weight_decay
        )

    elif optimizer_type == 'adam':
        # Standard Adam (not recommended, use AdamW instead)
        betas = kwargs.get('betas', (0.9, 0.999))
        return optim.Adam(
            params=model_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. "
                        f"Choose from: adamw, sgd, rmsprop, adam")


class OptimizerConfig:
    """
    Predefined optimizer configurations for different scenarios.
    """

    @staticmethod
    def default() -> Dict[str, Any]:
        """
        Default configuration (AdamW with standard hyperparameters).

        Returns:
            Config dict for create_optimizer()
        """
        return {
            'optimizer_type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)
        }

    @staticmethod
    def fast_convergence() -> Dict[str, Any]:
        """
        Configuration for fast convergence (higher learning rate).

        Returns:
            Config dict
        """
        return {
            'optimizer_type': 'adamw',
            'lr': 3e-3,  # Higher LR for faster training
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)
        }

    @staticmethod
    def strong_regularization() -> Dict[str, Any]:
        """
        Configuration with strong regularization (prevent overfitting).

        Returns:
            Config dict
        """
        return {
            'optimizer_type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-3,  # 10x stronger weight decay
            'betas': (0.9, 0.999)
        }

    @staticmethod
    def sgd_baseline() -> Dict[str, Any]:
        """
        SGD configuration (often best final accuracy with proper scheduling).

        Returns:
            Config dict
        """
        return {
            'optimizer_type': 'sgd',
            'lr': 1e-2,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'nesterov': True
        }


def get_parameter_groups(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    no_decay_bias: bool = True
) -> list:
    """
    Create parameter groups with different weight decay for biases and norms.

    Common practice: don't apply weight decay to bias and normalization layers.

    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay for conv/linear weights
        no_decay_bias: Whether to exclude bias and norm params from weight decay

    Returns:
        List of parameter group dicts

    Example:
        >>> model = CNN1D(num_classes=NUM_CLASSES)
        >>> param_groups = get_parameter_groups(model, lr=1e-3, weight_decay=1e-4)
        >>> optimizer = optim.AdamW(param_groups)
    """
    if not no_decay_bias:
        # All parameters with same weight decay
        return [{'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay}]

    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to bias and batch norm parameters
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0}
    ]

    return param_groups


def test_optimizers():
    """Test optimizer creation."""
    print("=" * 60)
    print("Testing CNN Optimizers")
    print("=" * 60)

    # Import constants
    from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

    # Create dummy model
    from models.cnn.cnn_1d import CNN1D
    model = CNN1D(num_classes=NUM_CLASSES)

    # Test AdamW
    print("\n1. Testing AdamW optimizer...")
    optimizer = create_adamw_optimizer(model.parameters(), lr=1e-3)
    print(f"   Created: {type(optimizer).__name__}")
    print(f"   LR: {optimizer.param_groups[0]['lr']}")
    print(f"   Weight decay: {optimizer.param_groups[0]['weight_decay']}")

    # Test SGD
    print("\n2. Testing SGD optimizer...")
    optimizer = create_sgd_optimizer(model.parameters(), lr=1e-2)
    print(f"   Created: {type(optimizer).__name__}")
    print(f"   Momentum: {optimizer.param_groups[0]['momentum']}")
    print(f"   Nesterov: {optimizer.param_groups[0]['nesterov']}")

    # Test RMSprop
    print("\n3. Testing RMSprop optimizer...")
    optimizer = create_rmsprop_optimizer(model.parameters())
    print(f"   Created: {type(optimizer).__name__}")

    # Test factory function
    print("\n4. Testing create_optimizer factory...")
    optimizer1 = create_optimizer('adamw', model.parameters(), lr=1e-3)
    optimizer2 = create_optimizer('sgd', model.parameters(), lr=1e-2)
    print(f"   Created: {type(optimizer1).__name__}, {type(optimizer2).__name__}")

    # Test OptimizerConfig presets
    print("\n5. Testing OptimizerConfig presets...")
    default_config = OptimizerConfig.default()
    print(f"   Default config: {default_config}")
    fast_config = OptimizerConfig.fast_convergence()
    print(f"   Fast config: lr={fast_config['lr']}")

    # Test parameter groups
    print("\n6. Testing parameter groups...")
    param_groups = get_parameter_groups(model, lr=1e-3, weight_decay=1e-4)
    print(f"   Number of groups: {len(param_groups)}")
    print(f"   Group 0 (decay): {len(param_groups[0]['params'])} params")
    print(f"   Group 1 (no decay): {len(param_groups[1]['params'])} params")

    # Test optimizer step
    print("\n7. Testing optimizer step...")
    optimizer = create_adamw_optimizer(model.parameters(), lr=1e-3)
    dummy_input = torch.randn(8, 1, SIGNAL_LENGTH)
    dummy_target = torch.randint(0, 11, (8,))
    output = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"   Optimizer step completed successfully")

    print("\n" + "=" * 60)
    print("✅ All optimizer tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_optimizers()
