"""Neural network models for bearing fault diagnosis."""

from .base_model import BaseModel
from .cnn_1d import CNN1D, create_cnn1d
from .resnet_1d import ResNet1D, create_resnet18_1d, create_resnet34_1d
from .transformer import SignalTransformer, create_transformer
from .hybrid_pinn import HybridPINN, create_hybrid_pinn
from .ensemble import (
    EnsembleModel,
    VotingEnsemble,
    StackedEnsemble,
    create_voting_ensemble,
    create_stacked_ensemble
)
from .model_factory import (
    create_model,
    create_model_from_config,
    load_pretrained,
    save_checkpoint,
    create_ensemble,
    list_available_models,
    register_model,
    get_model_info,
    print_model_summary
)

__all__ = [
    # Base
    'BaseModel',

    # CNN models
    'CNN1D',
    'create_cnn1d',

    # ResNet models
    'ResNet1D',
    'create_resnet18_1d',
    'create_resnet34_1d',

    # Transformer
    'SignalTransformer',
    'create_transformer',

    # Physics-informed
    'HybridPINN',
    'create_hybrid_pinn',

    # Ensemble
    'EnsembleModel',
    'VotingEnsemble',
    'StackedEnsemble',
    'create_voting_ensemble',
    'create_stacked_ensemble',

    # Factory functions
    'create_model',
    'create_model_from_config',
    'load_pretrained',
    'save_checkpoint',
    'create_ensemble',
    'list_available_models',
    'register_model',
    'get_model_info',
    'print_model_summary',
]
