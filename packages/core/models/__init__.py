"""
Neural network models for bearing fault diagnosis.

Curated zoo (Convergence Plan Part I §4):
  Tier 1 — CNN1D, AttentionCNN1D, CNNLSTM, ResNet1D(18), PatchTST,
           HybridPINN, PhysicsConstrainedCNN, MultitaskPINN, VotingEnsemble
           (+ classical RF/SVM/GB in .classical, sklearn-based)
  Tier 2 — MultiScaleCNN1D, SEResNet1D, SignalTransformer

Everything else was pruned 2026-06; recoverable from tag
`pre-convergence-2026-06`.
"""

from .base_model import BaseModel

# Tier 1
from .cnn.cnn_1d import CNN1D, create_cnn1d
from .cnn.attention_cnn import AttentionCNN1D
from .hybrid.cnn_lstm import CNNLSTM, create_cnn_lstm
from .resnet.resnet_1d import ResNet1D, create_resnet18_1d
from .transformer.patchtst import PatchTST
from .pinn.hybrid_pinn import HybridPINN, create_hybrid_pinn
from .pinn.physics_constrained_cnn import (
    PhysicsConstrainedCNN,
    create_physics_constrained_cnn,
)
from .pinn.multitask_pinn import MultitaskPINN, create_multitask_pinn
from .ensemble.voting_ensemble import VotingEnsemble, create_voting_ensemble

# Tier 2 (extension — smoke-tested, benchmark-optional)
from .cnn.multi_scale_cnn import MultiScaleCNN1D
from .resnet.se_resnet import SEResNet1D, create_se_resnet18_1d
from .transformer.signal_transformer import SignalTransformer, create_transformer

# Factory
from .model_factory import (
    MODEL_REGISTRY,
    create_model,
    create_model_from_config,
    create_attention_cnn,
    create_multi_scale_cnn,
    create_patchtst,
    load_pretrained,
    save_checkpoint,
    create_ensemble,
    list_available_models,
    register_model,
    get_model_info,
    print_model_summary,
)

__all__ = [
    'BaseModel',
    # Tier 1
    'CNN1D', 'create_cnn1d',
    'AttentionCNN1D', 'create_attention_cnn',
    'CNNLSTM', 'create_cnn_lstm',
    'ResNet1D', 'create_resnet18_1d',
    'PatchTST', 'create_patchtst',
    'HybridPINN', 'create_hybrid_pinn',
    'PhysicsConstrainedCNN', 'create_physics_constrained_cnn',
    'MultitaskPINN', 'create_multitask_pinn',
    'VotingEnsemble', 'create_voting_ensemble',
    # Tier 2
    'MultiScaleCNN1D', 'create_multi_scale_cnn',
    'SEResNet1D', 'create_se_resnet18_1d',
    'SignalTransformer', 'create_transformer',
    # Factory
    'MODEL_REGISTRY',
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
