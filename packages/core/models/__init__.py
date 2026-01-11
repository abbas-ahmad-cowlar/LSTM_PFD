"""Neural network models for bearing fault diagnosis."""

from .base_model import BaseModel
from .cnn_1d import CNN1D, create_cnn1d
from .resnet_1d import ResNet1D, create_resnet18_1d, create_resnet34_1d
from .transformer import SignalTransformer, create_transformer
from .hybrid_pinn import HybridPINN, create_hybrid_pinn
from .legacy_ensemble import (
    EnsembleModel,
    VotingEnsemble,
    StackedEnsemble,
    create_voting_ensemble,
    create_stacked_ensemble
)

# Phase 4: Advanced Transformer Variants
from .transformer.vision_transformer_1d import (
    VisionTransformer1D,
    create_vit_1d,
    vit_tiny_1d,
    vit_small_1d,
    vit_base_1d
)
from .hybrid.cnn_transformer import (
    CNNTransformerHybrid,
    create_cnn_transformer_hybrid,
    cnn_transformer_small,
    cnn_transformer_base,
    cnn_transformer_large
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

# Phase 8: Advanced Ensemble Methods
from .ensemble.voting_ensemble import (
    VotingEnsemble as VotingEnsembleV2,
    optimize_ensemble_weights,
    soft_voting,
    hard_voting
)
from .ensemble.stacking_ensemble import (
    StackingEnsemble as StackingEnsembleV2,
    train_stacking,
    create_meta_features
)
from .ensemble.boosting_ensemble import (
    BoostingEnsemble,
    AdaptiveBoosting,
    train_boosting
)
from .ensemble.mixture_of_experts import (
    MixtureOfExperts,
    GatingNetwork,
    ExpertModel,
    create_specialized_experts
)

# Phase 8: Multi-Modal Fusion
from .fusion.early_fusion import (
    EarlyFusion,
    SimpleEarlyFusion,
    MultiModalFeatureExtractor,
    create_early_fusion,
    extract_and_concatenate_features
)
from .fusion.late_fusion import (
    LateFusion,
    create_late_fusion,
    train_late_fusion_weights,
    late_fusion_weighted_average,
    late_fusion_max,
    late_fusion_product,
    late_fusion_borda_count
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

    # Phase 4: Advanced Transformer Variants
    'VisionTransformer1D',
    'create_vit_1d',
    'vit_tiny_1d',
    'vit_small_1d',
    'vit_base_1d',
    'CNNTransformerHybrid',
    'create_cnn_transformer_hybrid',
    'cnn_transformer_small',
    'cnn_transformer_base',
    'cnn_transformer_large',

    # Physics-informed
    'HybridPINN',
    'create_hybrid_pinn',

    # Ensemble (Legacy)
    'EnsembleModel',
    'VotingEnsemble',
    'StackedEnsemble',
    'create_voting_ensemble',
    'create_stacked_ensemble',

    # Phase 8: Advanced Ensemble Methods
    'VotingEnsembleV2',
    'optimize_ensemble_weights',
    'soft_voting',
    'hard_voting',
    'StackingEnsembleV2',
    'train_stacking',
    'create_meta_features',
    'BoostingEnsemble',
    'AdaptiveBoosting',
    'train_boosting',
    'MixtureOfExperts',
    'GatingNetwork',
    'ExpertModel',
    'create_specialized_experts',

    # Phase 8: Multi-Modal Fusion
    'EarlyFusion',
    'SimpleEarlyFusion',
    'MultiModalFeatureExtractor',
    'create_early_fusion',
    'extract_and_concatenate_features',
    'LateFusion',
    'create_late_fusion',
    'train_late_fusion_weights',
    'late_fusion_weighted_average',
    'late_fusion_max',
    'late_fusion_product',
    'late_fusion_borda_count',

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
