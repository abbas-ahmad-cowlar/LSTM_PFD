
import pytest
import torch
import torch.nn as nn
from packages.core.models.cnn.cnn_1d import CNN1D
from packages.core.models.cnn.attention_cnn import AttentionCNN1D
from packages.core.models.cnn.multi_scale_cnn import MultiScaleCNN1D
from packages.core.models.resnet.resnet_1d import ResNet1D
from packages.core.models.resnet.se_resnet import SEResNet1D
from packages.core.models.resnet.wide_resnet import WideResNet1D
# from packages.core.models.transformer.vision_transformer_1d import VisionTransformer1D
from packages.core.models.pinn.hybrid_pinn import HybridPINN
from packages.core.models.pinn.knowledge_graph_pinn import KnowledgeGraphPINN
from packages.core.models.pinn.multitask_pinn import MultitaskPINN
# from packages.core.models.pinn.physics_constrained_cnn import PhysicsConstrainedCNN
from packages.core.models.hybrid.cnn_lstm import CNNLSTM
from packages.core.models.hybrid.cnn_tcn import CNNTCN
from packages.core.models.hybrid.cnn_transformer import CNNTransformerHybrid
from packages.core.models.hybrid.multiscale_cnn import MultiScaleCNN as HybridMultiScaleCNN 
from packages.core.models.efficientnet.efficientnet_1d import EfficientNet1D

@pytest.mark.parametrize("model_class, config", [
    (CNN1D, {}),
    (AttentionCNN1D, {}),
    (MultiScaleCNN1D, {}),
    (ResNet1D, {'base_filters': 16, 'n_blocks': [2, 2, 2, 2]}),
    (SEResNet1D, {}),
    (WideResNet1D, {}),
    # (VisionTransformer1D, {}),
    (HybridPINN, {}),
    (KnowledgeGraphPINN, {}),
    (MultitaskPINN, {}),
    # (PhysicsConstrainedCNN, {}),
    (CNNLSTM, {}),
    (CNNTCN, {}),
    (CNNTransformerHybrid, {}),
    (HybridMultiScaleCNN, {}),
    (EfficientNet1D, {})
])
def test_model_forward_backward(model_class, config):
    """Test instantiation, forward pass, and backward pass."""
    batch_size = 2
    seq_len = 5000 # Use smaller length for speed, or 102400 if strictly required
    num_classes = 11
    
    # Instantiate
    try:
        model = model_class(**config) if config else model_class()
    except TypeError:
        # Some models might require args in init, assume default config for now
        # If failure, we will need to adjust specific configs
        model = model_class(num_classes=num_classes)

    # Move to CPU
    model = model.cpu()
    
    # Dummy input
    x = torch.randn(batch_size, 1, seq_len)
    
    # Forward
    y = model(x)
    
    # Check output shape
    assert y.shape[0] == batch_size
    assert y.shape[1] == num_classes
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    # Check gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad

if __name__ == "__main__":
    # Allow manual run
    pytest.main([__file__])
