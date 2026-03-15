"""Quick import smoke test - bypasses training package."""
import sys, os
sys.path.insert(0, '.')
os.environ['TESTING_MODELS_ONLY'] = '1'

# Test individual model imports (these don't go through training/__init__)
from packages.core.models.base_model import BaseModel
print("BaseModel imported OK")

models_to_test = [
    ('packages.core.models.transformer.patchtst', 'PatchTST'),
    ('packages.core.models.transformer.tsmixer', 'TSMixer'),
    ('packages.core.models.cnn.attention_cnn', 'AttentionCNN1D'),
    ('packages.core.models.cnn.attention_cnn', 'LightweightAttentionCNN'),
    ('packages.core.models.cnn.multi_scale_cnn', 'MultiScaleCNN1D'),
    ('packages.core.models.cnn.multi_scale_cnn', 'DilatedMultiScaleCNN'),
    ('packages.core.models.contrastive.signal_encoder', 'SignalEncoder'),
    ('packages.core.models.contrastive.classifier', 'ContrastiveClassifier'),
    ('packages.core.models.spectrogram_cnn.dual_stream_cnn', 'DualStreamCNN'),
    ('packages.core.models.cnn.cnn_1d', 'CNN1D'),
    ('packages.core.models.resnet.resnet_1d', 'ResNet1D'),
    ('packages.core.models.transformer.signal_transformer', 'SignalTransformer'),
    ('packages.core.models.transformer.vision_transformer_1d', 'VisionTransformer1D'),
    ('packages.core.models.hybrid.cnn_transformer', 'CNNTransformerHybrid'),
    ('packages.core.models.efficientnet.efficientnet_1d', 'EfficientNet1D'),
    ('packages.core.models.pinn.hybrid_pinn', 'HybridPINN'),
    ('packages.core.models.pinn.physics_constrained_cnn', 'PhysicsConstrainedCNN'),
    ('packages.core.models.pinn.multitask_pinn', 'MultitaskPINN'),
    ('packages.core.models.pinn.knowledge_graph_pinn', 'KnowledgeGraphPINN'),
]

import importlib

print("=" * 65)
print("Model Architecture Refactor - Import & Inheritance Smoke Test")
print("=" * 65)

all_ok = True
for mod_path, cls_name in models_to_test:
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        is_base = issubclass(cls, BaseModel)
        has_config = 'get_config' in dir(cls)
        status = "OK" if (is_base and has_config) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {status}: {cls_name:35s} BaseModel={is_base}, get_config={has_config}")
    except Exception as e:
        all_ok = False
        print(f"  ERR: {cls_name:35s} {type(e).__name__}: {e}")

print("=" * 65)
if all_ok:
    print("ALL 19 MODELS PASS")
else:
    print("SOME MODELS FAILED")
    sys.exit(1)
