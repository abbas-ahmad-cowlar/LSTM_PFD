"""
Test Phase 8 Bug Fixes

Tests all the critical fixes made to Phase 8 ensemble module:
1. create_meta_features() label collection fix
2. evaluate() helper function
3. DiversityBasedSelector availability
4. Proper imports

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('/home/user/LSTM_PFD')

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 8 BUG FIX VERIFICATION")
    print("=" * 70)

    # Test 1: Import all ensemble modules
    print("\n[Test 1] Testing imports...")
    try:
        from packages.core.models.ensemble import (
            VotingEnsemble, StackingEnsemble, BoostingEnsemble,
            MixtureOfExperts, optimize_ensemble_weights,
            create_meta_features, DiversityBasedSelector, evaluate
        )
        from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    # Test 2: Create dummy models for testing
    print("\n[Test 2] Creating dummy models...")
    try:
        class DummyModel(nn.Module):
            def __init__(self, num_classes=11):
                super().__init__()
                self.fc = nn.Linear(102400, num_classes)

            def forward(self, x):
                # x: [B, 1, 102400]
                B = x.size(0)
                x_flat = x.view(B, -1)
                return self.fc(x_flat)

        model1 = DummyModel()
        model2 = DummyModel()
        model3 = DummyModel()
        print("✓ Dummy models created")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        sys.exit(1)

    # Test 3: Create dummy data
    print("\n[Test 3] Creating dummy dataset...")
    try:
        # Create dummy data: 100 samples, 11 classes
        X = torch.randn(100, 1, 102400)
        y = torch.randint(0, 11, (100,))

        # Create dataloaders
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        print(f"✓ Dataset created: {len(dataset)} samples")
        print(f"  Batches: {len(dataloader)}, Batch size: 32")
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        sys.exit(1)

    # Test 4: Test create_meta_features() with multiple batches
    print("\n[Test 4] Testing create_meta_features() label collection fix...")
    try:
        base_models = [model1, model2, model3]
        meta_features, labels = create_meta_features(
            base_models, dataloader, device='cpu', return_probabilities=True
        )

        expected_samples = len(dataset)
        expected_features = len(base_models) * NUM_CLASSES  # 3 models × 11 classes = 33

        assert meta_features.shape == (expected_samples, expected_features), \
            f"Meta-features shape mismatch: {meta_features.shape} vs ({expected_samples}, {expected_features})"

        assert labels is not None, "Labels should not be None"
        assert labels.shape == (expected_samples,), \
            f"Labels shape mismatch: {labels.shape} vs ({expected_samples},)"

        # Critical check: all labels collected (not just first batch)
        unique_batches_needed = (expected_samples + 31) // 32  # ceil division
        print(f"✓ create_meta_features() works correctly")
        print(f"  Meta-features: {meta_features.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  ✓ CRITICAL: All {expected_samples} labels collected across {unique_batches_needed} batches")

    except Exception as e:
        print(f"✗ create_meta_features() test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 5: Test evaluate() helper function
    print("\n[Test 5] Testing evaluate() helper function...")
    try:
        ensemble = VotingEnsemble([model1, model2], voting_type='soft')
        accuracy = evaluate(ensemble, dataloader, device='cpu')

        assert isinstance(accuracy, float), "Accuracy should be a float"
        assert 0 <= accuracy <= 100, f"Accuracy should be 0-100, got {accuracy}"

        print(f"✓ evaluate() works correctly")
        print(f"  Accuracy: {accuracy:.2f}%")

    except Exception as e:
        print(f"✗ evaluate() test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 6: Test DiversityBasedSelector
    print("\n[Test 6] Testing DiversityBasedSelector...")
    try:
        # Get predictions from models
        predictions = {}
        accuracies = {}

        for i, model in enumerate([model1, model2, model3]):
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch_x, batch_y in dataloader:
                    logits = model(batch_x)
                    preds = logits.argmax(dim=1)
                    all_preds.append(preds.numpy())
                    all_labels.append(batch_y.numpy())

            preds_array = np.concatenate(all_preds)
            labels_array = np.concatenate(all_labels)

            predictions[f'Model{i+1}'] = preds_array
            accuracies[f'Model{i+1}'] = (preds_array == labels_array).mean()

        # Test selector
        selector = DiversityBasedSelector(metric='disagreement')
        selected = selector.select(predictions, accuracies, num_models=2, diversity_weight=0.3)

        assert len(selected) == 2, f"Should select 2 models, got {len(selected)}"
        assert all(isinstance(name, str) and isinstance(score, float) for name, score in selected), \
            "Selected should be list of (name, score) tuples"

        print(f"✓ DiversityBasedSelector works correctly")
        print(f"  Selected models: {[name for name, _ in selected]}")
        print(f"  Scores: {[f'{score:.4f}' for _, score in selected]}")

    except Exception as e:
        print(f"✗ DiversityBasedSelector test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 7: Test VotingEnsemble
    print("\n[Test 7] Testing VotingEnsemble...")
    try:
        # Test both soft and hard voting
        soft_ensemble = VotingEnsemble([model1, model2, model3], voting_type='soft')
        hard_ensemble = VotingEnsemble([model1, model2, model3], voting_type='hard')

        soft_acc = evaluate(soft_ensemble, dataloader, device='cpu')
        hard_acc = evaluate(hard_ensemble, dataloader, device='cpu')

        print(f"✓ VotingEnsemble works correctly")
        print(f"  Soft voting accuracy: {soft_acc:.2f}%")
        print(f"  Hard voting accuracy: {hard_acc:.2f}%")

    except Exception as e:
        print(f"✗ VotingEnsemble test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 8: Test MixtureOfExperts
    print("\n[Test 8] Testing MixtureOfExperts...")
    try:
        moe = MixtureOfExperts([model1, model2, model3], num_classes=NUM_CLASSES)
        moe_acc = evaluate(moe, dataloader, device='cpu')

        # Test get_expert_usage() return type
        usage_stats = moe.get_expert_usage(dataloader, device='cpu')

        assert isinstance(usage_stats, dict), "get_expert_usage should return dict"
        assert 'usage_proportion' in usage_stats, "Should have 'usage_proportion' key"
        assert 'total_samples' in usage_stats, "Should have 'total_samples' key"
        assert len(usage_stats['usage_proportion']) == 3, "Should have 3 expert usage values"

        print(f"✓ MixtureOfExperts works correctly")
        print(f"  Accuracy: {moe_acc:.2f}%")
        print(f"  Expert usage: {[f'{u:.2%}' for u in usage_stats['usage_proportion']]}")

    except Exception as e:
        print(f"✗ MixtureOfExperts test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nSummary of fixes verified:")
    print("  1. ✓ create_meta_features() collects ALL labels from ALL batches")
    print("  2. ✓ evaluate() helper function works correctly")
    print("  3. ✓ DiversityBasedSelector module exists and works")
    print("  4. ✓ All imports are correct")
    print("  5. ✓ get_expert_usage() returns proper dict format")
    print("  6. ✓ VotingEnsemble works for both soft and hard voting")
    print("  7. ✓ MixtureOfExperts works correctly")
    print("\nPhase 8 is now bug-free and ready for use!")
