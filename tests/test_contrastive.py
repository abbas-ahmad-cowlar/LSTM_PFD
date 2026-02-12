"""
Test: Contrastive Learning Integration
Verifies contrastive models, losses, datasets, and training utilities.
"""
import pytest
import torch
import numpy as np
from packages.core.models.model_factory import create_model, MODEL_REGISTRY


NUM_CLASSES = 11
SIGNAL_LENGTH = 5000
BATCH = 4


# ==================================================================
# Test Contrastive Models (via factory)
# ==================================================================
class TestContrastiveFactory:
    """Contrastive models should instantiate from factory."""

    def test_signal_encoder_create(self):
        model = create_model('signal_encoder', num_classes=NUM_CLASSES)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_signal_encoder_forward(self):
        model = create_model('signal_encoder', num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(BATCH, 1, SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, 128), f"Expected ({BATCH}, 128), got {out.shape}"

    def test_signal_encoder_return_features(self):
        model = create_model('signal_encoder', num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(BATCH, 1, SIGNAL_LENGTH)
        with torch.no_grad():
            features = model(x, return_features=True)
        assert features.shape == (BATCH, 256), f"Expected ({BATCH}, 256), got {features.shape}"

    def test_contrastive_classifier_create(self):
        model = create_model('contrastive_classifier', num_classes=NUM_CLASSES)
        assert model is not None

    def test_contrastive_classifier_forward(self):
        model = create_model('contrastive_classifier', num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(BATCH, 1, SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_registry_entries(self):
        assert 'signal_encoder' in MODEL_REGISTRY
        assert 'contrastive_classifier' in MODEL_REGISTRY


# ==================================================================
# Test Loss Functions
# ==================================================================
class TestContrastiveLosses:
    """Verify loss functions compute valid scalar losses."""

    def test_physics_infonce_loss(self):
        from packages.core.training.contrastive.losses import PhysicsInfoNCELoss
        loss_fn = PhysicsInfoNCELoss(temperature=0.07)
        anchor = torch.randn(BATCH, 128)
        positive = torch.randn(BATCH, 128)
        negatives = torch.randn(BATCH, 5, 128)
        loss = loss_fn(anchor, positive, negatives)
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_ntxent_loss(self):
        from packages.core.training.contrastive.losses import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.5)
        z1 = torch.randn(BATCH, 128)
        z2 = torch.randn(BATCH, 128)
        loss = loss_fn(z1, z2)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_simclr_loss(self):
        from packages.core.training.contrastive.losses import SimCLRLoss
        loss_fn = SimCLRLoss(temperature=0.5)
        z1 = torch.randn(BATCH, 128)
        z2 = torch.randn(BATCH, 128)
        loss = loss_fn(z1, z2)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_gradient(self):
        """Verify losses allow gradient backpropagation."""
        from packages.core.training.contrastive.losses import PhysicsInfoNCELoss
        loss_fn = PhysicsInfoNCELoss()
        anchor = torch.randn(BATCH, 128, requires_grad=True)
        positive = torch.randn(BATCH, 128)
        negatives = torch.randn(BATCH, 3, 128)
        loss = loss_fn(anchor, positive, negatives)
        loss.backward()
        assert anchor.grad is not None


# ==================================================================
# Test Physics Similarity
# ==================================================================
class TestPhysicsSimilarity:
    """Verify physics similarity functions."""

    def test_compute_similarity_identical(self):
        from packages.core.training.contrastive.physics_similarity import compute_physics_similarity
        p = {'eccentricity': 0.5, 'clearance': 0.1, 'viscosity': 100}
        sim = compute_physics_similarity(p, p)
        assert abs(sim - 1.0) < 1e-6, f"Identical params should give similarity ~1.0, got {sim}"

    def test_compute_similarity_different(self):
        from packages.core.training.contrastive.physics_similarity import compute_physics_similarity
        p1 = {'eccentricity': 0.1, 'clearance': 0.01}
        p2 = {'eccentricity': 0.9, 'clearance': 0.49}
        sim = compute_physics_similarity(p1, p2)
        assert 0.0 <= sim <= 1.0

    def test_select_pairs(self):
        from packages.core.training.contrastive.physics_similarity import select_positive_negative_pairs
        params = [
            {'eccentricity': 0.1, 'clearance': 0.02},
            {'eccentricity': 0.12, 'clearance': 0.02},
            {'eccentricity': 0.9, 'clearance': 0.4},
            {'eccentricity': 0.88, 'clearance': 0.39},
            {'eccentricity': 0.5, 'clearance': 0.2},
        ]
        pairs = select_positive_negative_pairs(params, similarity_threshold=0.9, num_negatives=2)
        # Should find some pairs
        assert isinstance(pairs, list)
        for anchor, pos, negs in pairs:
            assert isinstance(anchor, (int, np.integer))
            assert isinstance(pos, (int, np.integer))
            assert isinstance(negs, list)


# ==================================================================
# Test Datasets
# ==================================================================
class TestContrastiveDatasets:
    """Verify dataset classes produce correct output shapes."""

    def test_physics_contrastive_dataset(self):
        from packages.core.training.contrastive.dataset import PhysicsContrastiveDataset
        signals = np.random.randn(20, 1, 1024).astype(np.float32)
        params = [{'eccentricity': np.random.uniform(0, 1), 'clearance': np.random.uniform(0.01, 0.5)}
                  for _ in range(20)]
        ds = PhysicsContrastiveDataset(signals, params, similarity_threshold=0.7, augment=False)
        if len(ds) > 0:
            anchor, pos, negs = ds[0]
            assert anchor.shape[0] == 1  # channel dim
            assert pos.shape == anchor.shape

    def test_contrastive_spectrogram_dataset(self):
        from packages.core.training.contrastive.dataset import ContrastiveSpectrogramDataset
        specs = np.random.randn(10, 64, 64).astype(np.float32)
        ds = ContrastiveSpectrogramDataset(specs)
        v1, v2 = ds[0]
        assert v1.shape == (1, 64, 64)
        assert v2.shape == (1, 64, 64)

    def test_finetune_dataset(self):
        from packages.core.training.contrastive.dataset import FineTuneDataset
        signals = np.random.randn(10, 1024).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ds = FineTuneDataset(signals, labels)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (1, 1024)  # should have channel dim added


# ==================================================================
# Test Projection & Encoder
# ==================================================================
class TestContrastiveComponents:
    """Verify ProjectionHead and ContrastiveEncoder."""

    def test_projection_head(self):
        from packages.core.models.contrastive.projection import ProjectionHead
        proj = ProjectionHead(input_dim=512, output_dim=128)
        x = torch.randn(BATCH, 512)
        out = proj(x)
        assert out.shape == (BATCH, 128)

    def test_signal_encoder_module(self):
        from packages.core.models.contrastive.signal_encoder import SignalEncoder
        enc = SignalEncoder(embedding_dim=64)
        x = torch.randn(BATCH, 1, 2048)
        out = enc(x)
        assert out.shape == (BATCH, 64)
        # Check L2 normalized
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)


# ==================================================================
# Test Import Paths
# ==================================================================
class TestContrastiveImports:
    """Verify all contrastive modules are importable."""

    def test_import_training_package(self):
        from packages.core.training.contrastive import (
            compute_physics_similarity,
            select_positive_negative_pairs,
            PhysicsInfoNCELoss,
            NTXentLoss,
            SimCLRLoss,
            PhysicsContrastiveDataset,
            ContrastiveSpectrogramDataset,
            FineTuneDataset,
            ContrastivePretrainer,
            ContrastiveFineTuner,
            pretrain_contrastive,
        )

    def test_import_models_package(self):
        from packages.core.models.contrastive import (
            SignalEncoder,
            ContrastiveClassifier,
            ProjectionHead,
            ContrastiveEncoder,
        )
