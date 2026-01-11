
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from packages.core.evaluation.evaluator import ModelEvaluator

class MockModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class TestModelEvaluator(unittest.TestCase):
    """Test evaluation pipeline."""

    def setUp(self):
        """Setup mock model and data."""
        self.input_dim = 10
        self.num_classes = 3
        self.device = 'cpu'
        
        self.model = MockModel(self.input_dim, self.num_classes)
        
        # Create dummy data
        X = torch.randn(20, self.input_dim)
        y = torch.randint(0, self.num_classes, (20,))
        
        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=4)
        
        self.evaluator = ModelEvaluator(self.model, device=self.device)

    def test_evaluate(self):
        """Test evaluate method returns correct keys."""
        results = self.evaluator.evaluate(self.dataloader)
        
        self.assertIn('accuracy', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('per_class_metrics', results)
        self.assertIn('predictions', results)
        
        # Check integrity
        self.assertEqual(len(results['predictions']), 20)
        self.assertIsInstance(results['accuracy'], float)

    def test_per_class_metrics(self):
        """Test per-class metrics calculation."""
        # Use known predictions/targets
        preds = np.array([0, 1, 2, 0, 1])
        targets = np.array([0, 1, 2, 0, 0]) # Last one wrong
        probs = np.random.rand(5, 3)
        
        metrics = self.evaluator.compute_per_class_metrics(preds, targets, probs)
        
        self.assertIn('Class 0', metrics)
        self.assertIn('precision', metrics['Class 0'])
        
        # Class 0: 3 actuals, 2 predicted correctly (recall 2/3), 2 predicted (precision 1.0)
        # targets: 0, 0, 0 (indices 0, 3, 4). preds: 0, 0 (indices 0, 3). Pred index 4 is 1.
        # Wait, index 4 target is 0, predicted 1.
        # So recall = 2/3 = 0.66. Precision = 2/2 = 1.0.
        
        self.assertAlmostEqual(metrics['Class 0']['precision'], 1.0)
        self.assertAlmostEqual(metrics['Class 0']['recall'], 2/3)

    def test_classification_report(self):
        """Test report generation."""
        preds = np.array([0, 1, 0, 1])
        targets = np.array([0, 1, 1, 0])
        report = self.evaluator.generate_classification_report(preds, targets)
        self.assertIsInstance(report, str)
        self.assertIn('precision', report)

