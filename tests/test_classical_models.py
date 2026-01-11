
import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from sklearn.datasets import make_classification
from packages.core.models.classical.svm_classifier import SVMClassifier
from packages.core.models.classical.random_forest import RandomForestClassifier
from packages.core.models.classical.gradient_boosting import GradientBoostingClassifier

class TestClassicalBase(unittest.TestCase):
    """Base class for classical model tests."""

    def setUp(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # specific dummy data for testing
        self.X, self.y = make_classification(
            n_samples=100, 
            n_features=20, 
            n_informative=10, 
            n_classes=3, 
            random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=20, 
            n_features=20, 
            n_informative=10,
            n_classes=3, 
            random_state=43
        )
        
    def tearDown(self):
        """Cleanup."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

class TestSVMClassifier(TestClassicalBase):
    """Test SVM Classifier wrapper."""

    def test_train_predict(self):
        """Test training and prediction."""
        model = SVMClassifier(use_ecoc=True, random_state=42)
        model.train(self.X, self.y)
        
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), 20)
        
        acc = model.score(self.X_test, self.y_test)
        self.assertIsInstance(acc, float)
        self.assertTrue(0 <= acc <= 1)

    def test_save_load(self):
        """Test model persistence."""
        model = SVMClassifier(use_ecoc=True)
        model.train(self.X, self.y)
        
        save_path = self.temp_dir / "svm_test.joblib"
        model.save(save_path)
        
        self.assertTrue(save_path.exists())
        
        loaded_model = SVMClassifier()
        loaded_model.load(save_path)
        
        # Check if predictions match
        preds_orig = model.predict(self.X_test)
        preds_loaded = loaded_model.predict(self.X_test)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_predict_proba(self):
        """Test probability outputs."""
        # ECOC doesn't support predict_proba well, so test without it
        model = SVMClassifier(use_ecoc=False)
        model.train(self.X, self.y)
        
        proba = model.predict_proba(self.X_test)
        
        # Should sum to 1
        np.testing.assert_array_almost_equal(
            np.sum(proba, axis=1), 
            np.ones(len(self.X_test))
        )
        # Should match n_classes (we generated 3 classes, but implementation uses NUM_CLASSES constants maybe?)
        # Wait, the constants logic in svm_classifier uses NUM_CLASSES?
        # Reading file: imports NUM_CLASSES but doesn't seem to force it in fitting if sklearn is used.
        # Sklearn infers classes from y.
        # Let's check shape.
        self.assertEqual(proba.shape[1], 3) # We used n_classes=3 in make_classification

class TestRandomForest(TestClassicalBase):
    """Test Random Forest wrapper."""

    def test_train_predict(self):
        """Test training and prediction."""
        model = RandomForestClassifier(random_state=42)
        model.train(self.X, self.y, hyperparams={'n_estimators': 10})
        
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), 20)

    def test_feature_importance(self):
        """Test feature importance retrieval."""
        model = RandomForestClassifier(random_state=42)
        model.train(self.X, self.y, hyperparams={'n_estimators': 10})
        
        importances = model.get_feature_importances()
        self.assertEqual(len(importances), 20)
        self.assertAlmostEqual(sum(importances), 1.0, delta=0.01)

class TestGradientBoosting(TestClassicalBase):
    """Test Gradient Boosting wrapper."""

    def test_train_predict(self):
        """Test training and prediction."""
        model = GradientBoostingClassifier(random_state=42)
        model.train(self.X, self.y, hyperparams={'n_estimators': 10})
        
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), 20)
