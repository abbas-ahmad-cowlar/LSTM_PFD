"""
Ensemble Voting for Model Combination

Combines predictions from multiple models to improve overall accuracy.

Methods:
- Soft Voting: Weighted average of predicted probabilities
- Hard Voting: Majority vote of predicted classes
- Stacking: Train meta-learner on model predictions

Expected benefit: +1-2% accuracy over best individual model

Reference:
- Dietterich (2000). "Ensemble Methods in Machine Learning"
- Zhou (2012). "Ensemble Methods: Foundations and Algorithms"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class EnsembleVoting:
    """
    Ensemble voting for combining multiple models.

    Args:
        models: List of trained PyTorch models
        weights: Optional weights for each model (for soft voting)
        device: Device to run on
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        device: str = 'cpu'
    ):
        self.models = [model.to(device) for model in models]
        self.device = device

        # Set all models to evaluation mode
        for model in self.models:
            model.eval()

        # Validate or initialize weights
        if weights is None:
            # Equal weights for all models
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            # Normalize weights
            weight_sum = sum(weights)
            self.weights = [w / weight_sum for w in weights]

    def soft_voting(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft voting: Weighted average of predicted probabilities.

        Args:
            inputs: Input tensor [B, C, T]

        Returns:
            predictions: Predicted class indices [B]
            probabilities: Predicted probabilities [B, num_classes]
        """
        inputs = inputs.to(self.device)

        # Get predictions from all models
        all_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        # Weighted average of probabilities
        ensemble_probs = torch.zeros_like(all_probs[0])
        for probs, weight in zip(all_probs, self.weights):
            ensemble_probs += weight * probs

        # Get predicted class
        predictions = torch.argmax(ensemble_probs, dim=1)

        return predictions, ensemble_probs

    def hard_voting(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard voting: Majority vote of predicted classes.

        Args:
            inputs: Input tensor [B, C, T]

        Returns:
            predictions: Predicted class indices [B]
        """
        inputs = inputs.to(self.device)

        # Get predictions from all models
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())

        # Stack predictions: [num_models, batch_size]
        all_preds = np.stack(all_preds, axis=0)

        # Majority vote for each sample
        batch_size = all_preds.shape[1]
        predictions = []

        for i in range(batch_size):
            votes = all_preds[:, i]
            # Count votes with weights
            weighted_votes = {}
            for vote, weight in zip(votes, self.weights):
                weighted_votes[vote] = weighted_votes.get(vote, 0) + weight

            # Get class with most votes
            majority_class = max(weighted_votes, key=weighted_votes.get)
            predictions.append(majority_class)

        return torch.tensor(predictions)

    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        voting_method: str = 'soft'
    ) -> Dict[str, float]:
        """
        Evaluate ensemble on test set.

        Args:
            test_loader: Test data loader
            voting_method: 'soft' or 'hard' voting

        Returns:
            Dictionary with metrics
        """
        correct = 0
        total = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Get ensemble predictions
            if voting_method == 'soft':
                predictions, _ = self.soft_voting(inputs)
            else:
                predictions = self.hard_voting(inputs)
                predictions = predictions.to(self.device)

            # Accuracy
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        accuracy = 100.0 * correct / total

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }


class StackingEnsemble:
    """
    Stacking ensemble: Train meta-learner on base model predictions.

    Two-level architecture:
    1. Base models: Multiple diverse models
    2. Meta-learner: Learns how to combine base predictions

    Args:
        base_models: List of base PyTorch models
        meta_learner: Meta-learner (sklearn classifier)
        device: Device to run on
    """

    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: Optional = None,
        device: str = 'cpu'
    ):
        self.base_models = [model.to(device) for model in base_models]
        self.device = device

        # Set base models to evaluation mode
        for model in self.base_models:
            model.eval()

        # Initialize meta-learner if not provided
        if meta_learner is None:
            self.meta_learner = LogisticRegression(max_iter=1000)
        else:
            self.meta_learner = meta_learner

        self.is_trained = False

    def get_base_predictions(
        self,
        inputs: torch.Tensor
    ) -> np.ndarray:
        """
        Get predictions from all base models.

        Args:
            inputs: Input tensor [B, C, T]

        Returns:
            Base predictions [B, num_models * num_classes]
        """
        inputs = inputs.to(self.device)

        all_probs = []
        with torch.no_grad():
            for model in self.base_models:
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        # Concatenate predictions: [B, num_models * num_classes]
        base_predictions = np.concatenate(all_probs, axis=1)

        return base_predictions

    def train(
        self,
        train_loader: torch.utils.data.DataLoader
    ):
        """
        Train meta-learner on base model predictions.

        Args:
            train_loader: Training data loader
        """
        # Collect base predictions and labels
        all_base_predictions = []
        all_labels = []

        for inputs, labels in train_loader:
            base_preds = self.get_base_predictions(inputs)
            all_base_predictions.append(base_preds)
            all_labels.append(labels.numpy())

        # Concatenate all batches
        X_meta = np.vstack(all_base_predictions)
        y_meta = np.concatenate(all_labels)

        # Train meta-learner
        print(f"Training meta-learner on {len(y_meta)} samples...")
        self.meta_learner.fit(X_meta, y_meta)

        self.is_trained = True
        print("Meta-learner training complete!")

    def predict(
        self,
        inputs: torch.Tensor
    ) -> np.ndarray:
        """
        Predict using stacking ensemble.

        Args:
            inputs: Input tensor [B, C, T]

        Returns:
            Predictions [B]
        """
        if not self.is_trained:
            raise RuntimeError("Meta-learner not trained. Call train() first.")

        # Get base predictions
        base_predictions = self.get_base_predictions(inputs)

        # Meta-learner prediction
        predictions = self.meta_learner.predict(base_predictions)

        return predictions

    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate stacking ensemble on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with metrics
        """
        if not self.is_trained:
            raise RuntimeError("Meta-learner not trained. Call train() first.")

        correct = 0
        total = 0

        for inputs, labels in test_loader:
            predictions = self.predict(inputs)

            # Accuracy
            total += len(labels)
            correct += (predictions == labels.numpy()).sum()

        accuracy = 100.0 * correct / total

        return {
            'accuracy': accuracy,
            'correct': int(correct),
            'total': total
        }


def compare_ensemble_methods(
    models: List[nn.Module],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Compare different ensemble methods.

    Args:
        models: List of trained models
        train_loader: Training data loader (for stacking)
        test_loader: Test data loader
        device: Device to run on

    Returns:
        DataFrame with comparison results
    """
    results = []

    # 1. Individual model performance
    print("Evaluating individual models...")
    for i, model in enumerate(models):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = 100.0 * correct / total
        results.append({
            'method': f'Model {i+1} (individual)',
            'accuracy': accuracy
        })

    # 2. Soft voting
    print("\nEvaluating soft voting ensemble...")
    ensemble = EnsembleVoting(models, device=device)
    metrics = ensemble.evaluate(test_loader, voting_method='soft')
    results.append({
        'method': 'Soft Voting',
        'accuracy': metrics['accuracy']
    })

    # 3. Hard voting
    print("Evaluating hard voting ensemble...")
    metrics = ensemble.evaluate(test_loader, voting_method='hard')
    results.append({
        'method': 'Hard Voting',
        'accuracy': metrics['accuracy']
    })

    # 4. Stacking
    print("\nTraining and evaluating stacking ensemble...")
    stacking = StackingEnsemble(models, device=device)
    stacking.train(train_loader)
    metrics = stacking.evaluate(test_loader)
    results.append({
        'method': 'Stacking',
        'accuracy': metrics['accuracy']
    })

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('accuracy', ascending=False)

    return df


# Example usage
if __name__ == "__main__":
    print("Ensemble Voting Framework")
    print("\nExample usage:")
    print("""
    # Create ensemble with multiple models
    models = [
        create_resnet18_1d(num_classes=NUM_CLASSES),
        create_resnet50_1d(num_classes=NUM_CLASSES),
        create_efficientnet_b3(num_classes=NUM_CLASSES)
    ]

    # Load trained weights
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(f'model_{i}.pth'))

    # Soft voting ensemble
    ensemble = EnsembleVoting(models, device='cuda')
    predictions, probabilities = ensemble.soft_voting(inputs)

    # Evaluate ensemble
    metrics = ensemble.evaluate(test_loader, voting_method='soft')
    print(f"Ensemble accuracy: {metrics['accuracy']:.2f}%")

    # Compare methods
    comparison = compare_ensemble_methods(
        models, train_loader, test_loader, device='cuda'
    )
    print(comparison)
    """)
