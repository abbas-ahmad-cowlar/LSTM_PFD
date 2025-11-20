"""
End-to-end CNN experiment orchestration.

Purpose:
    Complete workflow for CNN model training and evaluation:
    - Data loading and preprocessing
    - Model creation and configuration
    - Training with all optimizations
    - Evaluation and metric reporting
    - Checkpoint management
    - Experiment tracking (MLflow integration ready)

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Any, List
import json
import time
from datetime import datetime

from models.cnn.cnn_1d import CNN1D
from data.cnn_dataset import create_cnn_datasets, create_cnn_datasets_from_arrays
from data.cnn_dataloader import create_cnn_dataloaders
from training.cnn_trainer import CNNTrainer
from training.cnn_optimizer import create_adamw_optimizer, create_sgd_optimizer
from training.cnn_losses import create_criterion
from evaluation.cnn_evaluator import CNNEvaluator
from utils.checkpoint_manager import CheckpointManager
from utils.early_stopping import EarlyStopping
from utils.logging import get_logger

logger = get_logger(__name__)


class CNNExperiment:
    """
    End-to-end CNN experiment orchestrator.

    Manages the complete ML pipeline:
    1. Data preparation
    2. Model initialization
    3. Training with checkpointing
    4. Evaluation on test set
    5. Results logging

    Args:
        config: Experiment configuration dictionary
        experiment_dir: Directory to save all experiment outputs

    Example:
        >>> config = {
        ...     'data': {
        ...         'train_path': './data/train.npy',
        ...         'batch_size': 32,
        ...         'num_workers': 4
        ...     },
        ...     'model': {
        ...         'num_classes': 11,
        ...         'dropout': 0.5
        ...     },
        ...     'training': {
        ...         'num_epochs': 50,
        ...         'learning_rate': 1e-3,
        ...         'patience': 10
        ...     }
        ... }
        >>>
        >>> experiment = CNNExperiment(config, './experiments/exp001')
        >>> results = experiment.run()
        >>> print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    """

    def __init__(
        self,
        config: Dict[str, Any],
        experiment_dir: Path
    ):
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup subdirectories
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.results_dir = self.experiment_dir / 'results'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Components (initialized in setup)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None
        self.trainer = None
        self.evaluator = None
        self.checkpoint_manager = None
        self.early_stopping = None

        # Results
        self.train_history = None
        self.test_metrics = None

        # Save config
        self._save_config()

        logger.info(f"CNNExperiment initialized at {experiment_dir}")

    def setup_data(self):
        """Prepare datasets and dataloaders."""
        logger.info("Setting up data...")

        data_config = self.config['data']

        # Check if data paths or arrays provided
        if 'train_path' in data_config:
            # Load from files
            train_ds, val_ds, test_ds = create_cnn_datasets(
                train_path=data_config['train_path'],
                val_path=data_config.get('val_path'),
                test_path=data_config.get('test_path'),
                val_split=data_config.get('val_split', 0.2),
                normalize=data_config.get('normalize', True),
                seed=data_config.get('seed', 42)
            )
        elif 'signals' in data_config and 'labels' in data_config:
            # Use provided arrays
            train_ds, val_ds, test_ds = create_cnn_datasets_from_arrays(
                signals=data_config['signals'],
                labels=data_config['labels'],
                val_split=data_config.get('val_split', 0.2),
                test_split=data_config.get('test_split', 0.1),
                normalize=data_config.get('normalize', True),
                seed=data_config.get('seed', 42)
            )
        else:
            raise ValueError("Must provide either 'train_path' or 'signals'/'labels' in data config")

        # Create dataloaders
        loaders = create_cnn_dataloaders(
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            batch_size=data_config.get('batch_size', 32),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True)
        )

        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']

        logger.info(f"Data loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    def setup_model(self):
        """Initialize model."""
        logger.info("Setting up model...")

        model_config = self.config.get('model', {})

        # Create model (currently only CNN1D supported, easy to extend)
        model_type = model_config.get('type', 'CNN1D')

        if model_type == 'CNN1D':
            self.model = CNN1D(
                num_classes=model_config.get('num_classes', 11),
                input_channels=model_config.get('input_channels', 1),
                dropout=model_config.get('dropout', 0.5),
                use_batch_norm=model_config.get('use_batch_norm', True)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model = self.model.to(self.device)

        # Log model info
        param_counts = self.model.count_parameters()
        logger.info(f"Model: {model_type}")
        logger.info(f"Parameters: {param_counts['total']:,} (trainable: {param_counts['trainable']:,})")

    def setup_training(self):
        """Initialize training components."""
        logger.info("Setting up training...")

        train_config = self.config.get('training', {})

        # Optimizer
        optimizer_type = train_config.get('optimizer', 'adamw')
        lr = train_config.get('learning_rate', 1e-3)
        weight_decay = train_config.get('weight_decay', 1e-4)

        if optimizer_type == 'adamw':
            self.optimizer = create_adamw_optimizer(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = create_sgd_optimizer(
                self.model.parameters(),
                lr=lr,
                momentum=train_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Loss function
        loss_type = train_config.get('loss', 'cross_entropy')
        self.criterion = create_criterion(
            loss_type=loss_type,
            num_classes=self.config['model'].get('num_classes', 11),
            smoothing=train_config.get('label_smoothing', 0.1)
        )

        # Learning rate scheduler (optional)
        lr_scheduler = None
        if train_config.get('use_scheduler', False):
            scheduler_type = train_config.get('scheduler_type', 'cosine')
            num_epochs = train_config.get('num_epochs', 50)

            if scheduler_type == 'cosine':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_epochs,
                    eta_min=train_config.get('min_lr', 1e-6)
                )
            elif scheduler_type == 'step':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=train_config.get('step_size', 10),
                    gamma=train_config.get('gamma', 0.1)
                )

        # Trainer
        self.trainer = CNNTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            lr_scheduler=lr_scheduler,
            max_grad_norm=train_config.get('max_grad_norm', 1.0),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
            mixed_precision=train_config.get('mixed_precision', True),
            checkpoint_dir=self.checkpoint_dir
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=lr_scheduler,
            mode='max',  # Monitor validation accuracy
            save_top_k=train_config.get('save_top_k', 3)
        )

        # Early stopping
        if train_config.get('use_early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=train_config.get('patience', 10),
                mode='max',  # Monitor validation accuracy
                min_delta=train_config.get('min_delta', 0.0),
                restore_best_weights=True,
                verbose=True
            )

        logger.info(f"Training setup complete: {optimizer_type} optimizer, {loss_type} loss")

    def train(self) -> Dict[str, List[float]]:
        """Run training loop."""
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)

        train_config = self.config.get('training', {})
        num_epochs = train_config.get('num_epochs', 50)

        start_time = time.time()

        # Training loop with early stopping integration
        for epoch in range(num_epochs):
            self.trainer.current_epoch = epoch

            # Train one epoch
            train_metrics = self.trainer.train_epoch()

            # Validate
            val_metrics = self.trainer.validate_epoch()

            # Update learning rate
            if self.trainer.lr_scheduler is not None:
                self.trainer.lr_scheduler.step()

            # Update history
            current_lr = self.trainer.get_current_lr()
            self.trainer.history['train_loss'].append(train_metrics['loss'])
            self.trainer.history['train_acc'].append(train_metrics['accuracy'])
            self.trainer.history['lr'].append(current_lr)

            if val_metrics:
                self.trainer.history['val_loss'].append(val_metrics['loss'])
                self.trainer.history['val_acc'].append(val_metrics['accuracy'])

            # Log epoch summary
            epoch_msg = f"Epoch {epoch + 1}/{num_epochs} - "
            epoch_msg += f"Train Loss: {train_metrics['loss']:.4f}, "
            epoch_msg += f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            if val_metrics:
                epoch_msg += f"Val Loss: {val_metrics['loss']:.4f}, "
                epoch_msg += f"Val Acc: {val_metrics['accuracy']:.2f}%, "
            epoch_msg += f"LR: {current_lr:.6f}"
            logger.info(epoch_msg)

            # Checkpoint saving
            if val_metrics:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    metric_value=val_metrics['accuracy'],
                    metric_name='val_acc'
                )

            # Early stopping check
            if self.early_stopping and val_metrics:
                if self.early_stopping(val_metrics['accuracy'], self.model, epoch):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        training_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"Training complete! Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info("=" * 60)

        self.train_history = self.trainer.history

        return self.train_history

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set."""
        logger.info("=" * 60)
        logger.info("Evaluating on test set...")
        logger.info("=" * 60)

        # Load best model
        if self.checkpoint_manager:
            self.checkpoint_manager.load_best_checkpoint(load_optimizer=False)
            logger.info("Best model loaded for evaluation")

        # Create evaluator
        self.evaluator = CNNEvaluator(
            model=self.model,
            device=self.device,
            class_names=self.config.get('class_names')
        )

        # Evaluate
        self.test_metrics = self.evaluator.evaluate(
            test_loader=self.test_loader,
            criterion=self.criterion,
            verbose=True
        )

        return self.test_metrics

    def run(self) -> Dict[str, Any]:
        """
        Run complete experiment pipeline.

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info(f"Running CNN Experiment: {self.experiment_dir.name}")
        logger.info("=" * 60)

        start_time = time.time()

        # Setup
        self.setup_data()
        self.setup_model()
        self.setup_training()

        # Train
        self.train()

        # Evaluate
        self.evaluate()

        # Collect results
        total_time = time.time() - start_time

        results = {
            'experiment_name': self.experiment_dir.name,
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'config': self.config,
            'train_history': self.train_history,
            'test_metrics': {
                'accuracy': self.test_metrics['accuracy'],
                'precision_macro': self.test_metrics['precision_macro'],
                'recall_macro': self.test_metrics['recall_macro'],
                'f1_macro': self.test_metrics['f1_macro'],
                'loss': self.test_metrics.get('loss', None)
            }
        }

        # Save results
        self._save_results(results)

        logger.info("=" * 60)
        logger.info(f"Experiment complete! Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        logger.info(f"Test Accuracy: {self.test_metrics['accuracy']:.2f}%")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("=" * 60)

        return results

    def _save_config(self):
        """Save experiment configuration."""
        config_path = self.experiment_dir / 'config.json'

        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Config saved to {config_path}")

    def _save_results(self, results: Dict):
        """Save experiment results."""
        # Save main results
        results_path = self.results_dir / 'results.json'

        # Prepare serializable results
        serializable_results = results.copy()

        # Convert numpy arrays if present
        if 'test_metrics' in serializable_results:
            test_metrics = serializable_results['test_metrics']
            for key in list(test_metrics.keys()):
                if hasattr(test_metrics[key], 'tolist'):
                    test_metrics[key] = test_metrics[key].tolist()

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

        # Save detailed evaluation results
        if self.test_metrics:
            eval_path = self.results_dir / 'evaluation.json'
            self.evaluator.save_results(self.test_metrics, eval_path)


def test_cnn_experiment():
    """Test CNN experiment with dummy data."""
    print("=" * 60)
    print("Testing CNN Experiment")
    print("=" * 60)

    import numpy as np
    import tempfile

    # Create dummy data
    num_samples = 200
    signal_length = 102400
    num_classes = 11

    signals = np.random.randn(num_samples, signal_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)

    # Create experiment config
    config = {
        'data': {
            'signals': signals,
            'labels': labels,
            'batch_size': 16,
            'num_workers': 0,  # Single-threaded for testing
            'normalize': True,
            'val_split': 0.2,
            'test_split': 0.1
        },
        'model': {
            'type': 'CNN1D',
            'num_classes': num_classes,
            'dropout': 0.5
        },
        'training': {
            'num_epochs': 3,  # Just 3 epochs for testing
            'learning_rate': 1e-3,
            'optimizer': 'adamw',
            'loss': 'cross_entropy',
            'mixed_precision': False,  # Disable for CPU testing
            'use_early_stopping': True,
            'patience': 2,
            'use_scheduler': False
        },
        'device': 'cpu'  # Force CPU for testing
    }

    # Run experiment
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'test_experiment'

        print("\n1. Creating experiment...")
        experiment = CNNExperiment(config, experiment_dir)

        print("\n2. Running experiment...")
        results = experiment.run()

        print("\n3. Results:")
        print(f"   Test Accuracy: {results['test_metrics']['accuracy']:.2f}%")
        print(f"   F1 Score: {results['test_metrics']['f1_macro']:.2f}%")
        print(f"   Total Time: {results['total_time_seconds']:.1f}s")

        print("\n4. Checking saved files...")
        assert (experiment_dir / 'config.json').exists()
        assert (experiment_dir / 'results' / 'results.json').exists()
        assert (experiment_dir / 'checkpoints' / 'best_model.pth').exists()
        print("   All files saved successfully")

    print("\n" + "=" * 60)
    print("✅ All CNN experiment tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cnn_experiment()
