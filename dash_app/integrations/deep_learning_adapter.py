"""
Deep Learning integration adapter (Phases 2-8).
Unified adapter for all deep learning models.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class DeepLearningAdapter:
    """Unified adapter for deep learning model training (Phases 2-8)."""

    @staticmethod
    def train(config: dict, progress_callback=None):
        """
        Train deep learning model.

        Args:
            config: Training configuration
                - model_type: Model architecture to use
                - dataset_path: Path to HDF5 dataset
                - hyperparameters: Model hyperparameters
                - num_epochs: Number of training epochs
                - batch_size: Batch size
                - learning_rate: Learning rate
                - optimizer: Optimizer type
                - scheduler: LR scheduler type
                - device: Training device (cuda/cpu/auto)
                - augmentation: List of augmentation types
                - early_stopping_patience: Patience for early stopping

            progress_callback: Optional callback(epoch, metrics) for progress updates

        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting deep learning training: {config['model_type']}")

            # Load data
            train_loader, val_loader, test_loader, num_classes = DeepLearningAdapter._load_data(config)

            # Create model
            model = DeepLearningAdapter._create_model(config, num_classes)

            # Setup device
            device = DeepLearningAdapter._get_device(config.get("device", "auto"))
            model = model.to(device)
            logger.info(f"Training on device: {device}")

            # Setup optimizer
            optimizer = DeepLearningAdapter._create_optimizer(model, config)

            # Setup scheduler
            scheduler = DeepLearningAdapter._create_scheduler(optimizer, config)

            # Setup loss function
            criterion = nn.CrossEntropyLoss()

            # Training loop
            num_epochs = config.get("num_epochs", 100)
            early_stopping_patience = config.get("early_stopping_patience", 15)
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            training_history = {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "val_accuracy": [],
                "learning_rates": [],
            }

            start_time = time.time()

            for epoch in range(1, num_epochs + 1):
                epoch_start = time.time()

                # Training phase
                train_loss, train_acc = DeepLearningAdapter._train_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                # Validation phase
                val_loss, val_acc = DeepLearningAdapter._validate_epoch(
                    model, val_loader, criterion, device
                )

                # Update learning rate
                current_lr = optimizer.param_groups[0]['lr']
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                # Record metrics
                training_history["train_loss"].append(train_loss)
                training_history["val_loss"].append(val_loss)
                training_history["train_accuracy"].append(train_acc)
                training_history["val_accuracy"].append(val_acc)
                training_history["learning_rates"].append(current_lr)

                # Progress callback
                if progress_callback:
                    progress_callback(epoch, {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "learning_rate": current_lr,
                        "epoch_time": time.time() - epoch_start,
                    })

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    logger.info(f"Epoch {epoch}: New best model (val_loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.2%}, LR: {current_lr:.2e}"
                )

            # Load best model for final evaluation
            if best_model_state:
                model.load_state_dict(best_model_state)

            # Final test evaluation
            test_loss, test_acc = DeepLearningAdapter._validate_epoch(
                model, test_loader, criterion, device
            )

            # Calculate detailed metrics (precision, recall, f1)
            from sklearn.metrics import precision_recall_fscore_support

            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, preds = outputs.max(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.numpy())

            # Calculate macro-averaged metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='macro', zero_division=0
            )

            total_time = time.time() - start_time

            logger.info(f"Training complete. Test Accuracy: {test_acc:.2%}, F1: {f1:.2%}")

            return {
                "success": True,
                "model_type": config["model_type"],
                "total_epochs": epoch,
                "best_epoch": epoch - patience_counter,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "best_val_loss": best_val_loss,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "training_time": total_time,
                "history": training_history,
            }

        except Exception as e:
            logger.error(f"Deep learning training failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def _load_data(config):
        """Load dataset from HDF5 and create data loaders."""
        import h5py

        cache_path = config.get("cache_path", "data/processed/signals_cache.h5")
        batch_size = config.get("batch_size", 32)

        with h5py.File(cache_path, 'r') as f:
            X_train = torch.FloatTensor(f['train']['signals'][:])
            y_train = torch.LongTensor(f['train']['labels'][:])
            X_val = torch.FloatTensor(f['val']['signals'][:])
            y_val = torch.LongTensor(f['val']['labels'][:])
            X_test = torch.FloatTensor(f['test']['signals'][:])
            y_test = torch.LongTensor(f['test']['labels'][:])
            num_classes = int(f.attrs.get('num_classes', 11))

        # Add channel dimension for CNNs if needed
        if X_train.dim() == 2:
            X_train = X_train.unsqueeze(1)
            X_val = X_val.unsqueeze(1)
            X_test = X_test.unsqueeze(1)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader, num_classes

    @staticmethod
    def _create_model(config, num_classes):
        """Create model based on configuration."""
        from models import create_model

        model_type = config["model_type"]
        hyperparams = config.get("hyperparameters", {})

        model = create_model(
            model_type,
            num_classes=num_classes,
            **hyperparams
        )

        return model

    @staticmethod
    def _get_device(device_str):
        """Get training device."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    @staticmethod
    def _create_optimizer(model, config):
        """Create optimizer."""
        optimizer_type = config.get("optimizer", "adam").lower()
        lr = config.get("learning_rate", 0.001)
        weight_decay = config.get("weight_decay", 0.0)

        if optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return optim.Adam(model.parameters(), lr=lr)

    @staticmethod
    def _create_scheduler(optimizer, config):
        """Create learning rate scheduler."""
        scheduler_type = config.get("scheduler", "plateau").lower()

        if scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.get("num_epochs", 100)
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            return None

    @staticmethod
    def _train_epoch(model, loader, criterion, optimizer, device):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @staticmethod
    def _validate_epoch(model, loader, criterion, device):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy
