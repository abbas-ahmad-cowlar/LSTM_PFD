"""
Contrastive Physics Pretraining Framework

This module implements a SimCLR-style contrastive learning approach where
physics similarity defines positive/negative pairs for bearing fault diagnosis.

Addresses Deficiency #26 (Priority 46): No Contrastive Physics Pretraining

Key Innovation:
- Signals with similar physics parameters (eccentricity, clearance, viscosity)
  are treated as positive pairs, even if from different fault types
- This encourages the encoder to learn physics-aware representations
- Fine-tuning on fault classification uses physics-informed features

Usage:
    # Pretraining
    python scripts/research/contrastive_physics.py pretrain --epochs 100
    
    # Fine-tuning
    python scripts/research/contrastive_physics.py finetune --checkpoint pretrained.pt
    
    # Full pipeline with benchmark
    python scripts/research/contrastive_physics.py --full-pipeline

Author: AI Research Team
Date: January 2026
"""

import argparse
import copy
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Error: PyTorch is required for contrastive pretraining")
    sys.exit(1)

try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('results/contrastive_physics')


# ============================================================================
# Physics Similarity Functions
# ============================================================================

def compute_physics_similarity(params1: Dict[str, float], 
                                params2: Dict[str, float],
                                weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute similarity between two sets of physics parameters.
    
    Physics parameters typically include:
    - eccentricity: shaft eccentricity ratio (0-1)
    - clearance: bearing clearance in mm
    - viscosity: lubricant viscosity in cSt
    - load: applied load in N
    - speed: rotational speed in RPM
    
    Returns similarity in [0, 1] where 1 = identical physics.
    """
    if weights is None:
        weights = {
            'eccentricity': 1.0,
            'clearance': 1.0,
            'viscosity': 0.5,
            'load': 0.5,
            'speed': 0.3
        }
    
    # Normalization ranges (typical values)
    ranges = {
        'eccentricity': (0.0, 1.0),
        'clearance': (0.01, 0.5),
        'viscosity': (10, 500),
        'load': (100, 10000),
        'speed': (500, 5000)
    }
    
    total_weight = 0
    total_similarity = 0
    
    for key in weights:
        if key in params1 and key in params2:
            v1 = params1[key]
            v2 = params2[key]
            
            # Normalize to [0, 1]
            min_val, max_val = ranges.get(key, (0, 1))
            v1_norm = (v1 - min_val) / (max_val - min_val + 1e-8)
            v2_norm = (v2 - min_val) / (max_val - min_val + 1e-8)
            
            # Euclidean distance in normalized space
            distance = abs(v1_norm - v2_norm)
            similarity = 1.0 - distance
            
            total_similarity += weights[key] * similarity
            total_weight += weights[key]
    
    if total_weight == 0:
        return 0.5
    
    return total_similarity / total_weight


def select_positive_negative_pairs(
    physics_params: List[Dict[str, float]],
    similarity_threshold: float = 0.8,
    num_negatives: int = 5
) -> List[Tuple[int, int, List[int]]]:
    """
    Select positive and negative pairs based on physics similarity.
    
    For each sample i:
    - Positive: sample j with physics_similarity(i, j) >= threshold
    - Negatives: samples with physics_similarity(i, k) < threshold
    
    Returns list of (anchor_idx, positive_idx, [negative_idxs])
    """
    n = len(physics_params)
    pairs = []
    
    # Precompute similarity matrix
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = compute_physics_similarity(physics_params[i], physics_params[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    for i in range(n):
        # Find positives (high similarity)
        positives = [j for j in range(n) if j != i and 
                     similarity_matrix[i, j] >= similarity_threshold]
        
        # Find negatives (low similarity)
        negatives = [j for j in range(n) if j != i and 
                     similarity_matrix[i, j] < similarity_threshold]
        
        if positives and negatives:
            # Random selection
            pos_idx = np.random.choice(positives)
            neg_idxs = np.random.choice(
                negatives, 
                size=min(num_negatives, len(negatives)),
                replace=False
            ).tolist()
            pairs.append((i, pos_idx, neg_idxs))
    
    return pairs


# ============================================================================
# Dataset Classes
# ============================================================================

class PhysicsContrastiveDataset(Dataset):
    """
    Dataset that yields triplets (anchor, positive, negatives) based on
    physics similarity rather than class labels.
    
    This enables the model to learn representations that capture
    underlying physics relationships.
    """
    
    def __init__(self, 
                 signals: np.ndarray,
                 physics_params: List[Dict[str, float]],
                 similarity_threshold: float = 0.8,
                 num_negatives: int = 5,
                 augment: bool = True):
        """
        Args:
            signals: Time-series signals (N, C, L) or (N, L)
            physics_params: List of physics parameter dicts for each signal
            similarity_threshold: Min similarity for positive pairs
            num_negatives: Number of negative samples per anchor
            augment: Whether to apply random augmentations
        """
        self.signals = torch.FloatTensor(signals)
        if self.signals.dim() == 2:
            self.signals = self.signals.unsqueeze(1)  # Add channel dim
            
        self.physics_params = physics_params
        self.similarity_threshold = similarity_threshold
        self.num_negatives = num_negatives
        self.augment = augment
        
        # Precompute pairs
        logger.info("Precomputing positive/negative pairs based on physics similarity...")
        self.pairs = select_positive_negative_pairs(
            physics_params, similarity_threshold, num_negatives
        )
        logger.info(f"Created {len(self.pairs)} triplets")
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx, pos_idx, neg_idxs = self.pairs[idx]
        
        anchor = self.signals[anchor_idx]
        positive = self.signals[pos_idx]
        negatives = self.signals[neg_idxs]  # (num_neg, C, L)
        
        # Apply augmentations
        if self.augment:
            anchor = self._augment(anchor)
            positive = self._augment(positive)
            negatives = torch.stack([self._augment(n) for n in negatives])
        
        return anchor, positive, negatives
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        # Random noise
        if np.random.random() < 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        
        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
        
        # Random temporal shift
        if np.random.random() < 0.5:
            shift = np.random.randint(-50, 50)
            x = torch.roll(x, shift, dims=-1)
        
        return x


class FineTuneDataset(Dataset):
    """Dataset for fine-tuning on downstream classification."""
    
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.FloatTensor(signals)
        if self.signals.dim() == 2:
            self.signals = self.signals.unsqueeze(1)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.signals[idx], self.labels[idx]


# ============================================================================
# Model Architecture
# ============================================================================

class SignalEncoder(nn.Module):
    """
    CNN-based signal encoder that produces embeddings for contrastive learning.
    Architecture inspired by SimCLR projection heads.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Backbone CNN
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Projection head (for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input signals (B, C, L)
            return_features: If True, return backbone features instead of projections
            
        Returns:
            embeddings: (B, embedding_dim) or (B, 256) if return_features
        """
        features = self.backbone(x).squeeze(-1)  # (B, 256)
        
        if return_features:
            return features
        
        embeddings = self.projection(features)  # (B, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
        
        return embeddings


class ContrastiveClassifier(nn.Module):
    """
    Classifier that uses pretrained encoder features.
    Encoder weights can be frozen or fine-tuned.
    """
    
    def __init__(self, 
                 encoder: SignalEncoder,
                 num_classes: int,
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x, return_features=True)
        return self.classifier(features)


# ============================================================================
# Loss Functions
# ============================================================================

class PhysicsInfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss adapted for physics-based pairs.
    
    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    
    where z_j is the physics-similar positive and z_k includes negatives.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, 
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: (B, D) anchor embeddings
            positive: (B, D) positive embeddings (physics-similar)
            negatives: (B, N, D) negative embeddings
            
        Returns:
            loss: scalar contrastive loss
        """
        batch_size = anchor.size(0)
        
        # Positive similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)  # (B,)
        pos_sim = pos_sim / self.temperature
        
        # Negative similarities
        # anchor: (B, 1, D), negatives: (B, N, D)
        neg_sim = F.cosine_similarity(
            anchor.unsqueeze(1), negatives, dim=2
        )  # (B, N)
        neg_sim = neg_sim / self.temperature
        
        # InfoNCE loss
        # log(softmax) = log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).
    
    Alternative to InfoNCE with in-batch negatives.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (B, D) embeddings from view 1
            z_j: (B, D) embeddings from view 2 (physics-similar)
            
        Returns:
            loss: NT-Xent loss
        """
        batch_size = z_i.size(0)
        
        # Combine all embeddings
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2B, 2B)
        sim = sim / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -float('inf'))
        
        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        
        loss = F.cross_entropy(sim, labels)
        
        return loss


# ============================================================================
# Training Functions
# ============================================================================

class ContrastivePretrainer:
    """Handles contrastive pretraining with physics-based pairs."""
    
    def __init__(self,
                 encoder: SignalEncoder,
                 learning_rate: float = 0.001,
                 temperature: float = 0.07,
                 device: str = None):
        self.encoder = encoder
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        
        self.optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.criterion = PhysicsInfoNCELoss(temperature=temperature)
        
        self.history = {'loss': [], 'epoch': []}
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.encoder.train()
        total_loss = 0
        
        for anchor, positive, negatives in dataloader:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negatives = negatives.to(self.device)
            
            # Get embeddings
            z_anchor = self.encoder(anchor)
            z_positive = self.encoder(positive)
            
            # Process negatives: (B, N, C, L) -> (B, N, D)
            B, N, C, L = negatives.shape
            negatives_flat = negatives.view(B * N, C, L)
            z_negatives = self.encoder(negatives_flat).view(B, N, -1)
            
            # Compute loss
            loss = self.criterion(z_anchor, z_positive, z_negatives)
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def pretrain(self, 
                 dataloader: DataLoader,
                 epochs: int = 100,
                 save_path: Optional[Path] = None) -> Dict[str, List[float]]:
        """Run full pretraining."""
        logger.info(f"Starting contrastive pretraining for {epochs} epochs")
        
        for epoch in range(epochs):
            loss = self.train_epoch(dataloader)
            self.history['loss'].append(loss)
            self.history['epoch'].append(epoch)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
                
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history
            }, save_path)
            logger.info(f"Saved pretrained model to {save_path}")
            
        return self.history


class ContrastiveFineTuner:
    """Handles fine-tuning pretrained encoder for classification."""
    
    def __init__(self,
                 encoder: SignalEncoder,
                 num_classes: int,
                 freeze_encoder: bool = False,
                 learning_rate: float = 0.0001,
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ContrastiveClassifier(
            encoder, num_classes, freeze_encoder
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.history = {'train_loss': [], 'val_acc': []}
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for signals, labels in train_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(self.device)
                outputs = self.model(signals)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return accuracy, f1
    
    def finetune(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 epochs: int = 50) -> Dict[str, List[float]]:
        """Run fine-tuning."""
        logger.info(f"Starting fine-tuning for {epochs} epochs")
        
        best_acc = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc, val_f1 = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Loss: {train_loss:.4f}, Acc: {val_acc:.4f}")
                
        logger.info(f"Best validation accuracy: {best_acc:.4f}")
        return self.history


# ============================================================================
# Benchmark: Contrastive vs Supervised
# ============================================================================

def run_benchmark(
    signals: np.ndarray,
    labels: np.ndarray,
    physics_params: List[Dict[str, float]],
    num_seeds: int = 3,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 30,
    device: str = None
) -> Dict[str, Any]:
    """
    Benchmark contrastive pretraining vs supervised-only baseline.
    
    Returns metrics for both approaches with confidence intervals.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(labels))
    
    results = {
        'supervised': {'accuracy': [], 'f1': []},
        'contrastive_frozen': {'accuracy': [], 'f1': []},
        'contrastive_finetuned': {'accuracy': [], 'f1': []}
    }
    
    for seed in range(num_seeds):
        logger.info(f"\n=== Seed {seed + 1}/{num_seeds} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Split data
        indices = np.arange(len(signals))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=seed
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, stratify=labels[train_idx], random_state=seed
        )
        
        # Create datasets
        train_signals = signals[train_idx]
        train_labels = labels[train_idx]
        train_physics = [physics_params[i] for i in train_idx]
        
        val_signals = signals[val_idx]
        val_labels = labels[val_idx]
        
        test_signals = signals[test_idx]
        test_labels = labels[test_idx]
        
        # 1. Supervised baseline
        logger.info("Training supervised baseline...")
        encoder_sup = SignalEncoder()
        classifier_sup = ContrastiveClassifier(encoder_sup, num_classes).to(device)
        optimizer_sup = optim.Adam(classifier_sup.parameters(), lr=0.001)
        
        train_loader = DataLoader(
            FineTuneDataset(train_signals, train_labels),
            batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            FineTuneDataset(test_signals, test_labels),
            batch_size=32
        )
        
        for _ in range(finetune_epochs):
            classifier_sup.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(classifier_sup(x), y)
                optimizer_sup.zero_grad()
                loss.backward()
                optimizer_sup.step()
                
        classifier_sup.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for x, y in test_loader:
                preds = classifier_sup(x.to(device)).argmax(1).cpu()
                all_preds.extend(preds.numpy())
                all_labels_list.extend(y.numpy())
                
        sup_acc = accuracy_score(all_labels_list, all_preds)
        sup_f1 = f1_score(all_labels_list, all_preds, average='weighted')
        results['supervised']['accuracy'].append(sup_acc)
        results['supervised']['f1'].append(sup_f1)
        logger.info(f"Supervised: Acc={sup_acc:.4f}, F1={sup_f1:.4f}")
        
        # 2. Contrastive pretraining
        logger.info("Contrastive pretraining...")
        encoder_con = SignalEncoder()
        
        contrastive_dataset = PhysicsContrastiveDataset(
            train_signals, train_physics,
            similarity_threshold=0.75,
            num_negatives=3
        )
        contrastive_loader = DataLoader(contrastive_dataset, batch_size=32, shuffle=True)
        
        pretrainer = ContrastivePretrainer(encoder_con, device=device)
        pretrainer.pretrain(contrastive_loader, epochs=pretrain_epochs)
        
        # 2a. Linear probe (frozen encoder)
        logger.info("Linear probe (frozen encoder)...")
        finetuner_frozen = ContrastiveFineTuner(
            encoder_con, num_classes, freeze_encoder=True, device=device
        )
        val_loader = DataLoader(FineTuneDataset(val_signals, val_labels), batch_size=32)
        finetuner_frozen.finetune(train_loader, val_loader, epochs=finetune_epochs)
        
        frozen_acc, frozen_f1 = finetuner_frozen.evaluate(test_loader)
        results['contrastive_frozen']['accuracy'].append(frozen_acc)
        results['contrastive_frozen']['f1'].append(frozen_f1)
        logger.info(f"Frozen encoder: Acc={frozen_acc:.4f}, F1={frozen_f1:.4f}")
        
        # 2b. Full fine-tuning
        logger.info("Full fine-tuning...")
        finetuner_full = ContrastiveFineTuner(
            encoder_con, num_classes, freeze_encoder=False, device=device
        )
        finetuner_full.finetune(train_loader, val_loader, epochs=finetune_epochs)
        
        full_acc, full_f1 = finetuner_full.evaluate(test_loader)
        results['contrastive_finetuned']['accuracy'].append(full_acc)
        results['contrastive_finetuned']['f1'].append(full_f1)
        logger.info(f"Fine-tuned: Acc={full_acc:.4f}, F1={full_f1:.4f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute statistics
    summary = {}
    for method, metrics in results.items():
        summary[method] = {
            'accuracy_mean': np.mean(metrics['accuracy']),
            'accuracy_std': np.std(metrics['accuracy']),
            'f1_mean': np.mean(metrics['f1']),
            'f1_std': np.std(metrics['f1'])
        }
        
    return {'results': results, 'summary': summary}


# ============================================================================
# Data Generation (for testing)
# ============================================================================

def generate_synthetic_data(
    n_samples: int = 1000,
    signal_length: int = 4096,
    n_classes: int = 5
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Generate synthetic signals with physics parameters."""
    
    signals = []
    labels = []
    physics_params = []
    
    for i in range(n_samples):
        label = i % n_classes
        
        # Generate physics parameters
        eccentricity = np.random.uniform(0.1, 0.9)
        clearance = np.random.uniform(0.02, 0.3)
        viscosity = np.random.uniform(20, 300)
        load = np.random.uniform(500, 5000)
        speed = np.random.uniform(1000, 4000)
        
        physics = {
            'eccentricity': eccentricity,
            'clearance': clearance,
            'viscosity': viscosity,
            'load': load,
            'speed': speed
        }
        
        # Generate signal based on physics + class
        t = np.linspace(0, 1, signal_length)
        freq = 50 + label * 30 + speed / 100
        
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t * (1 + eccentricity))
        signal *= (1 + 0.2 * clearance)
        signal += np.random.randn(signal_length) * 0.1
        
        signals.append(signal.astype(np.float32))
        labels.append(label)
        physics_params.append(physics)
        
    return np.array(signals), np.array(labels), physics_params


# ============================================================================
# Visualization
# ============================================================================

def plot_training_curves(pretrain_history: Dict, finetune_history: Dict, 
                         output_path: Path):
    """Plot training curves for pretraining and fine-tuning."""
    if not HAS_PLOTTING:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pretraining loss
    ax = axes[0]
    ax.plot(pretrain_history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Contrastive Loss')
    ax.set_title('Contrastive Pretraining')
    ax.grid(True, alpha=0.3)
    
    # Fine-tuning accuracy
    ax = axes[1]
    ax.plot(finetune_history['val_acc'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Fine-tuning')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def plot_benchmark_results(summary: Dict[str, Dict], output_path: Path):
    """Plot benchmark comparison bar chart."""
    if not HAS_PLOTTING:
        return
        
    methods = list(summary.keys())
    means = [summary[m]['accuracy_mean'] for m in methods]
    stds = [summary[m]['accuracy_std'] for m in methods]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(methods, means, yerr=stds, color=colors, 
                  capsize=5, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Contrastive Physics Pretraining vs Supervised Baseline')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved benchmark results to {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Contrastive Physics Pretraining Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run contrastive pretraining
    python scripts/research/contrastive_physics.py pretrain --epochs 100
    
    # Fine-tune pretrained model
    python scripts/research/contrastive_physics.py finetune --checkpoint results/contrastive.pt
    
    # Run full benchmark comparison
    python scripts/research/contrastive_physics.py benchmark --seeds 5
    
    # Quick test
    python scripts/research/contrastive_physics.py --quick
        """
    )
    
    parser.add_argument('mode', nargs='?', default='benchmark',
                        choices=['pretrain', 'finetune', 'benchmark'],
                        help='Mode: pretrain, finetune, or benchmark')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--seeds', '-s', type=int, default=3,
                        help='Number of seeds for benchmark (default: 3)')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Checkpoint path for fine-tuning')
    parser.add_argument('--output-dir', '-o', type=str, default='results/contrastive_physics',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced epochs and samples')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Quick mode
    if args.quick:
        args.epochs = 10
        args.seeds = 2
        n_samples = 200
    else:
        n_samples = 1000
    
    # Generate synthetic data
    logger.info("Generating synthetic data with physics parameters...")
    signals, labels, physics_params = generate_synthetic_data(n_samples=n_samples)
    logger.info(f"Generated {len(signals)} samples, {len(np.unique(labels))} classes")
    
    if args.mode == 'pretrain':
        logger.info("Running contrastive pretraining...")
        
        dataset = PhysicsContrastiveDataset(signals, physics_params)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        encoder = SignalEncoder()
        pretrainer = ContrastivePretrainer(encoder, device=device)
        history = pretrainer.pretrain(
            loader, 
            epochs=args.epochs,
            save_path=output_dir / 'pretrained_encoder.pt'
        )
        
        # Save history
        with open(output_dir / 'pretrain_history.json', 'w') as f:
            json.dump(history, f, indent=2)
            
    elif args.mode == 'finetune':
        if not args.checkpoint:
            logger.error("--checkpoint required for fine-tuning")
            return
            
        logger.info(f"Fine-tuning from {args.checkpoint}")
        
        # Load pretrained encoder
        encoder = SignalEncoder()
        checkpoint = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(signals)), test_size=0.2, stratify=labels
        )
        
        train_loader = DataLoader(
            FineTuneDataset(signals[train_idx], labels[train_idx]),
            batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            FineTuneDataset(signals[test_idx], labels[test_idx]),
            batch_size=32
        )
        
        finetuner = ContrastiveFineTuner(
            encoder, num_classes=len(np.unique(labels)), device=device
        )
        history = finetuner.finetune(train_loader, test_loader, epochs=args.epochs)
        
        acc, f1 = finetuner.evaluate(test_loader)
        logger.info(f"Final Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
    elif args.mode == 'benchmark':
        logger.info("Running benchmark: Contrastive vs Supervised")
        
        benchmark_results = run_benchmark(
            signals, labels, physics_params,
            num_seeds=args.seeds,
            pretrain_epochs=args.epochs,
            finetune_epochs=args.epochs // 2,
            device=device
        )
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        for method, stats in benchmark_results['summary'].items():
            print(f"\n{method}:")
            print(f"  Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
            print(f"  F1 Score: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
        print("="*60)
        
        # Save results
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=float)
            
        # Plot
        plot_benchmark_results(
            benchmark_results['summary'],
            output_dir / 'benchmark_comparison.png'
        )
        
    logger.info(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
