"""
Mixture of Experts (MoE) for Bearing Fault Diagnosis

Gating network dynamically selects which expert model to use for each sample.
Each expert specializes in different fault types.

Architecture:
  Signal → Gating Network → [B, n_experts]  # Which expert to use
             ↓
  Experts: [Expert1(signal), Expert2(signal), ..., ExpertN(signal)]
             ↓
  Weighted Sum: ∑ gate_weight_i × expert_i_prediction

Author: LSTM_PFD Team
Date: 2025-11-20
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import sys
sys.path.append('/home/user/LSTM_PFD')
from models.base_model import BaseModel


class GatingNetwork(nn.Module):
    """
    Gating network that decides which expert to use for each sample.

    Architecture:
        Input signal → 1D CNN → FC → Softmax → Gating weights

    Args:
        input_length: Signal length (default: 102400)
        n_experts: Number of experts (default: 3)
        hidden_dim: Hidden dimension (default: 128)
    """
    def __init__(
        self,
        input_length: int = SIGNAL_LENGTH,
        n_experts: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.input_length = input_length
        self.n_experts = n_experts

        # Simple 1D CNN to extract features for gating
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=7, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.AdaptiveAvgPool1d(1)
        )

        # Gating decision layers
        self.gate_fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights for each expert.

        Args:
            x: Input signal [B, C, T]

        Returns:
            Gating weights [B, n_experts] (softmax probabilities)
        """
        # Extract features
        features = self.feature_extractor(x)  # [B, 64, 1]
        features = features.squeeze(-1)  # [B, 64]

        # Compute gating weights
        logits = self.gate_fc(features)  # [B, n_experts]
        gates = F.softmax(logits, dim=1)  # [B, n_experts]

        return gates


class ExpertModel(BaseModel):
    """
    Expert model specialized for specific fault types.

    Simple wrapper around any BaseModel to add expert specialization.

    Args:
        model: Base model (CNN, ResNet, Transformer, etc.)
        specialization: Which fault types this expert specializes in (optional)
    """
    def __init__(
        self,
        model: nn.Module,
        specialization: Optional[List[int]] = None
    ):
        super().__init__()

        self.model = model
        self.specialization = specialization  # List of class indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert model."""
        return self.model(x)

    def get_config(self) -> dict:
        """Return expert configuration."""
        config = self.model.get_config() if hasattr(self.model, 'get_config') else {}
        config.update({
            'expert_type': self.__class__.__name__,
            'specialization': self.specialization
        })
        return config


class MixtureOfExperts(BaseModel):
    """
    Mixture of Experts ensemble.

    Gating network dynamically selects which expert(s) to use for each sample.

    Architecture:
        Signal → Gating Network → Expert Selection Weights
                      ↓
        Experts: [E1, E2, ..., EN] all process signal
                      ↓
        Weighted combination of expert predictions

    Args:
        experts: List of expert models
        gating_network: Gating network (created if not provided)
        num_classes: Number of classes (default: 11)
        input_length: Signal length (default: 102400)
        top_k: Use only top-k experts (default: None = use all)

    Example:
        >>> experts = [cnn_model, resnet_model, transformer_model]
        >>> moe = MixtureOfExperts(experts)
        >>> predictions = moe(x)
    """
    def __init__(
        self,
        experts: List[nn.Module],
        gating_network: Optional[GatingNetwork] = None,
        num_classes: int = NUM_CLASSES,
        input_length: int = SIGNAL_LENGTH,
        top_k: Optional[int] = None
    ):
        super().__init__()

        if not experts:
            raise ValueError("Must provide at least one expert")

        self.n_experts = len(experts)
        self.num_classes = num_classes
        self.input_length = input_length
        self.top_k = top_k if top_k is not None else self.n_experts

        # Store experts
        self.experts = nn.ModuleList(experts)

        # Create or use provided gating network
        if gating_network is None:
            gating_network = GatingNetwork(
                input_length=input_length,
                n_experts=self.n_experts
            )

        self.gating_network = gating_network

    def forward(self, x: torch.Tensor, return_gates: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with dynamic expert selection.

        Args:
            x: Input signal [B, C, T]
            return_gates: Whether to return gating weights

        Returns:
            Final predictions [B, num_classes]
            (Optional) Gating weights [B, n_experts]
        """
        # Compute gating weights
        gates = self.gating_network(x)  # [B, n_experts]

        # Top-k gating (sparse MoE)
        if self.top_k < self.n_experts:
            top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=1)
            # Renormalize top-k gates
            top_k_gates = top_k_gates / top_k_gates.sum(dim=1, keepdim=True)

            # Zero out non-top-k gates
            sparse_gates = torch.zeros_like(gates)
            sparse_gates.scatter_(1, top_k_indices, top_k_gates)
            gates = sparse_gates

        # Get predictions from all experts
        expert_outputs = []

        for expert in self.experts:
            expert.eval()  # Experts in eval mode
            with torch.no_grad():
                output = expert(x)  # [B, num_classes]
                expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, n_experts, num_classes]

        # Weighted combination
        gates_expanded = gates.unsqueeze(-1)  # [B, n_experts, 1]
        final_output = (expert_outputs * gates_expanded).sum(dim=1)  # [B, num_classes]

        if return_gates:
            return final_output, gates
        else:
            return final_output

    def train_gating_network(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 20,
        device: str = 'cuda',
        verbose: bool = True
    ) -> Dict:
        """
        Train only the gating network (experts are frozen).

        Args:
            dataloader: Training data loader
            optimizer: Optimizer for gating network
            criterion: Loss function
            num_epochs: Number of epochs
            device: Device to use
            verbose: Print progress

        Returns:
            Training history dict
        """
        self.gating_network.train()
        history = {'loss': [], 'accuracy': []}

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else dataloader

            for batch in pbar:
                x, y = batch
                x, y = x.to(device), y.to(device)

                # Forward pass
                logits = self.forward(x)
                loss = criterion(logits, y)

                # Backward pass (only gating network)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                if verbose:
                    pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

            avg_loss = total_loss / len(dataloader)
            accuracy = 100. * correct / total

            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return history

    def get_expert_usage(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ) -> Dict:
        """
        Analyze which experts are used most frequently.

        Args:
            dataloader: Data loader
            device: Device to use

        Returns:
            Expert usage statistics
        """
        self.eval()
        expert_usage = torch.zeros(self.n_experts)
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(device)
                _, gates = self.forward(x, return_gates=True)

                # Count which expert has highest weight for each sample
                max_expert = gates.argmax(dim=1)
                for expert_idx in max_expert:
                    expert_usage[expert_idx] += 1

                total_samples += x.size(0)

        expert_usage = expert_usage / total_samples

        return {
            'usage_proportion': expert_usage.tolist(),
            'total_samples': total_samples
        }

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'MixtureOfExperts',
            'n_experts': self.n_experts,
            'num_classes': self.num_classes,
            'top_k': self.top_k,
            'num_parameters': self.get_num_params()
        }


def create_specialized_experts(
    base_model_fn: callable,
    specializations: List[List[int]],
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 20,
    lr: float = 0.001,
    device: str = 'cuda',
    verbose: bool = True
) -> List[ExpertModel]:
    """
    Create expert models specialized for different fault types.

    Args:
        base_model_fn: Function to create base model
        specializations: List of fault type lists for each expert
            Example: [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9, 10]]
            Expert 1 specializes in classes 0, 1, 2
            Expert 2 specializes in classes 3, 4, 5
            Expert 3 specializes in classes 6, 7, 8, 9, 10
        train_loader: Training data loader
        num_epochs: Training epochs per expert
        lr: Learning rate
        device: Device to use
        verbose: Print progress

    Returns:
        List of trained expert models
    """
    experts = []

    for i, specialization in enumerate(specializations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Expert {i+1}/{len(specializations)}")
            print(f"Specialization: Classes {specialization}")
            print(f"{'='*60}")

        # Create model
        model = base_model_fn()
        model = model.to(device)

        # Filter dataset for this expert's specialization
        # (In practice, you might want to use the full dataset but with weighted loss)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Train model
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else train_loader

            for batch in pbar:
                x, y = batch
                x, y = x.to(device), y.to(device)

                # Forward pass
                logits = model(x)
                loss = criterion(logits, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                if verbose:
                    pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

        # Wrap as expert
        expert = ExpertModel(model, specialization)
        experts.append(expert)

    return experts
