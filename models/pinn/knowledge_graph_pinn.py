"""
Knowledge Graph PINN

Encodes fault relationships as a knowledge graph and uses graph neural network
concepts to leverage these relationships for improved classification.

Knowledge Graph Structure:
- Nodes: 11 fault types
- Edges: Physical relationships between faults
  * wear → lubrication (wear degrades oil quality)
  * clearance → cavitation (excessive clearance enables cavitation)
  * misalignment → imbalance (coupling effect)
  * etc.

Architecture:
    Input Signal → CNN → Features [B, 512]
    ↓
    Project to graph space → Node features [B, 11, 64]
    ↓
    Graph Convolution (aggregate neighboring fault info)
    ↓
    Classification [B, 11]

Benefits:
- Reduces confusion between related faults
- Leverages domain knowledge about fault relationships
- Improves performance on mixed fault conditions

Note: This implements a simplified GNN without requiring torch_geometric.
For production use with large graphs, consider using torch_geometric.
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


from models.base_model import BaseModel
from models.resnet.resnet_1d import ResNet1D
from models.cnn.cnn_1d import CNN1D


class FaultKnowledgeGraph:
    """
    Encodes domain knowledge about bearing fault relationships.

    Fault relationships based on tribology and bearing dynamics:
    - Physical causation (e.g., wear causes lubrication degradation)
    - Co-occurrence patterns (e.g., misalignment often with imbalance)
    - Similar frequency signatures (e.g., inner race vs outer race)
    """

    # Fault type names (for reference)
    FAULT_NAMES = [
        'healthy',        # 0
        'misalignment',   # 1
        'imbalance',      # 2
        'outer_race',     # 3
        'inner_race',     # 4
        'ball',           # 5
        'looseness',      # 6
        'oil_whirl',      # 7
        'cavitation',     # 8
        'wear',           # 9
        'lubrication'     # 10
    ]

    def __init__(self):
        """Initialize fault knowledge graph."""
        self.num_nodes = 11
        self._build_graph()

    def _build_graph(self):
        """
        Build adjacency matrix encoding fault relationships.

        Edge weights represent strength of relationship (0-1).
        Higher weights indicate stronger physical relationships.
        """
        # Initialize adjacency matrix (self-loops = 1.0)
        adj = np.eye(self.num_nodes, dtype=np.float32)

        # Define relationships (bidirectional edges)
        # Format: (fault1, fault2, weight, reason)
        relationships = [
            # Mechanical relationships
            (1, 2, 0.8, "misalignment causes imbalance"),
            (1, 6, 0.7, "misalignment can cause looseness"),
            (2, 7, 0.6, "imbalance can lead to oil whirl"),

            # Race and ball defects (similar signatures)
            (3, 4, 0.7, "outer and inner race defects similar"),
            (3, 5, 0.6, "outer race and ball defects related"),
            (4, 5, 0.6, "inner race and ball defects related"),

            # Wear progression
            (9, 10, 0.9, "wear degrades lubrication"),
            (9, 3, 0.7, "wear causes outer race damage"),
            (9, 4, 0.7, "wear causes inner race damage"),
            (9, 5, 0.6, "wear damages balls"),

            # Lubrication issues
            (10, 7, 0.8, "poor lubrication causes oil whirl"),
            (10, 8, 0.7, "poor lubrication causes cavitation"),
            (10, 3, 0.5, "poor lubrication damages races"),

            # Clearance-related faults
            (6, 8, 0.7, "looseness enables cavitation"),
            (6, 7, 0.6, "looseness affects oil whirl"),

            # Normal condition vs faults (weak negative relationships)
            # Healthy has weak connections to all faults
        ]

        # Add relationships to adjacency matrix
        for fault1, fault2, weight, _ in relationships:
            adj[fault1, fault2] = weight
            adj[fault2, fault1] = weight  # Bidirectional

        self.adjacency_matrix = adj

        # Compute normalized adjacency (for graph convolution)
        # D^(-1/2) * A * D^(-1/2) where D is degree matrix
        degrees = np.sum(adj, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-6))
        self.normalized_adjacency = D_inv_sqrt @ adj @ D_inv_sqrt

    def get_adjacency_matrix(self, normalized: bool = True) -> np.ndarray:
        """
        Get adjacency matrix.

        Args:
            normalized: If True, return normalized adjacency for GCN

        Returns:
            Adjacency matrix [num_nodes, num_nodes]
        """
        if normalized:
            return self.normalized_adjacency
        else:
            return self.adjacency_matrix

    def get_node_features(self) -> np.ndarray:
        """
        Get initial node features for each fault type.

        Features encode basic fault characteristics:
        - Frequency band (low/medium/high)
        - Periodicity (periodic/random)
        - Typical severity
        - Origin (mechanical/lubrication/clearance)

        Returns:
            Node features [num_nodes, num_features]
        """
        # Feature dimensions: [freq_low, freq_med, freq_high, periodic, random, severity, mech, lub, clear]
        node_features = np.array([
            # healthy: low energy, random, low severity
            [0.1, 0.1, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],

            # misalignment: low freq, periodic, medium severity, mechanical
            [1.0, 0.3, 0.1, 1.0, 0.0, 0.6, 1.0, 0.0, 0.0],

            # imbalance: low freq (1X), periodic, medium severity, mechanical
            [1.0, 0.2, 0.1, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0],

            # outer_race: medium freq, periodic, high severity, mechanical
            [0.2, 1.0, 0.4, 1.0, 0.0, 0.8, 1.0, 0.0, 0.0],

            # inner_race: medium freq, periodic, high severity, mechanical
            [0.2, 1.0, 0.4, 1.0, 0.0, 0.8, 1.0, 0.0, 0.0],

            # ball: medium-high freq, periodic, medium severity, mechanical
            [0.1, 0.7, 1.0, 1.0, 0.0, 0.7, 1.0, 0.0, 0.0],

            # looseness: multi-freq, chaotic, medium severity, clearance
            [0.8, 0.8, 0.5, 0.5, 0.5, 0.6, 0.5, 0.0, 1.0],

            # oil_whirl: low freq (subsync), periodic, medium severity, lubrication
            [1.0, 0.2, 0.1, 1.0, 0.0, 0.6, 0.0, 1.0, 0.0],

            # cavitation: high freq, random bursts, medium severity, lubrication
            [0.1, 0.3, 1.0, 0.2, 0.8, 0.6, 0.0, 1.0, 0.5],

            # wear: broadband, mixed, high severity, mechanical + lubrication
            [0.5, 0.7, 0.7, 0.5, 0.5, 0.9, 0.7, 0.7, 0.3],

            # lubrication: high freq, random, medium severity, lubrication
            [0.2, 0.4, 1.0, 0.3, 0.7, 0.5, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        return node_features


class GraphConvolutionLayer(nn.Module):
    """
    Simple graph convolution layer.

    Implements: H' = σ(A_norm * H * W)
    where A_norm is normalized adjacency matrix, H is node features, W is learnable weights
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize graph convolution layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph convolution.

        Args:
            node_features: Node features [B, num_nodes, in_features]
            adjacency: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Updated node features [B, num_nodes, out_features]
        """
        # Linear transformation
        support = torch.matmul(node_features, self.weight)  # [B, num_nodes, out_features]

        # Graph convolution (aggregate neighbors)
        output = torch.matmul(adjacency, support)  # [B, num_nodes, out_features]

        if self.bias is not None:
            output = output + self.bias

        return output


class KnowledgeGraphPINN(BaseModel):
    """
    Knowledge Graph PINN using graph neural network concepts.

    Leverages fault relationship knowledge to improve classification,
    especially for related or mixed faults.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_length: int = SIGNAL_LENGTH,
        backbone: str = 'resnet18',
        node_feature_dim: int = 64,
        gcn_hidden_dim: int = 128,
        num_gcn_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize Knowledge Graph PINN.

        Args:
            num_classes: Number of fault classes (11)
            input_length: Signal length
            backbone: CNN backbone
            node_feature_dim: Dimension of node features
            gcn_hidden_dim: Hidden dimension for GCN layers
            num_gcn_layers: Number of graph convolution layers
            dropout: Dropout probability
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_length = input_length
        self.backbone_name = backbone
        self.node_feature_dim = node_feature_dim
        self.gcn_hidden_dim = gcn_hidden_dim

        # Initialize knowledge graph
        self.kg = FaultKnowledgeGraph()

        # Register adjacency matrix as buffer (not a parameter)
        adj_normalized = torch.FloatTensor(self.kg.get_adjacency_matrix(normalized=True))
        self.register_buffer('adjacency', adj_normalized)

        # Register initial node features as buffer
        initial_node_features = torch.FloatTensor(self.kg.get_node_features())
        self.register_buffer('initial_node_features', initial_node_features)

        # ===== CNN BACKBONE =====
        if backbone == 'resnet18':
            self.encoder = ResNet1D(
                num_classes=num_classes,
                input_channels=1,
                layers=[2, 2, 2, 2],
                dropout=dropout,
                input_length=input_length
            )
            encoder_output_dim = 512
        elif backbone == 'resnet34':
            self.encoder = ResNet1D(
                num_classes=num_classes,
                input_channels=1,
                layers=[3, 4, 6, 3],
                dropout=dropout,
                input_length=input_length
            )
            encoder_output_dim = 512
        elif backbone == 'cnn1d':
            self.encoder = CNN1D(
                num_classes=num_classes,
                input_channels=1,
                dropout=dropout
            )
            encoder_output_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove final FC layer
        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()

        # ===== PROJECT TO GRAPH SPACE =====
        # Project signal features to per-node features
        self.feature_to_nodes = nn.Sequential(
            nn.Linear(encoder_output_dim, num_classes * node_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # ===== GRAPH CONVOLUTION LAYERS =====
        self.gcn_layers = nn.ModuleList()
        input_dim = node_feature_dim + self.initial_node_features.shape[1]  # Concat with static features

        for i in range(num_gcn_layers):
            if i == 0:
                self.gcn_layers.append(GraphConvolutionLayer(input_dim, gcn_hidden_dim))
            else:
                self.gcn_layers.append(GraphConvolutionLayer(gcn_hidden_dim, gcn_hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # ===== CLASSIFICATION HEAD =====
        self.classifier = nn.Linear(gcn_hidden_dim, 1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Knowledge Graph PINN.

        Args:
            signal: Input signal [B, 1, T]

        Returns:
            Class logits [B, num_classes]
        """
        batch_size = signal.shape[0]

        # Extract features from signal
        signal_features = self.encoder(signal)  # [B, 512]

        # Project to node features
        node_features_flat = self.feature_to_nodes(signal_features)  # [B, num_classes * node_feature_dim]
        node_features = node_features_flat.view(batch_size, self.num_classes, self.node_feature_dim)  # [B, 11, 64]

        # Concatenate with static node features (broadcast across batch)
        static_features = self.initial_node_features.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 11, 9]
        node_features = torch.cat([node_features, static_features], dim=2)  # [B, 11, 64+9]

        # Apply graph convolution layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            node_features = gcn_layer(node_features, self.adjacency)
            node_features = F.relu(node_features)
            node_features = self.dropout(node_features)

        # Classification: each node predicts its own class
        logits = self.classifier(node_features).squeeze(-1)  # [B, 11]

        return logits

    def forward_with_attention(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights showing fault relationships.

        Args:
            signal: Input signal [B, 1, T]

        Returns:
            logits: Class predictions [B, 11]
            attention: Learned attention over graph edges [B, 11, 11]
        """
        batch_size = signal.shape[0]

        # Get node features after GCN
        signal_features = self.encoder(signal)
        node_features_flat = self.feature_to_nodes(signal_features)
        node_features = node_features_flat.view(batch_size, self.num_classes, self.node_feature_dim)
        static_features = self.initial_node_features.unsqueeze(0).expand(batch_size, -1, -1)
        node_features = torch.cat([node_features, static_features], dim=2)

        # Compute attention scores (similarity between connected nodes)
        # This shows how much each fault type influences others
        attention_scores = []
        for gcn_layer in self.gcn_layers:
            node_features = gcn_layer(node_features, self.adjacency)
            node_features = F.relu(node_features)

            # Compute pairwise attention (for visualization)
            node_norm = F.normalize(node_features, p=2, dim=2)
            attention = torch.matmul(node_norm, node_norm.transpose(1, 2))  # [B, 11, 11]
            attention_scores.append(attention)

            node_features = self.dropout(node_features)

        # Use attention from last layer
        attention = attention_scores[-1]

        # Classification
        logits = self.classifier(node_features).squeeze(-1)

        return logits, attention

    def get_model_info(self) -> Dict[str, any]:
        """Get model configuration and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'KnowledgeGraphPINN',
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'node_feature_dim': self.node_feature_dim,
            'gcn_hidden_dim': self.gcn_hidden_dim,
            'num_gcn_layers': len(self.gcn_layers),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_graph_edges': int(np.sum(self.kg.adjacency_matrix > 0))
        }


if __name__ == "__main__":
    # Test Knowledge Graph PINN
    print("=" * 60)
    print("Knowledge Graph PINN - Validation")
    print("=" * 60)

    # Test knowledge graph
    print("\nFault Knowledge Graph:")
    kg = FaultKnowledgeGraph()
    adj = kg.get_adjacency_matrix(normalized=False)

    print(f"  Number of nodes: {kg.num_nodes}")
    print(f"  Number of edges: {int(np.sum(adj > 0))}")
    print(f"  Adjacency matrix shape: {adj.shape}")

    print("\n  Strong relationships (weight > 0.7):")
    for i in range(kg.num_nodes):
        for j in range(i+1, kg.num_nodes):
            if adj[i, j] > 0.7:
                print(f"    {kg.FAULT_NAMES[i]:15} <-> {kg.FAULT_NAMES[j]:15} : {adj[i, j]:.2f}")

    # Test model
    print("\nKnowledge Graph PINN Model:")
    model = KnowledgeGraphPINN(
        num_classes=NUM_CLASSES,
        backbone='resnet18',
        node_feature_dim=64,
        gcn_hidden_dim=128,
        num_gcn_layers=2,
        dropout=0.3
    )

    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    print("\nTesting Forward Pass:")
    batch_size = 4
    signal = torch.randn(batch_size, 1, SIGNAL_LENGTH)

    output = model(signal)
    print(f"  Input shape: {signal.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Test with attention
    print("\nTesting Forward with Attention:")
    output, attention = model.forward_with_attention(signal)
    print(f"  Output shape: {output.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Attention statistics:")
    print(f"    Mean: {attention.mean().item():.3f}")
    print(f"    Std: {attention.std().item():.3f}")
    print(f"    Max: {attention.max().item():.3f}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
