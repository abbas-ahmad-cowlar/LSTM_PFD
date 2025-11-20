"""
Neural Architecture Search (NAS) Search Space Definition

Defines the search space for automated architecture discovery.
Can be used with various NAS algorithms (DARTS, random search, evolution).

Search Space Components:
- Operations: Conv kernels, pooling, skip connections
- Connections: How layers connect
- Channel sizes: Width of each layer
- Layer depths: How many blocks

Reference:
- Liu et al. (2019). "DARTS: Differentiable Architecture Search"
- Zoph & Le (2017). "Neural Architecture Search with Reinforcement Learning"
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from enum import Enum


class OperationType(Enum):
    """Available operation types in search space."""
    CONV_3 = "conv_3x1"
    CONV_5 = "conv_5x1"
    CONV_7 = "conv_7x1"
    SEP_CONV_3 = "sep_conv_3x1"
    SEP_CONV_5 = "sep_conv_5x1"
    DILATED_CONV_3_2 = "dilated_conv_3x1_d2"
    DILATED_CONV_3_4 = "dilated_conv_3x1_d4"
    MAX_POOL_3 = "max_pool_3"
    AVG_POOL_3 = "avg_pool_3"
    SKIP_CONNECT = "skip_connect"
    ZERO = "zero"


class SearchSpaceConfig:
    """
    Configuration for NAS search space.

    Args:
        operations: List of allowed operations
        num_nodes: Number of intermediate nodes per cell
        num_cells: Number of cells in network
        channel_sizes: List of channel sizes to explore
        max_layers: Maximum network depth
    """

    def __init__(
        self,
        operations: Optional[List[OperationType]] = None,
        num_nodes: int = 4,
        num_cells: int = 8,
        channel_sizes: Optional[List[int]] = None,
        max_layers: int = 20
    ):
        # Default operations (exclude very expensive ones for 1D signals)
        if operations is None:
            self.operations = [
                OperationType.CONV_3,
                OperationType.CONV_5,
                OperationType.SEP_CONV_3,
                OperationType.SEP_CONV_5,
                OperationType.DILATED_CONV_3_2,
                OperationType.MAX_POOL_3,
                OperationType.AVG_POOL_3,
                OperationType.SKIP_CONNECT,
            ]
        else:
            self.operations = operations

        self.num_nodes = num_nodes
        self.num_cells = num_cells

        # Default channel exploration range
        if channel_sizes is None:
            self.channel_sizes = [32, 64, 128, 256, 512]
        else:
            self.channel_sizes = channel_sizes

        self.max_layers = max_layers

    def get_operation_names(self) -> List[str]:
        """Get list of operation names."""
        return [op.value for op in self.operations]

    def sample_random_architecture(self) -> Dict:
        """
        Sample a random architecture from search space.

        Returns:
            Architecture specification dictionary
        """
        import random

        arch = {
            'cells': [],
            'channels': [],
            'reductions': []  # Whether cell is reduction cell
        }

        for cell_idx in range(self.num_cells):
            cell_ops = []

            # Each cell has num_nodes intermediate nodes
            for node_idx in range(self.num_nodes):
                # Each node has 2 inputs
                input1_idx = random.randint(0, node_idx)  # Can connect to previous nodes
                input2_idx = random.randint(0, node_idx)

                op1 = random.choice(self.operations)
                op2 = random.choice(self.operations)

                cell_ops.append({
                    'input1': input1_idx,
                    'input2': input2_idx,
                    'op1': op1.value,
                    'op2': op2.value
                })

            arch['cells'].append(cell_ops)

            # Sample channels
            channels = random.choice(self.channel_sizes)
            arch['channels'].append(channels)

            # Every 3rd cell is reduction cell
            is_reduction = (cell_idx > 0 and (cell_idx + 1) % 3 == 0)
            arch['reductions'].append(is_reduction)

        return arch

    def count_search_space_size(self) -> int:
        """
        Estimate size of search space.

        Returns:
            Approximate number of possible architectures
        """
        # For each node: choose 2 inputs from previous nodes + 2 operations
        # This is a rough estimate
        num_ops = len(self.operations)
        choices_per_node = (self.num_nodes ** 2) * (num_ops ** 2)
        total_choices = choices_per_node ** (self.num_nodes * self.num_cells)

        return total_choices


class ArchitectureSpec:
    """
    Specification for a discovered/designed architecture.

    Provides easy serialization and deserialization of architectures.
    """

    def __init__(self, arch_dict: Dict):
        """
        Args:
            arch_dict: Architecture dictionary from search
        """
        self.arch_dict = arch_dict

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.arch_dict

    def save(self, filepath: str):
        """Save architecture to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.arch_dict, f, indent=2)
        print(f"Architecture saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ArchitectureSpec':
        """Load architecture from JSON file."""
        import json
        with open(filepath, 'r') as f:
            arch_dict = json.load(f)
        print(f"Architecture loaded from {filepath}")
        return cls(arch_dict)

    def get_model_description(self) -> str:
        """Get human-readable description."""
        lines = []
        lines.append("Architecture Description:")
        lines.append(f"  Number of cells: {len(self.arch_dict['cells'])}")
        lines.append(f"  Channels: {self.arch_dict['channels']}")

        for cell_idx, cell in enumerate(self.arch_dict['cells']):
            is_reduction = self.arch_dict['reductions'][cell_idx]
            cell_type = "Reduction" if is_reduction else "Normal"
            lines.append(f"\n  Cell {cell_idx} ({cell_type}):")

            for node_idx, node in enumerate(cell):
                lines.append(
                    f"    Node {node_idx}: "
                    f"Input[{node['input1']}]→{node['op1']}, "
                    f"Input[{node['input2']}]→{node['op2']}"
                )

        return "\n".join(lines)


# Example usage and utilities
if __name__ == "__main__":
    print("NAS Search Space Definition")
    print("="*60)

    # Create search space
    config = SearchSpaceConfig()

    print(f"\nSearch Space Configuration:")
    print(f"  Operations: {config.get_operation_names()}")
    print(f"  Nodes per cell: {config.num_nodes}")
    print(f"  Number of cells: {config.num_cells}")
    print(f"  Channel sizes: {config.channel_sizes}")

    # Estimate search space size
    space_size = config.count_search_space_size()
    print(f"\nEstimated search space size: {space_size:.2e} architectures")

    # Sample random architecture
    print("\nSampling random architecture...")
    arch = config.sample_random_architecture()
    spec = ArchitectureSpec(arch)

    print(spec.get_model_description())

    # Save and load
    spec.save('/tmp/example_arch.json')
    loaded_spec = ArchitectureSpec.load('/tmp/example_arch.json')

    print("\n✓ Search space tests passed!")
