"""
Physics Parameter Similarity and Pair Selection

Computes similarity between bearing operating conditions (eccentricity,
clearance, viscosity, load, speed) and selects positive/negative pairs
for contrastive learning.

Extracted from: scripts/research/contrastive_physics.py
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


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
