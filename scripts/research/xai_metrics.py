"""
XAI Quality Metrics Module

Implements evaluation metrics for explainability methods:
- Faithfulness: Do explanations match model behavior?
- Stability: Are explanations consistent for similar inputs?
- Sparsity: How concise are the explanations?
- Comprehensibility: Human evaluation support

Usage:
    from scripts.research.xai_metrics import XAIEvaluator
    evaluator = XAIEvaluator(model, explainer)
    metrics = evaluator.evaluate_all(test_signals)

Reference: Master Roadmap Chapter 3.2
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExplanationMetrics:
    """Container for XAI quality metrics."""
    faithfulness: float
    stability: float
    sparsity: float
    infidelity: Optional[float] = None
    sensitivity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class XAIEvaluator:
    """
    Evaluator for XAI explanation quality.
    
    Implements metrics from Nauta et al. (2023) and other XAI evaluation literature.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        explainer: Callable,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.explainer = explainer  # Should return attribution for input
        self.device = device
        self.model.eval()
    
    def compute_faithfulness(
        self,
        signal: np.ndarray,
        attribution: np.ndarray,
        n_steps: int = 10
    ) -> float:
        """
        Compute faithfulness by measuring accuracy drop when removing important features.
        
        Higher faithfulness = more accurate explanations
        
        Method: Remove top-k% important features and measure prediction change.
        """
        signal_tensor = torch.tensor(signal, dtype=torch.float32).to(self.device)
        if signal_tensor.ndim == 1:
            signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
        elif signal_tensor.ndim == 2:
            signal_tensor = signal_tensor.unsqueeze(0)
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(signal_tensor)
            original_prob = torch.softmax(original_output, dim=1).max().item()
            original_class = original_output.argmax(dim=1).item()
        
        # Sort features by importance
        importance_order = np.argsort(np.abs(attribution).flatten())[::-1]
        
        # Progressively remove features
        prob_drops = []
        for step in range(1, n_steps + 1):
            n_remove = int(len(importance_order) * step / n_steps)
            mask = np.ones_like(signal)
            mask.flat[importance_order[:n_remove]] = 0
            
            masked_signal = signal * mask
            masked_tensor = torch.tensor(masked_signal, dtype=torch.float32).to(self.device)
            if masked_tensor.ndim == 1:
                masked_tensor = masked_tensor.unsqueeze(0).unsqueeze(0)
            elif masked_tensor.ndim == 2:
                masked_tensor = masked_tensor.unsqueeze(0)
            
            with torch.no_grad():
                masked_output = self.model(masked_tensor)
                masked_prob = torch.softmax(masked_output, dim=1)[0, original_class].item()
            
            prob_drops.append(original_prob - masked_prob)
        
        # Faithfulness = average probability drop (higher = more faithful)
        return np.mean(prob_drops)
    
    def compute_stability(
        self,
        signal: np.ndarray,
        noise_std: float = 0.01,
        n_samples: int = 10
    ) -> float:
        """
        Compute stability by comparing explanations for similar inputs.
        
        Higher stability = more consistent explanations
        
        Method: Add small noise, compute explanations, measure similarity.
        """
        # Get baseline explanation
        baseline_attr = self.explainer(signal)
        
        # Generate noisy versions
        similarities = []
        for _ in range(n_samples):
            noise = np.random.normal(0, noise_std, signal.shape)
            noisy_signal = signal + noise
            noisy_attr = self.explainer(noisy_signal)
            
            # Compute cosine similarity
            sim = 1 - cosine(baseline_attr.flatten(), noisy_attr.flatten())
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def compute_sparsity(self, attribution: np.ndarray, threshold: float = 0.1) -> float:
        """
        Compute sparsity of explanation.
        
        Higher sparsity = fewer important features = more interpretable
        
        Method: Fraction of features below threshold.
        """
        normalized = np.abs(attribution) / (np.abs(attribution).max() + 1e-8)
        sparse_fraction = np.mean(normalized < threshold)
        return sparse_fraction
    
    def compute_infidelity(
        self,
        signal: np.ndarray,
        attribution: np.ndarray,
        n_perturbations: int = 50
    ) -> float:
        """
        Compute infidelity metric (Yeh et al., 2019).
        
        Lower infidelity = better explanation
        
        Method: Measure if explanation * perturbation correlates with output change.
        """
        signal_tensor = torch.tensor(signal, dtype=torch.float32).to(self.device)
        if signal_tensor.ndim == 1:
            signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
        elif signal_tensor.ndim == 2:
            signal_tensor = signal_tensor.unsqueeze(0)
        
        # Get original output
        with torch.no_grad():
            original_output = self.model(signal_tensor)
            original_prob = torch.softmax(original_output, dim=1).max().item()
        
        infidelity_scores = []
        for _ in range(n_perturbations):
            # Random perturbation
            perturbation = np.random.normal(0, 0.1, signal.shape)
            expected_change = np.sum(attribution * perturbation)
            
            # Actual change
            perturbed_signal = signal + perturbation
            perturbed_tensor = torch.tensor(perturbed_signal, dtype=torch.float32).to(self.device)
            if perturbed_tensor.ndim == 1:
                perturbed_tensor = perturbed_tensor.unsqueeze(0).unsqueeze(0)
            elif perturbed_tensor.ndim == 2:
                perturbed_tensor = perturbed_tensor.unsqueeze(0)
            
            with torch.no_grad():
                perturbed_output = self.model(perturbed_tensor)
                perturbed_prob = torch.softmax(perturbed_output, dim=1).max().item()
            
            actual_change = original_prob - perturbed_prob
            
            # Infidelity = squared difference
            infidelity_scores.append((expected_change - actual_change) ** 2)
        
        return np.mean(infidelity_scores)
    
    def evaluate_signal(self, signal: np.ndarray) -> ExplanationMetrics:
        """Evaluate all metrics for a single signal."""
        attribution = self.explainer(signal)
        
        return ExplanationMetrics(
            faithfulness=self.compute_faithfulness(signal, attribution),
            stability=self.compute_stability(signal),
            sparsity=self.compute_sparsity(attribution),
            infidelity=self.compute_infidelity(signal, attribution)
        )
    
    def evaluate_batch(
        self,
        signals: List[np.ndarray],
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate metrics across a batch of signals."""
        all_metrics = []
        
        for i, signal in enumerate(signals):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Evaluating signal {i+1}/{len(signals)}")
            
            metrics = self.evaluate_signal(signal)
            all_metrics.append(metrics.to_dict())
        
        # Aggregate
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated


def create_survey_template() -> str:
    """
    Generate expert validation survey template per Chapter 3.2.
    
    Returns markdown template for collecting domain expert feedback.
    """
    template = """# XAI Expert Validation Survey

## Instructions
Rate each explanation on a scale of 1-5 for the following criteria:

## Evaluation Criteria

### 1. Correctness (1-5)
Does the highlighted region correspond to known fault signatures?
- 1: Completely incorrect
- 5: Exactly matches expected fault pattern

### 2. Completeness (1-5)
Does the explanation capture all relevant features?
- 1: Misses critical features
- 5: Captures all diagnostic features

### 3. Actionability (1-5)
Would this explanation help a technician make a maintenance decision?
- 1: Not actionable
- 5: Clearly actionable

### 4. Trust (1-5)
How much would you trust a system that provides these explanations?
- 1: Would not trust
- 5: Fully trust

---

## Sample Evaluations

| Sample ID | Fault Type | Correctness | Completeness | Actionability | Trust | Notes |
|-----------|------------|-------------|--------------|---------------|-------|-------|
| S001 | Inner Race | | | | | |
| S002 | Outer Race | | | | | |
| S003 | Ball Fault | | | | | |
| S004 | Imbalance | | | | | |
| S005 | Misalignment | | | | | |

## Evaluator Information
- Name: _______________
- Years of Experience: ___
- Domain: _______________
- Date: _______________
"""
    return template


if __name__ == '__main__':
    # Generate survey template
    from pathlib import Path
    
    output_dir = Path('results/xai_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    survey = create_survey_template()
    with open(output_dir / 'expert_survey_template.md', 'w') as f:
        f.write(survey)
    
    print(f"Survey template saved to {output_dir / 'expert_survey_template.md'}")
