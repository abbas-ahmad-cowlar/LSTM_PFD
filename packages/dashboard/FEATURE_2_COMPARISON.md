# Feature #2: Experiment Comparison

## Overview

The Experiment Comparison feature allows users to compare 2-3 experiments side-by-side to identify which model performs best and understand performance differences across fault classes.

## Key Features

- ✅ Compare 2-3 experiments simultaneously
- ✅ Statistical significance testing (McNemar's test for 2, Friedman test for 3)
- ✅ Side-by-side confusion matrices
- ✅ Per-class performance comparison
- ✅ Training curve comparisons
- ✅ Configuration comparison
- ✅ Shareable comparison URLs
- ⏳ PDF export (coming soon)

## User Guide

### 1. Selecting Experiments for Comparison

**From Experiments Page:**
1. Navigate to `/experiments`
2. Select 2-3 experiments using checkboxes
3. Click the "Compare" button
4. Review selected experiments in the comparison cart
5. Click "View Comparison"

**Direct URL:**
- Navigate directly to: `/compare?ids=1,2,3`
- Replace `1,2,3` with your experiment IDs

### 2. Comparison Tabs

#### Overview Tab 📊
- Winner announcement (highest accuracy)
- Summary cards for each experiment (ranked)
- Quick metrics comparison table
- Key differences summary

#### Metrics Tab 📈
- Overall metrics bar chart
- Per-class performance table
- Per-class F1 scores chart

#### Visualizations Tab 📉
- Side-by-side confusion matrices
- Training loss curves comparison
- Validation accuracy curves comparison

#### Statistical Tests Tab 🔬
- **McNemar's Test** (2 experiments): Tests if models have significantly different error rates
- **Friedman Test** (3 experiments): Tests if at least one model performs significantly differently
- Contingency tables and rankings
- Interpretation guidance

#### Configuration Tab ⚙️
- Compare hyperparameters and settings
- Identify configuration differences

### 3. Sharing Comparisons

Click the "Share Link" button to:
1. Open the share modal
2. Copy the comparison URL
3. Share with team members

## Technical Architecture

### Backend Service

**File:** `dash_app/services/comparison_service.py`

**Key Methods:**
- `validate_comparison_request()`: Validates experiment IDs and permissions
- `get_comparison_data()`: Loads all data needed for comparison
- `_run_mcnemar_test()`: Performs McNemar's statistical test
- `_run_friedman_test()`: Performs Friedman statistical test
- `identify_key_differences()`: Auto-detects key differences

### Frontend Layout

**File:** `dash_app/layouts/experiment_comparison.py`

**Components:**
- `create_experiment_comparison_layout()`: Main comparison page
- `create_overview_tab()`: Overview tab with winner and summary
- `create_metrics_tab()`: Detailed metrics comparison
- `create_visualizations_tab()`: Visual comparisons
- `create_statistical_tab()`: Statistical test results
- `create_configuration_tab()`: Configuration comparison

### Callbacks

**File:** `dash_app/callbacks/comparison_callbacks.py`

**Callbacks:**
- Tab content rendering
- Key differences computation
- Share link modal
- PDF export (placeholder)

**File:** `dash_app/callbacks/experiments_callbacks.py`

**Callbacks:**
- Experiment loading and filtering
- Selection handling
- Navigation to comparison page

### Routing

**File:** `dash_app/callbacks/__init__.py`

URL pattern: `/compare?ids=1,2,3`

## Data Requirements

For statistical tests to work, experiments must have saved prediction data:

**Required Files:**
```
storage/results/experiment_{id}/
├── predictions.npz          # Contains: predictions, labels, probabilities
├── confusion_matrix.npy     # Confusion matrix
└── metrics.json            # Detailed metrics
```

### Saving Evaluation Results

Use the `EvaluationSaver` utility to save results:

```python
from dash_app.utils.evaluation_saver import EvaluationSaver
from evaluation.evaluator import ModelEvaluator

# After training, evaluate the model
evaluator = ModelEvaluator(model, device='cuda')
eval_results = evaluator.evaluate(test_loader, class_names=FAULT_CLASSES)

# Save results for dashboard
EvaluationSaver.save_from_evaluator_results(
    experiment_id=experiment.id,
    eval_results=eval_results
)
```

## Statistical Tests Explained

### McNemar's Test (2 Experiments)

**Purpose:** Tests if two paired models have significantly different error rates.

**How it works:**
1. Builds a contingency table of agreements/disagreements
2. Computes χ² statistic: `(b - c)² / (b + c)`
3. P-value < 0.05 indicates significant difference

**Interpretation:**
- "Significant" = Performance difference is unlikely due to chance
- Tells you IF there's a difference (not HOW MUCH)

### Friedman Test (3 Experiments)

**Purpose:** Tests if at least one model performs significantly differently.

**How it works:**
1. Ranks models for each test sample
2. Compares average rankings
3. P-value < 0.05 indicates significant difference exists

**Interpretation:**
- Lower average rank = better performance (1 is best)
- Tells you if models are significantly different
- Doesn't tell you which specific pairs differ

## Validation Rules

1. **Minimum:** 2 experiments required
2. **Maximum:** 3 experiments allowed
3. **Status:** Only completed experiments can be compared
4. **Test Set:** Experiments must use the same test set for statistical tests
5. **Authorization:** Users can only compare their own experiments

## Future Enhancements

- [ ] PDF export functionality
- [ ] Save comparison as template
- [ ] Compare more than 3 experiments (HPO campaign view)
- [ ] Post-hoc tests for Friedman (pairwise comparisons)
- [ ] Downloadable comparison reports
- [ ] Email sharing
- [ ] Comparison annotations

## API Reference

### ComparisonService

```python
class ComparisonService:
    @staticmethod
    def validate_comparison_request(
        experiment_ids: List[int],
        user_id: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate comparison request."""
        ...

    @staticmethod
    def get_comparison_data(experiment_ids: List[int]) -> Dict:
        """Load all data needed for comparison."""
        ...

    @staticmethod
    def identify_key_differences(comparison_data: Dict) -> List[str]:
        """Identify key differences between experiments."""
        ...
```

### EvaluationSaver

```python
class EvaluationSaver:
    @staticmethod
    def save_experiment_results(
        experiment_id: int,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        confusion_matrix: Optional[np.ndarray] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """Save evaluation results for dashboard."""
        ...

    @staticmethod
    def save_from_evaluator_results(
        experiment_id: int,
        eval_results: Dict
    ) -> None:
        """Save from ModelEvaluator output."""
        ...
```

## Troubleshooting

### "Prediction files not found" Error

**Solution:** Ensure predictions are saved using `EvaluationSaver` after training.

### "Experiments were evaluated on different test sets" Error

**Solution:** Only compare experiments that used the same test dataset.

### "Only completed experiments can be compared" Error

**Solution:** Wait for experiments to complete before comparing.

### Compare button is disabled

**Solution:** Select exactly 2-3 experiments (not 1, not 4+).

## Testing

To test the comparison feature:

1. Ensure you have at least 2 completed experiments
2. Run predictions and save using `EvaluationSaver`
3. Navigate to `/experiments`
4. Select 2-3 experiments
5. Click "Compare"
6. Verify all tabs render correctly
7. Check statistical tests (if predictions available)

## Credits

Implemented as part of Phase 11C: Feature #2 - Experiment Comparison
Based on specification in `feature_2.md`
