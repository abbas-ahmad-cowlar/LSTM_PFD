# FEATURE #2: EXPERIMENT COMPARISON (2-3 MODELS SIDE-BY-SIDE)

**Duration:** 2 weeks (10 days)  
**Priority:** P0 (High - Frequently requested by users)  
**Assigned To:** Full-Stack Developer

---

## 2.1 OBJECTIVES

### Primary Objective
Enable users to compare 2-3 experiments side-by-side to identify which model performs best and understand performance differences across fault classes.

### Success Criteria
- Users can select 2-3 experiments from experiment history page
- Comparison page displays metrics, confusion matrices, and training curves in unified view
- Statistical significance testing shows if differences are meaningful
- Users can identify which model is better for specific fault types
- Comparison can be bookmarked/shared via URL
- Export comparison as PDF report

### Business Value
- **Time Savings:** Users currently export data to Excel for comparison (30+ minutes) → Now 2 clicks (30 seconds)
- **Better Decisions:** Statistical tests remove guesswork ("Is 96.8% significantly better than 96.5%?")
- **Collaboration:** Share comparison URL with team for discussion
- **Reproducibility:** Documented comparisons for audit trail

---

## 2.2 TECHNICAL SPECIFICATIONS

### User Journey

```
STEP 1: SELECT EXPERIMENTS
User Path A (From Experiment History):
  /experiments → Select checkboxes [Exp 1234] [Exp 1567] → Click "Compare Selected"
  
User Path B (From Experiment Results):
  /experiment/1234/results → Click "Compare with Another" → Modal with dropdown
  
User Path C (Direct URL):
  /compare?ids=1234,1567,1890 → Load comparison directly

STEP 2: VIEW COMPARISON
  /compare?ids=1234,1567 → Loads comparison page
  
  Layout:
  ┌─────────────────────────────────────────────────────┐
  │ EXPERIMENT COMPARISON                                │
  │ Exp 1234 vs Exp 1567 vs Exp 1890                    │
  ├─────────────────────────────────────────────────────┤
  │ [Overview] [Metrics] [Visualizations] [Statistical] │
  │                                                      │
  │ TAB: OVERVIEW                                        │
  │ ┌────────┬────────┬────────┐                        │
  │ │ Exp 1234│ Exp 1567│ Exp 1890│                      │
  │ ├────────┼────────┼────────┤                        │
  │ │ ResNet34│Transform│ PINN   │                       │
  │ │ 96.8%  │ 96.5%  │ 97.1%  │                        │
  │ │ 14m 32s│ 22m 11s│ 18m 3s │                        │
  │ └────────┴────────┴────────┘                        │
  │                                                      │
  │ WINNER: Exp 1890 (PINN) - 97.1% accuracy ⭐         │
  │                                                      │
  │ ... (more tabs)                                      │
  └─────────────────────────────────────────────────────┘

STEP 3: ANALYZE DIFFERENCES
  User explores tabs:
  - Metrics: Per-class precision/recall comparison
  - Visualizations: Confusion matrices side-by-side
  - Statistical: McNemar's test results
  
STEP 4: EXPORT OR SHARE
  - Click "Export PDF" → Downloads comparison report
  - Click "Share Link" → Copies /compare?ids=... to clipboard
  - Click "Add to Report" → Saves to user's report collection
```

### URL Structure

```
Primary URL:
  /compare?ids=1234,1567,1890
  
Query Parameters:
  - ids: Comma-separated experiment IDs (required, 2-3 values)
  - tab: Active tab (optional, default: overview)
  - metric: Sort metric (optional, default: accuracy)
  
Examples:
  /compare?ids=1234,1567
  /compare?ids=1234,1567,1890&tab=metrics
  /compare?ids=1234,1567&metric=f1_score

URL Validation:
  - Min 2 experiments (error if <2)
  - Max 3 experiments (error if >3, suggest HPO campaign for >3)
  - All IDs must exist and belong to user (403 if unauthorized)
  - Redirect to /experiments if any validation fails
```

### Database Schema

```sql
-- New table for saved comparisons (optional feature, can be added later)
CREATE TABLE experiment_comparisons (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(200),  -- User-provided name (e.g., "ResNet vs Transformer")
    experiment_ids INTEGER[] NOT NULL,  -- Array of experiment IDs [1234, 1567, 1890]
    notes TEXT,  -- User notes about comparison
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_comparisons_user_id ON experiment_comparisons(user_id);
CREATE INDEX idx_comparisons_experiment_ids ON experiment_comparisons USING GIN(experiment_ids);

-- No changes to existing tables needed
-- Comparison page reads from existing 'experiments' table
```

---

## 2.3 IMPLEMENTATION TASKS

### Day 1-2: Backend Service Layer

**Task 2.1:** Create Comparison Service
- **File:** `services/comparison_service.py`
- **Purpose:** Load and aggregate data for multiple experiments
- **Code:**

```python
from typing import List, Dict, Optional
from models.experiment import Experiment
from models.training_run import TrainingRun
from database.connection import get_db_session
import numpy as np
from scipy import stats
import json

class ComparisonService:
    
    @staticmethod
    def validate_comparison_request(experiment_ids: List[int], user_id: int) -> tuple:
        """
        Validate that comparison request is valid.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            user_id: ID of requesting user (for authorization)
            
        Returns:
            (valid: bool, error_message: str or None)
        """
        
        # Validate count
        if len(experiment_ids) < 2:
            return False, "At least 2 experiments required for comparison"
        
        if len(experiment_ids) > 3:
            return False, "Maximum 3 experiments can be compared. For more, use HPO Campaign analysis."
        
        # Check for duplicates
        if len(experiment_ids) != len(set(experiment_ids)):
            return False, "Duplicate experiment IDs not allowed"
        
        # Validate existence and authorization
        session = get_db_session()
        experiments = session.query(Experiment).filter(
            Experiment.id.in_(experiment_ids)
        ).all()
        
        if len(experiments) != len(experiment_ids):
            missing = set(experiment_ids) - {e.id for e in experiments}
            return False, f"Experiments not found: {missing}"
        
        # Check authorization (user owns experiments or experiments are shared)
        for exp in experiments:
            if exp.user_id != user_id:
                # Check if experiment is shared with user (future feature)
                # For now, only owner can compare
                return False, f"Unauthorized access to experiment {exp.id}"
        
        return True, None
    
    @staticmethod
    def get_comparison_data(experiment_ids: List[int]) -> Dict:
        """
        Load all data needed for comparison.
        
        Args:
            experiment_ids: List of experiment IDs (2-3)
            
        Returns:
            Dictionary with structure:
            {
                'experiments': [
                    {
                        'id': 1234,
                        'name': 'ResNet34_Standard',
                        'model_type': 'resnet',
                        'created_at': '2025-06-15T14:32:11Z',
                        'status': 'completed',
                        'duration_seconds': 872,
                        'metrics': {
                            'accuracy': 0.968,
                            'precision': 0.965,
                            'recall': 0.967,
                            'f1_score': 0.966
                        },
                        'per_class_metrics': {
                            'normal': {'precision': 0.985, 'recall': 0.992, 'f1': 0.988, 'support': 130},
                            'misalignment': {...},
                            ...
                        },
                        'confusion_matrix': [[...], [...], ...],  # 11x11 matrix
                        'training_history': {
                            'epochs': [1, 2, ..., 100],
                            'train_loss': [0.52, 0.34, ..., 0.012],
                            'val_loss': [0.48, 0.41, ..., 0.039],
                            'val_accuracy': [0.72, 0.81, ..., 0.968]
                        },
                        'config': {
                            'batch_size': 32,
                            'learning_rate': 0.001,
                            ...
                        }
                    },
                    {...},  # Experiment 2
                    {...}   # Experiment 3 (optional)
                ],
                'statistical_tests': {
                    'mcnemar': {...},  # If 2 experiments
                    'friedman': {...}  # If 3 experiments
                }
            }
        """
        
        session = get_db_session()
        
        # Load experiments
        experiments = session.query(Experiment).filter(
            Experiment.id.in_(experiment_ids)
        ).order_by(Experiment.id).all()
        
        comparison_data = {
            'experiments': [],
            'statistical_tests': {}
        }
        
        # Load detailed data for each experiment
        for exp in experiments:
            # Load metrics from database (JSON field)
            metrics = json.loads(exp.metrics) if isinstance(exp.metrics, str) else exp.metrics
            
            # Load training history
            training_runs = session.query(TrainingRun).filter(
                TrainingRun.experiment_id == exp.id
            ).order_by(TrainingRun.epoch).all()
            
            training_history = {
                'epochs': [run.epoch for run in training_runs],
                'train_loss': [run.train_loss for run in training_runs],
                'val_loss': [run.val_loss for run in training_runs],
                'val_accuracy': [run.val_accuracy for run in training_runs]
            }
            
            # Load confusion matrix (stored as JSON in results table or file)
            confusion_matrix = ComparisonService._load_confusion_matrix(exp.id)
            
            # Load per-class metrics
            per_class_metrics = metrics.get('per_class', {})
            
            # Load config
            config = json.loads(exp.config) if isinstance(exp.config, str) else exp.config
            
            experiment_data = {
                'id': exp.id,
                'name': exp.name,
                'model_type': exp.model_type,
                'created_at': exp.created_at.isoformat(),
                'status': exp.status,
                'duration_seconds': exp.duration_seconds,
                'metrics': {
                    'accuracy': metrics.get('accuracy'),
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1_score': metrics.get('f1_score')
                },
                'per_class_metrics': per_class_metrics,
                'confusion_matrix': confusion_matrix,
                'training_history': training_history,
                'config': config
            }
            
            comparison_data['experiments'].append(experiment_data)
        
        # Run statistical tests
        if len(experiments) == 2:
            # McNemar's test for pairwise comparison
            comparison_data['statistical_tests']['mcnemar'] = ComparisonService._run_mcnemar_test(
                experiments[0].id, experiments[1].id
            )
        elif len(experiments) == 3:
            # Friedman test for 3+ models
            comparison_data['statistical_tests']['friedman'] = ComparisonService._run_friedman_test(
                [e.id for e in experiments]
            )
        
        return comparison_data
    
    @staticmethod
    def _load_confusion_matrix(experiment_id: int) -> List[List[int]]:
        """
        Load confusion matrix for an experiment.
        
        Confusion matrix is stored in:
          storage/experiments/{experiment_id}/results/confusion_matrix.npy
          
        Returns:
            11x11 matrix (list of lists)
        """
        import numpy as np
        import os
        from config import Config
        
        matrix_path = os.path.join(
            Config.STORAGE_PATH,
            'experiments',
            str(experiment_id),
            'results',
            'confusion_matrix.npy'
        )
        
        if not os.path.exists(matrix_path):
            # Fallback: Return empty matrix or raise error
            return [[0] * 11 for _ in range(11)]
        
        matrix = np.load(matrix_path)
        return matrix.tolist()
    
    @staticmethod
    def _run_mcnemar_test(exp1_id: int, exp2_id: int) -> Dict:
        """
        Run McNemar's test for paired comparison of two models.
        
        McNemar's test: Tests if two models have significantly different error rates.
        
        Contingency table:
                    Model 2 Correct  Model 2 Wrong
        Model 1 Correct      a              b
        Model 1 Wrong        c              d
        
        Test statistic: χ² = (b - c)² / (b + c)
        p-value: From chi-square distribution with 1 degree of freedom
        
        Args:
            exp1_id: First experiment ID
            exp2_id: Second experiment ID
            
        Returns:
            {
                'test_statistic': float,
                'p_value': float,
                'contingency_table': [[a, b], [c, d]],
                'interpretation': str,
                'significant': bool (p < 0.05)
            }
        """
        
        # Load predictions for both experiments on SAME test set
        # Predictions stored in: storage/experiments/{id}/results/predictions.npy
        import numpy as np
        import os
        from config import Config
        from scipy.stats import chi2
        
        def load_predictions(exp_id):
            pred_path = os.path.join(
                Config.STORAGE_PATH, 'experiments', str(exp_id), 
                'results', 'predictions.npy'
            )
            if not os.path.exists(pred_path):
                return None
            return np.load(pred_path)
        
        pred1 = load_predictions(exp1_id)
        pred2 = load_predictions(exp2_id)
        
        if pred1 is None or pred2 is None:
            return {
                'error': 'Prediction files not found',
                'test_statistic': None,
                'p_value': None
            }
        
        # Load ground truth labels (same for both)
        # Assuming predictions.npy contains {'predictions': [...], 'labels': [...]}
        # Or load from a shared test set file
        
        # For this example, assume predictions are dictionaries
        y_true = pred1['labels']
        y_pred1 = pred1['predictions']
        y_pred2 = pred2['predictions']
        
        # Build contingency table
        # a: both correct
        # b: model1 correct, model2 wrong
        # c: model1 wrong, model2 correct
        # d: both wrong
        
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        a = np.sum(correct1 & correct2)
        b = np.sum(correct1 & ~correct2)
        c = np.sum(~correct1 & correct2)
        d = np.sum(~correct1 & ~correct2)
        
        contingency_table = [[a, b], [c, d]]
        
        # McNemar's test statistic
        if b + c == 0:
            # No disagreements, models are identical
            return {
                'test_statistic': 0.0,
                'p_value': 1.0,
                'contingency_table': contingency_table,
                'interpretation': 'Models make identical predictions (no disagreements).',
                'significant': False
            }
        
        test_statistic = (b - c) ** 2 / (b + c)
        
        # p-value from chi-square distribution (1 degree of freedom)
        p_value = 1 - chi2.cdf(test_statistic, df=1)
        
        # Interpretation
        if p_value < 0.05:
            if b > c:
                winner = "Model 1"
            else:
                winner = "Model 2"
            interpretation = f"{winner} performs significantly better (p = {p_value:.4f})."
        else:
            interpretation = f"No significant difference between models (p = {p_value:.4f})."
        
        return {
            'test_statistic': float(test_statistic),
            'p_value': float(p_value),
            'contingency_table': contingency_table,
            'interpretation': interpretation,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def _run_friedman_test(experiment_ids: List[int]) -> Dict:
        """
        Run Friedman test for comparing 3+ models.
        
        Friedman test: Non-parametric test for repeated measures (like ANOVA but for ranks).
        
        Args:
            experiment_ids: List of 3+ experiment IDs
            
        Returns:
            {
                'test_statistic': float,
                'p_value': float,
                'rankings': [1.2, 2.8, 2.0],  # Average rank per model (1=best)
                'interpretation': str,
                'significant': bool
            }
        """
        
        from scipy.stats import friedmanchisquare
        import numpy as np
        
        # Load predictions for all experiments
        all_predictions = []
        y_true = None
        
        for exp_id in experiment_ids:
            pred_path = os.path.join(
                Config.STORAGE_PATH, 'experiments', str(exp_id),
                'results', 'predictions.npy'
            )
            if not os.path.exists(pred_path):
                return {'error': f'Predictions not found for experiment {exp_id}'}
            
            pred_data = np.load(pred_path, allow_pickle=True).item()
            all_predictions.append(pred_data['predictions'])
            
            if y_true is None:
                y_true = pred_data['labels']
        
        # Compute correctness for each model on each sample
        # correctness[i][j] = 1 if model i predicted sample j correctly, else 0
        correctness = []
        for preds in all_predictions:
            correctness.append((preds == y_true).astype(int))
        
        # Run Friedman test
        statistic, p_value = friedmanchisquare(*correctness)
        
        # Compute average rankings
        # Rank models for each sample (1=best, 3=worst)
        n_samples = len(y_true)
        n_models = len(experiment_ids)
        
        sample_ranks = []
        for i in range(n_samples):
            sample_correctness = [correctness[m][i] for m in range(n_models)]
            # Rank: 1 for correct, 2 for incorrect (ties handled by scipy)
            ranks = stats.rankdata([-c for c in sample_correctness], method='average')
            sample_ranks.append(ranks)
        
        # Average rank per model
        avg_ranks = np.mean(sample_ranks, axis=0).tolist()
        
        # Interpretation
        if p_value < 0.05:
            best_model_idx = np.argmin(avg_ranks)
            interpretation = f"Significant difference exists (p = {p_value:.4f}). Experiment {experiment_ids[best_model_idx]} ranks best."
        else:
            interpretation = f"No significant difference among models (p = {p_value:.4f})."
        
        return {
            'test_statistic': float(statistic),
            'p_value': float(p_value),
            'rankings': avg_ranks,
            'interpretation': interpretation,
            'significant': p_value < 0.05
        }
```

**Testing Criteria (Day 2):**
- ✅ `validate_comparison_request()` rejects <2 or >3 experiments
- ✅ `validate_comparison_request()` rejects unauthorized access
- ✅ `get_comparison_data()` returns all experiments with metrics
- ✅ McNemar's test computes correct contingency table
- ✅ Friedman test returns rankings in correct order
- ✅ Performance: Loading comparison data takes <500ms for 3 experiments

---

### Day 3-4: Layout & UI Components

**Task 2.2:** Create Comparison Page Layout
- **File:** `layouts/experiment_comparison.py`
- **Purpose:** Main comparison page with tabs
- **Code:**

```python
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from utils.constants import FAULT_CLASSES

def layout(experiment_ids):
    """
    Comparison page layout.
    
    Args:
        experiment_ids: List of experiment IDs from URL (e.g., [1234, 1567, 1890])
    """
    
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🔍 Experiment Comparison", className="mb-2"),
                html.P(
                    f"Comparing {len(experiment_ids)} experiments",
                    className="text-muted"
                )
            ], width=8),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("📥 Export PDF", id='export-comparison-pdf', color="secondary"),
                    dbc.Button("🔗 Share Link", id='share-comparison-link', color="info"),
                    dbc.Button("💾 Save Comparison", id='save-comparison', color="primary")
                ], className="float-end")
            ], width=4)
        ], className="mb-4"),
        
        # Breadcrumb navigation
        dbc.Row([
            dbc.Col([
                html.Nav([
                    html.A("Experiments", href="/experiments", className="breadcrumb-item"),
                    html.Span(" / ", className="breadcrumb-separator"),
                    html.Span("Comparison", className="breadcrumb-item active")
                ])
            ])
        ], className="mb-3"),
        
        # Loading indicator (shown while data loads)
        dcc.Loading(
            id="comparison-loading",
            type="default",
            children=[
                # Hidden div to store comparison data
                html.Div(id='comparison-data-store', style={'display': 'none'}),
                
                # Main content tabs
                dbc.Tabs(id='comparison-tabs', active_tab='overview', children=[
                    dbc.Tab(label="Overview", tab_id="overview"),
                    dbc.Tab(label="Metrics", tab_id="metrics"),
                    dbc.Tab(label="Visualizations", tab_id="visualizations"),
                    dbc.Tab(label="Statistical Tests", tab_id="statistical"),
                    dbc.Tab(label="Configuration", tab_id="configuration")
                ]),
                
                # Tab content container
                html.Div(id='comparison-tab-content', className="mt-4")
            ]
        ),
        
        # Modals
        create_share_link_modal(),
        create_save_comparison_modal()
        
    ], fluid=True, className="py-4")


def create_overview_tab(comparison_data):
    """
    Overview tab: High-level summary of compared experiments.
    
    Args:
        comparison_data: Dictionary from ComparisonService.get_comparison_data()
    """
    
    experiments = comparison_data['experiments']
    
    # Determine winner (highest accuracy)
    best_exp = max(experiments, key=lambda e: e['metrics']['accuracy'])
    
    return html.Div([
        # Winner announcement
        dbc.Alert([
            html.H4("🏆 Winner", className="alert-heading"),
            html.P(f"Experiment #{best_exp['id']}: {best_exp['name']}", className="mb-1"),
            html.P(f"Accuracy: {best_exp['metrics']['accuracy']:.2%}", className="mb-0 font-weight-bold")
        ], color="success", className="mb-4"),
        
        # Summary cards (one per experiment)
        dbc.Row([
            dbc.Col([
                create_experiment_summary_card(exp, rank=idx+1)
                for idx, exp in enumerate(
                    sorted(experiments, key=lambda e: e['metrics']['accuracy'], reverse=True)
                )
            ], width=4) for exp in experiments
        ], className="mb-4"),
        
        # Quick metrics comparison table
        html.H4("Quick Metrics Comparison", className="mb-3"),
        create_metrics_comparison_table(experiments),
        
        # Key differences summary
        html.H4("Key Differences", className="mt-4 mb-3"),
        create_key_differences_summary(experiments)
    ])


def create_experiment_summary_card(experiment, rank):
    """
    Card showing summary of single experiment.
    
    Args:
        experiment: Experiment data dictionary
        rank: Position in ranking (1, 2, 3)
    """
    
    # Medal emoji based on rank
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    medal = medals.get(rank, "")
    
    # Badge color based on rank
    badge_colors = {1: "success", 2: "info", 3: "warning"}
    badge_color = badge_colors.get(rank, "secondary")
    
    return dbc.Card([
        dbc.CardHeader([
            html.Span(medal, className="me-2"),
            html.Span(f"Rank #{rank}", className="badge bg-{badge_color}")
        ]),
        dbc.CardBody([
            html.H5(f"{experiment['name']}", className="card-title"),
            html.P(f"ID: {experiment['id']} | Type: {experiment['model_type']}", 
                   className="text-muted small"),
            html.Hr(),
            html.Div([
                create_metric_row("Accuracy", experiment['metrics']['accuracy'], format_pct=True),
                create_metric_row("F1-Score", experiment['metrics']['f1_score'], format_pct=True),
                create_metric_row("Precision", experiment['metrics']['precision'], format_pct=True),
                create_metric_row("Recall", experiment['metrics']['recall'], format_pct=True),
                create_metric_row("Duration", f"{experiment['duration_seconds'] // 60}m {experiment['duration_seconds'] % 60}s")
            ])
        ])
    ], className="mb-3")


def create_metric_row(label, value, format_pct=False):
    """Helper to create a metric row in card."""
    if format_pct and isinstance(value, (int, float)):
        value_str = f"{value:.2%}"
    else:
        value_str = str(value)
    
    return html.Div([
        html.Span(label, className="text-muted"),
        html.Span(value_str, className="float-end font-weight-bold")
    ], className="mb-2")


def create_metrics_comparison_table(experiments):
    """
    Table comparing all metrics side-by-side.
    
    Returns:
        Dash AG-Grid table or Bootstrap table
    """
    
    # Build table data
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
    
    table_header = [
        html.Thead(html.Tr([
            html.Th("Metric"),
            *[html.Th(f"Exp {exp['id']}") for exp in experiments],
            html.Th("Best")
        ]))
    ]
    
    table_rows = []
    for metric in metrics_to_compare:
        values = [exp['metrics'][metric] for exp in experiments]
        best_value = max(values)
        
        row = html.Tr([
            html.Td(metric.replace('_', ' ').title()),
            *[
                html.Td(
                    f"{val:.2%}",
                    className="font-weight-bold text-success" if val == best_value else ""
                )
                for val in values
            ],
            html.Td(f"{best_value:.2%}", className="text-success")
        ])
        table_rows.append(row)
    
    table_body = [html.Tbody(table_rows)]
    
    return dbc.Table(
        table_header + table_body,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True
    )


def create_key_differences_summary(experiments):
    """
    Automatically identify and highlight key differences.
    
    Examples:
    - "Exp 1234 (ResNet) is 0.3% more accurate than Exp 1567 (Transformer)"
    - "Exp 1234 takes 8 minutes less to train"
    - "Exp 1890 (PINN) excels at Oil Whirl detection (+5% recall)"
    """
    
    differences = []
    
    # Accuracy difference
    accuracies = [(exp['id'], exp['name'], exp['metrics']['accuracy']) for exp in experiments]
    accuracies_sorted = sorted(accuracies, key=lambda x: x[2], reverse=True)
    
    best = accuracies_sorted[0]
    second = accuracies_sorted[1]
    diff_pct = (best[2] - second[2]) * 100
    
    if diff_pct > 0.5:
        differences.append(
            html.Li(f"Exp {best[0]} ({best[1]}) is {diff_pct:.1f}% more accurate than Exp {second[0]} ({second[1]})")
        )
    
    # Training time difference
    durations = [(exp['id'], exp['name'], exp['duration_seconds']) for exp in experiments]
    fastest = min(durations, key=lambda x: x[2])
    slowest = max(durations, key=lambda x: x[2])
    time_diff_min = (slowest[2] - fastest[2]) // 60
    
    if time_diff_min > 5:
        differences.append(
            html.Li(f"Exp {fastest[0]} trains {time_diff_min} minutes faster than Exp {slowest[0]}")
        )
    
    # Per-class performance differences (find largest gap)
    # Compare recall for each fault class
    for fault_class in FAULT_CLASSES:
        recalls = []
        for exp in experiments:
            recall = exp['per_class_metrics'].get(fault_class, {}).get('recall', 0)
            recalls.append((exp['id'], exp['name'], recall))
        
        if recalls:
            best_recall = max(recalls, key=lambda x: x[2])
            worst_recall = min(recalls, key=lambda x: x[2])
            recall_diff = (best_recall[2] - worst_recall[2]) * 100
            
            if recall_diff > 5:  # >5% difference
                differences.append(
                    html.Li(f"Exp {best_recall[0]} excels at {fault_class} detection (+{recall_diff:.1f}% recall vs Exp {worst_recall[0]})")
                )
    
    if not differences:
        return html.P("Models perform similarly across all metrics.", className="text-muted")
    
    return html.Ul(differences[:5])  # Show top 5 differences


def create_metrics_tab(comparison_data):
    """
    Metrics tab: Per-class performance comparison.
    """
    
    experiments = comparison_data['experiments']
    
    return html.Div([
        html.H4("Per-Class Performance Comparison", className="mb-3"),
        
        # Tabs for each fault class
        dbc.Tabs([
            dbc.Tab(
                label=fault_class.replace('_', ' ').title(),
                tab_id=f"class-{fault_class}",
                children=[
                    create_per_class_comparison(experiments, fault_class)
                ]
            )
            for fault_class in FAULT_CLASSES
        ]),
        
        # Overall comparison heatmap
        html.H4("Recall Heatmap (All Classes)", className="mt-4 mb-3"),
        dcc.Graph(
            id='recall-heatmap',
            figure=create_recall_heatmap(experiments)
        )
    ])


def create_per_class_comparison(experiments, fault_class):
    """
    Comparison for a single fault class.
    """
    
    # Extract metrics for this class from all experiments
    class_data = []
    for exp in experiments:
        metrics = exp['per_class_metrics'].get(fault_class, {})
        class_data.append({
            'experiment_id': exp['id'],
            'experiment_name': exp['name'],
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'support': metrics.get('support', 0)
        })
    
    # Create bar chart
    fig = go.Figure()
    
    metrics_to_plot = ['precision', 'recall', 'f1']
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=[d['experiment_name'] for d in class_data],
            y=[d[metric] for d in class_data],
            text=[f"{d[metric]:.2%}" for d in class_data],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f"{fault_class.replace('_', ' ').title()} - Performance Metrics",
        xaxis_title="Experiment",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        barmode='group',
        height=400
    )
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.P(f"Support: {class_data[0]['support']} samples", className="text-muted")
    ], className="mt-3")


def create_recall_heatmap(experiments):
    """
    Heatmap showing recall for each experiment × fault class.
    """
    
    # Build matrix: rows = experiments, cols = fault classes
    recall_matrix = []
    experiment_names = []
    
    for exp in experiments:
        experiment_names.append(f"Exp {exp['id']}")
        recalls = []
        for fault_class in FAULT_CLASSES:
            recall = exp['per_class_metrics'].get(fault_class, {}).get('recall', 0)
            recalls.append(recall * 100)  # Convert to percentage
        recall_matrix.append(recalls)
    
    fig = go.Figure(data=go.Heatmap(
        z=recall_matrix,
        x=[fc.replace('_', ' ').title() for fc in FAULT_CLASSES],
        y=experiment_names,
        colorscale='RdYlGn',
        zmin=0,
        zmax=100,
        text=[[f"{val:.1f}%" for val in row] for row in recall_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Recall by Experiment and Fault Class (%)",
        xaxis_title="Fault Class",
        yaxis_title="Experiment",
        height=300 + len(experiments) * 50,  # Dynamic height based on # experiments
        xaxis={'side': 'bottom'}
    )
    
    return fig


def create_visualizations_tab(comparison_data):
    """
    Visualizations tab: Confusion matrices, training curves side-by-side.
    """
    
    experiments = comparison_data['experiments']
    
    return html.Div([
        # Confusion matrices (side-by-side)
        html.H4("Confusion Matrices", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H6(f"Exp {exp['id']}: {exp['name']}", className="text-center"),
                dcc.Graph(
                    figure=create_confusion_matrix_heatmap(exp),
                    config={'displayModeBar': False}
                )
            ], width=12 // len(experiments))
            for exp in experiments
        ], className="mb-4"),
        
        # Confusion matrix difference (if 2 experiments)
        html.Div([
            html.H4("Confusion Matrix Difference", className="mb-3"),
            html.P("Green: Exp 1 better | Red: Exp 2 better", className="text-muted"),
            dcc.Graph(
                figure=create_confusion_matrix_difference(experiments[0], experiments[1])
            )
        ], className="mb-4") if len(experiments) == 2 else html.Div(),
        
        # Training curves overlay
        html.H4("Training History", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_training_curves_overlay(experiments, metric='loss'))
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=create_training_curves_overlay(experiments, metric='accuracy'))
            ], width=6)
        ])
    ])


def create_confusion_matrix_heatmap(experiment):
    """Create confusion matrix heatmap for single experiment."""
    
    cm = experiment['confusion_matrix']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[fc.replace('_', ' ')[:10] for fc in FAULT_CLASSES],  # Truncate labels
        y=[fc.replace('_', ' ')[:10] for fc in FAULT_CLASSES],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title=f"Exp {experiment['id']}",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=400,
        width=400
    )
    
    return fig


def create_confusion_matrix_difference(exp1, exp2):
    """
    Show difference between two confusion matrices.
    Positive (green) = Exp 1 better, Negative (red) = Exp 2 better.
    """
    
    import numpy as np
    
    cm1 = np.array(exp1['confusion_matrix'])
    cm2 = np.array(exp2['confusion_matrix'])
    
    # Difference: cm1 - cm2
    diff = cm1 - cm2
    
    fig = go.Figure(data=go.Heatmap(
        z=diff,
        x=[fc.replace('_', ' ')[:10] for fc in FAULT_CLASSES],
        y=[fc.replace('_', ' ')[:10] for fc in FAULT_CLASSES],
        colorscale='RdYlGn',
        zmid=0,  # Center colorscale at 0
        text=diff,
        texttemplate="%{text:+d}",  # Show sign (+/-)
        textfont={"size": 9}
    ))
    
    fig.update_layout(
        title=f"Difference: Exp {exp1['id']} - Exp {exp2['id']}",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=500,
        width=500
    )
    
    return fig


def create_training_curves_overlay(experiments, metric='loss'):
    """
    Overlay training curves from multiple experiments.
    
    Args:
        experiments: List of experiment data
        metric: 'loss' or 'accuracy'
    """
    
    fig = go.Figure()
    
    for exp in experiments:
        history = exp['training_history']
        
        if metric == 'loss':
            # Plot both train and val loss
            fig.add_trace(go.Scatter(
                x=history['epochs'],
                y=history['train_loss'],
                mode='lines',
                name=f"Exp {exp['id']} (Train)",
                line=dict(dash='solid')
            ))
            fig.add_trace(go.Scatter(
                x=history['epochs'],
                y=history['val_loss'],
                mode='lines',
                name=f"Exp {exp['id']} (Val)",
                line=dict(dash='dash')
            ))
        else:  # accuracy
            fig.add_trace(go.Scatter(
                x=history['epochs'],
                y=history['val_accuracy'],
                mode='lines',
                name=f"Exp {exp['id']}"
            ))
    
    fig.update_layout(
        title=f"{'Loss' if metric == 'loss' else 'Validation Accuracy'} Over Epochs",
        xaxis_title="Epoch",
        yaxis_title='Loss' if metric == 'loss' else 'Accuracy',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_statistical_tab(comparison_data):
    """
    Statistical tests tab: McNemar's test (2 models) or Friedman test (3 models).
    """
    
    experiments = comparison_data['experiments']
    statistical_tests = comparison_data['statistical_tests']
    
    if len(experiments) == 2:
        # McNemar's test
        mcnemar = statistical_tests.get('mcnemar', {})
        
        return html.Div([
            html.H4("McNemar's Test (Pairwise Comparison)", className="mb-3"),
            html.P(
                "McNemar's test assesses whether two models have significantly different error rates.",
                className="text-muted"
            ),
            
            # Test results
            dbc.Alert([
                html.H5("Test Result", className="alert-heading"),
                html.P(mcnemar.get('interpretation', 'Test not available')),
                html.Hr(),
                html.Div([
                    html.Strong("Test Statistic: "),
                    html.Span(f"{mcnemar.get('test_statistic', 'N/A'):.4f}")
                ], className="mb-2"),
                html.Div([
                    html.Strong("p-value: "),
                    html.Span(f"{mcnemar.get('p_value', 'N/A'):.4f}"),
                    html.Span(
                        " (significant)" if mcnemar.get('significant') else " (not significant)",
                        className="ms-2 fst-italic"
                    )
                ])
            ], color="success" if mcnemar.get('significant') else "info"),
            
            # Contingency table
            html.H5("Contingency Table", className="mt-4 mb-3"),
            create_contingency_table_display(mcnemar.get('contingency_table', [[0,0],[0,0]])),
            
            # Explanation
            html.H5("Interpretation Guide", className="mt-4 mb-3"),
            html.Ul([
                html.Li("p-value < 0.05: Significant difference (reject null hypothesis)"),
                html.Li("p-value ≥ 0.05: No significant difference (fail to reject null)"),
                html.Li("Contingency table shows agreements and disagreements between models")
            ])
        ])
    
    elif len(experiments) == 3:
        # Friedman test
        friedman = statistical_tests.get('friedman', {})
        
        return html.Div([
            html.H4("Friedman Test (Multiple Model Comparison)", className="mb-3"),
            html.P(
                "Friedman test ranks models on each sample and tests if rankings differ significantly.",
                className="text-muted"
            ),
            
            # Test results
            dbc.Alert([
                html.H5("Test Result", className="alert-heading"),
                html.P(friedman.get('interpretation', 'Test not available')),
                html.Hr(),
                html.Div([
                    html.Strong("Test Statistic: "),
                    html.Span(f"{friedman.get('test_statistic', 'N/A'):.4f}")
                ], className="mb-2"),
                html.Div([
                    html.Strong("p-value: "),
                    html.Span(f"{friedman.get('p_value', 'N/A'):.4f}")
                ])
            ], color="success" if friedman.get('significant') else "info"),
            
            # Rankings
            html.H5("Model Rankings", className="mt-4 mb-3"),
            html.P("Lower rank = better performance (1 is best)", className="text-muted"),
            create_rankings_bar_chart(experiments, friedman.get('rankings', []))
        ])


def create_contingency_table_display(table):
    """Display McNemar's contingency table as HTML table."""
    
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th(""),
            html.Th("Model 2 Correct"),
            html.Th("Model 2 Wrong")
        ])),
        html.Tbody([
            html.Tr([
                html.Td("Model 1 Correct"),
                html.Td(table[0][0], className="text-center"),
                html.Td(table[0][1], className="text-center font-weight-bold text-primary")
            ]),
            html.Tr([
                html.Td("Model 1 Wrong"),
                html.Td(table[1][0], className="text-center font-weight-bold text-warning"),
                html.Td(table[1][1], className="text-center")
            ])
        ])
    ], bordered=True, hover=True)


def create_rankings_bar_chart(experiments, rankings):
    """Bar chart showing Friedman test rankings."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"Exp {exp['id']}" for exp in experiments],
        y=rankings,
        text=[f"{rank:.2f}" for rank in rankings],
        textposition='auto',
        marker_color=['gold' if i == 0 else 'silver' if i == 1 else 'brown' 
                      for i in range(len(rankings))]
    ))
    
    fig.update_layout(
        title="Average Rank by Model (Lower is Better)",
        xaxis_title="Experiment",
        yaxis_title="Average Rank",
        yaxis_autorange='reversed',  # Lower ranks at top
        height=300
    )
    
    return dcc.Graph(figure=fig)


def create_configuration_tab(comparison_data):
    """
    Configuration tab: Show hyperparameters side-by-side.
    """
    
    experiments = comparison_data['experiments']
    
    # Extract all unique config keys across experiments
    all_keys = set()
    for exp in experiments:
        all_keys.update(exp['config'].keys())
    
    # Build comparison table
    table_rows = []
    for key in sorted(all_keys):
        values = [exp['config'].get(key, 'N/A') for exp in experiments]
        
        # Check if all values are same
        all_same = len(set(str(v) for v in values)) == 1
        
        row = html.Tr([
            html.Td(key.replace('_', ' ').title()),
            *[
                html.Td(
                    str(val),
                    className="" if all_same else "font-weight-bold text-warning"
                )
                for val in values
            ]
        ])
        table_rows.append(row)
    
    return html.Div([
        html.H4("Hyperparameter Comparison", className="mb-3"),
        html.P(
            "Highlighted parameters differ across experiments (may explain performance differences).",
            className="text-muted"
        ),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Parameter"),
                *[html.Th(f"Exp {exp['id']}") for exp in experiments]
            ])),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, responsive=True, striped=True)
    ])


def create_share_link_modal():
    """Modal for sharing comparison link."""
    return dbc.Modal([
        dbc.ModalHeader("Share Comparison"),
        dbc.ModalBody([
            html.P("Share this link with your team:"),
            dbc.InputGroup([
                dbc.Input(id='share-link-input', readonly=True),
                dbc.Button("Copy", id='copy-link-btn', color="primary")
            ]),
            html.Div(id='copy-confirmation', className="mt-2")
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id='close-share-modal', className="ms-auto")
        ])
    ], id='share-link-modal', is_open=False)


def create_save_comparison_modal():
    """Modal for saving comparison."""
    return dbc.Modal([
        dbc.ModalHeader("Save Comparison"),
        dbc.ModalBody([
            dbc.Label("Name (optional)"),
            dbc.Input(id='comparison-name-input', placeholder="e.g., ResNet vs Transformer"),
            dbc.Label("Notes (optional)", className="mt-3"),
            dbc.Textarea(id='comparison-notes-input', placeholder="Add any observations...")
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id='cancel-save-comparison', className="me-2"),
            dbc.Button("Save", id='confirm-save-comparison', color="primary")
        ])
    ], id='save-comparison-modal', is_open=False)
```

**Testing Criteria (Day 4):**
- ✅ Overview tab displays all 3 experiments with correct rankings
- ✅ Metrics tab shows per-class performance for all 11 fault classes
- ✅ Visualizations tab displays confusion matrices side-by-side
- ✅ Training curves overlay correctly on same axes
- ✅ Statistical tab shows McNemar's test for 2 experiments
- ✅ Configuration tab highlights differing hyperparameters

---

### Day 5-7: Callbacks & Interactivity

**Task 2.3:** Implement Callbacks
- **File:** `callbacks/comparison_callbacks.py`
- **Purpose:** Handle data loading, tab switching, export, sharing
- **Code:**

```python
from dash import callback, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
import json
from services.comparison_service import ComparisonService
from layouts.experiment_comparison import (
    create_overview_tab,
    create_metrics_tab,
    create_visualizations_tab,
    create_statistical_tab,
    create_configuration_tab
)

@callback(
    Output('comparison-data-store', 'children'),
    Input('url', 'search'),
    State('url', 'pathname')
)
def load_comparison_data(url_search, pathname):
    """
    Load comparison data when page loads.
    
    Triggered by: URL change (when user navigates to /compare?ids=...)
    """
    
    # Only run on comparison page
    if not pathname or not pathname.startswith('/compare'):
        raise PreventUpdate
    
    # Parse experiment IDs from URL
    # url_search = "?ids=1234,1567,1890"
    if not url_search or 'ids=' not in url_search:
        return json.dumps({'error': 'No experiment IDs provided'})
    
    ids_str = url_search.split('ids=')[1].split('&')[0]
    try:
        experiment_ids = [int(id_str.strip()) for id_str in ids_str.split(',')]
    except ValueError:
        return json.dumps({'error': 'Invalid experiment IDs'})
    
    # Get user ID from session (assume we have auth)
    user_id = get_current_user_id()  # From auth system
    
    # Validate request
    valid, error_msg = ComparisonService.validate_comparison_request(experiment_ids, user_id)
    if not valid:
        return json.dumps({'error': error_msg})
    
    # Load comparison data
    comparison_data = ComparisonService.get_comparison_data(experiment_ids)
    
    # Store as JSON in hidden div
    return json.dumps(comparison_data)


@callback(
    Output('comparison-tab-content', 'children'),
    Input('comparison-tabs', 'active_tab'),
    State('comparison-data-store', 'children')
)
def render_tab_content(active_tab, comparison_data_json):
    """
    Render content for active tab.
    
    Triggered by: Tab selection
    """
    
    if not comparison_data_json:
        return html.Div("Loading...", className="text-center text-muted")
    
    comparison_data = json.loads(comparison_data_json)
    
    if 'error' in comparison_data:
        return dbc.Alert(
            comparison_data['error'],
            color="danger",
            className="mt-4"
        )
    
    # Render appropriate tab
    if active_tab == 'overview':
        return create_overview_tab(comparison_data)
    elif active_tab == 'metrics':
        return create_metrics_tab(comparison_data)
    elif active_tab == 'visualizations':
        return create_visualizations_tab(comparison_data)
    elif active_tab == 'statistical':
        return create_statistical_tab(comparison_data)
    elif active_tab == 'configuration':
        return create_configuration_tab(comparison_data)
    else:
        return html.Div("Invalid tab", className="text-muted")


@callback(
    Output('share-link-modal', 'is_open'),
    Output('share-link-input', 'value'),
    Input('share-comparison-link', 'n_clicks'),
    Input('close-share-modal', 'n_clicks'),
    State('url', 'href'),
    prevent_initial_call=True
)
def toggle_share_modal(share_click, close_click, current_url):
    """
    Show/hide share link modal and populate with current URL.
    """
    
    trigger_id = ctx.triggered_id
    
    if trigger_id == 'share-comparison-link':
        # Open modal, show current URL
        return True, current_url
    elif trigger_id == 'close-share-modal':
        # Close modal
        return False, ""
    
    return no_update, no_update


@callback(
    Output('copy-confirmation', 'children'),
    Input('copy-link-btn', 'n_clicks'),
    State('share-link-input', 'value'),
    prevent_initial_call=True
)
def copy_link_to_clipboard(n_clicks, link):
    """
    Copy link to clipboard (using JavaScript).
    
    Note: Actual clipboard copy requires JavaScript callback.
    This callback just shows confirmation message.
    """
    
    if not n_clicks:
        raise PreventUpdate
    
    return dbc.Alert(
        "✓ Link copied to clipboard!",
        color="success",
        duration=3000,  # Auto-dismiss after 3 seconds
        dismissable=True
    )


@callback(
    Output('save-comparison-modal', 'is_open'),
    Output('save-comparison-confirmation', 'children'),
    Input('save-comparison', 'n_clicks'),
    Input('cancel-save-comparison', 'n_clicks'),
    Input('confirm-save-comparison', 'n_clicks'),
    State('comparison-name-input', 'value'),
    State('comparison-notes-input', 'value'),
    State('comparison-data-store', 'children'),
    prevent_initial_call=True
)
def handle_save_comparison(save_click, cancel_click, confirm_click,
                           name, notes, comparison_data_json):
    """
    Save comparison to database for later retrieval.
    """
    
    trigger_id = ctx.triggered_id
    
    if trigger_id == 'save-comparison':
        # Open modal
        return True, ""
    elif trigger_id == 'cancel-save-comparison':
        # Close modal without saving
        return False, ""
    elif trigger_id == 'confirm-save-comparison':
        # Save comparison
        comparison_data = json.loads(comparison_data_json)
        experiment_ids = [exp['id'] for exp in comparison_data['experiments']]
        user_id = get_current_user_id()
        
        # Insert into database
        from models.experiment_comparison import ExperimentComparison
        from database.connection import get_db_session
        
        session = get_db_session()
        new_comparison = ExperimentComparison(
            user_id=user_id,
            name=name or f"Comparison {', '.join(map(str, experiment_ids))}",
            experiment_ids=experiment_ids,
            notes=notes
        )
        session.add(new_comparison)
        session.commit()
        
        # Close modal, show success message
        return False, dbc.Alert(
            f"✓ Comparison saved as '{new_comparison.name}'",
            color="success",
            duration=5000,
            dismissable=True
        )
    
    return no_update, no_update


@callback(
    Output('export-pdf-download', 'data'),
    Input('export-comparison-pdf', 'n_clicks'),
    State('comparison-data-store', 'children'),
    prevent_initial_call=True
)
def export_comparison_pdf(n_clicks, comparison_data_json):
    """
    Generate and download PDF report of comparison.
    
    Uses: WeasyPrint or ReportLab to generate PDF
    """
    
    if not n_clicks:
        raise PreventUpdate
    
    comparison_data = json.loads(comparison_data_json)
    
    # Generate PDF (delegated to service)
    from services.export_service import generate_comparison_pdf
    pdf_bytes = generate_comparison_pdf(comparison_data)
    
    # Return as download
    import base64
    from dash import dcc
    
    return dcc.send_bytes(
        pdf_bytes,
        filename=f"comparison_{'-'.join([str(e['id']) for e in comparison_data['experiments']])}.pdf"
    )


# Add JavaScript callback for clipboard copy
# File: assets/clipboard.js
"""
// Copy to clipboard when button clicked
document.addEventListener('DOMContentLoaded', function() {
    const copyBtn = document.getElementById('copy-link-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            const input = document.getElementById('share-link-input');
            input.select();
            document.execCommand('copy');
        });
    }
});
"""
```

**Testing Criteria (Day 7):**
- ✅ Navigate to `/compare?ids=1,2` → Loads comparison data
- ✅ Switch tabs → Content updates without reload
- ✅ Click "Share Link" → Modal opens with correct URL
- ✅ Click "Copy" → URL copied to clipboard
- ✅ Click "Save Comparison" → Saves to database, shows confirmation
- ✅ Click "Export PDF" → Downloads PDF report

---

### Day 8-9: Integration with Experiment History Page

**Task 2.4:** Add "Compare" Functionality to Experiment History
- **File:** `layouts/experiment_history.py` (enhance existing)
- **Add:**
  1. Checkboxes for each experiment row
  2. "Compare Selected" button (visible when 2-3 checked)
  3. Action handler to navigate to comparison page

**Code Addition:**

```python
# In experiment_history.py

# Add to table header
html.Tr([
    html.Th(dbc.Checkbox(id='select-all-experiments')),  # NEW
    html.Th("Date"),
    html.Th("Name"),
    # ... existing columns
])

# Add to each table row
html.Tr([
    html.Td(dbc.Checkbox(id={'type': 'exp-checkbox', 'index': exp.id})),  # NEW
    html.Td(exp.created_at.strftime('%Y-%m-%d')),
    # ... existing cells
])

# Add floating action button
html.Div([
    dbc.Button(
        "Compare Selected",
        id='compare-selected-btn',
        color="primary",
        size="lg",
        className="shadow",
        disabled=True  # Enabled when 2-3 selected
    )
], id='floating-compare-btn', className="position-fixed", 
   style={'bottom': '20px', 'right': '20px', 'display': 'none'})
```

**Callback:**

```python
# In callbacks/experiment_history_callbacks.py

@callback(
    Output('compare-selected-btn', 'disabled'),
    Output('floating-compare-btn', 'style'),
    Input({'type': 'exp-checkbox', 'index': ALL}, 'checked')
)
def update_compare_button_state(checked_states):
    """
    Enable compare button when 2-3 experiments selected.
    """
    
    num_selected = sum(1 for checked in checked_states if checked)
    
    if 2 <= num_selected <= 3:
        # Enable button, show floating div
        return False, {'bottom': '20px', 'right': '20px', 'display': 'block'}
    else:
        # Disable button, hide floating div
        return True, {'display': 'none'}


@callback(
    Output('url', 'pathname'),
    Output('url', 'search'),
    Input('compare-selected-btn', 'n_clicks'),
    State({'type': 'exp-checkbox', 'index': ALL}, 'checked'),
    State({'type': 'exp-checkbox', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def navigate_to_comparison(n_clicks, checked_states, checkbox_ids):
    """
    Navigate to comparison page with selected experiment IDs.
    """
    
    if not n_clicks:
        raise PreventUpdate
    
    # Get IDs of checked experiments
    selected_ids = [
        checkbox_id['index']
        for checkbox_id, checked in zip(checkbox_ids, checked_states)
        if checked
    ]
    
    if not (2 <= len(selected_ids) <= 3):
        # Invalid selection (shouldn't happen if button logic is correct)
        raise PreventUpdate
    
    # Navigate to comparison page
    ids_param = ','.join(map(str, selected_ids))
    return '/compare', f'?ids={ids_param}'
```

**Testing Criteria (Day 9):**
- ✅ Check 1 experiment → Compare button disabled/hidden
- ✅ Check 2 experiments → Compare button enabled
- ✅ Check 3 experiments → Compare button enabled
- ✅ Check 4 experiments → Compare button disabled
- ✅ Click "Compare Selected" → Navigates to `/compare?ids=1,2`
- ✅ "Select All" checkbox toggles all experiment checkboxes

---

### Day 10: Final Testing & Documentation

**Task 2.5:** End-to-End Testing

**Test Scenarios:**

```
SCENARIO 1: Compare 2 Experiments
1. Navigate to /experiments
2. Check experiment #1234 and #1567
3. Click "Compare Selected"
4. ✅ Loads /compare?ids=1234,1567
5. ✅ Overview tab shows 2 experiments ranked
6. ✅ Statistical tab shows McNemar's test
7. ✅ Visualizations show 2 confusion matrices side-by-side
8. Click "Share Link"
9. ✅ Modal shows URL
10. Click "Copy"
11. ✅ URL copied to clipboard
12. Open URL in new tab
13. ✅ Same comparison loads

SCENARIO 2: Compare 3 Experiments
1. Navigate to /experiments
2. Check experiments #1234, #1567, #1890
3. Click "Compare Selected"
4. ✅ Loads /compare?ids=1234,1567,1890
5. ✅ Overview shows 3 experiments
6. ✅ Statistical tab shows Friedman test with rankings
7. ✅ Visualizations show 3 confusion matrices in row

SCENARIO 3: Invalid Comparison
1. Navigate directly to /compare?ids=9999,8888
2. ✅ Shows error: "Experiments not found"
3. Navigate to /compare?ids=1234
4. ✅ Shows error: "At least 2 experiments required"
5. Navigate to /compare?ids=1234,1567,1890,2000
6. ✅ Shows error: "Maximum 3 experiments can be compared"

SCENARIO 4: Export PDF
1. Load valid comparison
2. Click "Export PDF"
3. ✅ PDF downloads with filename "comparison_1234-1567.pdf"
4. Open PDF
5. ✅ Contains: Overview metrics, confusion matrices, statistical test results

SCENARIO 5: Save Comparison
1. Load valid comparison
2. Click "Save Comparison"
3. ✅ Modal opens
4. Enter name: "ResNet vs Transformer"
5. Enter notes: "Transformer slower but better on oil whirl"
6. Click "Save"
7. ✅ Modal closes, success message shown
8. Navigate to /experiments/comparisons
9. ✅ Saved comparison appears in list
10. Click saved comparison
11. ✅ Loads /compare?ids=1234,1567 (same experiments)

SCENARIO 6: Per-Class Analysis
1. Load comparison
2. Click "Metrics" tab
3. ✅ Shows tabs for all 11 fault classes
4. Click "Oil Whirl" tab
5. ✅ Shows bar chart comparing precision/recall/F1 for oil whirl
6. ✅ Identifies which experiment is best for oil whirl

SCENARIO 7: Configuration Comparison
1. Load comparison of experiments with different learning rates
2. Click "Configuration" tab
3. ✅ Table highlights "learning_rate" row (different values)
4. ✅ Other parameters with same values not highlighted

SCENARIO 8: Direct URL Access
1. Share URL /compare?ids=1234,1567 with colleague
2. Colleague opens URL
3. ✅ If authorized: Loads comparison
4. ✅ If not authorized: Shows 403 Unauthorized error
```

**Task 2.6:** Create Documentation
- **File:** `docs/user_guides/experiment_comparison.md`
- **Sections:**
  - How to select experiments for comparison
  - Understanding statistical test results
  - Interpreting per-class performance differences
  - Sharing comparisons with team
  - Exporting comparison reports

---

## 2.4 DO'S AND DON'TS

### ✅ DO's

1. **DO validate experiment ownership**
   - Users can only compare their own experiments
   - Check authorization in service layer, not just UI

2. **DO handle missing data gracefully**
   - If confusion matrix file missing, show placeholder
   - If training history incomplete, show partial chart

3. **DO use consistent sorting**
   - Always sort experiments by accuracy (highest first)
   - Makes "winner" immediately obvious

4. **DO show statistical context**
   - Not just "96.8% vs 96.5%" but "Is this difference significant?"
   - McNemar's test answers this question

5. **DO highlight practical differences**
   - Auto-identify: "Exp 1234 excels at Oil Whirl detection"
   - Users care about per-class performance

6. **DO make sharing easy**
   - Copy URL button (one click)
   - PDF export for stakeholder presentations

7. **DO use color consistently**
   - Green = best/winner
   - Yellow = middle
   - Red = worst
   - Applies to rankings, heatmaps, etc.

8. **DO show training efficiency**
   - Include duration in comparison
   - "Exp 1567 is 0.3% more accurate but takes 8 minutes longer"

9. **DO cache comparison data**
   - Store in hidden div (avoids re-loading on tab switch)
   - Improves responsiveness

10. **DO provide context in statistical tests**
    - Explain what p-value means
    - "p < 0.05 = significant difference"

### ❌ DON'Ts

1. **DON'T allow comparing >3 experiments**
   - UI becomes cluttered
   - Suggest HPO campaign analysis instead

2. **DON'T compare experiments from different datasets**
   - Invalid comparison (different test sets)
   - Validate in service layer

3. **DON'T show raw confusion matrices**
   - Too large (11×11) for side-by-side
   - Use heatmaps with truncated labels

4. **DON'T forget mobile responsiveness**
   - Comparison page should work on tablets
   - Stack experiments vertically on small screens

5. **DON'T hardcode experiment IDs**
   - Parse from URL dynamically
   - Allows bookmarking/sharing

6. **DON'T reload data on tab switch**
   - Load once, store in hidden div
   - Tab callbacks just format differently

7. **DON'T skip error handling**
   - Invalid IDs, unauthorized access, missing files
   - Show user-friendly error messages

8. **DON'T make assumptions about test set**
   - Verify all experiments used SAME test set
   - Otherwise, McNemar's test is invalid

9. **DON'T forget to sort tables**
   - Per-class metrics: Sort by F1-score (descending)
   - Configuration: Sort alphabetically by parameter name

10. **DON'T overcomplicate statistical tests**
    - Show interpretation, not just numbers
    - "Model A is significantly better" > "p=0.032"

---

## 2.5 TESTING CHECKLIST

### Unit Tests (`tests/test_comparison_service.py`)

```python
def test_validate_comparison_request_requires_2_experiments():
    """Should reject comparison with <2 experiments"""
    valid, error = ComparisonService.validate_comparison_request([1], user_id=1)
    assert valid == False
    assert "at least 2" in error.lower()

def test_validate_comparison_request_max_3_experiments():
    """Should reject comparison with >3 experiments"""
    valid, error = ComparisonService.validate_comparison_request([1,2,3,4], user_id=1)
    assert valid == False
    assert "maximum 3" in error.lower()

def test_get_comparison_data_returns_correct_structure():
    """Should return dict with 'experiments' and 'statistical_tests' keys"""
    data = ComparisonService.get_comparison_data([1, 2])
    assert 'experiments' in data
    assert 'statistical_tests' in data
    assert len(data['experiments']) == 2

def test_mcnemar_test_runs_for_2_experiments():
    """McNemar's test should be present for 2 experiments"""
    data = ComparisonService.get_comparison_data([1, 2])
    assert 'mcnemar' in data['statistical_tests']
    assert 'p_value' in data['statistical_tests']['mcnemar']

def test_friedman_test_runs_for_3_experiments():
    """Friedman test should be present for 3 experiments"""
    data = ComparisonService.get_comparison_data([1, 2, 3])
    assert 'friedman' in data['statistical_tests']
    assert 'rankings' in data['statistical_tests']['friedman']
```

### Integration Tests (`tests/integration/test_comparison_page.py`)

```python
def test_comparison_page_loads(test_client):
    """Comparison page should load with valid IDs"""
    response = test_client.get('/compare?ids=1,2')
    assert response.status_code == 200
    assert 'Experiment Comparison' in response.text

def test_comparison_page_rejects_invalid_ids(test_client):
    """Should show error for non-existent IDs"""
    response = test_client.get('/compare?ids=9999,8888')
    assert 'not found' in response.text.lower()

def test_share_link_contains_ids(test_client):
    """Share link should contain experiment IDs"""
    # Simulate clicking "Share Link" button
    # ... (Dash callback testing requires dash.testing)
```

### Manual QA Checklist

- [ ] Compare 2 experiments → McNemar's test shown
- [ ] Compare 3 experiments → Friedman test shown
- [ ] Comparison page loads in <2 seconds
- [ ] Confusion matrices display correctly side-by-side
- [ ] Training curves overlay on same axes
- [ ] Per-class metrics show all 11 fault types
- [ ] Statistical test interpretation is clear
- [ ] Configuration tab highlights differing parameters
- [ ] Share link copies to clipboard
- [ ] PDF export downloads successfully
- [ ] PDF contains all comparison sections
- [ ] Save comparison stores to database
- [ ] Saved comparison can be re-loaded
- [ ] Experiment history "Compare Selected" button works
- [ ] Selecting 1 or 4+ experiments disables button
- [ ] Mobile/tablet: Layout adapts (stacks vertically)
- [ ] Error messages are user-friendly
- [ ] Unauthorized access returns 403

---

## 2.6 SUCCESS METRICS

### Quantitative
- Comparison page loads in <2 seconds (3 experiments)
- Statistical tests compute in <1 second
- PDF export generates in <5 seconds
- 100% of experiments have predictions.npy (required for tests)
- Zero "NaN" values in metrics display

### Qualitative
- Users can identify "winner" in <10 seconds
- Statistical test interpretation understandable without ML knowledge
- Comparison URL is shareable (works for authorized users)
- PDF report is presentation-ready (no manual formatting needed)

---

## 2.7 ROLLOUT PLAN

### Phase 1: Soft Launch (Day 1-2 of Week 3)
- Deploy to staging environment
- Internal testing with 5 power users
- Collect feedback on UI clarity

### Phase 2: Beta (Day 3-4 of Week 3)
- Deploy to production (feature flag: 20% of users)
- Monitor for errors (Sentry)
- A/B test: Do users find comparisons faster than Excel method?

### Phase 3: General Availability (Day 5 of Week 3)
- Enable for all users
- Announce in team meeting
- Create 3-minute video tutorial
- Update documentation

### Rollback Plan
If critical issues:
1. Disable "Compare Selected" button
2. Hide comparison routes (return 404)
3. Fix issues in dev
4. Redeploy next week

---

**END OF FEATURE #2 PLAN**

---

This completes the detailed implementation plan for **Feature #2: Experiment Comparison**. The plan includes:
- ✅ Clear objectives and success criteria
- ✅ Complete technical specifications (database, service layer, UI)
- ✅ Day-by-day implementation tasks (10 days)
- ✅ Comprehensive Do's and Don'ts (20 rules)
- ✅ Extensive testing checklist (unit, integration, manual QA)
- ✅ Success metrics and rollout plan

**Ready for your development team to execute with zero ambiguity.**