"""
Dataset Comparison Engine — compare training results across dataset versions.

Provides reusable comparison logic for CLI scripts and dashboard.

Usage:
    >>> from packages.core.evaluation.dataset_comparison import DatasetComparisonEngine
    >>> engine = DatasetComparisonEngine()
    >>> v1 = engine.load_results(Path('results/v1_basic/'))
    >>> v2 = engine.load_results(Path('results/v2_advanced/'))
    >>> report = engine.compare(v1, v2)
    >>> engine.print_comparison(report)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ModelResult:
    """Result for a single model."""
    model: str
    test_accuracy: float = 0.0
    best_val_accuracy: float = 0.0
    best_val_loss: float = float('inf')
    training_time_s: float = 0.0
    epochs_trained: int = 0
    params: int = 0
    checkpoint: str = ''
    error: Optional[str] = None


@dataclass
class DatasetResults:
    """Results from training on a single dataset version."""
    version_tag: str
    config_hash: str = ''
    advanced_physics: List[str] = field(default_factory=list)
    models: Dict[str, ModelResult] = field(default_factory=dict)
    total_training_time_s: float = 0.0
    timestamp: str = ''


@dataclass
class ComparisonReport:
    """Comparison report between two dataset versions."""
    v1_tag: str
    v2_tag: str
    per_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DatasetComparisonEngine:
    """Compare training results across different dataset versions."""

    def load_results(self, results_dir: Path) -> DatasetResults:
        """
        Load all training results from a results directory.

        Expects either:
        - Individual model results: {model_key}_results.json
        - Batch summaries: batch_*.json
        - A dataset_config.json for version info

        Args:
            results_dir: Path to results directory

        Returns:
            DatasetResults object with all model outcomes
        """
        results_dir = Path(results_dir)
        dataset_results = DatasetResults(
            version_tag=results_dir.name,
            timestamp=datetime.now().isoformat(),
        )

        # Try to load config info
        config_path = results_dir.parent / 'data' / 'generated' / 'dataset_config.json'
        if not config_path.exists():
            # Look in the results dir itself
            config_path = results_dir / 'dataset_config.json'
        if config_path.exists():
            with open(config_path) as f:
                config_info = json.load(f)
            dataset_results.config_hash = config_info.get('config_hash', '')
            dataset_results.advanced_physics = config_info.get(
                'advanced_physics_enabled', []
            )
            dataset_results.version_tag = config_info.get(
                'version_tag', dataset_results.version_tag
            )

        # Load batch results
        for batch_file in sorted(results_dir.glob('batch_*.json')):
            with open(batch_file) as f:
                batch = json.load(f)
            dataset_results.total_training_time_s += batch.get('total_time_s', 0)
            for r in batch.get('results', []):
                model_result = ModelResult(
                    model=r['model'],
                    test_accuracy=r.get('test_accuracy', 0),
                    best_val_accuracy=r.get('best_val_accuracy', 0),
                    best_val_loss=r.get('best_val_loss', float('inf')),
                    training_time_s=r.get('training_time_s', 0),
                    epochs_trained=r.get('epochs_trained', 0),
                    params=r.get('params', 0),
                    checkpoint=r.get('checkpoint', ''),
                    error=r.get('error'),
                )
                dataset_results.models[r['model']] = model_result

        # Load individual model results
        for result_file in sorted(results_dir.glob('*_results.json')):
            if result_file.name.startswith('batch_'):
                continue
            with open(result_file) as f:
                r = json.load(f)
            if r.get('model') and r['model'] not in dataset_results.models:
                model_result = ModelResult(
                    model=r['model'],
                    test_accuracy=r.get('test_accuracy', 0),
                    best_val_accuracy=r.get('best_val_accuracy', 0),
                    best_val_loss=r.get('best_val_loss', float('inf')),
                    training_time_s=r.get('training_time_s', 0),
                    epochs_trained=r.get('epochs_trained', 0),
                    params=r.get('params', 0),
                    checkpoint=r.get('checkpoint', ''),
                    error=r.get('error'),
                )
                dataset_results.models[r['model']] = model_result

        return dataset_results

    def compare(
        self,
        v1: DatasetResults,
        v2: DatasetResults,
    ) -> ComparisonReport:
        """
        Compare results from two dataset versions.

        Args:
            v1: Results from dataset version 1
            v2: Results from dataset version 2

        Returns:
            ComparisonReport with per-model deltas and summary
        """
        report = ComparisonReport(v1_tag=v1.version_tag, v2_tag=v2.version_tag)

        # Find common models
        common_models = set(v1.models.keys()) & set(v2.models.keys())
        v1_only = set(v1.models.keys()) - set(v2.models.keys())
        v2_only = set(v2.models.keys()) - set(v1.models.keys())

        total_improved = 0
        total_degraded = 0
        total_deltas = []

        for model_key in sorted(common_models):
            m1 = v1.models[model_key]
            m2 = v2.models[model_key]

            # Skip errored models
            if m1.error or m2.error:
                report.per_model[model_key] = {
                    'v1_error': m1.error,
                    'v2_error': m2.error,
                    'status': 'error',
                }
                continue

            delta = m2.test_accuracy - m1.test_accuracy
            total_deltas.append(delta)
            if delta > 0.001:
                total_improved += 1
            elif delta < -0.001:
                total_degraded += 1

            report.per_model[model_key] = {
                'v1_acc': m1.test_accuracy,
                'v2_acc': m2.test_accuracy,
                'delta': delta,
                'delta_pct': (delta / max(m1.test_accuracy, 1e-10)) * 100,
                'v1_time': m1.training_time_s,
                'v2_time': m2.training_time_s,
                'params': m1.params,
                'status': 'improved' if delta > 0.001 else (
                    'degraded' if delta < -0.001 else 'unchanged'
                ),
            }

        avg_delta = sum(total_deltas) / len(total_deltas) if total_deltas else 0

        report.summary = {
            'total_models_compared': len(common_models),
            'improved': total_improved,
            'degraded': total_degraded,
            'unchanged': len(common_models) - total_improved - total_degraded,
            'avg_accuracy_delta': avg_delta,
            'v1_only_models': sorted(v1_only),
            'v2_only_models': sorted(v2_only),
            'v1_physics': v1.advanced_physics,
            'v2_physics': v2.advanced_physics,
            'v1_config_hash': v1.config_hash,
            'v2_config_hash': v2.config_hash,
        }

        return report

    def generate_leaderboard(
        self,
        results: DatasetResults,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate a ranked leaderboard from dataset results.

        Args:
            results: Training results to rank
            top_n: Number of top models to return

        Returns:
            Sorted list of model results (best accuracy first)
        """
        ranked = []
        for model_key, m in results.models.items():
            if m.error:
                continue
            ranked.append({
                'rank': 0,
                'model': m.model,
                'test_accuracy': m.test_accuracy,
                'val_accuracy': m.best_val_accuracy,
                'params': m.params,
                'epochs': m.epochs_trained,
                'time_s': m.training_time_s,
            })

        ranked.sort(key=lambda x: x['test_accuracy'], reverse=True)
        for i, entry in enumerate(ranked, 1):
            entry['rank'] = i

        return ranked[:top_n] if top_n else ranked

    def print_comparison(self, report: ComparisonReport) -> str:
        """Format comparison report as a printable string."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"DATASET COMPARISON: {report.v1_tag} vs {report.v2_tag}")
        lines.append("=" * 80)

        s = report.summary
        lines.append(f"Models compared: {s['total_models_compared']}")
        lines.append(f"  Improved:  {s['improved']}")
        lines.append(f"  Degraded:  {s['degraded']}")
        lines.append(f"  Unchanged: {s['unchanged']}")
        lines.append(f"  Avg delta: {s['avg_accuracy_delta']:+.4f}")

        if s.get('v1_physics') or s.get('v2_physics'):
            lines.append(f"\nPhysics V1: {s.get('v1_physics', 'none')}")
            lines.append(f"Physics V2: {s.get('v2_physics', 'none')}")

        lines.append(f"\n{'Model':<30} {report.v1_tag:>10} {report.v2_tag:>10} {'Delta':>10} {'Status':>10}")
        lines.append("-" * 70)

        for model_key in sorted(report.per_model.keys()):
            m = report.per_model[model_key]
            if m.get('status') == 'error':
                lines.append(f"  {model_key:<28} {'ERROR':>10}")
                continue
            status_icon = {
                'improved': '[+]',
                'degraded': '[-]',
                'unchanged': '[=]',
            }.get(m['status'], '[ ]')
            lines.append(
                f"  {model_key:<28} {m['v1_acc']:>10.4f} {m['v2_acc']:>10.4f} "
                f"{m['delta']:>+10.4f} {status_icon:>10}"
            )

        output = '\n'.join(lines)
        return output

    def save_report(self, report: ComparisonReport, output_path: Path):
        """Save comparison report as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
