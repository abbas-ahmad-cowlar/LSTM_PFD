"""
Time-Domain vs. Frequency-Domain Model Comparison

Systematic comparison of models operating on different representations:
- Time-domain: 1D CNNs on raw signals
- Frequency-domain: 2D CNNs on spectrograms
- Dual-stream: Combined time + frequency processing

Analyzes:
- Overall accuracy comparison
- Per-fault performance differences
- Which faults benefit from frequency-domain learning
- Computational efficiency trade-offs

Usage:
    from evaluation.time_vs_frequency_comparison import compare_time_vs_frequency

    results = compare_time_vs_frequency(
        models={
            'CNN-1D': cnn_model,
            'ResNet-1D': resnet1d_model,
            'ResNet-2D (STFT)': resnet2d_model,
            'Dual-Stream': dual_stream_model
        },
        test_loader_time=time_loader,
        test_loader_freq=freq_loader,
        class_names=fault_types
    )
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm


class TimeVsFrequencyComparator:
    """
    Compare time-domain and frequency-domain models.

    Args:
        class_names: List of fault class names
        device: Device to run evaluation on (default: 'cuda')
    """

    def __init__(
        self,
        class_names: List[str],
        device: str = 'cuda'
    ):
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str
    ) -> Dict:
        """
        Evaluate a single model.

        Args:
            model: Model to evaluate
            dataloader: Test data loader
            model_name: Name of the model

        Returns:
            Dictionary of evaluation results
        """
        model.eval()
        model.to(self.device)

        all_predictions = []
        all_targets = []
        all_probs = []
        inference_times = []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                inference_time = time.time() - start_time

                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_predictions.append(predicted.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                inference_times.append(inference_time / inputs.size(0))  # Per sample

        # Concatenate results
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        probs = np.concatenate(all_probs)

        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix
        )

        accuracy = accuracy_score(targets, predictions) * 100
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        cm = confusion_matrix(targets, predictions)

        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            }

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probs,
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times)
        }

        return results

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict]:
        """
        Compare multiple models.

        Args:
            models: Dictionary of {model_name: model}
            dataloaders: Dictionary of {model_name: dataloader}

        Returns:
            Dictionary of results per model
        """
        all_results = {}

        for model_name, model in models.items():
            dataloader = dataloaders.get(model_name, list(dataloaders.values())[0])
            results = self.evaluate_model(model, dataloader, model_name)
            all_results[model_name] = results

        return all_results

    def create_comparison_table(
        self,
        results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Create comparison table of all models.

        Args:
            results: Dictionary of results from compare_models

        Returns:
            DataFrame with comparison metrics
        """
        rows = []

        for model_name, result in results.items():
            per_class = result['per_class_metrics']

            # Compute macro-averaged metrics
            f1_scores = [m['f1_score'] for m in per_class.values()]
            precisions = [m['precision'] for m in per_class.values()]
            recalls = [m['recall'] for m in per_class.values()]

            rows.append({
                'Model': model_name,
                'Accuracy (%)': result['accuracy'],
                'Macro F1': np.mean(f1_scores),
                'Macro Precision': np.mean(precisions),
                'Macro Recall': np.mean(recalls),
                'Inference Time (ms)': result['avg_inference_time'] * 1000
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('Accuracy (%)', ascending=False)

        return df

    def analyze_per_fault_performance(
        self,
        results: Dict[str, Dict],
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Analyze which faults benefit from frequency-domain models.

        Args:
            results: Dictionary of results from compare_models
            save_path: Optional path to save plot

        Returns:
            DataFrame with per-fault comparison
        """
        # Extract per-class F1 scores for each model
        data = []

        for fault in self.class_names:
            row = {'Fault': fault}
            for model_name, result in results.items():
                f1_score = result['per_class_metrics'][fault]['f1_score']
                row[model_name] = f1_score
            data.append(row)

        df = pd.DataFrame(data)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(self.class_names))
        width = 0.8 / len(results)  # Width of bars

        for i, model_name in enumerate(results.keys()):
            offset = (i - len(results) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                df[model_name],
                width,
                label=model_name
            )

        ax.set_xlabel('Fault Type', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('Per-Fault Performance Comparison', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved per-fault comparison to {save_path}")
        else:
            plt.show()

        plt.close()

        return df

    def identify_frequency_sensitive_faults(
        self,
        results: Dict[str, Dict],
        time_models: List[str],
        freq_models: List[str],
        threshold: float = 0.02
    ) -> Dict[str, Dict]:
        """
        Identify which faults benefit most from frequency-domain processing.

        Args:
            results: Dictionary of results from compare_models
            time_models: List of time-domain model names
            freq_models: List of frequency-domain model names
            threshold: Minimum improvement threshold (default: 2%)

        Returns:
            Dictionary of fault improvements
        """
        improvements = {}

        for fault in self.class_names:
            # Average F1 for time-domain models
            time_f1 = np.mean([
                results[m]['per_class_metrics'][fault]['f1_score']
                for m in time_models if m in results
            ])

            # Average F1 for frequency-domain models
            freq_f1 = np.mean([
                results[m]['per_class_metrics'][fault]['f1_score']
                for m in freq_models if m in results
            ])

            improvement = freq_f1 - time_f1

            if improvement > threshold:
                improvements[fault] = {
                    'time_f1': time_f1,
                    'freq_f1': freq_f1,
                    'improvement': improvement,
                    'relative_improvement': (improvement / time_f1) * 100
                }

        return improvements

    def plot_confusion_matrices(
        self,
        results: Dict[str, Dict],
        save_dir: Optional[Path] = None
    ):
        """
        Plot confusion matrices for all models side-by-side.

        Args:
            results: Dictionary of results from compare_models
            save_dir: Optional directory to save plots
        """
        num_models = len(results)
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))

        if num_models == 1:
            axes = [axes]

        for ax, (model_name, result) in zip(axes, results.items()):
            cm = result['confusion_matrix']
            accuracy = result['accuracy']

            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

            # Plot
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
                cbar_kws={'label': 'Proportion'}
            )

            ax.set_title(f'{model_name}\nAccuracy: {accuracy:.2f}%', fontsize=12, weight='bold')
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)

        plt.tight_layout()

        if save_dir:
            save_path = save_dir / 'confusion_matrices_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrices to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_comparison_report(
        self,
        results: Dict[str, Dict],
        output_dir: Path
    ):
        """
        Generate comprehensive comparison report.

        Args:
            results: Dictionary of results from compare_models
            output_dir: Directory to save report and figures
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("TIME-DOMAIN VS. FREQUENCY-DOMAIN MODEL COMPARISON REPORT")
        print("="*80)

        # Overall comparison table
        print("\n1. OVERALL PERFORMANCE COMPARISON")
        print("-" * 80)
        comparison_table = self.create_comparison_table(results)
        print(comparison_table.to_string(index=False))
        comparison_table.to_csv(output_dir / 'overall_comparison.csv', index=False)

        # Per-fault analysis
        print("\n2. PER-FAULT PERFORMANCE ANALYSIS")
        print("-" * 80)
        per_fault_df = self.analyze_per_fault_performance(
            results,
            save_path=output_dir / 'per_fault_comparison.png'
        )
        per_fault_df.to_csv(output_dir / 'per_fault_metrics.csv', index=False)

        # Identify frequency-sensitive faults
        time_models = [m for m in results.keys() if '1D' in m or 'CNN' in m]
        freq_models = [m for m in results.keys() if '2D' in m or 'STFT' in m or 'CWT' in m]

        if time_models and freq_models:
            print("\n3. FREQUENCY-SENSITIVE FAULTS")
            print("-" * 80)
            improvements = self.identify_frequency_sensitive_faults(
                results, time_models, freq_models
            )

            if improvements:
                for fault, metrics in improvements.items():
                    print(f"{fault}:")
                    print(f"  Time-domain F1: {metrics['time_f1']:.4f}")
                    print(f"  Freq-domain F1: {metrics['freq_f1']:.4f}")
                    print(f"  Improvement: +{metrics['improvement']:.4f} ({metrics['relative_improvement']:.1f}%)")
                    print()
            else:
                print("No significant improvements found for any fault type.")

        # Confusion matrices
        self.plot_confusion_matrices(results, save_dir=output_dir)

        print("\n" + "="*80)
        print(f"Report saved to {output_dir}")
        print("="*80)


def compare_time_vs_frequency(
    models: Dict[str, nn.Module],
    dataloaders: Dict[str, DataLoader],
    class_names: List[str],
    output_dir: Optional[Path] = None,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    Convenience function for time vs. frequency comparison.

    Args:
        models: Dictionary of {model_name: model}
        dataloaders: Dictionary of {model_name: dataloader}
        class_names: List of fault class names
        output_dir: Optional directory to save report
        device: Device to run evaluation on

    Returns:
        Dictionary of results per model
    """
    comparator = TimeVsFrequencyComparator(class_names, device)

    # Compare models
    results = comparator.compare_models(models, dataloaders)

    # Generate report if output directory provided
    if output_dir:
        comparator.generate_comparison_report(results, output_dir)

    return results


if __name__ == '__main__':
    # Test comparison
    from models.cnn.cnn_1d import CNN1D
    from models.spectrogram_cnn import resnet18_2d
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

    class_names = [
        'Normal', 'Ball Fault', 'Inner Race', 'Outer Race',
        'Combined', 'Imbalance', 'Misalignment', 'Oil Whirl',
        'Cavitation', 'Looseness', 'Oil Deficiency'
    ]

    # Create dummy models
    cnn_1d = CNN1D(num_classes=NUM_CLASSES)
    resnet_2d = resnet18_2d(num_classes=NUM_CLASSES)

    models = {
        'CNN-1D (Time)': cnn_1d,
        'ResNet-2D (Frequency)': resnet_2d
    }

    print("Time vs. Frequency Comparison Framework Ready!")
    print(f"Models to compare: {list(models.keys())}")
