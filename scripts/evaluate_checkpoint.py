"""
Evaluate a trained checkpoint on a dataset split and write a results artifact.

Convergence Plan P1.8 / Phase 4 building block: every evaluation produces
a results directory containing metrics.json (with provenance: checkpoint,
data, git SHA) and a confusion-matrix PNG.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint checkpoints/cnn/best_model.pth \
        --data data/generated/dataset.h5 \
        --split test \
        --output results/cnn1d_v1_baseline
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BearingFaultDataset  # noqa: E402
from packages.core.evaluation.evaluator import ModelEvaluator  # noqa: E402
from packages.core.models.model_factory import create_model  # noqa: E402
from utils.constants import FAULT_TYPES, FAULT_TYPE_DISPLAY_NAMES  # noqa: E402


def build_model_from_checkpoint(ckpt: dict) -> torch.nn.Module:
    """Reconstruct the model from the checkpoint's embedded model_config."""
    config = ckpt.get('model_config')
    if not config:
        raise ValueError("Checkpoint has no model_config; cannot reconstruct model")

    model_type = config['model_type']
    factory_keys = {
        'CNN1D': 'cnn1d',
        'ResNet1D': 'resnet18',
    }
    key = factory_keys.get(model_type)
    if key is None:
        raise ValueError(f"No factory mapping for model_type '{model_type}'")

    model = create_model(key, num_classes=config['num_classes'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def git_sha() -> str:
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            cwd=PROJECT_ROOT, check=True,
        ).stdout.strip()
    except Exception:
        return 'unknown'


def save_confusion_matrix_png(cm: np.ndarray, class_names: list, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    threshold = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > threshold else 'black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model_from_checkpoint(ckpt)

    print(f"Loading {args.split} split from {args.data}")
    dataset = BearingFaultDataset.from_hdf5(Path(args.data), split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    display_names = [FAULT_TYPE_DISPLAY_NAMES[ft] for ft in FAULT_TYPES]
    evaluator = ModelEvaluator(model, device=args.device)
    results = evaluator.evaluate(loader, class_names=display_names)

    cm = np.asarray(results['confusion_matrix'])
    artifact = {
        'accuracy': float(results['accuracy']),
        'per_class_metrics': {
            cls: {k: float(v) for k, v in m.items()}
            for cls, m in results['per_class_metrics'].items()
        },
        'confusion_matrix': cm.tolist(),
        'class_names': display_names,
        'num_samples': int(len(dataset)),
        'provenance': {
            'checkpoint': str(args.checkpoint),
            'checkpoint_epoch': ckpt.get('epoch'),
            'checkpoint_best_val_acc': ckpt.get('best_val_acc'),
            'data': str(args.data),
            'split': args.split,
            'git_sha': git_sha(),
            'evaluated_at': datetime.now(timezone.utc).isoformat(),
            'device': args.device,
        },
    }

    metrics_path = output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(artifact, indent=2))
    save_confusion_matrix_png(cm, display_names, output_dir / 'confusion_matrix.png')

    print(f"\nTest accuracy: {artifact['accuracy']:.2f}% on {artifact['num_samples']} samples")
    print(f"Artifacts written to {output_dir}")


if __name__ == '__main__':
    main()
