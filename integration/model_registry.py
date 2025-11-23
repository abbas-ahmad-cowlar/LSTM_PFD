"""
Model Registry for Tracking All Trained Models

Central database of all models with metadata for comparison and selection.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

import json
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all trained models with metadata.

    Stores:
    - Model metadata (name, phase, accuracy, hyperparameters)
    - File paths (PyTorch .pth, ONNX .onnx)
    - Performance metrics (accuracy, latency, size)
    - Training information (date, duration, dataset)

    Example:
        >>> registry = ModelRegistry('models/registry.db')
        >>> registry.register_model(
        ...     model_name='ResNet34_1D',
        ...     phase='Phase 3',
        ...     accuracy=0.967,
        ...     model_path='checkpoints/resnet34.pth'
        ... )
        >>> best_model = registry.get_best_model(metric='accuracy')
    """

    def __init__(self, db_path: str = 'models/model_registry.db'):
        """
        Initialize model registry.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                phase TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                training_date TEXT,
                hyperparameters TEXT,
                model_path TEXT,
                onnx_path TEXT,
                size_mb REAL,
                inference_latency_ms REAL,
                num_parameters INTEGER,
                dataset_name TEXT,
                notes TEXT
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"✓ Registry initialized: {self.db_path}")

    def register_model(
        self,
        model_name: str,
        phase: str,
        accuracy: float,
        model_path: str,
        **kwargs
    ) -> int:
        """
        Register a new model in the registry.

        Args:
            model_name: Model identifier (e.g., 'ResNet34_1D')
            phase: Phase name (e.g., 'Phase 3')
            accuracy: Test accuracy (0-1)
            model_path: Path to saved model (.pth file)
            **kwargs: Additional metadata (precision, recall, hyperparameters, etc.)

        Returns:
            model_id: Unique ID assigned to the model

        Example:
            >>> model_id = registry.register_model(
            ...     model_name='ResNet34_1D',
            ...     phase='Phase 3',
            ...     accuracy=0.967,
            ...     model_path='checkpoints/resnet34.pth',
            ...     hyperparameters={'lr': 0.001, 'epochs': 100}
            ... )
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Prepare hyperparameters as JSON
        hyperparameters = kwargs.get('hyperparameters', {})
        hyperparameters_json = json.dumps(hyperparameters)

        cursor.execute('''
            INSERT INTO models (
                model_name, phase, accuracy, precision, recall, f1_score,
                training_date, hyperparameters, model_path, onnx_path,
                size_mb, inference_latency_ms, num_parameters,
                dataset_name, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            phase,
            accuracy,
            kwargs.get('precision'),
            kwargs.get('recall'),
            kwargs.get('f1_score'),
            kwargs.get('training_date', datetime.now().isoformat()),
            hyperparameters_json,
            model_path,
            kwargs.get('onnx_path'),
            kwargs.get('size_mb'),
            kwargs.get('inference_latency_ms'),
            kwargs.get('num_parameters'),
            kwargs.get('dataset_name'),
            kwargs.get('notes')
        ))

        model_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"✓ Registered model: {model_name} (ID: {model_id})")
        return model_id

    def get_best_model(self, metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Get best performing model by specified metric.

        Args:
            metric: Metric to maximize ('accuracy', 'f1_score', etc.)

        Returns:
            Dictionary with model metadata

        Example:
            >>> best = registry.get_best_model('accuracy')
            >>> print(f"Best model: {best['model_name']} ({best['accuracy']:.2%})")
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute(f'''
            SELECT * FROM models
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT 1
        ''')

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        else:
            logger.warning("No models found in registry")
            return {}

    def compare_models(
        self,
        model_names: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models side-by-side.

        Args:
            model_names: List of model names to compare
            metrics: List of metrics to include (default: all)

        Returns:
            DataFrame with comparison

        Example:
            >>> df = registry.compare_models(
            ...     ['ResNet34', 'EfficientNet_B2', 'Transformer'],
            ...     metrics=['accuracy', 'inference_latency_ms', 'size_mb']
            ... )
            >>> print(df)
        """
        conn = sqlite3.connect(self.db_path)

        placeholders = ','.join(['?' for _ in model_names])
        query = f'SELECT * FROM models WHERE model_name IN ({placeholders})'

        df = pd.read_sql_query(query, conn, params=model_names)
        conn.close()

        if metrics:
            available_metrics = [m for m in metrics if m in df.columns]
            df = df[['model_name'] + available_metrics]

        return df

    def export_registry_report(self, output_path: str = 'models/registry_report.html'):
        """
        Generate HTML report of all models.

        Args:
            output_path: Path to save HTML report

        Example:
            >>> registry.export_registry_report('results/model_comparison.html')
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM models', conn)
        conn.close()

        # Generate HTML report
        html = df.to_html(index=False)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(f'''
<!DOCTYPE html>
<html>
<head>
    <title>Model Registry Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Model Registry Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total models: {len(df)}</p>
    {html}
</body>
</html>
            ''')

        logger.info(f"✓ Report exported: {output_path}")

    def list_all_models(self) -> pd.DataFrame:
        """
        List all registered models.

        Returns:
            DataFrame with all models

        Example:
            >>> models = registry.list_all_models()
            >>> print(models[['model_name', 'phase', 'accuracy']])
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM models ORDER BY accuracy DESC', conn)
        conn.close()

        return df
