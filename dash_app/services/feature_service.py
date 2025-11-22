"""
Feature Engineering Service.
Business logic for feature extraction, selection, and importance analysis.
"""
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import h5py
from database.connection import get_db_session
from models.dataset import Dataset
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureService:
    """Service for feature engineering operations."""

    @staticmethod
    def extract_features(
        dataset_id: int,
        domain: str,  # 'all', 'time', 'frequency', 'wavelet', 'bispectrum', 'envelope'
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract features from dataset.

        Args:
            dataset_id: Dataset ID
            domain: Feature domain to extract
            config: Optional configuration

        Returns:
            Dictionary with features, feature_names, extraction_time
        """
        try:
            import time
            from features.feature_extractor import FeatureExtractor
            from features.time_domain import extract_time_domain_features
            from features.frequency_domain import extract_frequency_domain_features
            from features.wavelet_features import extract_wavelet_features
            from features.bispectrum import extract_bispectrum_features
            from features.envelope_analysis import extract_envelope_features

            # Get dataset
            with get_db_session() as session:
                dataset = session.query(Dataset).filter_by(id=dataset_id).first()

                if not dataset or not Path(dataset.file_path).exists():
                    logger.error(f"Dataset {dataset_id} not found or file missing")
                    return {'success': False, 'error': 'Dataset not found'}

                # Load signals
                with h5py.File(dataset.file_path, 'r') as f:
                    signals = f['signals'][:]
                    labels = f['labels'][:]

                # Get sampling rate
                config = config or {}
                fs = config.get('sampling_rate', 20480)

                start_time = time.time()

                if domain == 'all':
                    # Extract all features using FeatureExtractor
                    extractor = FeatureExtractor(fs=fs)
                    features_list = []

                    for signal in signals:
                        feats = extractor.extract_features(signal)
                        features_list.append(feats)

                    features = np.array(features_list)
                    feature_names = extractor.feature_names_

                else:
                    # Extract domain-specific features
                    features_list = []
                    feature_names = []

                    for signal in signals:
                        if domain == 'time':
                            feats_dict = extract_time_domain_features(signal)
                        elif domain == 'frequency':
                            feats_dict = extract_frequency_domain_features(signal, fs)
                        elif domain == 'wavelet':
                            feats_dict = extract_wavelet_features(signal, fs)
                        elif domain == 'bispectrum':
                            feats_dict = extract_bispectrum_features(signal)
                        elif domain == 'envelope':
                            feats_dict = extract_envelope_features(signal, fs)
                        else:
                            return {'success': False, 'error': f'Unknown domain: {domain}'}

                        # Convert dict to array
                        if not feature_names:
                            feature_names = list(feats_dict.keys())

                        feats = np.array([feats_dict[name] for name in feature_names])
                        features_list.append(feats)

                    features = np.array(features_list)

                extraction_time = time.time() - start_time

                return {
                    'success': True,
                    'features': features,
                    'feature_names': feature_names,
                    'labels': labels,
                    'num_features': len(feature_names),
                    'num_samples': len(signals),
                    'extraction_time': extraction_time,
                    'domain': domain
                }

        except Exception as e:
            logger.error(f"Failed to extract features: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    @staticmethod
    def compute_feature_importance(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        method: str = 'mutual_info'  # 'mutual_info', 'random_forest', 'permutation'
    ) -> Dict[str, Any]:
        """
        Compute feature importance.

        Args:
            features: Feature matrix [N, num_features]
            labels: Target labels [N]
            feature_names: List of feature names
            method: Importance method

        Returns:
            Dictionary with importance scores
        """
        try:
            if method == 'mutual_info':
                from sklearn.feature_selection import mutual_info_classif

                scores = mutual_info_classif(features, labels, random_state=42)

                importance_dict = {
                    name: float(score)
                    for name, score in zip(feature_names, scores)
                }

                # Sort by importance
                sorted_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                return {
                    'success': True,
                    'method': method,
                    'importances': importance_dict,
                    'sorted_features': sorted_features,
                    'feature_names': feature_names,
                    'scores': scores.tolist()
                }

            elif method == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                from features.feature_importance import get_random_forest_importances

                # Train a quick RF model
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(features, labels)

                importance_dict = get_random_forest_importances(rf, feature_names)

                sorted_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                return {
                    'success': True,
                    'method': method,
                    'importances': importance_dict,
                    'sorted_features': sorted_features,
                    'feature_names': feature_names
                }

            elif method == 'permutation':
                from sklearn.ensemble import RandomForestClassifier
                from features.feature_importance import get_permutation_importances
                from sklearn.model_selection import train_test_split

                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    features, labels,
                    test_size=0.2,
                    random_state=42,
                    stratify=labels
                )

                # Train model
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)

                # Compute permutation importance
                perm_importances = get_permutation_importances(
                    rf, X_val, y_val,
                    feature_names=feature_names,
                    n_repeats=10
                )

                # Extract mean scores
                importance_dict = {
                    name: mean_score
                    for name, (mean_score, _) in perm_importances.items()
                }

                sorted_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                return {
                    'success': True,
                    'method': method,
                    'importances': importance_dict,
                    'sorted_features': sorted_features,
                    'feature_names': feature_names
                }

            else:
                return {'success': False, 'error': f'Unknown method: {method}'}

        except Exception as e:
            logger.error(f"Failed to compute feature importance: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    @staticmethod
    def select_features(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        method: str,  # 'mrmr', 'variance', 'mutual_info', 'rfe'
        num_features: int = 15,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Select top features.

        Args:
            features: Feature matrix [N, num_features]
            labels: Target labels [N]
            feature_names: List of feature names
            method: Selection method
            num_features: Number of features to select
            threshold: Threshold for variance method

        Returns:
            Dictionary with selected features
        """
        try:
            if method == 'mrmr':
                from features.feature_selector import FeatureSelector

                selector = FeatureSelector(
                    method='mrmr',
                    n_features=num_features,
                    random_state=42
                )
                selector.fit(features, labels)

                selected_indices = selector.selected_indices_
                selected_names = [feature_names[i] for i in selected_indices]

                return {
                    'success': True,
                    'method': method,
                    'selected_indices': selected_indices,
                    'selected_names': selected_names,
                    'num_selected': len(selected_indices),
                    'relevance_scores': selector.relevance_scores_.tolist() if selector.relevance_scores_ is not None else None
                }

            elif method == 'variance':
                from sklearn.feature_selection import VarianceThreshold

                selector = VarianceThreshold(threshold=threshold)
                selector.fit(features)

                selected_indices = np.where(selector.get_support())[0].tolist()
                selected_names = [feature_names[i] for i in selected_indices]

                return {
                    'success': True,
                    'method': method,
                    'selected_indices': selected_indices,
                    'selected_names': selected_names,
                    'num_selected': len(selected_indices),
                    'variances': selector.variances_.tolist()
                }

            elif method == 'mutual_info':
                from sklearn.feature_selection import SelectKBest, mutual_info_classif

                selector = SelectKBest(mutual_info_classif, k=num_features)
                selector.fit(features, labels)

                selected_indices = np.where(selector.get_support())[0].tolist()
                selected_names = [feature_names[i] for i in selected_indices]

                return {
                    'success': True,
                    'method': method,
                    'selected_indices': selected_indices,
                    'selected_names': selected_names,
                    'num_selected': len(selected_indices),
                    'scores': selector.scores_.tolist()
                }

            elif method == 'rfe':
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestClassifier

                estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                selector = RFE(estimator, n_features_to_select=num_features)
                selector.fit(features, labels)

                selected_indices = np.where(selector.get_support())[0].tolist()
                selected_names = [feature_names[i] for i in selected_indices]

                return {
                    'success': True,
                    'method': method,
                    'selected_indices': selected_indices,
                    'selected_names': selected_names,
                    'num_selected': len(selected_indices),
                    'ranking': selector.ranking_.tolist()
                }

            else:
                return {'success': False, 'error': f'Unknown method: {method}'}

        except Exception as e:
            logger.error(f"Failed to select features: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    @staticmethod
    def compute_feature_correlation(
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Compute feature correlation matrix.

        Args:
            features: Feature matrix [N, num_features]
            feature_names: List of feature names

        Returns:
            Dictionary with correlation matrix
        """
        try:
            # Compute correlation matrix
            corr_matrix = np.corrcoef(features.T)

            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    if abs(corr_matrix[i, j]) > 0.8:
                        high_corr_pairs.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(corr_matrix[i, j])
                        })

            # Sort by absolute correlation
            high_corr_pairs = sorted(
                high_corr_pairs,
                key=lambda x: abs(x['correlation']),
                reverse=True
            )

            return {
                'success': True,
                'correlation_matrix': corr_matrix.tolist(),
                'feature_names': feature_names,
                'high_corr_pairs': high_corr_pairs[:20],  # Top 20
                'num_features': len(feature_names)
            }

        except Exception as e:
            logger.error(f"Failed to compute correlation: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    @staticmethod
    def save_feature_set(
        dataset_id: int,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        domain: str,
        metadata: Optional[Dict] = None,
        output_dir: str = 'feature_sets'
    ) -> Optional[str]:
        """
        Save extracted features for later use.

        Args:
            dataset_id: Dataset ID
            features: Feature matrix
            labels: Target labels
            feature_names: Feature names
            domain: Feature domain
            metadata: Optional metadata
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"features_dataset{dataset_id}_{domain}_{timestamp}.h5"
            file_path = output_path / filename

            # Save to HDF5
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('features', data=features)
                f.create_dataset('labels', data=labels)
                f.create_dataset('feature_names', data=np.array(feature_names, dtype='S'))

                # Attributes
                f.attrs['dataset_id'] = dataset_id
                f.attrs['domain'] = domain
                f.attrs['num_features'] = len(feature_names)
                f.attrs['num_samples'] = len(features)
                f.attrs['timestamp'] = timestamp

                if metadata:
                    for key, value in metadata.items():
                        f.attrs[key] = value

            logger.info(f"Saved feature set to: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save feature set: {e}", exc_info=True)
            return None
