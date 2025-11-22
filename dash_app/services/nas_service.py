"""
Neural Architecture Search (NAS) Service.
Handles NAS campaign creation, architecture sampling, and result management.
"""
from typing import Dict, List, Optional, Any
import hashlib
import json
import random
from pathlib import Path

from utils.logger import setup_logger
from database.connection import get_db_session
from models.nas_campaign import NASCampaign, NASTrial
from models.dataset import Dataset

logger = setup_logger(__name__)


class NASService:
    """Service for Neural Architecture Search operations."""

    @staticmethod
    def create_nas_campaign(
        name: str,
        dataset_id: int,
        search_space_config: Dict,
        search_algorithm: str = 'random',
        num_trials: int = 20,
        max_epochs_per_trial: int = 10
    ) -> int:
        """
        Create a new NAS campaign.

        Args:
            name: Campaign name
            dataset_id: Dataset to use for training
            search_space_config: SearchSpaceConfig parameters
            search_algorithm: 'random', 'bayesian', 'evolution'
            num_trials: Number of architectures to try
            max_epochs_per_trial: Training epochs per architecture

        Returns:
            Campaign ID
        """
        try:
            with get_db_session() as session:
                # Verify dataset exists
                dataset = session.query(Dataset).get(dataset_id)
                if not dataset:
                    raise ValueError(f"Dataset {dataset_id} not found")

                # Create campaign
                campaign = NASCampaign(
                    name=name,
                    dataset_id=dataset_id,
                    search_algorithm=search_algorithm,
                    num_trials=num_trials,
                    max_epochs_per_trial=max_epochs_per_trial,
                    search_space_config=search_space_config,
                    status='pending'
                )
                session.add(campaign)
                session.commit()
                session.refresh(campaign)

                logger.info(f"Created NAS campaign: {name} (ID: {campaign.id})")
                return campaign.id

        except Exception as e:
            logger.error(f"Failed to create NAS campaign: {e}")
            raise

    @staticmethod
    def sample_architecture(search_space_config: Dict) -> Dict:
        """
        Sample a random architecture from search space.

        Args:
            search_space_config: Search space configuration

        Returns:
            Architecture dict with:
            - operations: List of operation types
            - channels: List of channel sizes
            - num_layers: Total number of layers
        """
        operations = search_space_config.get('operations', ['conv_3', 'conv_5', 'max_pool_3', 'skip_connect'])
        channel_sizes = search_space_config.get('channel_sizes', [32, 64, 128, 256])
        min_layers = search_space_config.get('min_layers', 3)
        max_layers = search_space_config.get('max_layers', 10)

        # Sample number of layers
        num_layers = random.randint(min_layers, max_layers)

        # Sample operations for each layer
        sampled_operations = [random.choice(operations) for _ in range(num_layers)]

        # Sample channel sizes (increasing or stable)
        sampled_channels = []
        current_channels = random.choice(channel_sizes[:2])  # Start with smaller channels
        for i in range(num_layers):
            sampled_channels.append(current_channels)
            # Optionally increase channels
            if random.random() > 0.7 and current_channels < channel_sizes[-1]:
                idx = channel_sizes.index(current_channels)
                current_channels = channel_sizes[min(idx + 1, len(channel_sizes) - 1)]

        architecture = {
            'operations': sampled_operations,
            'channels': sampled_channels,
            'num_layers': num_layers,
            'search_space_config': search_space_config
        }

        return architecture

    @staticmethod
    def compute_architecture_hash(architecture: Dict) -> str:
        """
        Compute hash of architecture for deduplication.

        Args:
            architecture: Architecture dict

        Returns:
            SHA-256 hash string
        """
        # Create canonical string representation
        arch_str = json.dumps({
            'operations': architecture['operations'],
            'channels': architecture['channels']
        }, sort_keys=True)

        return hashlib.sha256(arch_str.encode()).hexdigest()

    @staticmethod
    def get_campaign_details(campaign_id: int) -> Optional[Dict]:
        """
        Get NAS campaign details with all trials.

        Args:
            campaign_id: Campaign ID

        Returns:
            Campaign dict with trials, or None if not found
        """
        try:
            with get_db_session() as session:
                campaign = session.query(NASCampaign).get(campaign_id)
                if not campaign:
                    return None

                # Get all trials
                trials = session.query(NASTrial).filter(
                    NASTrial.campaign_id == campaign_id
                ).order_by(NASTrial.trial_number.asc()).all()

                return {
                    'id': campaign.id,
                    'name': campaign.name,
                    'dataset_id': campaign.dataset_id,
                    'search_algorithm': campaign.search_algorithm,
                    'num_trials': campaign.num_trials,
                    'max_epochs_per_trial': campaign.max_epochs_per_trial,
                    'status': campaign.status,
                    'best_accuracy': campaign.best_accuracy,
                    'created_at': campaign.created_at.isoformat() if campaign.created_at else None,
                    'trials': [
                        {
                            'id': trial.id,
                            'trial_number': trial.trial_number,
                            'validation_accuracy': trial.validation_accuracy,
                            'validation_loss': trial.validation_loss,
                            'training_time': trial.training_time,
                            'num_parameters': trial.num_parameters,
                            'flops': trial.flops,
                            'architecture': trial.architecture
                        }
                        for trial in trials
                    ]
                }

        except Exception as e:
            logger.error(f"Failed to get campaign details: {e}")
            return None

    @staticmethod
    def get_best_architecture(campaign_id: int) -> Optional[Dict]:
        """
        Get best performing architecture from campaign.

        Args:
            campaign_id: Campaign ID

        Returns:
            Best trial dict, or None if no trials completed
        """
        try:
            with get_db_session() as session:
                # Get best trial (highest accuracy)
                best_trial = session.query(NASTrial).filter(
                    NASTrial.campaign_id == campaign_id,
                    NASTrial.validation_accuracy.isnot(None)
                ).order_by(NASTrial.validation_accuracy.desc()).first()

                if not best_trial:
                    return None

                return {
                    'id': best_trial.id,
                    'trial_number': best_trial.trial_number,
                    'validation_accuracy': best_trial.validation_accuracy,
                    'validation_loss': best_trial.validation_loss,
                    'num_parameters': best_trial.num_parameters,
                    'flops': best_trial.flops,
                    'architecture': best_trial.architecture,
                    'training_time': best_trial.training_time
                }

        except Exception as e:
            logger.error(f"Failed to get best architecture: {e}")
            return None

    @staticmethod
    def export_architecture(trial_id: int, format: str = 'pytorch') -> Optional[str]:
        """
        Export discovered architecture as code.

        Args:
            trial_id: NAS trial ID
            format: 'pytorch', 'json'

        Returns:
            Architecture code/config as string, or None if trial not found
        """
        try:
            with get_db_session() as session:
                trial = session.query(NASTrial).get(trial_id)
                if not trial:
                    return None

                architecture = trial.architecture

                if format == 'json':
                    return json.dumps(architecture, indent=2)

                elif format == 'pytorch':
                    # Generate PyTorch code
                    operations = architecture['operations']
                    channels = architecture['channels']

                    code = "import torch\nimport torch.nn as nn\n\n"
                    code += "class DiscoveredModel(nn.Module):\n"
                    code += "    def __init__(self, num_classes=10):\n"
                    code += "        super().__init__()\n"
                    code += "        layers = []\n\n"

                    in_channels = 1  # Assuming single-channel input
                    for i, (op, out_channels) in enumerate(zip(operations, channels)):
                        if 'conv' in op:
                            kernel_size = int(op.split('_')[1])
                            code += f"        # Layer {i}: {op}\n"
                            code += f"        layers.append(nn.Conv1d({in_channels}, {out_channels}, kernel_size={kernel_size}, padding={kernel_size//2}))\n"
                            code += f"        layers.append(nn.ReLU())\n"
                            code += f"        layers.append(nn.BatchNorm1d({out_channels}))\n"
                            in_channels = out_channels
                        elif 'pool' in op:
                            code += f"        # Layer {i}: {op}\n"
                            code += f"        layers.append(nn.MaxPool1d(3, stride=2, padding=1))\n"
                        # Skip 'skip_connect' for simplicity in code generation

                    code += "\n        self.features = nn.Sequential(*layers)\n"
                    code += f"        self.classifier = nn.Linear({in_channels}, num_classes)\n\n"
                    code += "    def forward(self, x):\n"
                    code += "        x = self.features(x)\n"
                    code += "        x = torch.mean(x, dim=2)  # Global average pooling\n"
                    code += "        x = self.classifier(x)\n"
                    code += "        return x\n"

                    return code

                else:
                    return None

        except Exception as e:
            logger.error(f"Failed to export architecture: {e}")
            return None

    @staticmethod
    def list_campaigns(limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        List all NAS campaigns.

        Args:
            limit: Maximum number of campaigns to return
            offset: Offset for pagination

        Returns:
            List of campaign dicts
        """
        try:
            with get_db_session() as session:
                campaigns = session.query(NASCampaign).order_by(
                    NASCampaign.created_at.desc()
                ).limit(limit).offset(offset).all()

                return [
                    {
                        'id': campaign.id,
                        'name': campaign.name,
                        'dataset_id': campaign.dataset_id,
                        'search_algorithm': campaign.search_algorithm,
                        'num_trials': campaign.num_trials,
                        'status': campaign.status,
                        'best_accuracy': campaign.best_accuracy,
                        'created_at': campaign.created_at.isoformat() if campaign.created_at else None
                    }
                    for campaign in campaigns
                ]

        except Exception as e:
            logger.error(f"Failed to list campaigns: {e}")
            return []
