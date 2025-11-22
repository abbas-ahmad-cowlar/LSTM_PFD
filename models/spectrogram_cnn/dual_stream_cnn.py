"""
Dual-Stream CNN: Time + Frequency Domain Fusion

Combines 1D CNN (time-domain) and 2D CNN (frequency-domain) processing
in parallel streams with late fusion for improved performance.

Architecture:
    Signal Input [B, 1, T]
         ├─────────────┬─────────────┐
         │             │             │
    1D CNN        Spectrogram    (optional)
   (time branch)   Generator
         │             │
         │         2D CNN
         │      (freq branch)
         │             │
    [B, 512]      [B, 512]
         └──── Concat ─┘
               │
          [B, 1024]
               │
            FC Layer
               │
           [B, 11]

Reference:
- Two-stream networks for action recognition (Simonyan & Zisserman, 2014)
- Adapted for bearing fault diagnosis with time-frequency fusion

Usage:
    from models.spectrogram_cnn.dual_stream_cnn import DualStreamCNN

    model = DualStreamCNN(
        time_branch='resnet18_1d',
        freq_branch='resnet18_2d',
        num_classes=NUM_CLASSES
    )

    signal = torch.randn(4, 1, SIGNAL_LENGTH)
    output = model(signal)  # [4, 11]
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


from models.base_model import BaseModel


class DualStreamCNN(BaseModel):
    """
    Dual-stream CNN with time and frequency domain processing.

    Args:
        time_branch: 1D CNN architecture name or pretrained model path
        freq_branch: 2D CNN architecture name or pretrained model path
        num_classes: Number of output classes (default: 11)
        fusion_dim: Dimension for fusion layer (default: 1024)
        fusion_type: Type of fusion ('concat', 'add', 'attention') (default: 'concat')
        freeze_time_branch: Freeze time branch weights (default: False)
        freeze_freq_branch: Freeze freq branch weights (default: False)
        tfr_type: Time-frequency representation type (default: 'stft')
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        time_branch: Union[str, nn.Module] = 'resnet18_1d',
        freq_branch: Union[str, nn.Module] = 'resnet18_2d',
        num_classes: int = NUM_CLASSES,
        fusion_dim: int = 1024,
        fusion_type: str = 'concat',
        freeze_time_branch: bool = False,
        freeze_freq_branch: bool = False,
        tfr_type: str = 'stft',
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim
        self.tfr_type = tfr_type

        # Create or load time branch (1D CNN)
        if isinstance(time_branch, str):
            self.time_branch = self._create_time_branch(time_branch)
        else:
            self.time_branch = time_branch

        # Create or load frequency branch (2D CNN)
        if isinstance(freq_branch, str):
            self.freq_branch = self._create_freq_branch(freq_branch)
        else:
            self.freq_branch = freq_branch

        # Freeze branches if requested
        if freeze_time_branch:
            for param in self.time_branch.parameters():
                param.requires_grad = False

        if freeze_freq_branch:
            for param in self.freq_branch.parameters():
                param.requires_grad = False

        # Extract feature dimensions
        self.time_feat_dim = self._get_feature_dim(self.time_branch)
        self.freq_feat_dim = self._get_feature_dim(self.freq_branch)

        # Create spectrogram generator (for on-the-fly TFR)
        if tfr_type == 'stft':
            from data.spectrogram_generator import STFTSpectrogram
            self.tfr_generator = STFTSpectrogram(
                n_fft=256,
                hop_length=128,
                win_length=256
            )
        elif tfr_type == 'cwt':
            from data.wavelet_transform import CWTTransform
            self.tfr_generator = CWTTransform(
                wavelet='morl',
                scales=128
            )
        else:
            self.tfr_generator = None

        # Fusion layer
        if fusion_type == 'concat':
            fusion_input_dim = self.time_feat_dim + self.freq_feat_dim
        elif fusion_type == 'add':
            assert self.time_feat_dim == self.freq_feat_dim, \
                "Feature dimensions must match for 'add' fusion"
            fusion_input_dim = self.time_feat_dim
        elif fusion_type == 'attention':
            fusion_input_dim = self.time_feat_dim + self.freq_feat_dim
            self.attention = AttentionFusion(self.time_feat_dim, self.freq_feat_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Classification head
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )

    def _create_time_branch(self, model_name: str) -> nn.Module:
        """Create 1D CNN for time-domain processing."""
        if 'resnet' in model_name.lower():
            from models.resnet.resnet_1d import ResNet1D
            from models.resnet.residual_blocks import BasicBlock1D

            if '18' in model_name:
                layers = [2, 2, 2, 2]
            elif '34' in model_name:
                layers = [3, 4, 6, 3]
            else:
                layers = [2, 2, 2, 2]

            model = ResNet1D(
                num_classes=self.num_classes,
                block=BasicBlock1D,
                layers=layers
            )

            # Remove final FC layer (we'll use fusion head instead)
            model.fc = nn.Identity()

        elif 'cnn' in model_name.lower():
            from models.cnn.cnn_1d import CNN1D
            model = CNN1D(num_classes=self.num_classes)
            model.fc = nn.Identity()

        else:
            raise ValueError(f"Unknown time branch model: {model_name}")

        return model

    def _create_freq_branch(self, model_name: str) -> nn.Module:
        """Create 2D CNN for frequency-domain processing."""
        from models.spectrogram_cnn import get_model

        if 'resnet' in model_name.lower():
            if '18' in model_name:
                model = get_model('resnet18_2d', num_classes=self.num_classes)
            elif '34' in model_name:
                model = get_model('resnet34_2d', num_classes=self.num_classes)
            else:
                model = get_model('resnet18_2d', num_classes=self.num_classes)

        elif 'efficientnet' in model_name.lower():
            if 'b0' in model_name:
                model = get_model('efficientnet_b0', num_classes=self.num_classes)
            elif 'b1' in model_name:
                model = get_model('efficientnet_b1', num_classes=self.num_classes)
            else:
                model = get_model('efficientnet_b0', num_classes=self.num_classes)

        else:
            raise ValueError(f"Unknown freq branch model: {model_name}")

        # Remove final FC layer
        model.fc = nn.Identity()

        return model

    def _get_feature_dim(self, model: nn.Module) -> int:
        """Get output feature dimension of a model."""
        # Create dummy input and get output shape
        dummy_input = torch.randn(1, 1, SIGNAL_LENGTH)

        # Check if model expects 2D input (spectrogram)
        try:
            with torch.no_grad():
                output = model(dummy_input)
        except:
            # Try with 2D input
            dummy_input_2d = torch.randn(1, 1, 129, 400)
            with torch.no_grad():
                output = model(dummy_input_2d)

        return output.shape[1]

    def forward(
        self,
        x: torch.Tensor,
        spectrogram: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through dual-stream network.

        Args:
            x: Time-domain signal [B, 1, T]
            spectrogram: Optional precomputed spectrogram [B, 1, H, W]
                        If None, will be computed on-the-fly

        Returns:
            Class logits [B, num_classes]
        """
        # Time branch: process raw signal
        time_features = self.time_branch(x)  # [B, time_feat_dim]

        # Frequency branch: process spectrogram
        if spectrogram is None:
            # Generate spectrogram on-the-fly
            if self.tfr_generator is not None:
                with torch.no_grad():
                    spectrogram = self._generate_spectrogram(x)
            else:
                raise ValueError(
                    "No spectrogram provided and no TFR generator configured. "
                    "Either provide precomputed spectrogram or configure tfr_type."
                )

        freq_features = self.freq_branch(spectrogram)  # [B, freq_feat_dim]

        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([time_features, freq_features], dim=1)
        elif self.fusion_type == 'add':
            fused = time_features + freq_features
        elif self.fusion_type == 'attention':
            fused = self.attention(time_features, freq_features)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Classification
        output = self.fusion_fc(fused)

        return output

    def _generate_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Generate spectrogram from signal on-the-fly.

        Args:
            signal: Time-domain signal [B, 1, T]

        Returns:
            Spectrogram [B, 1, H, W]
        """
        B = signal.shape[0]
        spectrograms = []

        for i in range(B):
            sig = signal[i, 0].cpu().numpy()
            spec = self.tfr_generator.generate(sig)
            spectrograms.append(torch.from_numpy(spec))

        spectrograms = torch.stack(spectrograms).unsqueeze(1)  # [B, 1, H, W]
        return spectrograms.to(signal.device)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for time and frequency features.

    Learns to weight time and frequency features dynamically.

    Args:
        time_dim: Dimension of time features
        freq_dim: Dimension of frequency features
    """

    def __init__(self, time_dim: int, freq_dim: int):
        super().__init__()

        self.time_dim = time_dim
        self.freq_dim = freq_dim
        total_dim = time_dim + freq_dim

        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim // 2, 2),  # 2 weights (time, freq)
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        time_features: torch.Tensor,
        freq_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention fusion.

        Args:
            time_features: Time-domain features [B, time_dim]
            freq_features: Frequency-domain features [B, freq_dim]

        Returns:
            Fused features [B, time_dim + freq_dim]
        """
        # Concatenate features
        concat_features = torch.cat([time_features, freq_features], dim=1)

        # Compute attention weights
        weights = self.attention(concat_features)  # [B, 2]

        # Apply weights
        time_weight = weights[:, 0:1]  # [B, 1]
        freq_weight = weights[:, 1:2]  # [B, 1]

        weighted_time = time_features * time_weight
        weighted_freq = freq_features * freq_weight

        # Concatenate weighted features
        fused = torch.cat([weighted_time, weighted_freq], dim=1)

        return fused


def dual_stream_resnet18(num_classes: int = NUM_CLASSES, **kwargs) -> DualStreamCNN:
    """Dual-stream with ResNet-18 in both branches."""
    return DualStreamCNN(
        time_branch='resnet18_1d',
        freq_branch='resnet18_2d',
        num_classes=num_classes,
        **kwargs
    )


def dual_stream_mixed(num_classes: int = NUM_CLASSES, **kwargs) -> DualStreamCNN:
    """Dual-stream with ResNet-18 (time) and EfficientNet-B0 (freq)."""
    return DualStreamCNN(
        time_branch='resnet18_1d',
        freq_branch='efficientnet_b0',
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Test dual-stream model
    print("Testing Dual-Stream CNN...")

    model = dual_stream_resnet18(num_classes=NUM_CLASSES, fusion_type='concat')

    # Test with time-domain input only
    signal = torch.randn(4, 1, SIGNAL_LENGTH)
    output = model(signal)

    print(f"Input signal shape: {signal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with precomputed spectrogram
    spectrogram = torch.randn(4, 1, 129, 400)
    output = model(signal, spectrogram=spectrogram)
    print(f"\nWith precomputed spectrogram: {output.shape}")

    # Test attention fusion
    print("\nTesting attention fusion...")
    model_attn = dual_stream_resnet18(num_classes=NUM_CLASSES, fusion_type='attention')
    output_attn = model_attn(signal)
    print(f"Attention fusion output: {output_attn.shape}")
