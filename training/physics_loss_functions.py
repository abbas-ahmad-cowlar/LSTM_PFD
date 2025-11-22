"""
Physics-Based Loss Functions for PINNs

This module provides various physics-informed loss terms that constrain neural
networks to make predictions consistent with bearing physics and fault dynamics.

These loss functions complement standard classification loss (cross-entropy) to:
1. Enforce frequency domain consistency
2. Ensure operating condition validity
3. Maintain temporal coherence
4. Respect physical constraints (Sommerfeld number, Reynolds number, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.physics.bearing_dynamics import BearingDynamics
from models.physics.fault_signatures import FaultSignatureDatabase
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class FrequencyConsistencyLoss(nn.Module):
    """
    Penalizes predictions that are inconsistent with expected fault frequencies.

    For a predicted fault class, we expect certain frequencies to be dominant
    in the vibration spectrum. This loss computes the mismatch between observed
    and expected frequency signatures.
    """

    def __init__(
        self,
        sample_rate: int = 51200,
        n_fft: int = 2048,
        top_k: int = 5,
        tolerance: float = 0.1
    ):
        """
        Initialize frequency consistency loss.

        Args:
            sample_rate: Sampling rate in Hz
            n_fft: FFT size for frequency analysis
            top_k: Number of top frequencies to compare
            tolerance: Frequency matching tolerance (fraction of expected freq)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.top_k = top_k
        self.tolerance = tolerance

        self.signature_db = FaultSignatureDatabase()
        self.bearing_dynamics = BearingDynamics()

    def forward(
        self,
        signal: torch.Tensor,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute frequency consistency loss.

        Args:
            signal: Raw vibration signal [B, 1, T] or [B, T]
            predictions: Predicted class logits [B, num_classes]
            metadata: Optional dict with 'rpm' key (default: 3600 RPM)

        Returns:
            Frequency consistency loss (scalar)
        """
        batch_size = signal.shape[0]

        # Get predicted classes
        predicted_classes = torch.argmax(predictions, dim=1)  # [B]

        # Default RPM if not provided
        if metadata is None or 'rpm' not in metadata:
            rpm = 3600.0
        else:
            rpm = metadata['rpm']
            if isinstance(rpm, torch.Tensor):
                rpm = rpm.cpu().numpy()
            if isinstance(rpm, np.ndarray):
                rpm = rpm[0] if len(rpm.shape) > 0 else float(rpm)

        # Compute FFT of signal
        if signal.dim() == 3:
            signal = signal.squeeze(1)  # [B, T]

        # Use rfft for real signals
        fft = torch.fft.rfft(signal, n=self.n_fft, dim=-1)
        magnitude = torch.abs(fft)  # [B, n_fft//2 + 1]

        # Frequency bins
        freq_bins = torch.fft.rfftfreq(self.n_fft, d=1.0/self.sample_rate)  # [n_fft//2 + 1]
        freq_bins = freq_bins.to(signal.device)

        # Compute loss for each sample
        total_loss = 0.0

        for i in range(batch_size):
            pred_class = predicted_classes[i].item()

            # Get expected frequencies for this fault type
            expected_freqs = self.signature_db.get_expected_frequencies(
                pred_class, rpm, top_k=self.top_k
            )
            expected_freqs = torch.tensor(expected_freqs, device=signal.device)

            # Find peaks in observed spectrum
            spectrum_i = magnitude[i]  # [n_fft//2 + 1]

            # Find top_k peaks
            peak_values, peak_indices = torch.topk(spectrum_i, k=min(self.top_k * 2, len(spectrum_i)))
            peak_freqs = freq_bins[peak_indices]

            # Compute minimum distance from each expected freq to any peak
            for expected_freq in expected_freqs:
                if expected_freq == 0:
                    continue

                # Distance to closest peak
                distances = torch.abs(peak_freqs - expected_freq) / (expected_freq + 1e-6)
                min_distance = torch.min(distances)

                # Penalize if no peak within tolerance
                loss_i = F.relu(min_distance - self.tolerance)
                total_loss += loss_i

        # Average over batch and expected frequencies
        avg_loss = total_loss / (batch_size * self.top_k)

        return avg_loss


class SommerfeldConsistencyLoss(nn.Module):
    """
    Ensures predicted severity is consistent with operating conditions.

    The Sommerfeld number relates load, speed, and viscosity to lubrication regime.
    High Sommerfeld → good lubrication → lower severity expected
    Low Sommerfeld → boundary lubrication → higher severity expected
    """

    def __init__(self):
        """Initialize Sommerfeld consistency loss."""
        super().__init__()
        self.bearing_dynamics = BearingDynamics()

    def forward(
        self,
        predicted_severity: torch.Tensor,
        metadata: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Sommerfeld consistency loss.

        Args:
            predicted_severity: Predicted severity scores [B, num_severity_levels]
                               or [B] if regression
            metadata: Dict with 'load', 'rpm', 'viscosity' (optional)

        Returns:
            Sommerfeld consistency loss (scalar)
        """
        # Extract metadata
        load = metadata.get('load', torch.tensor(500.0))  # Default 500N
        rpm = metadata.get('rpm', torch.tensor(3600.0))  # Default 3600 RPM
        viscosity = metadata.get('viscosity', torch.tensor(0.03))  # Default 0.03 Pa·s

        # Ensure tensors
        if not isinstance(load, torch.Tensor):
            load = torch.tensor(load, dtype=torch.float32)
        if not isinstance(rpm, torch.Tensor):
            rpm = torch.tensor(rpm, dtype=torch.float32)
        if not isinstance(viscosity, torch.Tensor):
            viscosity = torch.tensor(viscosity, dtype=torch.float32)

        # Move to same device
        device = predicted_severity.device
        load = load.to(device)
        rpm = rpm.to(device)
        viscosity = viscosity.to(device)

        # Compute Sommerfeld number
        S = self.bearing_dynamics.sommerfeld_number(load, rpm, viscosity, return_torch=True)
        S = S.to(device)

        # Expected severity based on Sommerfeld number
        # S < 0.1: boundary → high severity
        # 0.1 < S < 1: mixed → medium severity
        # S > 1: hydrodynamic → low severity

        if predicted_severity.dim() == 2:
            # Classification output [B, num_levels]
            # Assume severity levels: 0=low, 1=medium, 2=high, 3=critical
            predicted_level = torch.argmax(predicted_severity, dim=1).float()  # [B]
        else:
            # Regression output [B]
            predicted_level = predicted_severity

        # Expected severity level from Sommerfeld number
        expected_severity = torch.zeros_like(predicted_level)
        expected_severity[S < 0.1] = 3.0  # Critical (boundary lubrication)
        expected_severity[(S >= 0.1) & (S < 1.0)] = 1.5  # Medium (mixed)
        expected_severity[S >= 1.0] = 0.5  # Low (hydrodynamic)

        # MSE loss between predicted and expected
        loss = F.mse_loss(predicted_level, expected_severity)

        return loss


class TemporalSmoothness Loss(nn.Module):
    """
    Penalizes erratic temporal changes in predictions.

    For a sequence of consecutive samples from the same bearing, predictions
    should be temporally smooth (bearings don't instantly switch fault types).
    """

    def __init__(self, smoothness_weight: float = 1.0):
        """
        Initialize temporal smoothness loss.

        Args:
            smoothness_weight: Weight for smoothness penalty
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        predictions_sequence: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute temporal smoothness loss.

        Args:
            predictions_sequence: Sequence of predictions [T, B, num_classes]
            time_delta: Time between samples in seconds (optional)

        Returns:
            Temporal smoothness loss (scalar)
        """
        if predictions_sequence.shape[0] < 2:
            return torch.tensor(0.0, device=predictions_sequence.device)

        # Get predicted probabilities
        probs = F.softmax(predictions_sequence, dim=-1)  # [T, B, num_classes]

        # Compute differences between consecutive predictions
        prob_diffs = probs[1:] - probs[:-1]  # [T-1, B, num_classes]

        # L2 norm of differences
        smoothness_loss = torch.mean(torch.sum(prob_diffs ** 2, dim=-1))

        # Weight by time delta if provided
        if time_delta is not None:
            # Normalize by time: faster changes over longer time are OK
            smoothness_loss = smoothness_loss / (time_delta.mean() + 1e-6)

        return self.smoothness_weight * smoothness_loss


class PhysicalConstraintLoss(nn.Module):
    """
    Combined physics-based loss with multiple constraint terms.

    This is the main loss function for PINN training, combining:
    1. Frequency consistency
    2. Sommerfeld consistency (if severity prediction available)
    3. Temporal smoothness (if sequence available)
    """

    def __init__(
        self,
        lambda_freq: float = 1.0,
        lambda_sommerfeld: float = 0.5,
        lambda_temporal: float = 0.1,
        sample_rate: int = 51200
    ):
        """
        Initialize combined physics constraint loss.

        Args:
            lambda_freq: Weight for frequency consistency loss
            lambda_sommerfeld: Weight for Sommerfeld consistency loss
            lambda_temporal: Weight for temporal smoothness loss
            sample_rate: Signal sampling rate
        """
        super().__init__()
        self.lambda_freq = lambda_freq
        self.lambda_sommerfeld = lambda_sommerfeld
        self.lambda_temporal = lambda_temporal

        self.freq_loss = FrequencyConsistencyLoss(sample_rate=sample_rate)
        self.sommerfeld_loss = SommerfeldConsistencyLoss()
        self.temporal_loss = TemporalSmoothnessLoss()

    def forward(
        self,
        signal: torch.Tensor,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
        severity_predictions: Optional[torch.Tensor] = None,
        predictions_sequence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined physics constraint loss.

        Args:
            signal: Raw vibration signal [B, 1, T]
            predictions: Class predictions [B, num_classes]
            metadata: Operating conditions (rpm, load, etc.)
            severity_predictions: Severity predictions (optional)
            predictions_sequence: Temporal sequence of predictions (optional)

        Returns:
            total_loss: Combined physics loss
            loss_dict: Dictionary of individual loss components
        """
        losses = {}
        total_loss = 0.0

        # Frequency consistency loss (always computed)
        if self.lambda_freq > 0:
            freq_loss = self.freq_loss(signal, predictions, metadata)
            losses['freq_consistency'] = freq_loss.item()
            total_loss += self.lambda_freq * freq_loss

        # Sommerfeld consistency loss (if severity available)
        if self.lambda_sommerfeld > 0 and severity_predictions is not None:
            somm_loss = self.sommerfeld_loss(severity_predictions, metadata or {})
            losses['sommerfeld_consistency'] = somm_loss.item()
            total_loss += self.lambda_sommerfeld * somm_loss

        # Temporal smoothness loss (if sequence available)
        if self.lambda_temporal > 0 and predictions_sequence is not None:
            temp_loss = self.temporal_loss(predictions_sequence)
            losses['temporal_smoothness'] = temp_loss.item()
            total_loss += self.lambda_temporal * temp_loss

        losses['total_physics'] = total_loss.item()

        return total_loss, losses


class SpectralDistanceLoss(nn.Module):
    """
    Computes distance between observed and expected frequency spectrum.

    This is an alternative to frequency consistency loss that directly compares
    the full spectrum rather than just peak frequencies.
    """

    def __init__(self, sample_rate: int = 51200, n_fft: int = 2048):
        """
        Initialize spectral distance loss.

        Args:
            sample_rate: Sampling rate in Hz
            n_fft: FFT size
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.signature_db = FaultSignatureDatabase()

    def forward(
        self,
        signal: torch.Tensor,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute spectral distance loss.

        Args:
            signal: Raw vibration signal [B, 1, T] or [B, T]
            predictions: Predicted class logits [B, num_classes]
            metadata: Optional dict with 'rpm'

        Returns:
            Spectral distance loss (scalar)
        """
        batch_size = signal.shape[0]
        predicted_classes = torch.argmax(predictions, dim=1)

        # Default RPM
        rpm = 3600.0
        if metadata is not None and 'rpm' in metadata:
            rpm_val = metadata['rpm']
            if isinstance(rpm_val, torch.Tensor):
                rpm = rpm_val.cpu().numpy()
            if isinstance(rpm, np.ndarray):
                rpm = float(rpm.mean())

        # Compute observed spectrum
        if signal.dim() == 3:
            signal = signal.squeeze(1)

        fft = torch.fft.rfft(signal, n=self.n_fft, dim=-1)
        observed_spectrum = torch.abs(fft)  # [B, n_fft//2 + 1]

        # Normalize
        observed_spectrum = observed_spectrum / (torch.max(observed_spectrum, dim=-1, keepdim=True)[0] + 1e-6)

        # Frequency bins
        freq_bins = np.fft.rfftfreq(self.n_fft, d=1.0/self.sample_rate)

        # Compute expected spectrum for each sample
        total_loss = 0.0

        for i in range(batch_size):
            pred_class = predicted_classes[i].item()

            # Generate expected spectrum
            expected = self.signature_db.compute_expected_spectrum(
                pred_class, rpm, freq_bins, amplitude=1.0
            )
            expected = torch.tensor(expected, device=signal.device, dtype=torch.float32)

            # Normalize
            expected = expected / (torch.max(expected) + 1e-6)

            # MSE between observed and expected spectra
            loss_i = F.mse_loss(observed_spectrum[i], expected)
            total_loss += loss_i

        return total_loss / batch_size


def frequency_consistency_loss(
    signal: torch.Tensor,
    predictions: torch.Tensor,
    metadata: Optional[Dict[str, torch.Tensor]] = None,
    sample_rate: int = 51200
) -> torch.Tensor:
    """
    Convenience function for frequency consistency loss.

    Args:
        signal: Vibration signal [B, 1, T]
        predictions: Class predictions [B, num_classes]
        metadata: Operating conditions
        sample_rate: Sampling rate

    Returns:
        Frequency consistency loss
    """
    loss_fn = FrequencyConsistencyLoss(sample_rate=sample_rate)
    return loss_fn(signal, predictions, metadata)


def sommerfeld_consistency_loss(
    predicted_severity: torch.Tensor,
    metadata: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Convenience function for Sommerfeld consistency loss.

    Args:
        predicted_severity: Severity predictions
        metadata: Operating conditions

    Returns:
        Sommerfeld consistency loss
    """
    loss_fn = SommerfeldConsistencyLoss()
    return loss_fn(predicted_severity, metadata)


def temporal_smoothness_loss(
    predictions_sequence: torch.Tensor
) -> torch.Tensor:
    """
    Convenience function for temporal smoothness loss.

    Args:
        predictions_sequence: Sequence of predictions [T, B, num_classes]

    Returns:
        Temporal smoothness loss
    """
    loss_fn = TemporalSmoothnessLoss()
    return loss_fn(predictions_sequence)


if __name__ == "__main__":
    # Test physics loss functions
    print("=" * 60)
    print("Physics Loss Functions - Validation")
    print("=" * 60)

    # Create synthetic data
    batch_size = 4
    signal_length = 10240
    num_classes=NUM_CLASSES

    # Random signal
    signal = torch.randn(batch_size, 1, signal_length)

    # Random predictions (logits)
    predictions = torch.randn(batch_size, num_classes)

    # Metadata
    metadata = {
        'rpm': torch.tensor([3600.0] * batch_size),
        'load': torch.tensor([500.0] * batch_size),
        'viscosity': torch.tensor([0.03] * batch_size)
    }

    print("\nTesting Frequency Consistency Loss:")
    freq_loss_fn = FrequencyConsistencyLoss(sample_rate=51200, n_fft=2048)
    freq_loss = freq_loss_fn(signal, predictions, metadata)
    print(f"   Loss value: {freq_loss.item():.4f}")

    print("\nTesting Sommerfeld Consistency Loss:")
    severity_preds = torch.randn(batch_size, 4)  # 4 severity levels
    somm_loss_fn = SommerfeldConsistencyLoss()
    somm_loss = somm_loss_fn(severity_preds, metadata)
    print(f"   Loss value: {somm_loss.item():.4f}")

    print("\nTesting Temporal Smoothness Loss:")
    sequence_length = 10
    pred_sequence = torch.randn(sequence_length, batch_size, num_classes)
    temp_loss_fn = TemporalSmoothnessLoss()
    temp_loss = temp_loss_fn(pred_sequence)
    print(f"   Loss value: {temp_loss.item():.4f}")

    print("\nTesting Combined Physics Loss:")
    combined_loss_fn = PhysicalConstraintLoss(
        lambda_freq=1.0,
        lambda_sommerfeld=0.5,
        lambda_temporal=0.1
    )
    total_loss, loss_dict = combined_loss_fn(
        signal, predictions, metadata,
        severity_predictions=severity_preds,
        predictions_sequence=pred_sequence
    )
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Loss components: {loss_dict}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
