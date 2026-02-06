"""
Communication Theory - Extended from Claude Shannon

Implements:
- Error Correction for Robust ML
- Channel Capacity
- Noise Reduction
- Robust Communication Protocols

Source: ML-Toolbox ml_toolbox.textbook_concepts.communication_theory
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ErrorCorrectingPredictions:
    """
    Error-correcting predictions using redundancy
    """

    def __init__(self, redundancy_factor: int = 3):
        """
        Initialize error correction

        Args:
            redundancy_factor: Number of redundant predictions
        """
        self.redundancy_factor = redundancy_factor

    def correct_predictions(
        self,
        predictions: np.ndarray,
        method: str = "majority_vote",
    ) -> np.ndarray:
        """
        Correct predictions using redundancy

        Args:
            predictions: Array of shape (n_samples, redundancy_factor)
            method: 'majority_vote', 'median', 'mean'

        Returns:
            Corrected predictions
        """
        if method == "majority_vote":
            # Majority vote for classification
            from scipy.stats import mode

            corrected, _ = mode(predictions, axis=1)
            return corrected.flatten()

        elif method == "median":
            # Median for regression
            return np.median(predictions, axis=1)

        elif method == "mean":
            # Mean for regression
            return np.mean(predictions, axis=1)

        else:
            raise ValueError(f"Unknown method: {method}")

    def robust_predict(
        self,
        models: List[Any],
        X: np.ndarray,
        method: str = "majority_vote",
    ) -> np.ndarray:
        """
        Robust prediction using multiple models

        Args:
            models: List of models
            X: Input data
            method: Correction method

        Returns:
            Corrected predictions
        """
        predictions = []
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        return self.correct_predictions(predictions, method=method)


class NoiseRobustModel:
    """
    Model robust to input noise
    """

    def __init__(self, base_model: Any, noise_level: float = 0.1):
        """
        Initialize noise-robust model

        Args:
            base_model: Base model
            noise_level: Noise level for training
        """
        self.base_model = base_model
        self.noise_level = noise_level

    def fit(self, X: np.ndarray, y: np.ndarray, n_augmentations: int = 5):
        """
        Train with noise augmentation

        Args:
            X: Training features
            y: Training labels
            n_augmentations: Number of noise augmentations per sample
        """
        # Augment data with noise
        X_augmented = [X]
        y_augmented = [y]

        for _ in range(n_augmentations):
            noise = np.random.normal(0, self.noise_level, X.shape)
            X_augmented.append(X + noise)
            y_augmented.append(y)

        X_augmented = np.vstack(X_augmented)
        y_augmented = np.concatenate(y_augmented)

        # Train on augmented data
        self.base_model.fit(X_augmented, y_augmented)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using base model"""
        return self.base_model.predict(X)


def channel_capacity(
    signal_power: float, noise_power: float, bandwidth: float = 1.0
) -> float:
    """
    Calculate channel capacity (Shannon's theorem)

    C = B * log2(1 + S/N)

    Args:
        signal_power: Signal power
        noise_power: Noise power
        bandwidth: Channel bandwidth

    Returns:
        Channel capacity (bits per second)
    """
    snr = signal_power / (noise_power + 1e-10)
    return bandwidth * np.log2(1 + snr)


def signal_to_noise_ratio(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate signal-to-noise ratio

    Args:
        signal: Signal values
        noise: Noise values

    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return float("inf")

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    return snr_db


class RobustMLProtocol:
    """
    Robust ML protocol with error detection and correction
    """

    def __init__(self, error_threshold: float = 0.1):
        """
        Initialize robust protocol

        Args:
            error_threshold: Error threshold for detection
        """
        self.error_threshold = error_threshold

    def detect_errors(
        self,
        predictions: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Detect potential errors in predictions

        Args:
            predictions: Model predictions
            confidence_scores: Optional confidence scores

        Returns:
            Boolean array indicating potential errors
        """
        errors = np.zeros(len(predictions), dtype=bool)

        if confidence_scores is not None:
            # Low confidence indicates potential error
            errors = confidence_scores < (1 - self.error_threshold)
        else:
            # Use prediction variance as proxy
            # (for ensemble predictions)
            if len(predictions.shape) > 1:
                variance = np.var(predictions, axis=1)
                errors = variance > self.error_threshold

        return errors

    def correct_errors(
        self,
        predictions: np.ndarray,
        error_mask: np.ndarray,
        method: str = "fallback",
    ) -> np.ndarray:
        """
        Correct detected errors

        Args:
            predictions: Original predictions
            error_mask: Boolean mask of errors
            method: Correction method ('fallback', 'smooth', 'interpolate')

        Returns:
            Corrected predictions
        """
        corrected = predictions.copy()

        if method == "fallback":
            # Use mean/median as fallback
            if len(np.unique(predictions)) < 20:  # Classification
                fallback = np.bincount(predictions.astype(int)).argmax()
            else:  # Regression
                fallback = np.median(predictions)
            corrected[error_mask] = fallback

        elif method == "smooth":
            # Smooth predictions using neighbors
            from scipy.ndimage import gaussian_filter1d

            corrected = gaussian_filter1d(predictions.astype(float), sigma=1.0)

        elif method == "interpolate":
            # Interpolate from non-error predictions
            from scipy.interpolate import interp1d

            valid_indices = np.where(~error_mask)[0]
            if len(valid_indices) > 1:
                f = interp1d(
                    valid_indices,
                    predictions[valid_indices],
                    kind="linear",
                    fill_value="extrapolate",
                )
                error_indices = np.where(error_mask)[0]
                corrected[error_indices] = f(error_indices)

        return corrected
