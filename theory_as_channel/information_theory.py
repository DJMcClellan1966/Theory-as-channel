"""
Information Theory (Mathematics for ML - Deisenroth et al.)

Implements:
- Entropy
- Mutual Information
- KL Divergence
- Information Gain

Source: ML-Toolbox ml_toolbox.textbook_concepts.information_theory
"""
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def entropy(probabilities: np.ndarray, base: float = 2.0) -> float:
    """
    Shannon Entropy

    H(X) = -Σ p(x) * log(p(x))

    Parameters
    ----------
    probabilities : array
        Probability distribution
    base : float
        Logarithm base

    Returns
    -------
    entropy : float
        Entropy value
    """
    probabilities = np.asarray(probabilities)
    probabilities = probabilities[probabilities > 0]  # Remove zeros

    if len(probabilities) == 0:
        return 0.0

    return -np.sum(probabilities * np.log(probabilities) / np.log(base))


def mutual_information(X: np.ndarray, Y: np.ndarray, n_bins: int = 10) -> float:
    """
    Mutual Information

    I(X; Y) = H(X) + H(Y) - H(X, Y)

    Parameters
    ----------
    X : array
        First variable
    Y : array
        Second variable
    n_bins : int
        Number of bins for discretization

    Returns
    -------
    mi : float
        Mutual information
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()

    # Discretize
    X_binned = np.digitize(X, np.linspace(X.min(), X.max(), n_bins))
    Y_binned = np.digitize(Y, np.linspace(Y.min(), Y.max(), n_bins))

    # Joint distribution
    joint_counts = np.zeros((n_bins, n_bins))
    for x, y in zip(X_binned, Y_binned):
        if 0 <= x < n_bins and 0 <= y < n_bins:
            joint_counts[x, y] += 1

    joint_probs = joint_counts / joint_counts.sum()

    # Marginal distributions
    X_probs = joint_probs.sum(axis=1)
    Y_probs = joint_probs.sum(axis=0)

    # Entropies
    H_X = entropy(X_probs)
    H_Y = entropy(Y_probs)
    H_XY = entropy(joint_probs.flatten())

    # Mutual information
    return H_X + H_Y - H_XY


def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Kullback-Leibler Divergence

    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

    Parameters
    ----------
    P : array
        True distribution
    Q : array
        Approximate distribution

    Returns
    -------
    kl : float
        KL divergence
    """
    P = np.asarray(P)
    Q = np.asarray(Q)

    # Normalize
    P = P / P.sum()
    Q = Q / Q.sum()

    # Avoid division by zero
    mask = P > 0
    P = P[mask]
    Q = Q[mask]

    if len(P) == 0:
        return 0.0

    # Avoid log(0)
    Q = np.maximum(Q, 1e-10)

    return np.sum(P * np.log(P / Q))


def information_gain(y: np.ndarray, y_after_split: List[np.ndarray]) -> float:
    """
    Information Gain

    IG = H(Y) - Σ (|S_i| / |S|) * H(Y_i)

    Parameters
    ----------
    y : array
        Original target values
    y_after_split : list of arrays
        Target values after split

    Returns
    -------
    ig : float
        Information gain
    """
    y = np.asarray(y)

    # Original entropy
    y_counts = np.bincount(y.astype(int))
    y_probs = y_counts / y_counts.sum()
    H_Y = entropy(y_probs)

    # Weighted entropy after split
    total_samples = len(y)
    weighted_entropy = 0.0

    for y_split in y_after_split:
        if len(y_split) == 0:
            continue

        split_counts = np.bincount(y_split.astype(int))
        split_probs = split_counts / split_counts.sum()
        H_split = entropy(split_probs)

        weight = len(y_split) / total_samples
        weighted_entropy += weight * H_split

    return H_Y - weighted_entropy


class Entropy:
    """Entropy class (wrapper)"""

    @staticmethod
    def shannon(probabilities: np.ndarray, base: float = 2.0) -> float:
        """Shannon entropy"""
        return entropy(probabilities, base)

    @staticmethod
    def conditional(X: np.ndarray, Y: np.ndarray, n_bins: int = 10) -> float:
        """
        Conditional Entropy

        H(X|Y) = H(X, Y) - H(Y)
        """
        X = np.asarray(X).ravel()
        Y = np.asarray(Y).ravel()

        # Discretize
        X_binned = np.digitize(X, np.linspace(X.min(), X.max(), n_bins))
        Y_binned = np.digitize(Y, np.linspace(Y.min(), Y.max(), n_bins))

        # Joint distribution
        joint_counts = np.zeros((n_bins, n_bins))
        for x, y in zip(X_binned, Y_binned):
            if 0 <= x < n_bins and 0 <= y < n_bins:
                joint_counts[x, y] += 1

        joint_probs = joint_counts / joint_counts.sum()
        Y_probs = joint_probs.sum(axis=0)

        H_XY = entropy(joint_probs.flatten())
        H_Y = entropy(Y_probs)

        return H_XY - H_Y


class MutualInformation:
    """Mutual Information class (wrapper)"""

    @staticmethod
    def compute(X: np.ndarray, Y: np.ndarray, n_bins: int = 10) -> float:
        """Compute mutual information"""
        return mutual_information(X, Y, n_bins)


class KLDivergence:
    """KL Divergence class (wrapper)"""

    @staticmethod
    def compute(P: np.ndarray, Q: np.ndarray) -> float:
        """Compute KL divergence"""
        return kl_divergence(P, Q)


class InformationGain:
    """Information Gain class (wrapper)"""

    @staticmethod
    def compute(y: np.ndarray, y_after_split: List[np.ndarray]) -> float:
        """Compute information gain"""
        return information_gain(y, y_after_split)
