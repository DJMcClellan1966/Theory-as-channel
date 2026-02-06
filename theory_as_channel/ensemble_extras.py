"""
Unusual / underused ensemble methods for theory-as-channel and toolbox use.

- Entropy-weighted soft voting (confident models count more)
- Median-of-probas (robust to one bad channel)
- Channel-capacity weighting (Shannon-style weight by C from accuracy)
- Trimmed-mean soft (drop extreme probas per class)
- Simple stacking (meta-learner on base predictions/probas)
- Diversity-weighted voting (upweight models that disagree with the crowd when accurate)
"""
import numpy as np
from typing import List, Optional, Any, Tuple



def _proba_entropy(proba: np.ndarray, base: float = 2.0) -> float:
    """Mean entropy of rows of proba (each row = distribution)."""
    proba = np.asarray(proba)
    proba = np.clip(proba, 1e-12, 1.0)
    H = -np.sum(proba * np.log(proba) / np.log(base), axis=1)
    return float(np.mean(H))


def entropy_weighted_soft_weights(probas: np.ndarray) -> np.ndarray:
    """
    Per-model weights = 1 / (1 + mean_entropy). Confident (low-entropy) models get higher weight.
    Unusual: rarely used in standard sklearn ensembles.
    """
    # probas: (n_samples, n_models, n_classes)
    n_models = probas.shape[1]
    w = np.zeros(n_models)
    for m in range(n_models):
        w[m] = 1.0 / (1.0 + _proba_entropy(probas[:, m, :]))
    w = w / (w.sum() + 1e-12)
    return w


def capacity_weighted_weights(accuracies: List[float]) -> np.ndarray:
    """
    Weight each channel by Shannon-style capacity: C = log2(1 + p/(1-p)) for p = accuracy.
    Treats accuracy as "signal rate" vs error rate. Unusual in ensemble practice.
    """
    acc = np.asarray(accuracies, dtype=float)
    p = np.clip(acc, 1e-6, 1 - 1e-6)
    # C proportional to log2(1 + SNR) with SNR ~ p/(1-p)
    C = np.log2(1 + p / (1 - p))
    C = C / (C.sum() + 1e-12)
    return C


def soft_vote_median(probas: np.ndarray) -> np.ndarray:
    """
    Robust soft voting: median probability per class across models, then argmax.
    Reduces impact of one badly calibrated or adversarial channel. Unusual (mean is standard).
    """
    # probas: (n_samples, n_models, n_classes)
    med = np.median(probas, axis=1)
    return np.argmax(med, axis=1)


def soft_vote_trimmed_mean(probas: np.ndarray, trim: int = 1) -> np.ndarray:
    """
    Trimmed mean of probabilities: drop `trim` smallest and `trim` largest per class across models, then average.
    Robust to a few extreme channels. Unusual.
    """
    # probas: (n_samples, n_models, n_classes)
    n_samples, n_models, n_classes = probas.shape
    out = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        for c in range(n_classes):
            col = np.sort(probas[i, :, c])
            if 2 * trim < len(col):
                out[i, c] = np.mean(col[trim:-trim])
            else:
                out[i, c] = np.mean(col)
    return np.argmax(out, axis=1)


def diversity_weighted_weights(predictions: np.ndarray, accuracies: List[float]) -> np.ndarray:
    """
    Weight = accuracy * (1 - mean_agreement_with_others). Upweights accurate models that disagree.
    "Diverse" correct voters can break ties. Unusual (most methods downweight disagreement).
    """
    preds = np.asarray(predictions)
    n_samples, n_models = preds.shape
    acc = np.asarray(accuracies)
    agreement = np.zeros(n_models)
    for m in range(n_models):
        # Agreement of model m with others (excluding self)
        rest = [j for j in range(n_models) if j != m]
        same = np.mean(np.array([preds[:, m] == preds[:, j] for j in rest]), axis=0)
        agreement[m] = np.mean(same)
    # Weight: accurate and disagreeing
    w = acc * (1.0 - agreement)
    w = np.maximum(w, 0)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones(n_models) / n_models
    return w


class SimpleStackingEnsemble:
    """
    Stacking: meta-learner trained on base model outputs (probas or one-hot).
    ML-Toolbox has full StackingEnsemble with CV; this is a minimal version for theory-as-channel.
    """

    def __init__(self, meta_model: Any = None, use_proba: bool = True):
        if meta_model is None:
            try:
                from sklearn.linear_model import LogisticRegression
                meta_model = LogisticRegression(max_iter=500)
            except ImportError:
                meta_model = None
        self.meta_model = meta_model
        self.use_proba = use_proba

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> "SimpleStackingEnsemble":
        """
        X_meta: (n_samples, n_features) where each row is concatenated base outputs.
                If use_proba, n_features = n_models * n_classes; else n_models (one-hot or class indices).
        """
        if self.meta_model is None:
            raise RuntimeError("No meta_model and sklearn not available")
        self.meta_model.fit(X_meta, y)
        return self

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        return self.meta_model.predict(X_meta)


def build_meta_features_from_probas(probas: np.ndarray) -> np.ndarray:
    """Flatten probas to (n_samples, n_models * n_classes) for stacking."""
    n_samples, n_models, n_classes = probas.shape
    return probas.reshape(n_samples, n_models * n_classes)


def build_meta_features_from_predictions(predictions: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot encode predictions to (n_samples, n_models * n_classes) for stacking."""
    n_samples, n_models = predictions.shape
    onehot = np.zeros((n_samples, n_models * n_classes))
    for m in range(n_models):
        for c in range(n_classes):
            onehot[:, m * n_classes + c] = (predictions[:, m] == c).astype(float)
    return onehot
