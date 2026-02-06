"""
Theory as Channel - Information & communication theory for ML.

From ML-Toolbox textbook_concepts. Treat ML pipelines as communication channels:
- Error correction via redundancy (ensemble as channel)
- Channel capacity bounds (Shannon: C = B * log2(1 + S/N))
- Compression / distillation as capacity-limited channel
"""
from .communication_theory import (
    ErrorCorrectingPredictions,
    NoiseRobustModel,
    channel_capacity,
    signal_to_noise_ratio,
    RobustMLProtocol,
)
from .information_theory import (
    entropy,
    mutual_information,
    kl_divergence,
    information_gain,
    Entropy,
    MutualInformation,
    KLDivergence,
    InformationGain,
)
from .ensemble_extras import (
    entropy_weighted_soft_weights,
    capacity_weighted_weights,
    soft_vote_median,
    soft_vote_trimmed_mean,
    diversity_weighted_weights,
    SimpleStackingEnsemble,
    build_meta_features_from_probas,
    build_meta_features_from_predictions,
)

__all__ = [
    "ErrorCorrectingPredictions",
    "NoiseRobustModel",
    "channel_capacity",
    "signal_to_noise_ratio",
    "RobustMLProtocol",
    "entropy",
    "mutual_information",
    "kl_divergence",
    "information_gain",
    "Entropy",
    "MutualInformation",
    "KLDivergence",
    "InformationGain",
    "entropy_weighted_soft_weights",
    "capacity_weighted_weights",
    "soft_vote_median",
    "soft_vote_trimmed_mean",
    "diversity_weighted_weights",
    "SimpleStackingEnsemble",
    "build_meta_features_from_probas",
    "build_meta_features_from_predictions",
]
