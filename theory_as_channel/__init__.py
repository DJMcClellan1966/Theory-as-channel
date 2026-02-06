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
]
