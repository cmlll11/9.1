"""Logit features and lightweight Ridge probes for robustness prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


FEATURE_NAMES = tuple([f"logit_gap_{index}_vs_target" for index in range(1, 10)] + ["top1_top2_margin", "entropy"])

# Target-free schema used by Stage 1C.  Sorting logits removes dependence on
# the numeric identity of a possible backdoor target class.
UNTARGETED_FEATURE_NAMES = tuple(
    [f"top1_gap_top{index}" for index in range(2, 11)] + ["entropy", "logit_std"]
)


def target_feature_names(target: int, num_classes: int = 10) -> tuple[str, ...]:
    """Return the 11-feature schema for one target class.

    The first nine entries are the logit gaps ``z_k-z_target`` for all
    non-target classes in ascending class-id order.  The target itself is not
    included because that column would be identically zero.
    """

    target = int(target)
    num_classes = int(num_classes)
    if num_classes < 2 or not 0 <= target < num_classes:
        raise ValueError("target must be a valid class and num_classes must be at least 2")
    return tuple(
        [f"logit_gap_class_{index}_vs_target_{target}" for index in range(num_classes) if index != target]
        + ["top1_top2_margin", "entropy"]
    )


def target_conditioned_logits_features(logits: Tensor, target: int) -> Tensor:
    """Extract target-relative logits features for any valid target class.

    The nine non-target columns are ``z_k - z_target`` in ascending class-id
    order.  The final two values are the global top-1/top-2 margin and
    softmax entropy.
    """

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(f"expected logits with shape [N, K] and K >= 2, got {tuple(logits.shape)}")
    target = int(target)
    if target < 0 or target >= logits.shape[1]:
        raise ValueError(f"target {target} is outside logits dimension {logits.shape[1]}")
    gaps = torch.cat(
        [logits[:, index : index + 1] - logits[:, target : target + 1]
         for index in range(logits.shape[1]) if index != target],
        dim=1,
    )
    top2 = logits.topk(k=2, dim=1).values
    probabilities = logits.softmax(dim=1)
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
    return torch.cat((gaps, (top2[:, 0] - top2[:, 1]).unsqueeze(1), entropy), dim=1)


def target_margin(logits: Tensor, target: int) -> Tensor:
    """Return ``max(non-target logits) - target logit`` for each sample."""

    other = logits.clone()
    other[:, int(target)] = float("-inf")
    return other.max(dim=1).values - logits[:, int(target)]


def logits_features(logits: Tensor, target: int) -> Tensor:
    """Extract the fixed 11-dimensional feature vector from classifier logits.

    The first nine features are ``z_k - z_target`` for non-target CIFAR-10
    classes ``k=1..9``.  The final two are the top-1/top-2 logit gap and the
    softmax entropy.  No gradients are required for feature extraction.
    """

    target = int(target)
    gaps = logits[:, 1:] - logits[:, target : target + 1] if target == 0 else torch.cat(
        (logits[:, :target] - logits[:, target : target + 1], logits[:, target + 1 :] - logits[:, target : target + 1]),
        dim=1,
    )
    # The protocol names gaps for classes 1..9, so target=0 is the intended
    # setting.  Keep a clear error for accidental use with another target.
    if target != 0 or gaps.shape[1] != 9:
        raise ValueError("the pilot feature schema is defined for target=0")
    top2 = logits.topk(k=2, dim=1).values
    probabilities = logits.softmax(dim=1)
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return torch.cat((gaps, (top2[:, 0] - top2[:, 1]).unsqueeze(1), entropy.unsqueeze(1)), dim=1)


def untargeted_margin(logits: Tensor) -> Tensor:
    """Return the top-1/top-2 logit margin for each sample.

    This is the target-free baseline: a large margin generally indicates that
    a larger perturbation is needed to change the current prediction.
    """

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(f"expected logits with shape [N, K] and K >= 2, got {tuple(logits.shape)}")
    top2 = logits.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def untargeted_logits_features(logits: Tensor) -> Tensor:
    """Extract the fixed 11-dimensional target-free logit feature vector.

    The first nine values are ``z_(1)-z_(j)`` for ``j=2..10`` after sorting
    logits in descending order.  The final two values are softmax entropy and
    the standard deviation of the raw logits.  No class label, target class,
    or intermediate feature is used.
    """

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(f"expected logits with shape [N, K] and K >= 2, got {tuple(logits.shape)}")
    sorted_logits = logits.sort(dim=1, descending=True).values
    gaps = sorted_logits[:, :1] - sorted_logits[:, 1:]
    probabilities = logits.softmax(dim=1)
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
    logit_std = logits.std(dim=1, unbiased=False, keepdim=True)
    return torch.cat((gaps, entropy, logit_std), dim=1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks without requiring SciPy."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Spearman rank correlation, returning NaN for constant inputs."""

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.size < 2 or np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
        return float("nan")
    return float(np.corrcoef(_average_ranks(actual), _average_ranks(predicted))[0, 1])


@dataclass
class RidgeProbe:
    """A standardized linear regression model predicting estimated radius."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    weight: np.ndarray
    bias: float
    alpha: float = 1.0
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        alpha: float = 1.0,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> "RidgeProbe":
        """Fit Ridge regression with an unregularized intercept.

        ``features`` must contain finite rows with shape ``[N, 11]`` and
        ``labels`` must contain the corresponding finite coarse PGD radii.
        Standardization statistics are learned only from this fitting data.
        """

        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        feature_names = tuple(feature_names)
        if x.ndim != 2 or x.shape[1] != len(feature_names) or len(x) != len(y):
            raise ValueError("features must have shape [N, 11] and match labels")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Ridge training data must be finite")
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-12] = 1.0
        standardized = (x - mean) / std
        design = np.column_stack((standardized, np.ones(len(standardized))))
        regularizer = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
        regularizer[-1, -1] = 0.0
        parameters = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        return cls(mean, std, parameters[:-1], float(parameters[-1]), float(alpha), feature_names)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict radius for an array with shape ``[N, 11]``."""

        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("features must have shape [N, 11]")
        return ((x - self.feature_mean) / self.feature_std) @ self.weight + self.bias

    def save(self, path: str | Path) -> None:
        """Save weights, scaler and schema in a portable NumPy archive."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, feature_mean=self.feature_mean, feature_std=self.feature_std, weight=self.weight,
                 bias=np.asarray(self.bias), alpha=np.asarray(self.alpha), feature_names=np.asarray(self.feature_names))

    @classmethod
    def load(cls, path: str | Path) -> "RidgeProbe":
        """Load and validate a previously saved Ridge probe."""

        archive = np.load(path, allow_pickle=False)
        names = tuple(str(item) for item in archive["feature_names"].tolist())
        if archive["weight"].shape[0] != len(names):
            raise ValueError("probe archive has inconsistent feature dimensions")
        return cls(
            archive["feature_mean"],
            archive["feature_std"],
            archive["weight"],
            float(archive["bias"]),
            float(archive["alpha"]),
            names,
        )
