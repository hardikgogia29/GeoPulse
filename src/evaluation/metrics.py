
from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|e| / sum|actual|.

    Returns NaN when the window has no demand at all, rather than 0 or infinity.
    """
    denominator = float(np.sum(np.abs(actual)))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean signed error - positive means the model over-predicts."""
    return float(np.mean(predicted - actual))


def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "wape": wape(actual, predicted),
        "bias": bias(actual, predicted),
        "n": int(actual.size),
        "actual_mean": float(np.mean(actual)),
        "zero_share": float(np.mean(actual == 0)),
    }


def hotspot_f1(
    actual: np.ndarray, predicted: np.ndarray, groups: np.ndarray, top_share: float = 0.10
) -> dict[str, float]:
    """Per-timestamp top-k overlap between actual and predicted demand.

    `groups` identifies the timestamp each row belongs to. For every timestamp the
    top `top_share` of regions by actual demand is the truth set and the top
    `top_share` by predicted demand is the prediction; precision/recall/F1 are
    micro-averaged over timestamps.
    """
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    order = np.argsort(groups, kind="stable")
    actual, predicted, groups = actual[order], predicted[order], np.asarray(groups)[order]
    boundaries = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True])

    true_positive = false_positive = false_negative = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        size = hi - lo
        k = max(1, int(round(size * top_share)))
        actual_top = set(np.argpartition(-actual[lo:hi], k - 1)[:k])
        predicted_top = set(np.argpartition(-predicted[lo:hi], k - 1)[:k])
        overlap = len(actual_top & predicted_top)
        true_positive += overlap
        false_positive += len(predicted_top) - overlap
        false_negative += len(actual_top) - overlap

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"hotspot_precision": precision, "hotspot_recall": recall, "hotspot_f1": f1}
