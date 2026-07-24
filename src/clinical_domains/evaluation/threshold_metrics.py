from __future__ import annotations

from sklearn.metrics import precision_score, recall_score


def precision_at_threshold(y_true, y_score, threshold: float) -> float:
    return float(precision_score(y_true, [score >= threshold for score in y_score], zero_division=0))


def recall_at_threshold(y_true, y_score, threshold: float) -> float:
    return float(recall_score(y_true, [score >= threshold for score in y_score], zero_division=0))
