from __future__ import annotations

from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def average_precision(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))
