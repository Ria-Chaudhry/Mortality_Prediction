from __future__ import annotations

import numpy as np


def net_benefit(y_true, y_score, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_score) >= threshold
    n = len(y_true)
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    odds = threshold / (1 - threshold)
    return float((true_positive / n) - (false_positive / n) * odds)
