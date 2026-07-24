from __future__ import annotations

from sklearn.metrics import brier_score_loss


def brier_score(y_true, y_score) -> float:
    return float(brier_score_loss(y_true, y_score))
