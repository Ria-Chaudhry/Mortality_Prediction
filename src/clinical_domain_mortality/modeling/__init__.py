"""Frozen estimators and outer-fold OOF prediction."""

from .runner import FitResult, fit_predict_fold, validate_oof_predictions
from .shap_analysis import fold_shap_aggregate

__all__ = [
    "FitResult",
    "fit_predict_fold",
    "fold_shap_aggregate",
    "validate_oof_predictions",
]
