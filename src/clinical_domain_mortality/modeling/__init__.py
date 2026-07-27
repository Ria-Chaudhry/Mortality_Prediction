"""Frozen estimators and outer-fold OOF prediction."""

from .runner import FitResult, fit_predict_fold, validate_oof_predictions

__all__ = ["FitResult", "fit_predict_fold", "validate_oof_predictions"]
