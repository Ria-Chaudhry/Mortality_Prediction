"""OOF performance, calibration, utility, and paired analyses."""

from .analysis import EvaluationResult, evaluate_predictions
from .bootstrap import bootstrap_indices
from .metrics import binary_metrics

__all__ = ["EvaluationResult", "binary_metrics", "bootstrap_indices", "evaluate_predictions"]
