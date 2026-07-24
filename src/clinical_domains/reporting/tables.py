from __future__ import annotations

import pandas as pd

from clinical_domains.evaluation.calibration import brier_score
from clinical_domains.evaluation.discrimination import auroc, average_precision


def metric_table(
    predictions: pd.DataFrame,
    domain_set: str = "all_domains",
    model_name: str = "logistic_regression",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    y_true = predictions[outcome_col]
    y_score = predictions["y_score"]
    rows = [
        {"metric": "auroc", "estimate": auroc(y_true, y_score)},
        {"metric": "average_precision", "estimate": average_precision(y_true, y_score)},
        {"metric": "brier_score", "estimate": brier_score(y_true, y_score)},
    ]
    table = pd.DataFrame(rows)
    table["domain_set"] = domain_set
    table["model_name"] = model_name
    return table[["model_name", "domain_set", "metric", "estimate"]]
