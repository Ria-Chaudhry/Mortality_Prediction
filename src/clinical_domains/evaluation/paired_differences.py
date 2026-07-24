from __future__ import annotations

import pandas as pd


def paired_difference(metrics: pd.DataFrame, left: str, right: str, metric: str) -> float:
    subset = metrics.loc[metrics["metric"] == metric]
    left_value = subset.loc[subset["domain_set"] == left, "estimate"].iloc[0]
    right_value = subset.loc[subset["domain_set"] == right, "estimate"].iloc[0]
    return float(right_value - left_value)
