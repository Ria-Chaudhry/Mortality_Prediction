from __future__ import annotations

import pandas as pd


def top_variance_features(frame: pd.DataFrame, max_features: int = 200) -> list[str]:
    """Simple deterministic feature filter intended for exploratory baselines."""
    numeric = frame.select_dtypes(include="number")
    variances = numeric.var(numeric_only=True).sort_values(ascending=False)
    return list(variances.head(max_features).index)
