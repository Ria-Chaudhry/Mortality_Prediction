from __future__ import annotations

import pandas as pd


def threshold_at_percentile(scores, percentile: float) -> float:
    return float(pd.Series(scores).quantile(1 - percentile / 100))
