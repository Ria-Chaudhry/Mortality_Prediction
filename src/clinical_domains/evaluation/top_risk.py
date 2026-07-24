from __future__ import annotations

import pandas as pd


def top_risk_capture(y_true, y_score, percentile: float = 5) -> float:
    frame = pd.DataFrame({"y_true": y_true, "y_score": y_score}).sort_values(
        "y_score", ascending=False
    )
    n_top = max(1, round(len(frame) * percentile / 100))
    total_events = frame["y_true"].sum()
    if total_events == 0:
        return 0.0
    return float(frame.head(n_top)["y_true"].sum() / total_events)
