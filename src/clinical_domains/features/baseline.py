from __future__ import annotations

import pandas as pd


def prepare_baseline_features(baseline: pd.DataFrame) -> pd.DataFrame:
    """Convert standardized baseline data to encounter-level matrix columns."""
    if baseline.empty:
        return pd.DataFrame(columns=["encounter_id"])

    if {"feature_name", "value"}.issubset(baseline.columns):
        wide = baseline.pivot_table(
            index="encounter_id", columns="feature_name", values="value", aggfunc="first"
        ).reset_index()
    else:
        wide = baseline.copy()

    renamed = {
        col: f"baseline__{col}"
        for col in wide.columns
        if col not in {"encounter_id", "patient_id"}
        and not str(col).startswith("baseline__")
    }
    return wide.rename(columns=renamed)
