from __future__ import annotations

import pandas as pd


def build_mortality_labels(
    encounters: pd.DataFrame,
    mortality: pd.DataFrame,
    outcome_col: str = "died",
) -> pd.DataFrame:
    """Attach binary mortality labels to standardized encounters."""
    required = {"encounter_id"}
    if not required.issubset(encounters.columns) or not required.issubset(mortality.columns):
        raise ValueError("Both encounters and mortality must include encounter_id")

    label_cols = ["encounter_id", outcome_col]
    labels = encounters[["encounter_id", "patient_id"]].merge(
        mortality[label_cols], on="encounter_id", how="left"
    )
    labels[outcome_col] = labels[outcome_col].fillna(False).astype(int)
    return labels
