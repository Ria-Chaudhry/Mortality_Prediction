from __future__ import annotations

import pandas as pd


def build_cohort(encounters: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    """Apply basic dataset-agnostic encounter eligibility rules."""
    cohort = encounters.copy()
    if "age" in cohort.columns:
        cohort = cohort.loc[cohort["age"].fillna(-1) >= min_age]
    if {"admit_time", "discharge_time"}.issubset(cohort.columns):
        cohort["admit_time"] = pd.to_datetime(cohort["admit_time"])
        cohort["discharge_time"] = pd.to_datetime(cohort["discharge_time"])
        cohort = cohort.loc[cohort["discharge_time"] >= cohort["admit_time"]]
    return cohort.reset_index(drop=True)
