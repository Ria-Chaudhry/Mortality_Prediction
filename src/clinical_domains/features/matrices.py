from __future__ import annotations

import pandas as pd

from clinical_domains.features.baseline import prepare_baseline_features


def build_feature_matrix(
    encounters: pd.DataFrame,
    baseline: pd.DataFrame | None = None,
    event_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge standardized encounter identifiers with baseline and event-derived features."""
    matrix = encounters[["encounter_id", "patient_id"]].copy()
    if baseline is not None:
        matrix = matrix.merge(prepare_baseline_features(baseline), on="encounter_id", how="left")
    if event_features is not None:
        matrix = matrix.merge(event_features, on="encounter_id", how="left")
    return matrix


def select_domain_columns(matrix: pd.DataFrame, domains: list[str]) -> list[str]:
    prefixes = tuple(f"{domain}__" for domain in domains)
    id_columns = {"encounter_id", "patient_id", "outcome", "fold"}
    return [col for col in matrix.columns if col not in id_columns and str(col).startswith(prefixes)]
