"""Eight source-neutral feature matrix definitions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..errors import IntegrityError
from .construction import DomainFeatures
from .validation import assert_no_forbidden_features


def assemble_matrix(
    baseline: pd.DataFrame,
    domains: dict[str, DomainFeatures],
    matrix_name: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Join a fold's selected domains to baseline without changing row identity."""
    components = config["matrices"][matrix_name]
    frame = baseline.copy()
    expected_order = frame["cohort_visit_number"].tolist()
    for component in components:
        component_order = domains[component].frame["cohort_visit_number"].tolist()
        if component_order != expected_order:
            raise IntegrityError(
                f"{matrix_name}/{component} lost, duplicated, or reordered cohort rows"
            )
        frame = frame.merge(
            domains[component].frame,
            on="cohort_visit_number",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    if frame["cohort_visit_number"].tolist() != expected_order:
        raise IntegrityError(f"{matrix_name} changed cohort row order")
    if len(frame.columns) != len(set(frame.columns)):
        raise IntegrityError(f"{matrix_name} contains duplicate feature names")
    features = frame.set_index("cohort_visit_number", verify_integrity=True)
    assert_no_forbidden_features(features, config)
    return features
