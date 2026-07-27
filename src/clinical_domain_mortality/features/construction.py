"""Exact per-domain fold-specific feature definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..errors import IntegrityError
from ..hashing import hash_object
from .selection import ConceptSelection, feature_safe_key


@dataclass
class DomainFeatures:
    domain: str
    fold: int
    frame: pd.DataFrame
    feature_names: list[str]
    feature_hash: str


def build_fold_domain_features(
    cohort: pd.DataFrame,
    selection: ConceptSelection,
    all_qualifying_events: pd.DataFrame,
    config: dict[str, Any],
) -> DomainFeatures:
    """Construct one fold's selected columns for all frozen visits."""
    selected_keys = selection.selected["concept_key"].astype(str).tolist()
    if selection.domain == "measurements":
        frame = _measurement_features(
            cohort, selection.eligible_events, selected_keys, config
        )
    elif selection.domain == "medications":
        frame = _medication_features(
            cohort, selection.eligible_events, all_qualifying_events, selected_keys
        )
    elif selection.domain == "procedures":
        frame = _procedure_features(
            cohort, selection.eligible_events, all_qualifying_events, selected_keys
        )
    else:
        raise IntegrityError(f"Unknown feature domain: {selection.domain}")
    if frame["cohort_visit_number"].tolist() != cohort["cohort_visit_number"].tolist():
        raise IntegrityError(f"{selection.domain} feature construction changed cohort row order")
    names = [column for column in frame if column != "cohort_visit_number"]
    if len(names) != len(set(names)):
        raise IntegrityError(f"{selection.domain} contains duplicate feature names")
    expected = int(config["features"][selection.domain]["expected_count"])
    if len(names) != expected:
        raise IntegrityError(
            f"{selection.domain} produced {len(names)} features; expected {expected}"
        )
    return DomainFeatures(
        domain=selection.domain,
        fold=selection.fold,
        frame=frame,
        feature_names=names,
        feature_hash=hash_object(names),
    )


def _measurement_features(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    selected_keys: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    visit_index = pd.Index(cohort["cohort_visit_number"], name="cohort_visit_number")
    selected_events = events.loc[events["concept_key"].astype(str).isin(selected_keys)].copy()
    grouped = (
        selected_events.groupby(["cohort_visit_number", "concept_key"])["value_numeric"]
        .agg(["mean", "max", "min", "count"])
        .reset_index()
    )
    sd = (
        selected_events.groupby(["cohort_visit_number", "concept_key"])["value_numeric"]
        .std(ddof=int(config["features"]["measurements"]["sd_ddof"]))
        .fillna(float(config["features"]["measurements"]["single_value_sd"]))
        .rename("sd")
        .reset_index()
    )
    grouped = grouped.merge(sd, on=["cohort_visit_number", "concept_key"], validate="one_to_one")
    used_names: set[str] = set()
    columns: dict[str, pd.Series] = {}
    for key in selected_keys:
        safe = feature_safe_key(key)
        if safe in used_names:
            raise IntegrityError("Selected measurement concepts create duplicate feature names")
        used_names.add(safe)
        concept = grouped.loc[grouped["concept_key"].astype(str) == str(key)].set_index(
            "cohort_visit_number"
        )
        prefix = f"measurement__{safe}"
        for statistic in ("mean", "max", "min", "sd"):
            columns[f"{prefix}__{statistic}"] = concept[statistic].reindex(visit_index)
        count = concept["count"].reindex(visit_index).fillna(0).astype("int64")
        columns[f"{prefix}__count"] = count
        columns[f"{prefix}__missing"] = (count == 0).astype("int8")
    return pd.DataFrame(columns, index=visit_index).reset_index()


def _medication_features(
    cohort: pd.DataFrame,
    eligible_events: pd.DataFrame,
    all_events: pd.DataFrame,
    selected_keys: list[str],
) -> pd.DataFrame:
    visit_index = pd.Index(cohort["cohort_visit_number"], name="cohort_visit_number")
    counts = (
        eligible_events.loc[eligible_events["concept_key"].astype(str).isin(selected_keys)]
        .groupby(["cohort_visit_number", "concept_key"])
        .size()
        .rename("count")
        .reset_index()
    )
    used_names: set[str] = set()
    columns: dict[str, pd.Series] = {}
    for key in selected_keys:
        safe = feature_safe_key(key)
        if safe in used_names:
            raise IntegrityError("Selected medication concepts create duplicate feature names")
        used_names.add(safe)
        series = (
            counts.loc[counts["concept_key"].astype(str) == str(key)]
            .set_index("cohort_visit_number")["count"]
            .reindex(visit_index)
            .fillna(0)
            .astype("int64")
        )
        columns[f"medication__{safe}__exposure"] = (series > 0).astype("int8")
        columns[f"medication__{safe}__count"] = series
    all_counts = all_events.groupby(["cohort_visit_number", "concept_key"]).size()
    by_visit = all_counts.groupby(level=0)
    total = all_events.groupby("cohort_visit_number").size().reindex(visit_index).fillna(0)
    unique = (
        all_events.groupby("cohort_visit_number")["concept_key"]
        .nunique()
        .reindex(visit_index)
        .fillna(0)
    )
    repeat = by_visit.apply(lambda values: int(np.maximum(values.to_numpy() - 1, 0).sum()))
    first = (
        all_events.groupby("cohort_visit_number")["hours_from_start"]
        .min()
        .reindex(visit_index)
    )
    columns["any_drug_24h"] = (total > 0).astype("int8")
    columns["unique_drug_count_24h"] = unique.astype("int64")
    columns["repeat_drug_exposure_count_24h"] = (
        repeat.reindex(visit_index).fillna(0).astype("int64")
    )
    columns["time_to_first_drug_in_hours"] = first
    return pd.DataFrame(columns, index=visit_index).reset_index()


def _procedure_features(
    cohort: pd.DataFrame,
    eligible_events: pd.DataFrame,
    all_events: pd.DataFrame,
    selected_keys: list[str],
) -> pd.DataFrame:
    visit_index = pd.Index(cohort["cohort_visit_number"], name="cohort_visit_number")
    counts = (
        eligible_events.loc[eligible_events["concept_key"].astype(str).isin(selected_keys)]
        .groupby(["cohort_visit_number", "concept_key"])
        .size()
        .rename("count")
        .reset_index()
    )
    used_names: set[str] = set()
    columns: dict[str, pd.Series] = {}
    for key in selected_keys:
        safe = feature_safe_key(key)
        if safe in used_names:
            raise IntegrityError("Selected procedure concepts create duplicate feature names")
        used_names.add(safe)
        series = (
            counts.loc[counts["concept_key"].astype(str) == str(key)]
            .set_index("cohort_visit_number")["count"]
            .reindex(visit_index)
            .fillna(0)
            .astype("int64")
        )
        columns[f"procedure__{safe}__exposure"] = (series > 0).astype("int8")
        columns[f"procedure__{safe}__count"] = series
    total = all_events.groupby("cohort_visit_number").size().reindex(visit_index).fillna(0)
    unique = (
        all_events.groupby("cohort_visit_number")["concept_key"]
        .nunique()
        .reindex(visit_index)
        .fillna(0)
    )
    columns["any_procedure_24h"] = (total > 0).astype("int8")
    columns["unique_procedure_count_24h"] = unique.astype("int64")
    columns["procedure_count_total_24h"] = total.astype("int64")
    return pd.DataFrame(columns, index=visit_index).reset_index()
