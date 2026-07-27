"""Frozen acute-care cohort, outcome, and baseline construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..adapters import StandardizedData
from ..config import PROJECT_ROOT
from ..errors import IntegrityError
from ..hashing import hash_frame


@dataclass
class CohortResult:
    cohort: pd.DataFrame
    baseline: pd.DataFrame
    attrition: pd.DataFrame
    cohort_hash: str
    row_order_hash: str


def build_cohort(data: StandardizedData, config: dict[str, Any]) -> CohortResult:
    """Apply the landmark design once and freeze the reusable cohort."""
    rules = config["cohort"]
    patients = data.tables["patients"].copy()
    encounters = data.tables["encounters"].copy()
    deaths = data.tables["deaths"].copy()
    attrition: list[dict[str, Any]] = []

    frame = encounters.merge(patients, on="patient_id", how="left", validate="many_to_one")
    _record(attrition, "source encounters", frame)
    if frame["birth_datetime"].isna().any():
        raise IntegrityError("An encounter has no matching patient birth date")

    frame["age"] = (
        (frame["start_datetime"] - frame["birth_datetime"]).dt.total_seconds()
        / (365.2425 * 24 * 3600)
    )
    eligible = frame["age"].between(rules["min_age_years"], rules["max_age_years"], inclusive="both")
    frame = frame.loc[eligible].copy()
    _record(attrition, "adult age range", frame)

    acute_types = {str(value).casefold() for value in rules["acute_visit_types"]}
    frame = frame.loc[frame["visit_type"].str.casefold().isin(acute_types)].copy()
    _record(attrition, "acute encounter type", frame)

    excluded_elective = {
        str(value).strip().casefold() for value in rules["excluded_elective_values"]
    }
    elective_text = frame["elective"].astype("string").str.strip().str.casefold()
    frame = frame.loc[~elective_text.isin(excluded_elective)].copy()
    _record(attrition, "non-elective", frame)

    frame = frame.merge(deaths, on="patient_id", how="left", validate="many_to_one")
    frame["landmark_datetime"] = frame["start_datetime"] + pd.to_timedelta(
        rules["landmark_hours"], unit="h"
    )
    frame["outcome_horizon_datetime"] = frame["start_datetime"] + pd.to_timedelta(
        rules["outcome_horizon_days"], unit="D"
    )
    frame["predictor_end_datetime"] = frame[["end_datetime", "landmark_datetime"]].min(axis=1)
    frame["predictor_end_datetime"] = frame["predictor_end_datetime"].fillna(
        frame["landmark_datetime"]
    )
    frame["short_visit"] = frame["end_datetime"].notna() & (
        frame["end_datetime"] < frame["landmark_datetime"]
    )
    if not rules["retain_short_visits"]:
        frame = frame.loc[~frame["short_visit"]].copy()
    _record(attrition, "short-visit policy", frame)

    early_death = frame["death_datetime"].notna() & (
        frame["death_datetime"] <= frame["landmark_datetime"]
    )
    frame = frame.loc[~early_death].copy()
    _record(attrition, "alive after 24-hour landmark", frame)

    frame["outcome"] = (
        frame["death_datetime"].notna()
        & (frame["death_datetime"] > frame["landmark_datetime"])
        & (frame["death_datetime"] <= frame["outcome_horizon_datetime"])
    ).astype("int8")
    verified = (
        (frame["outcome"] == 1)
        | (
            frame["death_datetime"].notna()
            & (frame["death_datetime"] > frame["outcome_horizon_datetime"])
        )
        | (
            frame["followup_end_datetime"].notna()
            & (frame["followup_end_datetime"] >= frame["outcome_horizon_datetime"])
        )
    )
    if rules["require_verified_followup"]:
        frame = frame.loc[verified].copy()
    frame["followup_verified_30d"] = verified.loc[frame.index].astype("int8")
    _record(attrition, "verified 30-day follow-up", frame)

    frame = _mimic_subsample(frame, config)
    _record(attrition, "deterministic dataset subsample", frame)
    if frame["visit_id"].duplicated().any():
        raise IntegrityError("Eligible cohort has duplicate visit identifiers")

    frame = frame.sort_values(
        ["start_datetime", "patient_id", "visit_id"], kind="stable"
    ).reset_index(drop=True)
    frame.insert(0, "cohort_visit_number", np.arange(1, len(frame) + 1, dtype=np.int64))
    prior = _prior_features(frame, encounters, data.tables["diagnoses"], rules, config)
    frame = frame.merge(prior, on="cohort_visit_number", validate="one_to_one", how="left")
    for column in ("prior_visit_count", "prior_acute_visit_count", "prior_charlson_score"):
        frame[column] = frame[column].fillna(0).astype("int64")
    frame["prior_visit_indicator"] = (frame["prior_visit_count"] > 0).astype("int8")

    expected = rules.get("expected_counts")
    if config.get("paper_run") and expected:
        observed = {
            "visits": len(frame),
            "patients": int(frame["patient_id"].nunique()),
            "deaths": int(frame["outcome"].sum()),
        }
        if observed != expected:
            raise IntegrityError(f"Paper cohort count mismatch: expected={expected}, observed={observed}")

    baseline_columns = [
        "cohort_visit_number",
        "age",
        "sex",
        "race",
        "ethnicity",
        "visit_type",
        "prior_visit_count",
        "prior_acute_visit_count",
        "prior_visit_indicator",
        "prior_charlson_score",
    ]
    baseline = frame[baseline_columns].copy()
    baseline["age"] = baseline["age"].round(8)
    row_hash = hash_frame(frame, ["cohort_visit_number", "visit_id", "patient_id"])
    cohort_hash = hash_frame(
        frame,
        [
            "cohort_visit_number",
            "visit_id",
            "patient_id",
            "start_datetime",
            "landmark_datetime",
            "predictor_end_datetime",
            "outcome",
        ],
    )
    return CohortResult(
        cohort=frame,
        baseline=baseline,
        attrition=pd.DataFrame(attrition),
        cohort_hash=cohort_hash,
        row_order_hash=row_hash,
    )


def _record(attrition: list[dict[str, Any]], step: str, frame: pd.DataFrame) -> None:
    attrition.append(
        {
            "step": step,
            "visits": len(frame),
            "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else 0,
        }
    )


def _mimic_subsample(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if config["adapter"] != "mimiciv":
        return frame
    settings = config["source"].get("deterministic_subsample", {})
    if not settings.get("enabled"):
        return frame
    maximum = int(settings["max_visits"])
    if len(frame) <= maximum:
        return frame
    seed = int(settings["seed"])
    scored = frame.assign(
        _subsample_hash=frame["visit_id"].map(
            lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()
        )
    )
    return scored.nsmallest(maximum, "_subsample_hash").drop(columns="_subsample_hash")


def _prior_features(
    cohort: pd.DataFrame,
    encounters: pd.DataFrame,
    diagnoses: pd.DataFrame,
    rules: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    index = cohort[["cohort_visit_number", "patient_id", "start_datetime"]].rename(
        columns={"start_datetime": "index_start"}
    )
    previous = encounters[
        ["visit_id", "patient_id", "start_datetime", "visit_type"]
    ].rename(columns={"start_datetime": "prior_start", "visit_type": "prior_visit_type"})
    pairs = index.merge(previous, on="patient_id", how="left")
    lookback = pd.to_timedelta(rules["prior_lookback_days"], unit="D")
    pairs = pairs.loc[
        (pairs["prior_start"] < pairs["index_start"])
        & (pairs["prior_start"] >= pairs["index_start"] - lookback)
    ].copy()
    counts = (
        pairs.groupby("cohort_visit_number", sort=False)["visit_id"]
        .nunique()
        .rename("prior_visit_count")
    )
    acute_types = {str(value).casefold() for value in rules["acute_visit_types"]}
    acute_pairs = pairs.loc[pairs["prior_visit_type"].str.casefold().isin(acute_types)].copy()
    acute_counts = (
        acute_pairs.groupby("cohort_visit_number", sort=False)["visit_id"]
        .nunique()
        .rename("prior_acute_visit_count")
    )

    mapping = pd.read_csv(PROJECT_ROOT / "mappings" / "charlson_mapping.csv")
    mapping = mapping.sort_values(
        "code_prefix", key=lambda values: values.str.len(), ascending=False, kind="stable"
    )
    prior_diagnoses = acute_pairs[["cohort_visit_number", "visit_id"]].merge(
        diagnoses[["visit_id", "code"]], on="visit_id", how="inner"
    )
    prior_diagnoses["category"] = prior_diagnoses["code"].map(
        lambda code: _charlson_category(str(code), mapping)
    )
    prior_diagnoses = prior_diagnoses.dropna(subset=["category"]).drop_duplicates(
        ["cohort_visit_number", "category"]
    )
    weight_by_category = mapping.drop_duplicates("category").set_index("category")["weight"]
    prior_diagnoses["weight"] = prior_diagnoses["category"].map(weight_by_category)
    scores = (
        prior_diagnoses.groupby("cohort_visit_number", sort=False)["weight"]
        .sum()
        .rename("prior_charlson_score")
    )
    result = index[["cohort_visit_number"]].copy()
    return (
        result.join(counts, on="cohort_visit_number")
        .join(acute_counts, on="cohort_visit_number")
        .join(scores, on="cohort_visit_number")
    )


def _charlson_category(code: str, mapping: pd.DataFrame) -> str | None:
    normalized = code.upper().replace(".", "").strip()
    for row in mapping.itertuples(index=False):
        if normalized.startswith(str(row.code_prefix).upper().replace(".", "")):
            return str(row.category)
    return None
