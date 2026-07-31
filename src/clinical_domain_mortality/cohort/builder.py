"""Frozen acute-care cohort, outcome, and baseline construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..adapters import StandardizedData
from ..errors import CountMismatchError, IntegrityError
from ..hashing import hash_frame
from .charlson import score_diagnosis_frame


@dataclass
class CohortResult:
    cohort: pd.DataFrame
    baseline: pd.DataFrame
    attrition: pd.DataFrame
    cohort_hash: str
    row_order_hash: str
    count_comparison: pd.DataFrame


def build_cohort(data: StandardizedData, config: dict[str, Any]) -> CohortResult:
    """Apply the landmark design once and freeze the reusable cohort."""
    rules = config["cohort"]
    patients = data.tables["patients"].copy()
    encounters = data.tables["encounters"].copy()
    deaths = data.tables["deaths"].copy()
    attrition: list[dict[str, Any]] = []

    frame = encounters.merge(patients, on="patient_id", how="left", validate="many_to_one")
    _record(attrition, "source encounters", frame)
    if (
        {"age_anchor", "age_anchor_year"}.issubset(frame.columns)
        and frame["age_anchor"].notna().any()
    ):
        if frame[["age_anchor", "age_anchor_year"]].isna().any().any():
            raise IntegrityError("A MIMIC-IV encounter has missing anchor age fields")
        frame["age"] = frame["age_anchor"] + (
            frame["start_datetime"].dt.year - frame["age_anchor_year"]
        )
    else:
        if frame["birth_datetime"].isna().any():
            raise IntegrityError("An encounter has no matching patient birth date")
        frame["age"] = (
            (frame["start_datetime"] - frame["birth_datetime"]).dt.total_seconds()
            / (365.2425 * 24 * 3600)
        )
    if "race_at_admission" in frame:
        frame["race"] = frame["race_at_admission"].astype(object).where(
            frame["race_at_admission"].notna(), frame["race"].astype(object)
        )
    if "ethnicity_at_admission" in frame:
        frame["ethnicity"] = frame["ethnicity_at_admission"].astype(object).where(
            frame["ethnicity_at_admission"].notna(),
            frame["ethnicity"].astype(object),
        )
    eligible = frame["age"].between(rules["min_age_years"], rules["max_age_years"], inclusive="both")
    frame = frame.loc[eligible].copy()
    _record(attrition, "configured age range", frame)

    acute_types = {str(value).casefold() for value in rules["acute_visit_types"]}
    frame = frame.loc[frame["visit_type"].str.casefold().isin(acute_types)].copy()
    _record(attrition, "acute encounter type", frame)

    excluded_elective = {
        str(value).strip().casefold() for value in rules["excluded_elective_values"]
    }
    elective_text = frame["elective"].astype("string").str.strip().str.casefold()
    frame = frame.loc[~elective_text.isin(excluded_elective)].copy()
    _record(attrition, "non-elective", frame)

    if "visit_id" in deaths and deaths["visit_id"].notna().any():
        visit_deaths = deaths.loc[deaths["visit_id"].notna()].copy()
        if visit_deaths.duplicated(["patient_id", "visit_id"]).any():
            raise IntegrityError(
                "Visit-specific death ascertainment contains duplicate visit rows"
            )
        frame = frame.merge(
            visit_deaths,
            on=["patient_id", "visit_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        frame = frame.merge(
            deaths.drop(columns=["visit_id"], errors="ignore"),
            on="patient_id",
            how="left",
            validate="many_to_one",
        )
    frame["landmark_datetime"] = frame["start_datetime"] + pd.to_timedelta(
        rules["landmark_hours"], unit="h"
    )
    frame["outcome_horizon_datetime"] = frame["start_datetime"] + pd.to_timedelta(
        rules["outcome_horizon_days"], unit="D"
    )
    frame["configured_predictor_end_datetime"] = frame[
        "start_datetime"
    ] + pd.to_timedelta(rules["predictor_window_hours"], unit="h")
    predictor_end_policy = rules.get(
        "predictor_window_end_policy",
        "earliest_discharge_or_window_v1",
    )
    if predictor_end_policy == "admission_plus_window_v1":
        frame["predictor_end_datetime"] = frame[
            "configured_predictor_end_datetime"
        ]
    elif predictor_end_policy == "earliest_discharge_or_window_v1":
        frame["predictor_end_datetime"] = frame[
            ["configured_predictor_end_datetime", "end_datetime"]
        ].min(axis=1)
    else:
        raise IntegrityError(
            f"Unsupported predictor-window end policy: {predictor_end_policy}"
        )
    frame["short_visit"] = frame["end_datetime"].notna() & (
        frame["end_datetime"] < frame["landmark_datetime"]
    )
    invalid_time = frame["end_datetime"].notna() & (
        frame["end_datetime"] < frame["start_datetime"]
    )
    frame = frame.loc[~invalid_time].copy()
    if not rules["retain_short_visits"]:
        frame = frame.loc[~frame["short_visit"]].copy()
    _record(attrition, "short-visit policy", frame)

    precise_death = frame["death_time_precision"].eq("datetime")
    date_only_death = frame["death_time_precision"].eq("date")
    early_death = (
        precise_death
        & frame["death_datetime"].notna()
        & (frame["death_datetime"] <= frame["landmark_datetime"])
    ) | (
        date_only_death
        & frame["death_date"].notna()
        & (frame["death_date"] <= frame["landmark_datetime"].dt.normalize())
    )
    frame = frame.loc[~early_death].copy()
    _record(attrition, "alive after 24-hour landmark", frame)

    frame["outcome"] = (
        (
            frame["death_time_precision"].eq("datetime")
            & frame["death_datetime"].notna()
            & (frame["death_datetime"] > frame["landmark_datetime"])
            & (frame["death_datetime"] <= frame["outcome_horizon_datetime"])
        )
        | (
            frame["death_time_precision"].eq("date")
            & frame["death_date"].notna()
            & (frame["death_date"] > frame["landmark_datetime"].dt.normalize())
            & (frame["death_date"] <= frame["outcome_horizon_datetime"].dt.normalize())
        )
    ).astype("int8")
    death_after_horizon = (
        (
            frame["death_time_precision"].eq("datetime")
            & frame["death_datetime"].notna()
            & (frame["death_datetime"] > frame["outcome_horizon_datetime"])
        )
        | (
            frame["death_time_precision"].eq("date")
            & frame["death_date"].notna()
            & (frame["death_date"] > frame["outcome_horizon_datetime"].dt.normalize())
        )
    )
    verified = (
        (frame["outcome"] == 1)
        | death_after_horizon
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
    server_attrition = data.audit.get("server_side_cohort_attrition")
    if isinstance(server_attrition, list) and server_attrition:
        attrition = server_attrition
    if frame["visit_id"].duplicated().any():
        raise IntegrityError("Eligible cohort has duplicate visit identifiers")

    row_order_policy = rules.get(
        "row_order_policy",
        "start_patient_visit_v1",
    )
    if row_order_policy == "patient_start_visit_v1":
        row_order = ["patient_id", "start_datetime", "visit_id"]
    elif row_order_policy == "start_patient_visit_v1":
        row_order = ["start_datetime", "patient_id", "visit_id"]
    else:
        raise IntegrityError(f"Unsupported cohort row-order policy: {row_order_policy}")
    frame = frame.sort_values(row_order, kind="stable").reset_index(drop=True)
    frame.insert(0, "cohort_visit_number", np.arange(1, len(frame) + 1, dtype=np.int64))
    prior = _prior_features(
        frame,
        data.tables.get("prior_encounters", encounters),
        data.tables["diagnoses"],
        rules,
        config,
    )
    frame = frame.merge(prior, on="cohort_visit_number", validate="one_to_one", how="left")
    for column in ("prior_visit_count", "prior_acute_visit_count", "prior_charlson_score"):
        frame[column] = frame[column].fillna(0).astype("int64")
    frame["prior_visit_indicator"] = (frame["prior_visit_count"] > 0).astype("int8")

    count_comparison = pd.DataFrame()
    expected = rules.get("expected_counts")
    if (config.get("paper_run") or rules.get("enforce_expected_counts")) and expected:
        count_comparison = _cohort_count_comparison(
            frame, attrition, config, expected
        )
        if not count_comparison["matches"].all():
            observed = {
                row.measure: int(row.observed)
                for row in count_comparison.loc[
                    count_comparison["category"].eq("final_cohort")
                ].itertuples(index=False)
            }
            raise CountMismatchError(
                f"Paper cohort count mismatch: expected={expected}, observed={observed}",
                attrition=pd.DataFrame(attrition),
                comparison=count_comparison,
            )

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
    for column in ("sex", "race", "ethnicity", "visit_type"):
        baseline[column] = (
            baseline[column].astype(object).where(baseline[column].notna(), np.nan)
        )
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
            "death_datetime",
            "death_date",
            "death_time_precision",
            "death_source",
            "death_source_conflict",
            "outcome",
        ],
    )
    return CohortResult(
        cohort=frame,
        baseline=baseline,
        attrition=pd.DataFrame(attrition),
        cohort_hash=cohort_hash,
        row_order_hash=row_hash,
        count_comparison=count_comparison,
    )


def _record(attrition: list[dict[str, Any]], step: str, frame: pd.DataFrame) -> None:
    attrition.append(
        {
            "step": step,
            "visits": len(frame),
            "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else 0,
        }
    )


def _cohort_count_comparison(
    frame: pd.DataFrame,
    attrition: list[dict[str, Any]],
    config: dict[str, Any],
    expected_final: dict[str, Any],
) -> pd.DataFrame:
    """Compare final and, in paper mode, every configured attrition count."""
    default_tolerance = int(
        config.get("paper", {})
        .get("expected_count_tolerances", {})
        .get("default", 0)
    )
    observed_final = {
        "visits": len(frame),
        "patients": int(frame["patient_id"].nunique()),
        "deaths": int(frame["outcome"].sum()),
    }
    rows: list[dict[str, Any]] = []
    for measure in ("visits", "patients", "deaths"):
        expected, tolerance = _count_target(
            expected_final[measure], default_tolerance
        )
        observed = observed_final[measure]
        rows.append(
            _count_row(
                "final_cohort",
                "final eligible cohort",
                measure,
                expected,
                observed,
                tolerance,
            )
        )

    if config.get("paper_run"):
        expected_attrition = config.get("paper", {}).get(
            "expected_attrition_counts"
        )
        if not isinstance(expected_attrition, dict):
            raise IntegrityError(
                "Paper attrition expectations must enumerate every attrition step"
            )
        observed_attrition = {str(row["step"]): row for row in attrition}
        missing = sorted(set(observed_attrition) - set(expected_attrition))
        unexpected = sorted(set(expected_attrition) - set(observed_attrition))
        if missing or unexpected:
            raise IntegrityError(
                "Paper attrition expectations do not match implemented stages: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for step, observed_row in observed_attrition.items():
            target = expected_attrition[step]
            if not isinstance(target, dict):
                raise IntegrityError(
                    f"Paper attrition expectation for {step!r} must be a mapping"
                )
            for measure in ("visits", "patients"):
                if measure not in target:
                    raise IntegrityError(
                        f"Paper attrition expectation for {step!r} lacks {measure}"
                    )
                expected, tolerance = _count_target(
                    target[measure], default_tolerance
                )
                rows.append(
                    _count_row(
                        "attrition",
                        step,
                        measure,
                        expected,
                        int(observed_row[measure]),
                        tolerance,
                    )
                )
    return pd.DataFrame(rows)


def _count_target(value: Any, default_tolerance: int) -> tuple[int, int]:
    if isinstance(value, dict):
        if "expected" not in value:
            raise IntegrityError("Count target mapping lacks expected")
        return int(value["expected"]), int(value.get("tolerance", default_tolerance))
    return int(value), default_tolerance


def _count_row(
    category: str,
    stage: str,
    measure: str,
    expected: int,
    observed: int,
    tolerance: int,
) -> dict[str, Any]:
    difference = abs(observed - expected)
    return {
        "category": category,
        "stage": stage,
        "measure": measure,
        "expected": expected,
        "observed": observed,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "matches": difference <= tolerance,
    }


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
    if settings.get("method") == "historical_random_patient_order_boundary_v1":
        work = frame.reset_index(drop=True)
        rng = np.random.default_rng(seed)
        patient_order = rng.permutation(work["patient_id"].dropna().unique())
        grouped = {
            patient_id: np.asarray(positions, dtype=int)
            for patient_id, positions in work.groupby(
                "patient_id", sort=False
            ).indices.items()
        }
        selected: list[int] = []
        for patient_id in patient_order:
            positions = grouped[patient_id].copy()
            remaining = maximum - len(selected)
            if remaining <= 0:
                break
            if len(positions) <= remaining:
                selected.extend(positions.tolist())
            else:
                rng.shuffle(positions)
                selected.extend(positions[:remaining].tolist())
                break
        sampled = work.iloc[selected].copy()
        sampled = sampled.sort_values(
            ["patient_id", "start_datetime", "visit_id"], kind="stable"
        ).reset_index(drop=True)
        if len(sampled) != maximum or sampled["visit_id"].duplicated().any():
            raise IntegrityError(
                "Historical MIMIC patient-order subsampling failed its invariants"
            )
        return sampled
    if settings.get("method") != "seeded_sha256_visit_rank":
        raise IntegrityError(
            f"Unsupported MIMIC subsampling method: {settings.get('method')}"
        )
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

    prior_diagnoses = acute_pairs[["cohort_visit_number", "visit_id"]].merge(
        diagnoses[["visit_id", "code", "icd_version"]], on="visit_id", how="inner"
    )
    if not prior_diagnoses.empty and prior_diagnoses["icd_version"].isna().any():
        raise IntegrityError("Prior diagnoses contain missing ICD versions")
    scores = (
        prior_diagnoses.groupby("cohort_visit_number", sort=False, group_keys=False)
        .apply(lambda group: score_diagnosis_frame(group).score, include_groups=False)
        .rename("prior_charlson_score")
        if not prior_diagnoses.empty
        else pd.Series(dtype="int64", name="prior_charlson_score")
    )
    result = index[["cohort_visit_number"]].copy()
    return (
        result.join(counts, on="cohort_visit_number")
        .join(acute_counts, on="cohort_visit_number")
        .join(scores, on="cohort_visit_number")
    )
