"""Exact per-domain fold-specific feature definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from ..errors import IntegrityError
from ..hashing import hash_frame, hash_frame_values, hash_object
from .selection import ConceptSelection, feature_safe_key


@dataclass
class DomainFeatures:
    domain: str
    fold: int
    frame: pd.DataFrame
    feature_names: list[str]
    full_feature_count: int
    selection_audit: pd.DataFrame
    feature_schema_hash: str
    feature_value_hash: str


def build_fold_domain_features(
    cohort: pd.DataFrame,
    selection: ConceptSelection,
    all_qualifying_events: pd.DataFrame,
    training_visits: set[int],
    config: dict[str, Any],
) -> DomainFeatures:
    """Construct top-50 columns, then retain training-ranked derived features."""
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
    frame = _canonicalize_numeric_features(frame, config)
    if frame["cohort_visit_number"].tolist() != cohort["cohort_visit_number"].tolist():
        raise IntegrityError(f"{selection.domain} feature construction changed cohort row order")
    full_names = [column for column in frame if column != "cohort_visit_number"]
    if len(full_names) != len(set(full_names)):
        raise IntegrityError(f"{selection.domain} contains duplicate feature names")
    constructed = int(config["features"][selection.domain]["constructed_count"])
    if len(full_names) != constructed:
        raise IntegrityError(
            f"{selection.domain} produced {len(full_names)} derived features; "
            f"expected {constructed}"
        )
    audit = _select_derived_features(
        frame,
        training_visits,
        selection.domain,
        selection.fold,
        config,
        cohort.set_index("cohort_visit_number")["outcome"],
        {
            feature_safe_key(key): str(key)
            for key in selected_keys
        },
    )
    names = audit.loc[audit["selected"], "candidate_feature_name"].tolist()
    frame = frame[["cohort_visit_number", *names]].copy()
    expected = int(config["features"][selection.domain]["expected_count"])
    if len(names) != expected:
        raise IntegrityError(
            f"{selection.domain} retained {len(names)} features; expected {expected}"
        )
    schema_material = [
        {"name": name, "dtype": str(frame[name].dtype)}
        for name in names
    ]
    return DomainFeatures(
        domain=selection.domain,
        fold=selection.fold,
        frame=frame,
        feature_names=names,
        full_feature_count=len(full_names),
        selection_audit=audit,
        feature_schema_hash=hash_object(schema_material),
        feature_value_hash=hash_frame_values(
            frame.loc[:, ["cohort_visit_number", *names]],
            identity_columns=["cohort_visit_number"],
        ),
    )


def _canonicalize_numeric_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Freeze derived floating values before selection, hashing, or modeling."""
    settings = config["features"]["numeric_canonicalization"]
    if settings["identifier"] != "derived_numeric_decimal_round_v1":
        raise IntegrityError("Unsupported derived numeric canonicalization rule")
    decimal_places = int(settings["decimal_places"])
    canonical = frame.copy()
    for column in canonical.columns:
        if column == "cohort_visit_number" or not pd.api.types.is_float_dtype(
            canonical[column]
        ):
            continue
        rounded = canonical[column].round(decimal_places)
        canonical[column] = rounded.mask(rounded.eq(0), 0.0)
    return canonical


def _select_derived_features(
    frame: pd.DataFrame,
    training_visits: set[int],
    domain: str,
    fold: int,
    config: dict[str, Any],
    outcomes_by_visit: pd.Series,
    concept_lookup: dict[str, str],
) -> pd.DataFrame:
    """Rank raw derived columns using one configured training-fold-only rule.

    Measurement summaries are available when nonmissing; measurement counts occur
    when positive; and a measurement missingness flag is considered available
    when it is zero (the concept was observed). Medication/procedure counts,
    exposures, and aggregates occur when positive; time-to-first is available
    when nonmissing. Outcomes and validation-row distributions are never read.
    """
    training = frame.loc[frame["cohort_visit_number"].isin(training_visits)].copy()
    if set(training["cohort_visit_number"]) != set(training_visits):
        raise IntegrityError(f"{domain} derived-feature ranking lost training visits")
    rule = str(
        config["features"].get(
            "derived_feature_selection_rule",
            "training_support_prevalence_v1",
        )
    )
    if rule not in {
        "training_support_prevalence_v1",
        "mutual_information_after_training_median_v1",
    }:
        raise IntegrityError(f"Unsupported derived-feature selection rule: {rule!r}")
    candidate_names = [
        column for column in frame if column != "cohort_visit_number"
    ]
    rows: list[dict[str, Any]] = []
    for candidate_order, name in enumerate(candidate_names, start=1):
        values = training[name]
        if name.endswith("__missing"):
            occurred = values.eq(0)
            definition = "measurement_concept_observed"
        elif name.endswith("__count") or name.endswith("__exposure"):
            occurred = pd.to_numeric(values, errors="coerce").fillna(0).gt(0)
            definition = "positive_occurrence"
        elif name == "time_to_first_drug_in_hours":
            occurred = values.notna()
            definition = "nonmissing_availability"
        elif domain == "measurements":
            occurred = values.notna()
            definition = "nonmissing_availability"
        else:
            occurred = pd.to_numeric(values, errors="coerce").fillna(0).gt(0)
            definition = "positive_occurrence"
        source_concept, summary_type = _feature_provenance(
            name, domain, concept_lookup
        )
        support_count = int(occurred.sum())
        support_proportion = support_count / len(training)
        rows.append(
            {
                "fold": fold,
                "domain": domain,
                "candidate_feature_name": name,
                "source_concept": source_concept,
                "summary_type": summary_type,
                "training_support_count": support_count,
                "training_support_proportion": support_proportion,
                "selection_score": support_proportion,
                "tie_break_value": (
                    candidate_order
                    if rule == "training_support_prevalence_v1"
                    else name
                ),
                "training_visit_count": len(training),
                "support_definition": definition,
                "selection_rule_identifier": rule,
                "selection_rule_version": "1",
                "eligibility_status": "eligible",
            }
        )
    ranking = pd.DataFrame(rows)
    if rule == "mutual_information_after_training_median_v1":
        ranking = _score_training_mutual_information(
            training,
            ranking,
            outcomes_by_visit,
            training_visits,
            domain,
            fold,
            config,
        )
    ranking = ranking.sort_values(
        ["selection_score", "tie_break_value"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1, dtype=np.int64)
    keep = int(config["features"]["retained_derived_feature_count"])
    if len(ranking) < keep:
        raise IntegrityError(
            f"{domain} fold {fold} has only {len(ranking)} derived features; {keep} required"
        )
    ranking["selected"] = ranking["rank"].le(keep)
    if int(ranking["selected"].sum()) != keep or ranking.loc[
        ranking["selected"], "selection_score"
    ].isna().any():
        eligible_count = int(ranking["selection_score"].notna().sum())
        raise IntegrityError(
            f"{domain} fold {fold} has only {eligible_count} eligible candidate "
            f"features; exactly {keep} required"
        )
    selection_hash = hash_frame(
        ranking,
        [
            "fold",
            "domain",
            "rank",
            "candidate_feature_name",
            "source_concept",
            "summary_type",
            "training_support_count",
            "training_support_proportion",
            "selection_score",
            "tie_break_value",
            "training_visit_count",
            "support_definition",
            "selected",
            "selection_rule_identifier",
            "selection_rule_version",
            "eligibility_status",
        ],
    )
    ranking["derived_selection_hash"] = selection_hash
    return ranking[
        [
            "fold",
            "domain",
            "rank",
            "candidate_feature_name",
            "source_concept",
            "summary_type",
            "training_support_count",
            "training_support_proportion",
            "selection_score",
            "tie_break_value",
            "training_visit_count",
            "support_definition",
            "selected",
            "selection_rule_identifier",
            "selection_rule_version",
            "eligibility_status",
            "derived_selection_hash",
        ]
    ]


def _score_training_mutual_information(
    training: pd.DataFrame,
    ranking: pd.DataFrame,
    outcomes_by_visit: pd.Series,
    training_visits: set[int],
    domain: str,
    fold: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply the recovered MIMIC selector without reading validation rows."""
    minimum = int(
        config["features"].get("minimum_training_support_for_selection", 1)
    )
    candidate_names = ranking["candidate_feature_name"].tolist()
    eligible: list[str] = []
    reasons: dict[str, str] = {}
    for name in candidate_names:
        values = pd.to_numeric(training[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if values.notna().sum() == 0:
            reasons[name] = "no_training_values"
            continue
        if values.nunique(dropna=True) <= 1:
            reasons[name] = "zero_training_variance"
            continue
        summary = ranking.loc[
            ranking["candidate_feature_name"].eq(name), "summary_type"
        ].iloc[0]
        requires_minority_support = (
            domain == "measurements" and summary == "missing"
        ) or (
            domain == "procedures"
            and (summary == "exposure" or name == "any_procedure_24h")
        )
        if requires_minority_support:
            counts = values.fillna(0).value_counts()
            if len(counts) < 2 or int(counts.min()) < minimum:
                reasons[name] = "insufficient_training_minority_support"
                continue
        elif int(
            ranking.loc[
                ranking["candidate_feature_name"].eq(name),
                "training_support_count",
            ].iloc[0]
        ) < minimum:
            reasons[name] = "insufficient_training_support"
            continue
        eligible.append(name)
        reasons[name] = "eligible"
    keep = int(config["features"]["retained_derived_feature_count"])
    if len(eligible) < keep:
        raise IntegrityError(
            f"{domain} fold {fold} has only {len(eligible)} MI-eligible features; "
            f"exactly {keep} required"
        )
    ordered_training = training.set_index("cohort_visit_number").loc[
        sorted(training_visits), eligible
    ]
    y_train = (
        pd.to_numeric(
            outcomes_by_visit.loc[sorted(training_visits)], errors="raise"
        )
        .astype(int)
        .to_numpy()
    )
    if set(np.unique(y_train)) != {0, 1}:
        raise IntegrityError(
            f"{domain} fold {fold} feature selection requires both outcome classes"
        )
    imputed = SimpleImputer(strategy="median").fit_transform(ordered_training)
    discrete = np.asarray(
        [
            name.endswith("__missing")
            or name.endswith("__count")
            or name.endswith("__exposure")
            or name
            in {
                "any_drug_24h",
                "unique_drug_count_24h",
                "repeat_drug_exposure_count_24h",
                "any_procedure_24h",
                "unique_procedure_count_24h",
                "procedure_count_total_24h",
            }
            for name in eligible
        ],
        dtype=bool,
    )
    scores = mutual_info_classif(
        imputed,
        y_train,
        discrete_features=discrete,
        random_state=int(config["models"]["seed"]) + int(fold) + 1,
    )
    score_by_name = dict(zip(eligible, scores, strict=True))
    result = ranking.copy()
    result["selection_score"] = result["candidate_feature_name"].map(
        score_by_name
    )
    result["eligibility_status"] = result["candidate_feature_name"].map(reasons)
    return result


def _feature_provenance(
    feature_name: str,
    domain: str,
    concept_lookup: dict[str, str],
) -> tuple[str, str]:
    """Recover source-concept and summary labels from deterministic column names."""
    prefix = {
        "measurements": "measurement__",
        "medications": "medication__",
        "procedures": "procedure__",
    }[domain]
    if feature_name.startswith(prefix):
        remainder = feature_name.removeprefix(prefix)
        source_concept, separator, summary = remainder.rpartition("__")
        if not separator:
            raise IntegrityError(
                f"Cannot parse {domain} candidate feature provenance: {feature_name}"
            )
        if source_concept not in concept_lookup:
            raise IntegrityError(
                f"Cannot map {domain} feature to selected source concept: {feature_name}"
            )
        return concept_lookup[source_concept], summary
    return "__domain_aggregate__", feature_name


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
