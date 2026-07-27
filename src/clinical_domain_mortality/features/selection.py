"""Outer-training-fold concept ranking and unit definitions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..errors import IntegrityError, UnitError
from ..hashing import hash_frame


@dataclass
class ConceptSelection:
    domain: str
    fold: int
    selected: pd.DataFrame
    eligible_events: pd.DataFrame
    selection_hash: str
    unit_audit: pd.DataFrame


def normalize_concept_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text)


def feature_safe_key(value: Any) -> str:
    normalized = normalize_concept_key(value)
    safe = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return safe or "empty"


def select_concepts(
    events: pd.DataFrame,
    training_visits: set[int],
    domain: str,
    fold: int,
    config: dict[str, Any],
) -> ConceptSelection:
    """Rank concepts by training-visit prevalence only, with deterministic ties."""
    if domain == "measurements":
        eligible, unit_audit = _measurement_eligibility(events, training_visits, config)
    else:
        eligible = events.copy()
        unit_audit = pd.DataFrame(
            columns=["concept_key", "unit", "status", "count"]
        )
    training = eligible.loc[eligible["cohort_visit_number"].isin(training_visits)].copy()
    training["_normalized_key"] = training["concept_key"].map(normalize_concept_key)
    collisions = (
        training[["concept_key", "_normalized_key"]]
        .drop_duplicates()
        .groupby("_normalized_key")["concept_key"]
        .nunique()
    )
    if (collisions > 1).any():
        raise IntegrityError(f"{domain} has concept keys that collide after normalization")
    prevalence = (
        training.drop_duplicates(["cohort_visit_number", "concept_key"])
        .groupby(["concept_key", "_normalized_key"], dropna=False)
        .agg(training_visit_prevalence=("cohort_visit_number", "size"))
        .reset_index()
    )
    metadata = (
        training.sort_values(["concept_key", "source_table", "semantics"], kind="stable")
        .groupby("concept_key", as_index=False)
        .agg(
            concept_name=("concept_name", "first"),
            source_table=("source_table", lambda values: "|".join(sorted(set(map(str, values))))),
            semantics=("semantics", lambda values: "|".join(sorted(set(map(str, values))))),
            units=("unit", lambda values: "|".join(sorted(set(map(str, values.dropna()))))),
        )
    )
    prevalence = prevalence.merge(metadata, on="concept_key", how="left", validate="one_to_one")
    prevalence = prevalence.sort_values(
        ["training_visit_prevalence", "_normalized_key"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    prevalence["rank"] = range(1, len(prevalence) + 1)
    required = int(config["features"]["concept_count"])
    if len(prevalence) < required:
        raise IntegrityError(
            f"{domain} fold {fold} has {len(prevalence)} eligible concepts; {required} required"
        )
    selected = prevalence.iloc[:required].copy()
    selected.insert(0, "fold", fold)
    selected.insert(1, "domain", domain)
    selected["selected"] = True
    selected = selected[
        [
            "fold",
            "domain",
            "rank",
            "concept_key",
            "concept_name",
            "training_visit_prevalence",
            "source_table",
            "semantics",
            "units",
            "selected",
        ]
    ]
    selection_hash = hash_frame(selected)
    return ConceptSelection(
        domain=domain,
        fold=fold,
        selected=selected,
        eligible_events=eligible,
        selection_hash=selection_hash,
        unit_audit=unit_audit,
    )


def _measurement_eligibility(
    events: pd.DataFrame,
    training_visits: set[int],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = config["features"]["measurements"]
    result = events.copy()
    result["value_numeric"] = pd.to_numeric(result["value"], errors="coerce")
    result["_unit_status"] = "valid"
    audit_rows = []
    policy = rules["unit_policy"]
    if policy == "confirmed_rules":
        allowed_map = rules.get("allowed_units", {})
        conversions = rules.get("conversions", {})
        valid = []
        converted = []
        canonical_units = []
        for row in result.itertuples(index=False):
            concept = str(row.concept_key)
            unit = "" if pd.isna(row.unit) else str(row.unit)
            allowed = allowed_map.get(concept, allowed_map.get("default", []))
            conversion = conversions.get(concept, {}).get(unit)
            numeric = row.value_numeric
            if conversion is not None and pd.notna(numeric):
                factor = float(conversion.get("factor", 1.0))
                offset = float(conversion.get("offset", 0.0))
                converted.append(float(numeric) * factor + offset)
                canonical_units.append(str(conversion["canonical_unit"]))
                valid.append(True)
            elif unit in allowed and pd.notna(numeric):
                converted.append(float(numeric))
                canonical_units.append(unit)
                valid.append(True)
            else:
                converted.append(float("nan"))
                canonical_units.append(pd.NA)
                valid.append(False)
        result["value_numeric"] = converted
        result["canonical_unit"] = canonical_units
        result["_unit_valid"] = valid
    elif policy == "training_mode":
        training = result.loc[
            result["cohort_visit_number"].isin(training_visits) & result["value_numeric"].notna()
        ]
        counts = (
            training.groupby(["concept_key", "unit"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        counts["_unit_key"] = counts["unit"].map(normalize_concept_key)
        choices = (
            counts.sort_values(["concept_key", "count", "_unit_key"], ascending=[True, False, True])
            .drop_duplicates("concept_key")
            .set_index("concept_key")["unit"]
        )
        chosen = result["concept_key"].map(choices)
        result["_unit_valid"] = result["unit"].eq(chosen) & result["value_numeric"].notna()
        result["canonical_unit"] = result["unit"].where(result["_unit_valid"])
    else:
        raise UnitError(f"Unsupported measurement unit policy: {policy}")

    invalid = result.loc[~result["_unit_valid"]]
    if not invalid.empty:
        audit_rows = (
            invalid.groupby(["concept_key", "unit"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .assign(status="excluded_incompatible_or_non_numeric")
            .to_dict(orient="records")
        )
    eligible = result.loc[result["_unit_valid"]].copy()
    unit_counts = (
        eligible.groupby("concept_key")["canonical_unit"].nunique(dropna=True)
        if not eligible.empty
        else pd.Series(dtype=int)
    )
    if (unit_counts > 1).any():
        bad = sorted(unit_counts[unit_counts > 1].index.astype(str))
        raise UnitError(f"Confirmed conversion leaves incompatible units for concepts: {bad[:5]}")
    return eligible, pd.DataFrame(audit_rows)
