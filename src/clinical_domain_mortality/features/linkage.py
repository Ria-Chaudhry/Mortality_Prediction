"""Audited direct, bridge, and patient-time event linkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..adapters import StandardizedData
from ..errors import IntegrityError, LinkageError


@dataclass
class PreparedEvents:
    events: dict[str, pd.DataFrame]
    audit: pd.DataFrame


def prepare_domain_events(
    data: StandardizedData, cohort: pd.DataFrame, config: dict[str, Any]
) -> PreparedEvents:
    """Link raw events once, retain only [start, predictor_end), and preserve semantics."""
    prepared: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    bridge = data.tables["bridge"].copy()
    if not bridge.empty and bridge["bridge_key"].duplicated(keep=False).any():
        duplicates = sorted(bridge.loc[bridge["bridge_key"].duplicated(keep=False), "bridge_key"].unique())
        raise LinkageError(f"Bridge keys are ambiguous: {duplicates[:5]}")

    cohort_links = cohort[
        [
            "cohort_visit_number",
            "visit_id",
            "patient_id",
            "start_datetime",
            "predictor_end_datetime",
        ]
    ].copy()
    for domain in ("measurements", "medications", "procedures"):
        source = data.tables[domain].copy()
        if source["event_id"].duplicated().any():
            raise IntegrityError(f"{domain} contains duplicate event identifiers")
        _validate_semantics(source, domain, config)
        linked, audit = _link_one(source, cohort_links, bridge, domain, config)
        prepared[domain] = linked.sort_values(
            ["cohort_visit_number", "event_date", "event_datetime", "event_id"],
            kind="stable",
        ).reset_index(drop=True)
        audits.extend(audit)
    return PreparedEvents(events=prepared, audit=pd.DataFrame(audits))


def _link_one(
    source: pd.DataFrame,
    cohort: pd.DataFrame,
    bridge: pd.DataFrame,
    domain: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = source.copy()
    work["row_number"] = range(len(work))
    direct_map = cohort.set_index("visit_id")
    work["cohort_visit_number"] = pd.NA
    work["linkage_strategy"] = pd.NA

    direct_mask = work["source_visit_id"].notna() & work["source_visit_id"].isin(direct_map.index)
    if direct_mask.any():
        direct_rows = work.loc[direct_mask]
        mapped_patient = direct_rows["source_visit_id"].map(direct_map["patient_id"])
        mismatch = direct_rows["patient_id"].notna() & (
            direct_rows["patient_id"].astype("string") != mapped_patient.astype("string")
        )
        if mismatch.any():
            raise LinkageError(f"{domain} direct visit link disagrees with patient identifier")
        work.loc[direct_mask, "cohort_visit_number"] = direct_rows["source_visit_id"].map(
            direct_map["cohort_visit_number"]
        )
        work.loc[direct_mask, "linkage_strategy"] = "direct_visit"

    remaining = work["cohort_visit_number"].isna()
    if remaining.any() and not bridge.empty:
        bridge_map = bridge.set_index("bridge_key")["visit_id"]
        bridged_visit = work.loc[remaining, "bridge_key"].map(bridge_map)
        bridge_mask = remaining.copy()
        bridge_mask.loc[remaining] = bridged_visit.isin(direct_map.index).to_numpy()
        if bridge_mask.any():
            mapped_visits = work.loc[bridge_mask, "bridge_key"].map(bridge_map)
            mapped_patient = mapped_visits.map(direct_map["patient_id"])
            patient = work.loc[bridge_mask, "patient_id"]
            if (patient.notna() & (patient.astype("string") != mapped_patient.astype("string"))).any():
                raise LinkageError(f"{domain} bridge link disagrees with patient identifier")
            work.loc[bridge_mask, "cohort_visit_number"] = mapped_visits.map(
                direct_map["cohort_visit_number"]
            )
            work.loc[bridge_mask, "linkage_strategy"] = "approved_bridge"

    remaining_rows = work.loc[work["cohort_visit_number"].isna()]
    for row in remaining_rows.itertuples():
        if (
            pd.isna(row.patient_id)
            or pd.isna(row.event_datetime)
            or str(row.event_time_precision) != "datetime"
        ):
            continue
        candidates = cohort.loc[
            (cohort["patient_id"] == str(row.patient_id))
            & (cohort["start_datetime"] <= row.event_datetime)
            & (row.event_datetime < cohort["predictor_end_datetime"])
        ]
        if len(candidates) > 1:
            raise LinkageError(
                f"{domain} event {row.event_id!r} has ambiguous patient-time linkage"
            )
        if len(candidates) == 1:
            work.loc[work["row_number"] == row.row_number, "cohort_visit_number"] = int(
                candidates.iloc[0]["cohort_visit_number"]
            )
            work.loc[work["row_number"] == row.row_number, "linkage_strategy"] = "patient_time"

    linked = work.loc[work["cohort_visit_number"].notna()].copy()
    linked["cohort_visit_number"] = linked["cohort_visit_number"].astype("int64")
    linked = linked.merge(
        cohort[
            [
                "cohort_visit_number",
                "patient_id",
                "start_datetime",
                "predictor_end_datetime",
            ]
        ].rename(columns={"patient_id": "_cohort_patient"}),
        on="cohort_visit_number",
        how="left",
        validate="many_to_one",
    )
    exact_time = linked["event_time_precision"].eq("datetime")
    exact_in_window = (linked["event_datetime"] >= linked["start_datetime"]) & (
        linked["event_datetime"] < linked["predictor_end_datetime"]
    )
    date_only = linked["event_time_precision"].eq("date")
    date_rule = config.get("source", {}).get("native", {}).get(
        "procedure_date_rule"
    )
    if date_only.any() and (
        domain != "procedures"
        or date_rule != "calendar_dates_spanned_inclusive_v1"
    ):
        raise LinkageError(
            f"{domain} contains date-only events without the approved calendar-date rule"
        )
    date_in_window = (
        linked["event_date"].notna()
        & linked["event_date"].ge(linked["start_datetime"].dt.normalize())
        & linked["event_date"].le(linked["predictor_end_datetime"].dt.normalize())
    )
    in_window = (exact_time & exact_in_window) | (date_only & date_in_window)
    out_of_window = int((~in_window).sum())
    linked = linked.loc[in_window].copy()
    linked["hours_from_start"] = np.nan
    exact_linked = linked["event_time_precision"].eq("datetime")
    if exact_linked.any():
        linked.loc[exact_linked, "hours_from_start"] = (
            pd.to_datetime(linked.loc[exact_linked, "event_datetime"])
            - linked.loc[exact_linked, "start_datetime"]
        ).dt.total_seconds() / 3600
    linked = linked.drop(
        columns=["row_number", "_cohort_patient", "start_datetime", "predictor_end_datetime"]
    )
    if linked.duplicated(["event_id"]).any():
        raise LinkageError(f"{domain} produced duplicate linked events")
    audit = [
        {"domain": domain, "status": "source", "count": len(source)},
        {
            "domain": domain,
            "status": "linked_direct",
            "count": int((work["linkage_strategy"] == "direct_visit").sum()),
        },
        {
            "domain": domain,
            "status": "linked_bridge",
            "count": int((work["linkage_strategy"] == "approved_bridge").sum()),
        },
        {
            "domain": domain,
            "status": "linked_patient_time",
            "count": int((work["linkage_strategy"] == "patient_time").sum()),
        },
        {
            "domain": domain,
            "status": "unmatched",
            "count": int(work["cohort_visit_number"].isna().sum()),
        },
        {"domain": domain, "status": "outside_predictor_window", "count": out_of_window},
        {"domain": domain, "status": "qualifying", "count": len(linked)},
    ]
    return linked, audit


def _validate_semantics(
    events: pd.DataFrame, domain: str, config: dict[str, Any]
) -> None:
    if domain == "measurements":
        return
    rules = config["features"][domain]
    if events["semantics"].isna().any() and rules.get("require_explicit_semantics"):
        raise LinkageError(f"{domain} contains events without explicit source semantics")
    allowed = set(rules["qualifying_semantics"])
    observed = set(events["semantics"].dropna().astype(str))
    unknown = observed - allowed
    if unknown:
        raise LinkageError(
            f"{domain} contains unapproved source semantics: {sorted(unknown)}"
        )
