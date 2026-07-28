from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from clinical_domain_mortality.errors import LinkageError, SchemaError
from clinical_domain_mortality.features import prepare_domain_events
from clinical_domain_mortality.schemas import validate_standardized


def test_both_adapters_satisfy_same_schema(chorus_data, mimic_data):
    assert set(chorus_data.tables) == set(mimic_data.tables)
    for name in chorus_data.tables:
        assert list(chorus_data.tables[name].columns) == list(
            mimic_data.tables[name].columns
        )
        if name not in {"bridge", "metadata"}:
            assert len(chorus_data.tables[name]) == len(mimic_data.tables[name])
    assert len(chorus_data.tables["bridge"]) > 0
    assert mimic_data.tables["bridge"].empty


def test_direct_bridge_patient_time_and_unmatched_are_audited(prepared_events):
    audit = prepared_events.audit
    for domain in ("measurements", "medications", "procedures"):
        domain_audit = audit.loc[audit["domain"] == domain].set_index("status")["count"]
        assert domain_audit["linked_direct"] > 0
        assert domain_audit["linked_bridge"] > 0
        assert domain_audit["linked_patient_time"] > 0
        assert domain_audit["unmatched"] > 0
        assert domain_audit["outside_predictor_window"] > 0


def test_event_window_is_start_inclusive_end_exclusive(prepared_events):
    measurement = prepared_events.events["measurements"]
    assert (measurement["hours_from_start"] >= 0).all()
    assert not measurement["concept_name"].isin(["Boundary at", "Boundary after"]).any()
    assert measurement["concept_name"].eq("Boundary start").any()
    assert measurement["concept_name"].eq("Boundary before").any()


def test_semantics_are_preserved(prepared_events):
    medication = set(prepared_events.events["medications"]["semantics"])
    procedure = set(prepared_events.events["procedures"]["semantics"])
    assert {"order", "administration"} <= medication
    assert {"performed", "coded", "claim", "order"} <= procedure


def test_unapproved_semantics_hard_fail(
    chorus_data, cohort_result, chorus_config
):
    data = deepcopy(chorus_data)
    data.tables["medications"] = data.tables["medications"].copy()
    data.tables["medications"].loc[0, "semantics"] = "uncertain"
    with pytest.raises(LinkageError):
        prepare_domain_events(data, cohort_result.cohort, chorus_config)


def test_ambiguous_patient_time_link_hard_fails(
    chorus_data, cohort_result, chorus_config
):
    data = deepcopy(chorus_data)
    cohort = cohort_result.cohort.copy()
    original = cohort.iloc[0].copy()
    original["cohort_visit_number"] = 999
    original["visit_id"] = "OVERLAP"
    cohort = pd.concat([cohort, original.to_frame().T], ignore_index=True)
    event = data.tables["measurements"].iloc[[0]].copy()
    event["event_id"] = "AMBIG"
    event["source_visit_id"] = pd.NA
    event["bridge_key"] = pd.NA
    event["patient_id"] = original["patient_id"]
    event["event_datetime"] = original["start_datetime"] + pd.Timedelta(hours=1)
    data.tables["measurements"] = event
    with pytest.raises(LinkageError):
        prepare_domain_events(data, cohort, chorus_config)


def test_duplicate_source_visit_hard_fails(chorus_data):
    data = deepcopy(chorus_data)
    data.tables["encounters"] = pd.concat(
        [data.tables["encounters"], data.tables["encounters"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(SchemaError):
        validate_standardized(data.tables)
