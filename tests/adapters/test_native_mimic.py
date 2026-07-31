from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from clinical_domain_mortality.adapters import MIMICIVAdapter
from clinical_domain_mortality.adapters.mimic_iv import (
    _historical_patient_subsample,
)
from clinical_domain_mortality.cohort import build_cohort
from clinical_domain_mortality.config import PROJECT_ROOT
from clinical_domain_mortality.errors import ConfigurationError
from clinical_domain_mortality.features import prepare_domain_events


def native_config(mimic_config):
    config = deepcopy(mimic_config)
    config["source"] = {
        "layout": "native",
        "root": str(PROJECT_ROOT / "tests" / "fixtures" / "native_mimic"),
        "release_or_snapshot": "fixture-native-v1",
        "mapping_confirmed": True,
        "file_format": "auto",
        "csv_chunksize": 1,
        "tables": {
            "patients": "patients",
            "encounters": "admissions",
            "diagnoses": "diagnoses_icd",
            "labevents": "labevents",
            "medications": "prescriptions",
            "procedures": "procedures_icd",
        },
        "native": {
            "event_linkage_policy": "direct_hadm_then_patient_time_v1",
            "measurement_value_policy": "preserve_source_values_v1",
            "measurement_sources": ["labevents"],
            "measurement_time_fields": {"labevents": ["charttime"]},
            "medication_concept_rule": "single_native_field_v1",
            "medication_concept_field": "formulary_drug_cd",
            "medication_semantics": "prescription",
            "procedure_semantics": "coded",
            "procedure_date_rule": "calendar_dates_spanned_inclusive_v1",
            "death_rule": {
                "identifier": "precise_admission_deathtime_then_patient_dod_v1",
                "precise_source": "admissions.deathtime",
                "date_only_fallback_source": "patients.dod",
                "date_only_landmark_day": "exclude_as_on_or_before_landmark",
                "inconsistent_sources": "prefer_precise_and_audit",
            },
            "race_harmonization": {"mode": "identity"},
            "ethnicity_harmonization": {"mode": "unavailable"},
            "admission_type_harmonization": {
                "EMERGENCY": "inpatient",
                "URGENT": "inpatient",
            },
            "elective_admission_types": ["ELECTIVE"],
            "followup_policy": "assume_complete_death_capture",
        },
    }
    config["features"]["medications"]["qualifying_semantics"].append("prescription")
    config["features"]["measurements"]["allowed_units"]["default"] = ["u"]
    return config


def historical_native_config(mimic_config):
    config = native_config(mimic_config)
    native = config["source"]["native"]
    native["event_linkage_policy"] = "direct_hadm_only_v1"
    native["measurement_value_policy"] = "historical_numeric_only_v1"
    native["medication_concept_rule"] = (
        "historical_gsn_ndc_formulary_drug_v1"
    )
    native["medication_concept_field"] = None
    native["death_rule"] = {
        "identifier": "historical_date_normalized_earliest_v1",
        "sources": ["admissions.deathtime", "patients.dod"],
        "precision": "calendar_date",
        "combination": "earliest_nonmissing_date",
        "date_only_landmark_day": "exclude_as_on_or_before_landmark",
        "inconsistent_sources": "use_earliest_date_and_audit",
    }
    native["race_harmonization"] = {
        "mode": "historical_combined_race_v1"
    }
    native["ethnicity_harmonization"] = {
        "mode": "historical_combined_race_v1"
    }
    native["admission_type_harmonization"] = {
        "mode": "historical_uppercase_identity_v1"
    }
    native["elective_admission_types"] = [
        "ELECTIVE",
        "SURGICAL SAME DAY ADMISSION",
    ]
    config["cohort"]["acute_visit_types"] = [
        "AMBULATORY OBSERVATION",
        "DIRECT EMER.",
        "DIRECT OBSERVATION",
        "EU OBSERVATION",
        "EW EMER.",
        "OBSERVATION ADMIT",
        "URGENT",
    ]
    config["cohort"]["min_age_years"] = 0
    return config


def test_native_mimic_fixture_loads_official_fields(mimic_config):
    result = MIMICIVAdapter(native_config(mimic_config)).load()
    assert len(result.tables["patients"]) == 2
    assert len(result.tables["encounters"]) == 2
    assert set(result.tables["diagnoses"]["icd_version"].astype(int)) == {9, 10}
    assert set(result.tables["measurements"]["concept_key"]) == {"labevents:50868"}
    assert all(
        value.startswith("prescriptions:")
        for value in result.tables["medications"]["concept_key"]
    )
    assert result.tables["medications"]["event_id"].is_unique
    assert result.tables["procedures"]["event_id"].is_unique
    assert set(result.tables["procedures"]["concept_key"]) == {
        "icd10:0WQF0ZZ",
        "icd9:3995",
    }
    assert result.tables["procedures"]["event_datetime"].isna().all()
    assert result.tables["procedures"]["event_time_precision"].eq("date").all()
    assert result.audit["cohort_first_candidate_hadm_count"] == 2


def test_historical_event_filter_requires_direct_eligible_hadm():
    direct = MIMICIVAdapter._native_event_filter(
        "direct_hadm_only_v1",
        {2001, 2002},
        {1001, 1002},
    )
    fallback = MIMICIVAdapter._native_event_filter(
        "direct_hadm_then_patient_time_v1",
        {2001, 2002},
        {1001, 1002},
    )
    assert direct == {"allowed_values": {"hadm_id": {2001, 2002}}}
    assert "primary_or_fallback" not in direct
    assert fallback["primary_or_fallback"] == (
        "hadm_id",
        {2001, 2002},
        "subject_id",
        {1001, 1002},
    )


def test_historical_measurement_policy_retains_numeric_values_only():
    raw = pd.DataFrame(
        {
            "labevent_id": [1, 2, 3],
            "subject_id": [1001, 1001, 1001],
            "hadm_id": [2001.0, 2001.0, 2001.0],
            "itemid": [50868, 50868, 50868],
            "charttime": [
                "2020-01-01 09:00:00",
                "2020-01-01 10:00:00",
                "2020-01-01 11:00:00",
            ],
            "valuenum": ["14.0", None, "not-numeric"],
            "valueuom": ["u", "u", "u"],
        }
    )
    historical = MIMICIVAdapter._native_measurements(
        raw,
        "labevents",
        "labevent_id",
        ["charttime"],
        "historical_numeric_only_v1",
    )
    generic = MIMICIVAdapter._native_measurements(
        raw,
        "labevents",
        "labevent_id",
        ["charttime"],
        "preserve_source_values_v1",
    )
    assert historical["event_id"].tolist() == ["labevents:1"]
    assert historical["source_visit_id"].tolist() == ["2001"]
    assert historical["value"].tolist() == [14.0]
    assert len(generic) == 3


def test_native_death_procedure_and_linkage_rules_enter_mapping_hash(mimic_config):
    config = native_config(mimic_config)
    adapter = MIMICIVAdapter(config)
    loaded = adapter.load()
    baseline = loaded.mapping_hash
    death_changed = deepcopy(config)
    death_changed["source"]["native"]["death_rule"][
        "identifier"
    ] = "different-death-rule"
    procedure_changed = deepcopy(config)
    procedure_changed["source"]["native"][
        "procedure_date_rule"
    ] = "different-procedure-rule"
    linkage_changed = deepcopy(config)
    linkage_changed["source"]["native"][
        "event_linkage_policy"
    ] = "direct_hadm_only_v1"
    source_rows = {
        name: len(frame)
        for name, frame in loaded.tables.items()
        if name != "metadata"
    }
    assert (
        MIMICIVAdapter(death_changed)._finalize_standardized(
            deepcopy(loaded.tables), source_rows
        ).mapping_hash
        != baseline
    )
    assert (
        MIMICIVAdapter(procedure_changed)._finalize_standardized(
            deepcopy(loaded.tables), source_rows
        ).mapping_hash
        != baseline
    )
    assert (
        MIMICIVAdapter(linkage_changed)._finalize_standardized(
            deepcopy(loaded.tables), source_rows
        ).mapping_hash
        != baseline
    )


def test_native_mimic_requires_explicit_medication_concept(mimic_config):
    config = native_config(mimic_config)
    config["source"]["native"]["medication_concept_rule"] = "UNCONFIRMED"
    with pytest.raises(ConfigurationError, match="medication_concept_rule"):
        MIMICIVAdapter(config).load()


def test_native_mimic_validates_required_columns(tmp_path, mimic_config):
    config = native_config(mimic_config)
    config["source"]["root"] = str(tmp_path)
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "native_mimic"
    for source in fixture.iterdir():
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
    admissions = tmp_path / "admissions.csv"
    text = admissions.read_text(encoding="utf-8").replace("admission_type,", "")
    admissions.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="required native columns"):
        MIMICIVAdapter(config).load()


def test_precise_deathtime_is_never_overridden_by_midnight_dod(mimic_config):
    config = native_config(mimic_config)
    adapter = MIMICIVAdapter(config)
    patients = pd.DataFrame(
        {
            "subject_id": ["p"],
            "gender": ["F"],
            "anchor_age": [50],
            "anchor_year": [2020],
            "anchor_year_group": ["2017 - 2019"],
            "dod": ["2020-01-02"],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": ["p"],
            "hadm_id": ["v"],
            "admittime": ["2020-01-01 08:00:00"],
            "dischtime": ["2020-01-03 08:00:00"],
            "deathtime": ["2020-01-02 10:00:00"],
            "admission_type": ["EMERGENCY"],
            "race": ["WHITE"],
        }
    )
    patients_standard, encounters_standard, deaths = adapter._native_core(
        patients, admissions, config["source"]["native"]
    )
    assert deaths.iloc[0]["death_datetime"] == pd.Timestamp("2020-01-02 10:00:00")
    assert deaths.iloc[0]["death_time_precision"] == "datetime"
    assert deaths.iloc[0]["death_source"] == "admissions.deathtime"
    assert not bool(deaths.iloc[0]["death_source_conflict"])
    data = MIMICIVAdapter(config).load()
    data.tables["patients"] = patients_standard
    data.tables["encounters"] = encounters_standard
    data.tables["prior_encounters"] = encounters_standard.copy()
    data.tables["deaths"] = deaths
    cohort = build_cohort(data, config).cohort
    assert len(cohort) == 1
    assert int(cohort.iloc[0]["outcome"]) == 1
    assert (
        cohort.iloc[0]["death_datetime"] - cohort.iloc[0]["start_datetime"]
    ) == pd.Timedelta(hours=26)


@pytest.mark.parametrize(
    ("deathtime", "dod", "precision", "source", "conflict"),
    [
        ("2020-01-03 01:00:00", None, "datetime", "admissions.deathtime", False),
        (None, "2020-01-03", "date", "patients.dod", False),
        ("2020-01-03 01:00:00", "2020-01-03", "datetime", "admissions.deathtime", False),
        ("2020-01-03 01:00:00", "2020-01-04", "datetime", "admissions.deathtime", True),
    ],
)
def test_native_death_source_and_precision(
    mimic_config, deathtime, dod, precision, source, conflict
):
    config = native_config(mimic_config)
    adapter = MIMICIVAdapter(config)
    patients = pd.DataFrame(
        {
            "subject_id": ["p"],
            "gender": ["F"],
            "anchor_age": [50],
            "anchor_year": [2020],
            "anchor_year_group": ["2017 - 2019"],
            "dod": [dod],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": ["p"],
            "hadm_id": ["v"],
            "admittime": ["2020-01-01 08:00:00"],
            "dischtime": ["2020-01-02 12:00:00"],
            "deathtime": [deathtime],
            "admission_type": ["EMERGENCY"],
            "race": ["WHITE"],
        }
    )
    _patients, _encounters, deaths = adapter._native_core(
        patients, admissions, config["source"]["native"]
    )
    assert deaths.iloc[0]["death_time_precision"] == precision
    assert deaths.iloc[0]["death_source"] == source
    assert bool(deaths.iloc[0]["death_source_conflict"]) is conflict


@pytest.mark.parametrize(
    ("deathtime", "dod", "expected_rows", "expected_outcome"),
    [
        ("2019-12-31 12:00:00", None, 0, None),
        ("2020-01-02 08:00:00", None, 0, None),
        ("2020-01-02 10:00:00", "2020-01-02", 0, None),
        ("2020-01-03 01:00:00", None, 1, 1),
        (None, "2020-01-03", 1, 1),
        ("2020-02-01 08:00:01", None, 1, 0),
        (None, None, 1, 0),
    ],
)
def test_historical_date_normalized_mortality_boundaries(
    mimic_config, deathtime, dod, expected_rows, expected_outcome
):
    config = historical_native_config(mimic_config)
    adapter = MIMICIVAdapter(config)
    patients = pd.DataFrame(
        {
            "subject_id": ["p"],
            "gender": ["F"],
            "anchor_age": [0],
            "anchor_year": [2020],
            "anchor_year_group": ["2017 - 2019"],
            "dod": [dod],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": ["p"],
            "hadm_id": ["v"],
            "admittime": ["2020-01-01 08:00:00"],
            "dischtime": ["2020-01-01 12:00:00"],
            "deathtime": [deathtime],
            "admission_type": ["URGENT"],
            "race": ["HISPANIC/LATINO - PUERTO RICAN"],
        }
    )
    patient_table, encounter_table, deaths = adapter._native_core(
        patients, admissions, config["source"]["native"]
    )
    data = MIMICIVAdapter(native_config(mimic_config)).load()
    data.tables["patients"] = patient_table
    data.tables["encounters"] = encounter_table
    data.tables["prior_encounters"] = encounter_table.copy()
    data.tables["deaths"] = deaths
    cohort = build_cohort(data, config).cohort
    assert len(cohort) == expected_rows
    if expected_outcome is not None:
        assert int(cohort.iloc[0]["outcome"]) == expected_outcome
        assert int(cohort.iloc[0]["age"]) == 0


def test_historical_death_disagreement_uses_earliest_calendar_date(
    mimic_config,
):
    config = historical_native_config(mimic_config)
    adapter = MIMICIVAdapter(config)
    patients = pd.DataFrame(
        {
            "subject_id": ["p"],
            "gender": ["F"],
            "anchor_age": [50],
            "anchor_year": [2020],
            "anchor_year_group": ["2017 - 2019"],
            "dod": ["2020-01-05"],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": ["p"],
            "hadm_id": ["v"],
            "admittime": ["2020-01-01 08:00:00"],
            "dischtime": ["2020-01-04 08:00:00"],
            "deathtime": ["2020-01-03 11:12:13"],
            "admission_type": ["URGENT"],
            "race": ["WHITE"],
        }
    )
    _patients, _encounters, deaths = adapter._native_core(
        patients, admissions, config["source"]["native"]
    )
    assert deaths.iloc[0]["death_date"] == pd.Timestamp("2020-01-03")
    assert pd.isna(deaths.iloc[0]["death_datetime"])
    assert deaths.iloc[0]["death_time_precision"] == "date"
    assert deaths.iloc[0]["death_source"] == "admissions.deathtime"
    assert bool(deaths.iloc[0]["death_source_conflict"])


def test_historical_race_ethnicity_and_admission_identity(mimic_config):
    config = historical_native_config(mimic_config)
    adapter = MIMICIVAdapter(config)
    patients = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c"],
            "gender": ["F", "M", "F"],
            "anchor_age": [10, 40, 50],
            "anchor_year": [2020, 2020, 2020],
            "anchor_year_group": ["2017 - 2019"] * 3,
            "dod": [None, None, None],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c"],
            "hadm_id": ["v1", "v2", "v3"],
            "admittime": ["2020-01-01"] * 3,
            "dischtime": ["2020-01-03"] * 3,
            "deathtime": [None, None, None],
            "admission_type": [
                " urgent ",
                "ELECTIVE",
                "SURGICAL SAME DAY ADMISSION",
            ],
            "race": [
                "HISPANIC/LATINO - PUERTO RICAN",
                "BLACK/AFRICAN AMERICAN",
                "PORTUGUESE",
            ],
        }
    )
    _patients, encounters, _deaths = adapter._native_core(
        patients, admissions, config["source"]["native"]
    )
    assert encounters["visit_type"].tolist() == [
        "URGENT",
        "ELECTIVE",
        "SURGICAL SAME DAY ADMISSION",
    ]
    assert encounters["race_at_admission"].tolist() == [
        "OTHER",
        "BLACK",
        "UNKNOWN",
    ]
    assert encounters["ethnicity_at_admission"].tolist() == [
        "HISPANIC_OR_LATINO",
        "NOT_HISPANIC_OR_LATINO",
        "UNKNOWN",
    ]
    assert encounters["elective"].tolist() == [False, True, True]


def test_historical_medication_identifier_priority(mimic_config):
    raw = pd.DataFrame(
        {
            "subject_id": ["p"] * 4,
            "hadm_id": ["v"] * 4,
            "pharmacy_id": ["1", "2", "3", "4"],
            "poe_id": ["a", "b", "c", "d"],
            "poe_seq": [1, 2, 3, 4],
            "starttime": ["2020-01-01"] * 4,
            "stoptime": ["2020-01-02"] * 4,
            "drug": ["Drug A", "Drug B", "Drug C", "Drug D"],
            "formulary_drug_cd": ["f1", "f2", "f3", None],
            "gsn": ["g1", None, None, None],
            "ndc": ["n1", "n2", None, None],
        }
    )
    events = MIMICIVAdapter._native_medications(
        raw,
        "historical_gsn_ndc_formulary_drug_v1",
        None,
        "prescription",
    )
    assert events["concept_key"].tolist() == [
        "prescriptions:gsn:g1",
        "prescriptions:ndc:n2",
        "prescriptions:formulary:f3",
        "prescriptions:drug:drug d",
    ]


def test_date_only_procedure_rule_includes_calendar_dates_spanned(mimic_config):
    config = native_config(mimic_config)
    data = MIMICIVAdapter(config).load()
    raw = pd.DataFrame(
        {
            "subject_id": ["1001"] * 4,
            "hadm_id": ["2001"] * 4,
            "seq_num": [1, 2, 3, 4],
            "chartdate": ["2020-01-01", "2020-01-02", "2020-01-03", None],
            "icd_code": ["ADMIT", "FOLLOWING", "OUTSIDE", "MISSING"],
            "icd_version": [10, 10, 10, 10],
        }
    )
    data.tables["procedures"] = MIMICIVAdapter._native_procedures(raw, "coded")
    cohort = build_cohort(data, config)
    prepared = prepare_domain_events(data, cohort.cohort, config)
    procedures = prepared.events["procedures"]
    assert set(procedures["concept_key"]) == {
        "icd10:ADMIT",
        "icd10:FOLLOWING",
    }
    assert procedures["hours_from_start"].isna().all()


def test_date_only_procedure_rule_for_admission_crossing_midnight(
    mimic_config,
):
    config = native_config(mimic_config)
    data = MIMICIVAdapter(config).load()
    data.tables["encounters"].loc[
        data.tables["encounters"]["visit_id"].eq("2001"),
        ["start_datetime", "end_datetime", "followup_end_datetime"],
    ] = [
        pd.Timestamp("2020-01-01 23:30:00"),
        pd.Timestamp("2020-01-04"),
        pd.Timestamp("2020-02-02"),
    ]
    data.tables["prior_encounters"] = data.tables["encounters"].copy()
    raw = pd.DataFrame(
        {
            "subject_id": ["1001"] * 4,
            "hadm_id": ["2001"] * 4,
            "seq_num": [1, 2, 3, 4],
            "chartdate": ["2020-01-01", "2020-01-02", "2020-01-03", None],
            "icd_code": ["ADMISSION", "NEXT", "OUTSIDE", "MISSING"],
            "icd_version": [10, 10, 10, 10],
        }
    )
    data.tables["procedures"] = MIMICIVAdapter._native_procedures(raw, "coded")
    cohort = build_cohort(data, config)
    prepared = prepare_domain_events(data, cohort.cohort, config)
    assert set(prepared.events["procedures"]["concept_key"]) == {
        "icd10:ADMISSION",
        "icd10:NEXT",
    }


def test_date_only_procedure_landmark_date_survives_early_discharge(
    mimic_config,
):
    config = native_config(mimic_config)
    data = MIMICIVAdapter(config).load()
    data.tables["encounters"].loc[
        data.tables["encounters"]["visit_id"].eq("2001"), "end_datetime"
    ] = pd.Timestamp("2020-01-01 12:00:00")
    data.tables["prior_encounters"] = data.tables["encounters"].copy()
    raw = pd.DataFrame(
        {
            "subject_id": ["1001", "1001"],
            "hadm_id": ["2001", "2001"],
            "seq_num": [1, 2],
            "chartdate": ["2020-01-02", "2020-01-03"],
            "icd_code": ["LANDMARK", "AFTER"],
            "icd_version": [10, 10],
        }
    )
    data.tables["procedures"] = MIMICIVAdapter._native_procedures(raw, "coded")
    cohort = build_cohort(data, config)
    prepared = prepare_domain_events(data, cohort.cohort, config)
    assert set(prepared.events["procedures"]["concept_key"]) == {
        "icd10:LANDMARK"
    }


def test_duplicate_date_only_procedures_get_deterministic_unique_keys(mimic_config):
    raw = pd.DataFrame(
        {
            "subject_id": ["p", "p"],
            "hadm_id": ["v", "v"],
            "seq_num": [1, 1],
            "chartdate": ["2020-01-01", "2020-01-01"],
            "icd_code": ["0WQF0ZZ", "0WQF0ZZ"],
            "icd_version": [10, 10],
        }
    )
    first = MIMICIVAdapter._native_procedures(raw, "coded")
    second = MIMICIVAdapter._native_procedures(raw, "coded")
    assert first["event_id"].is_unique
    assert first["event_id"].tolist() == second["event_id"].tolist()


def test_historical_patient_order_subsample_is_exact_and_deterministic():
    cohort = pd.DataFrame(
        {
            "subject_id": [
                "p1",
                "p1",
                "p2",
                "p2",
                "p2",
                "p3",
                "p4",
                "p4",
                "p4",
                "p4",
            ],
            "hadm_id": [f"v{index}" for index in range(10)],
            "admittime": pd.date_range("2020-01-01", periods=10, freq="D"),
        }
    )
    first = _historical_patient_subsample(
        cohort,
        5,
        42,
        patient_column="subject_id",
        visit_column="hadm_id",
        time_column="admittime",
    )
    second = _historical_patient_subsample(
        cohort.copy(),
        5,
        42,
        patient_column="subject_id",
        visit_column="hadm_id",
        time_column="admittime",
    )
    assert len(first) == 5
    assert first["hadm_id"].is_unique
    pd.testing.assert_frame_equal(first, second)
    retained = first.groupby("subject_id").size()
    available = cohort.groupby("subject_id").size()
    partially_retained = [
        patient
        for patient, count in retained.items()
        if count < available.loc[patient]
    ]
    assert len(partially_retained) <= 1
