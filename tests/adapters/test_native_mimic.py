from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from clinical_domain_mortality.adapters import MIMICIVAdapter
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
            "measurement_sources": ["labevents"],
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


def test_native_mimic_fixture_loads_official_fields(mimic_config):
    result = MIMICIVAdapter(native_config(mimic_config)).load()
    assert len(result.tables["patients"]) == 2
    assert len(result.tables["encounters"]) == 2
    assert set(result.tables["diagnoses"]["icd_version"].astype(int)) == {9, 10}
    assert set(result.tables["measurements"]["concept_key"]) == {"labevents:50868"}
    assert all(
        value.startswith("prescriptions:formulary_drug_cd:")
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


def test_native_death_and_procedure_rules_enter_mapping_hash(mimic_config):
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


def test_native_mimic_requires_explicit_medication_concept(mimic_config):
    config = native_config(mimic_config)
    config["source"]["native"]["medication_concept_field"] = "UNCONFIRMED"
    with pytest.raises(ConfigurationError, match="medication_concept_field"):
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
