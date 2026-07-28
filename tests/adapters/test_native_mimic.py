from __future__ import annotations

from copy import deepcopy

import pytest

from clinical_domain_mortality.adapters import MIMICIVAdapter
from clinical_domain_mortality.config import PROJECT_ROOT
from clinical_domain_mortality.errors import ConfigurationError


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
    assert result.audit["cohort_first_candidate_hadm_count"] == 2


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
