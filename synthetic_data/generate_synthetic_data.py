#!/usr/bin/env python3
"""Generate deterministic, privacy-safe CHoRUS-like and MIMIC-IV-like tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
CHORUS = ROOT / "synthetic_data" / "chorus_like"
MIMIC = ROOT / "synthetic_data" / "mimic_like"
SEED = 20260727


def eligible_fold_map() -> dict[str, int]:
    rows = []
    for index in range(5, 75):
        rows.append(
            {
                "cohort_visit_number": index - 4,
                "patient_id": f"P{index:03d}",
                "outcome": 0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["_patient_order"] = frame["patient_id"].map(
        lambda value: hashlib.sha256(f"42|{value}".encode()).hexdigest()
    )
    frame = frame.sort_values(
        ["_patient_order", "patient_id", "cohort_visit_number"], kind="stable"
    ).reset_index(drop=True)
    frame["fold"] = -1
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, validation) in enumerate(
        splitter.split(frame, frame["outcome"], groups=frame["patient_id"])
    ):
        frame.loc[validation, "fold"] = fold
    return frame.set_index("patient_id")["fold"].astype(int).to_dict()


def clinical_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    fold_map = eligible_fold_map()
    outcome_patients = {
        patient
        for fold in range(5)
        for patient in sorted(key for key, value in fold_map.items() if value == fold)[:3]
    }
    outside_patients = {
        sorted(key for key, value in fold_map.items() if value == fold)[3] for fold in range(5)
    }
    patients = []
    encounters = []
    deaths = []
    diagnoses = []
    measurements = []
    medications = []
    procedures = []
    bridge = []
    observations = []
    counters = {"measurement": 0, "medication": 0, "procedure": 0, "diagnosis": 0}

    for index in range(75):
        patient = f"P{index:03d}"
        birth = pd.Timestamp("1970-01-01") + pd.Timedelta(days=(index % 20) * 365)
        start = pd.Timestamp("2020-01-01 08:00:00") + pd.Timedelta(days=index)
        visit = f"V{index:03d}"
        end = start + pd.Timedelta(hours=12 if index % 10 == 0 else 72)
        patients.append(
            {
                "patient_id": patient,
                "birth_datetime": birth,
                "sex": "F" if index % 2 else "M",
                "race": ["White", "Black", "Asian", "Other"][index % 4],
                "ethnicity": "Hispanic" if index % 5 == 0 else "Not Hispanic",
            }
        )
        encounters.append(
            {
                "visit_id": visit,
                "patient_id": patient,
                "start_datetime": start,
                "end_datetime": end,
                "visit_type": "inpatient" if index % 3 else "observation",
                "elective": False,
                "followup_end_datetime": start + pd.Timedelta(days=60),
            }
        )
        if index % 3 == 0:
            prior_visit = f"PA{index:03d}"
            prior_start = start - pd.Timedelta(days=90)
            encounters.append(
                {
                    "visit_id": prior_visit,
                    "patient_id": patient,
                    "start_datetime": prior_start,
                    "end_datetime": prior_start + pd.Timedelta(days=2),
                    "visit_type": "inpatient",
                    "elective": True,
                    "followup_end_datetime": start + pd.Timedelta(days=60),
                }
            )
            counters["diagnosis"] += 1
            diagnoses.append(
                {
                    "diagnosis_id": f"D{counters['diagnosis']:05d}",
                    "visit_id": prior_visit,
                    "patient_id": patient,
                    "diagnosis_datetime": prior_start + pd.Timedelta(hours=30),
                    "code": "I50" if index % 2 else "N18",
                }
            )
        if index % 5 == 0:
            outpatient = f"PO{index:03d}"
            outpatient_start = start - pd.Timedelta(days=40)
            encounters.append(
                {
                    "visit_id": outpatient,
                    "patient_id": patient,
                    "start_datetime": outpatient_start,
                    "end_datetime": outpatient_start + pd.Timedelta(hours=2),
                    "visit_type": "outpatient",
                    "elective": False,
                    "followup_end_datetime": start + pd.Timedelta(days=60),
                }
            )
        if index < 5:
            deaths.append({"patient_id": patient, "death_datetime": start + pd.Timedelta(hours=12)})
        elif patient in outcome_patients:
            deaths.append({"patient_id": patient, "death_datetime": start + pd.Timedelta(days=10)})
        elif patient in outside_patients:
            deaths.append({"patient_id": patient, "death_datetime": start + pd.Timedelta(days=45)})

        # Current-admission diagnoses are present to prove they are never used for Charlson.
        counters["diagnosis"] += 1
        diagnoses.append(
            {
                "diagnosis_id": f"D{counters['diagnosis']:05d}",
                "visit_id": visit,
                "patient_id": patient,
                "diagnosis_datetime": start + pd.Timedelta(days=2),
                "code": "C77",
            }
        )
        bridge_key = f"B{index:03d}"
        bridge.append({"bridge_key": bridge_key, "visit_id": visit})
        observations.append(
            {
                "observation_id": f"O{index:04d}",
                "visit_id": visit,
                "patient_id": patient,
                "observation_datetime": start + pd.Timedelta(hours=2),
                "observation_concept_id": "smoking_status",
                "value_as_string": "never" if index % 2 else "former",
                "record_semantics": "observation",
            }
        )

        # Two eligible visits deliberately have no clinical-domain records.
        if index >= 73:
            continue
        patient_fold = fold_map.get(patient, index % 5)
        selected_concepts = [*range(49), 49 + patient_fold]
        if index % 17 == 0:
            selected_concepts.append(54)
        for concept in selected_concepts:
            hour = 1 + (concept % 10) * 0.7
            strategy = concept % 19
            direct_visit = visit
            event_bridge = pd.NA
            if strategy == 1:
                direct_visit = pd.NA
                event_bridge = bridge_key
            elif strategy == 2:
                direct_visit = pd.NA
            counters["measurement"] += 1
            measurements.append(
                {
                    "event_id": f"M{counters['measurement']:06d}",
                    "source_visit_id": direct_visit,
                    "bridge_key": event_bridge,
                    "patient_id": patient,
                    "event_datetime": start + pd.Timedelta(hours=hour),
                    "concept_key": f"m{concept:03d}",
                    "concept_name": f"Synthetic measurement {concept:03d}",
                    "value": round(float(50 + concept + index * 0.1 + rng.normal(0, 0.2)), 4),
                    "unit": "u",
                    "semantics": "result",
                }
            )
            counters["medication"] += 1
            medications.append(
                {
                    "event_id": f"RX{counters['medication']:06d}",
                    "source_visit_id": direct_visit,
                    "bridge_key": event_bridge,
                    "patient_id": patient,
                    "event_datetime": start + pd.Timedelta(hours=hour + 0.1),
                    "concept_key": f"d{concept:03d}",
                    "concept_name": f"Synthetic medication {concept:03d}",
                    "value": 1,
                    "unit": "dose",
                    "semantics": "administration" if concept % 2 else "order",
                }
            )
            counters["procedure"] += 1
            procedures.append(
                {
                    "event_id": f"PR{counters['procedure']:06d}",
                    "source_visit_id": direct_visit,
                    "bridge_key": event_bridge,
                    "patient_id": patient,
                    "event_datetime": start + pd.Timedelta(hours=hour + 0.2),
                    "concept_key": f"p{concept:03d}",
                    "concept_name": f"Synthetic procedure {concept:03d}",
                    "value": 1,
                    "unit": "event",
                    "semantics": ["performed", "coded", "claim", "order"][concept % 4],
                }
            )

        # Repeat events exercise sample SD and repeated-exposure aggregates.
        for collection, prefix, concept_key, semantics in (
            (measurements, "M", "m000", "result"),
            (medications, "RX", "d000", "administration"),
            (procedures, "PR", "p000", "performed"),
        ):
            domain = {
                "M": "measurement",
                "RX": "medication",
                "PR": "procedure",
            }[prefix]
            counters[domain] += 1
            event = {
                "event_id": f"{prefix}{counters[domain]:06d}",
                "source_visit_id": visit,
                "bridge_key": pd.NA,
                "patient_id": patient,
                "event_datetime": start + pd.Timedelta(hours=5.5),
                "concept_key": concept_key,
                "concept_name": f"Synthetic repeat {concept_key}",
                "value": 55.0 if domain == "measurement" else 1,
                "unit": "u" if domain == "measurement" else ("dose" if domain == "medication" else "event"),
                "semantics": semantics,
            }
            collection.append(event)

        # Predictor-window boundaries and incompatible/non-numeric measurement records.
        for suffix, hour, value, unit in (
            ("start", 0, 1.0, "u"),
            ("before", 23.999, 2.0, "u"),
            ("at", 24, 3.0, "u"),
            ("after", 24.001, 4.0, "u"),
            ("badunit", 3, 5.0, "incompatible_unit"),
            ("categorical", 4, "positive", "u"),
        ):
            counters["measurement"] += 1
            measurements.append(
                {
                    "event_id": f"M{counters['measurement']:06d}",
                    "source_visit_id": visit,
                    "bridge_key": pd.NA,
                    "patient_id": patient,
                    "event_datetime": start + pd.Timedelta(hours=hour),
                    "concept_key": (
                        "m000"
                        if suffix in {"start", "before", "at", "after"}
                        else f"m_{suffix}"
                    ),
                    "concept_name": f"Boundary {suffix}",
                    "value": value,
                    "unit": unit,
                    "semantics": "result",
                }
            )
        counters["medication"] += 1
        medications.append(
            {
                "event_id": f"RX{counters['medication']:06d}",
                "source_visit_id": visit,
                "bridge_key": pd.NA,
                "patient_id": patient,
                "event_datetime": start + pd.Timedelta(hours=24),
                "concept_key": "d000",
                "concept_name": "Medication exactly at exclusive boundary",
                "value": 1,
                "unit": "dose",
                "semantics": "order",
            }
        )
        counters["procedure"] += 1
        procedures.append(
            {
                "event_id": f"PR{counters['procedure']:06d}",
                "source_visit_id": visit,
                "bridge_key": pd.NA,
                "patient_id": patient,
                "event_datetime": start + pd.Timedelta(hours=24),
                "concept_key": "p000",
                "concept_name": "Procedure exactly at exclusive boundary",
                "value": 1,
                "unit": "event",
                "semantics": "order",
            }
        )

    # Unmatched records remain audit counts, and separate ambiguous files are failure fixtures.
    for collection, domain, prefix, concept, semantics, unit in (
        (measurements, "measurement", "M", "m_unmatched", "result", "u"),
        (medications, "medication", "RX", "d_unmatched", "order", "dose"),
        (procedures, "procedure", "PR", "p_unmatched", "order", "event"),
    ):
        counters[domain] += 1
        collection.append(
            {
                "event_id": f"{prefix}{counters[domain]:06d}",
                "source_visit_id": pd.NA,
                "bridge_key": pd.NA,
                "patient_id": "P_UNMATCHED",
                "event_datetime": pd.Timestamp("2020-02-01"),
                "concept_key": concept,
                "concept_name": "Unmatched audit record",
                "value": 1,
                "unit": unit,
                "semantics": semantics,
            }
        )
    return {
        "patients": pd.DataFrame(patients),
        "encounters": pd.DataFrame(encounters),
        "deaths": pd.DataFrame(deaths),
        "diagnoses": pd.DataFrame(diagnoses),
        "measurements": pd.DataFrame(measurements),
        "medications": pd.DataFrame(medications),
        "procedures": pd.DataFrame(procedures),
        "bridge": pd.DataFrame(bridge),
        "observations": pd.DataFrame(observations),
    }


def write_chorus(tables: dict[str, pd.DataFrame]) -> None:
    CHORUS.mkdir(parents=True, exist_ok=True)
    renames = {
        "patients": (
            "person.csv",
            {
                "patient_id": "person_id",
            },
        ),
        "encounters": (
            "visit_occurrence.csv",
            {
                "visit_id": "visit_occurrence_id",
                "patient_id": "person_id",
                "start_datetime": "visit_start_datetime",
                "end_datetime": "visit_end_datetime",
            },
        ),
        "deaths": ("death.csv", {"patient_id": "person_id"}),
        "diagnoses": (
            "condition_occurrence.csv",
            {
                "diagnosis_id": "condition_occurrence_id",
                "visit_id": "visit_occurrence_id",
                "patient_id": "person_id",
                "diagnosis_datetime": "condition_start_datetime",
                "code": "condition_source_value",
            },
        ),
        "measurements": (
            "measurement.csv",
            {
                "event_id": "measurement_id",
                "source_visit_id": "visit_occurrence_id",
                "patient_id": "person_id",
                "event_datetime": "measurement_datetime",
                "concept_key": "measurement_concept_id",
                "concept_name": "measurement_name",
                "value": "value_as_number",
                "unit": "unit_source_value",
                "semantics": "record_semantics",
            },
        ),
        "medications": (
            "drug_exposure.csv",
            {
                "event_id": "drug_exposure_id",
                "source_visit_id": "visit_occurrence_id",
                "patient_id": "person_id",
                "event_datetime": "drug_exposure_start_datetime",
                "concept_key": "drug_concept_id",
                "concept_name": "drug_name",
                "value": "quantity",
                "unit": "dose_unit_source_value",
                "semantics": "record_semantics",
            },
        ),
        "procedures": (
            "procedure_occurrence.csv",
            {
                "event_id": "procedure_occurrence_id",
                "source_visit_id": "visit_occurrence_id",
                "patient_id": "person_id",
                "event_datetime": "procedure_datetime",
                "concept_key": "procedure_concept_id",
                "concept_name": "procedure_name",
                "value": "quantity",
                "unit": "unit_source_value",
                "semantics": "record_semantics",
            },
        ),
        "bridge": (
            "event_visit_bridge.csv",
            {"visit_id": "visit_occurrence_id"},
        ),
    }
    for name, (filename, mapping) in renames.items():
        tables[name].rename(columns=mapping).to_csv(CHORUS / filename, index=False)
    tables["observations"].rename(
        columns={"patient_id": "person_id", "visit_id": "visit_occurrence_id"}
    ).to_csv(CHORUS / "observation.csv", index=False)
    ambiguous_fixture(CHORUS / "ambiguous_events_expected_failure.csv", chorus=True)


def write_mimic(tables: dict[str, pd.DataFrame]) -> None:
    MIMIC.mkdir(parents=True, exist_ok=True)
    tables["patients"].rename(
        columns={
            "patient_id": "subject_id",
            "birth_datetime": "anchor_birth_datetime",
            "sex": "gender",
        }
    ).to_parquet(MIMIC / "patients.parquet", index=False)
    tables["encounters"].rename(
        columns={
            "visit_id": "hadm_id",
            "patient_id": "subject_id",
            "start_datetime": "admittime",
            "end_datetime": "dischtime",
            "visit_type": "admission_type_normalized",
        }
    ).to_csv(
        MIMIC / "admissions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    tables["deaths"].rename(
        columns={"patient_id": "subject_id", "death_datetime": "dod"}
    ).to_csv(MIMIC / "deaths.csv", index=False)
    tables["diagnoses"].rename(
        columns={
            "visit_id": "hadm_id",
            "patient_id": "subject_id",
            "diagnosis_datetime": "diagnosis_datetime",
            "code": "icd_code",
        }
    ).to_csv(MIMIC / "diagnoses_icd.csv", index=False)
    tables["measurements"].rename(
        columns={
            "event_id": "labevent_id",
            "source_visit_id": "hadm_id",
            "patient_id": "subject_id",
            "event_datetime": "charttime",
            "concept_key": "itemid",
            "concept_name": "label",
            "value": "valuenum",
            "unit": "valueuom",
            "semantics": "record_semantics",
        }
    ).to_csv(
        MIMIC / "labevents.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    tables["medications"].rename(
        columns={
            "event_id": "medication_event_id",
            "source_visit_id": "hadm_id",
            "patient_id": "subject_id",
            "event_datetime": "starttime",
            "concept_key": "formulary_drug_cd",
            "concept_name": "drug",
            "unit": "dose_unit_rx",
            "semantics": "record_semantics",
        }
    ).to_csv(MIMIC / "prescriptions.csv", index=False)
    tables["procedures"].rename(
        columns={
            "event_id": "procedure_event_id",
            "source_visit_id": "hadm_id",
            "patient_id": "subject_id",
            "event_datetime": "chartdate",
            "concept_key": "icd_code",
            "concept_name": "long_title",
            "semantics": "record_semantics",
        }
    ).to_csv(
        MIMIC / "procedures_icd.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    tables["bridge"].rename(columns={"visit_id": "hadm_id"}).to_csv(
        MIMIC / "event_visit_bridge.csv", index=False
    )
    ambiguous_fixture(MIMIC / "ambiguous_events_expected_failure.csv", chorus=False)


def ambiguous_fixture(path: Path, chorus: bool) -> None:
    columns = {
        "event_id": "measurement_id" if chorus else "labevent_id",
        "source_visit_id": "visit_occurrence_id" if chorus else "hadm_id",
        "patient_id": "person_id" if chorus else "subject_id",
        "event_datetime": "measurement_datetime" if chorus else "charttime",
    }
    pd.DataFrame(
        [
            {
                "event_id": "AMBIGUOUS",
                "source_visit_id": pd.NA,
                "bridge_key": pd.NA,
                "patient_id": "P_AMBIGUOUS",
                "event_datetime": "2020-01-01 10:00:00",
                "reason": "fixture requires two overlapping visits and must hard fail",
            }
        ]
    ).rename(columns=columns).to_csv(path, index=False)


def main() -> None:
    tables = clinical_tables()
    write_chorus(tables)
    write_mimic(tables)
    print(
        f"Generated {len(tables['patients'])} synthetic patients and "
        f"{len(tables['measurements'])} measurement rows per adapter."
    )


if __name__ == "__main__":
    main()
