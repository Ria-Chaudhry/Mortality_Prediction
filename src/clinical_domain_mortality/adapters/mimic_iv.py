"""Native and explicitly mapped MIMIC-IV local-file adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import require_environment_reference, resolve_project_path
from ..errors import ConfigurationError, SchemaError
from ..hashing import hash_file
from ..io import find_table, read_table
from ..schemas import EVENT_COLUMNS, STANDARD_COLUMNS
from .base import LocalFileAdapter, StandardizedData


class MIMICIVAdapter(LocalFileAdapter):
    """Normalize official MIMIC-IV CSV/CSV.GZ/Parquet tables directly.

    ``source.layout: native`` uses only official source columns. The older
    ``mapped`` layout remains available for non-native site extracts with an
    explicit field mapping.
    """

    def load(self) -> StandardizedData:
        if self.source.get("layout", "mapped") == "native":
            result = self._load_native()
        else:
            raw = self._load_local_tables()
            result = self._build_result(raw)
        result.audit["mimic_release_or_snapshot"] = self.source.get(
            "release_or_snapshot", self.source.get("expected_version")
        )
        result.audit["medication_source_semantics"] = self.source.get(
            "source_semantics", {}
        ).get("medications", {})
        result.audit["procedure_source_semantics"] = self.source.get(
            "source_semantics", {}
        ).get("procedures", {})
        return result

    def _load_native(self) -> StandardizedData:
        root_value = (
            require_environment_reference(self.source["root_env"])
            if self.source.get("root_env")
            else self.source["root"]
        )
        root = resolve_project_path(root_value)
        if not root.is_dir():
            raise ConfigurationError(f"MIMIC-IV source root does not exist: {root}")
        native = self.source.get("native", {})
        medication_concept = native.get("medication_concept_field")
        if medication_concept not in {
            "formulary_drug_cd",
            "gsn",
            "ndc",
            "drug",
        }:
            raise ConfigurationError(
                "Native MIMIC-IV requires medication_concept_field to be one of "
                "formulary_drug_cd, gsn, ndc, or drug"
            )
        race_rules = native.get("race_harmonization")
        if not isinstance(race_rules, dict) or race_rules.get("mode") not in {
            "identity",
            "mapping",
        }:
            raise ConfigurationError(
                "Native MIMIC-IV requires explicit race_harmonization mode"
            )
        ethnicity_rules = native.get("ethnicity_harmonization")
        if not isinstance(ethnicity_rules, dict) or ethnicity_rules.get("mode") not in {
            "unavailable",
            "mapping_from_race",
        }:
            raise ConfigurationError(
                "Native MIMIC-IV requires explicit ethnicity_harmonization mode"
            )
        followup_policy = native.get("followup_policy")
        if followup_policy not in {"assume_complete_death_capture", "source_column"}:
            raise ConfigurationError(
                "Native MIMIC-IV requires an explicit supported followup_policy"
            )

        table_names = self.source["tables"]
        patients_raw = self._native_read(
            root,
            table_names["patients"],
            [
                "subject_id",
                "gender",
                "anchor_age",
                "anchor_year",
                "anchor_year_group",
                "dod",
            ],
        )
        admissions_columns = [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "race",
        ]
        if followup_policy == "source_column":
            followup_column = native.get("followup_end_column")
            if not followup_column:
                raise ConfigurationError(
                    "source_column followup policy requires followup_end_column"
                )
            admissions_columns.append(str(followup_column))
        admissions_raw = self._native_read(
            root, table_names["encounters"], admissions_columns
        )
        patients, encounters, deaths = self._native_core(
            patients_raw, admissions_raw, native
        )
        candidate_hadm = set(
            encounters.loc[
                encounters["visit_type"].isin(
                    {
                        str(value).casefold()
                        for value in self.config["cohort"]["acute_visit_types"]
                    }
                )
                & ~encounters["elective"].astype(bool),
                "visit_id",
            ].tolist()
        )
        candidate_subjects = set(
            encounters.loc[
                encounters["visit_id"].isin(candidate_hadm), "patient_id"
            ].tolist()
        )
        starts = encounters.loc[
            encounters["visit_id"].isin(candidate_hadm), "start_datetime"
        ]
        if starts.empty:
            raise SchemaError("No native MIMIC-IV admissions satisfy configured acute predicates")
        upper = starts.max() + pd.to_timedelta(
            self.config["cohort"]["predictor_window_hours"], unit="h"
        )

        diagnoses_raw = self._native_read(
            root,
            table_names["diagnoses"],
            ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
            # Charlson needs prior admissions, but only for patients who can enter
            # the acute-care cohort; never scan diagnoses for unrelated patients.
            allowed_values={"subject_id": candidate_subjects},
        )
        diagnoses = self._native_diagnoses(diagnoses_raw, encounters)

        measurement_parts = []
        measurement_sources = native.get("measurement_sources", ["labevents"])
        if not measurement_sources:
            raise ConfigurationError("At least one native measurement source is required")
        for source_name in measurement_sources:
            if source_name not in {"labevents", "chartevents"}:
                raise ConfigurationError(
                    f"Unsupported native measurement source: {source_name}"
                )
            source_table = table_names.get(source_name, source_name)
            event_id = "labevent_id" if source_name == "labevents" else None
            columns = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
            if event_id:
                columns.insert(0, event_id)
            raw = self._native_read(
                root,
                source_table,
                columns,
                allowed_any={
                    "hadm_id": candidate_hadm,
                    "subject_id": candidate_subjects,
                },
                time_bounds=("charttime", starts.min(), upper),
            )
            measurement_parts.append(
                self._native_measurements(raw, source_name, event_id)
            )
        measurements = pd.concat(measurement_parts, ignore_index=True)

        prescription_columns = [
            "subject_id",
            "hadm_id",
            "pharmacy_id",
            "poe_id",
            "poe_seq",
            "starttime",
            "stoptime",
            "drug",
            "formulary_drug_cd",
            "gsn",
            "ndc",
        ]
        medications_raw = self._native_read(
            root,
            table_names["medications"],
            prescription_columns,
            allowed_any={
                "hadm_id": candidate_hadm,
                "subject_id": candidate_subjects,
            },
            time_bounds=("starttime", starts.min(), upper),
        )
        medications = self._native_medications(
            medications_raw,
            medication_concept,
            native.get("medication_semantics"),
        )

        procedures_raw = self._native_read(
            root,
            table_names["procedures"],
            ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
            allowed_any={
                "hadm_id": candidate_hadm,
                "subject_id": candidate_subjects,
            },
            time_bounds=("chartdate", starts.min().normalize(), upper.normalize() + pd.Timedelta(days=1)),
        )
        procedures = self._native_procedures(
            procedures_raw, native.get("procedure_semantics")
        )
        tables = {
            "patients": patients,
            "encounters": encounters,
            "deaths": deaths,
            "diagnoses": diagnoses,
            "measurements": measurements,
            "medications": medications,
            "procedures": procedures,
            "bridge": pd.DataFrame(columns=STANDARD_COLUMNS["bridge"]),
        }
        self._normalize_strings(tables)
        result = self._finalize_standardized(
            tables,
            {
                "patients": len(patients_raw),
                "encounters": len(admissions_raw),
                "diagnoses": len(diagnoses_raw),
                "measurements": len(measurements),
                "medications": len(medications_raw),
                "procedures": len(procedures_raw),
            },
        )
        result.audit["native_tables"] = {
            "patients": table_names["patients"],
            "admissions": table_names["encounters"],
            "diagnoses_icd": table_names["diagnoses"],
            "measurements": measurement_sources,
            "prescriptions": table_names["medications"],
            "procedures_icd": table_names["procedures"],
        }
        result.audit["cohort_first_candidate_hadm_count"] = len(candidate_hadm)
        return result

    def _native_read(
        self,
        root: Path,
        table_name: str,
        columns: list[str],
        *,
        allowed_values: dict[str, set[Any]] | None = None,
        allowed_any: dict[str, set[Any]] | None = None,
        time_bounds: tuple[str, pd.Timestamp, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        path = find_table(root, table_name, self.source.get("file_format", "auto"))
        self._input_hashes[path.relative_to(root).as_posix()] = hash_file(path)
        return read_table(
            path,
            columns=columns,
            allowed_values=allowed_values,
            allowed_any=allowed_any,
            time_bounds=time_bounds,
            chunksize=int(self.source.get("csv_chunksize", 250_000)),
        )

    def _native_core(
        self,
        patients_raw: pd.DataFrame,
        admissions_raw: pd.DataFrame,
        native: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        patients = pd.DataFrame(
            {
                "patient_id": patients_raw["subject_id"],
                # Kept only for the shared schema; cohort age uses anchor fields below.
                "birth_datetime": pd.to_datetime(
                    (patients_raw["anchor_year"] - patients_raw["anchor_age"])
                    .astype("Int64")
                    .astype("string")
                    + "-01-01",
                    errors="coerce",
                ),
                "sex": patients_raw["gender"],
                "race": pd.NA,
                "ethnicity": pd.NA,
                "age_anchor": pd.to_numeric(patients_raw["anchor_age"], errors="coerce"),
                "age_anchor_year": pd.to_numeric(
                    patients_raw["anchor_year"], errors="coerce"
                ),
                "anchor_year_group": patients_raw["anchor_year_group"],
            }
        )
        race = admissions_raw["race"].astype("string")
        harmonization = native["race_harmonization"]
        if harmonization["mode"] == "mapping":
            mapping = harmonization.get("values")
            if not isinstance(mapping, dict):
                raise ConfigurationError("race_harmonization mapping mode requires values")
            unknown = set(race.dropna().unique()) - set(mapping)
            if unknown and not harmonization.get("allow_unmapped", False):
                raise ConfigurationError(
                    f"Unmapped MIMIC-IV race values: {sorted(map(str, unknown))[:10]}"
                )
            race = race.map(mapping).fillna(race)
        ethnicity_rules = native["ethnicity_harmonization"]
        if ethnicity_rules["mode"] == "mapping_from_race":
            ethnicity_mapping = ethnicity_rules.get("values")
            if not isinstance(ethnicity_mapping, dict):
                raise ConfigurationError(
                    "mapping_from_race ethnicity harmonization requires values"
                )
            ethnicity = admissions_raw["race"].astype("string").map(
                ethnicity_mapping
            )
            if ethnicity.isna().any() and not ethnicity_rules.get(
                "allow_unmapped", False
            ):
                raise ConfigurationError(
                    "MIMIC-IV race values remain unmapped for ethnicity"
                )
        else:
            ethnicity = pd.Series(pd.NA, index=admissions_raw.index, dtype="string")
        admission_type = admissions_raw["admission_type"].astype("string")
        type_map = native.get("admission_type_harmonization")
        if not isinstance(type_map, dict):
            raise ConfigurationError(
                "Native MIMIC-IV requires admission_type_harmonization"
            )
        normalized_type = admission_type.map(type_map)
        unknown_types = admission_type[normalized_type.isna()].dropna().unique()
        if len(unknown_types):
            raise ConfigurationError(
                f"Unmapped MIMIC-IV admission types: {sorted(map(str, unknown_types))}"
            )
        admittime = pd.to_datetime(admissions_raw["admittime"], errors="coerce")
        dischtime = pd.to_datetime(admissions_raw["dischtime"], errors="coerce")
        if native["followup_policy"] == "source_column":
            followup = pd.to_datetime(
                admissions_raw[native["followup_end_column"]], errors="coerce"
            )
        else:
            followup = admittime + pd.Timedelta(
                days=int(self.config["cohort"]["outcome_horizon_days"])
            )
        encounters = pd.DataFrame(
            {
                "visit_id": admissions_raw["hadm_id"],
                "patient_id": admissions_raw["subject_id"],
                "start_datetime": admittime,
                "end_datetime": dischtime,
                "visit_type": normalized_type,
                "elective": admission_type.str.casefold().isin(
                    {
                        str(value).casefold()
                        for value in native.get(
                            "elective_admission_types", ["ELECTIVE"]
                        )
                    }
                ),
                "followup_end_datetime": followup,
                "race_at_admission": race,
                "ethnicity_at_admission": ethnicity,
            }
        )
        death_rows = [
            pd.DataFrame(
                {
                    "patient_id": patients_raw["subject_id"],
                    "death_datetime": pd.to_datetime(patients_raw["dod"], errors="coerce"),
                }
            ),
            pd.DataFrame(
                {
                    "patient_id": admissions_raw["subject_id"],
                    "death_datetime": pd.to_datetime(
                        admissions_raw["deathtime"], errors="coerce"
                    ),
                }
            ),
        ]
        deaths = (
            pd.concat(death_rows, ignore_index=True)
            .dropna(subset=["death_datetime"])
            .sort_values(["patient_id", "death_datetime"], kind="stable")
            .drop_duplicates("patient_id")
            .reset_index(drop=True)
        )
        return patients, encounters, deaths

    @staticmethod
    def _native_diagnoses(
        raw: pd.DataFrame, encounters: pd.DataFrame
    ) -> pd.DataFrame:
        times = encounters.set_index("visit_id")["start_datetime"]
        frame = pd.DataFrame(
            {
                "diagnosis_id": _stable_event_keys(
                    raw,
                    "diagnoses_icd",
                    ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
                ),
                "visit_id": raw["hadm_id"],
                "patient_id": raw["subject_id"],
                "diagnosis_datetime": raw["hadm_id"].map(times),
                "code": raw["icd_code"],
                "icd_version": pd.to_numeric(raw["icd_version"], errors="coerce").astype(
                    "Int64"
                ),
                "source_table": "diagnoses_icd",
            }
        )
        return frame

    @staticmethod
    def _native_measurements(
        raw: pd.DataFrame, source_name: str, event_id: str | None
    ) -> pd.DataFrame:
        keys = (
            source_name + ":" + raw[event_id].astype("string")
            if event_id
            else _stable_event_keys(
                raw,
                source_name,
                ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
            )
        )
        return pd.DataFrame(
            {
                "event_id": keys,
                "source_visit_id": raw["hadm_id"],
                "bridge_key": pd.NA,
                "patient_id": raw["subject_id"],
                "event_datetime": pd.to_datetime(raw["charttime"], errors="coerce"),
                "concept_key": source_name + ":" + raw["itemid"].astype("string"),
                "concept_name": source_name + " item " + raw["itemid"].astype("string"),
                "value": pd.to_numeric(raw["valuenum"], errors="coerce"),
                "unit": raw["valueuom"],
                "source_table": source_name,
                "semantics": "measured_result",
            },
            columns=EVENT_COLUMNS,
        )

    @staticmethod
    def _native_medications(
        raw: pd.DataFrame, concept_field: str, semantics: str | None
    ) -> pd.DataFrame:
        if semantics not in {"prescription", "order", "dispensing", "administration"}:
            raise ConfigurationError(
                "Native MIMIC medication_semantics must explicitly describe the source"
            )
        concept = raw[concept_field].astype("string")
        if concept.isna().any():
            raise SchemaError(
                f"Configured medication concept field {concept_field} contains missing values"
            )
        return pd.DataFrame(
            {
                "event_id": _stable_event_keys(
                    raw,
                    "prescriptions",
                    [
                        "subject_id",
                        "hadm_id",
                        "pharmacy_id",
                        "poe_id",
                        "poe_seq",
                        "starttime",
                        concept_field,
                    ],
                ),
                "source_visit_id": raw["hadm_id"],
                "bridge_key": pd.NA,
                "patient_id": raw["subject_id"],
                "event_datetime": pd.to_datetime(raw["starttime"], errors="coerce"),
                "concept_key": "prescriptions:"
                + concept_field
                + ":"
                + concept,
                "concept_name": raw["drug"],
                "value": 1,
                "unit": "record",
                "source_table": "prescriptions",
                "semantics": semantics,
            },
            columns=EVENT_COLUMNS,
        )

    @staticmethod
    def _native_procedures(raw: pd.DataFrame, semantics: str | None) -> pd.DataFrame:
        if semantics not in {"coded", "performed", "order", "imaging", "other"}:
            raise ConfigurationError(
                "Native MIMIC procedure_semantics must explicitly describe the source"
            )
        version = pd.to_numeric(raw["icd_version"], errors="coerce").astype("Int64")
        concept = (
            "procedures_icd:"
            + version.astype("string")
            + ":"
            + raw["icd_code"].astype("string")
        )
        return pd.DataFrame(
            {
                "event_id": _stable_event_keys(
                    raw,
                    "procedures_icd",
                    ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
                ),
                "source_visit_id": raw["hadm_id"],
                "bridge_key": pd.NA,
                "patient_id": raw["subject_id"],
                "event_datetime": pd.to_datetime(raw["chartdate"], errors="coerce"),
                "concept_key": concept,
                "concept_name": concept,
                "value": 1,
                "unit": "coded_record",
                "source_table": "procedures_icd",
                "semantics": semantics,
            },
            columns=EVENT_COLUMNS,
        )


def _stable_event_keys(
    frame: pd.DataFrame, namespace: str, columns: list[str]
) -> pd.Series:
    """Create stable unique keys while preserving duplicate-row multiplicity."""
    canonical = frame.loc[:, columns].astype("string").fillna("<NA>").agg("|".join, axis=1)
    digest = canonical.map(
        lambda value: hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()
    )
    occurrence = digest.groupby(digest, sort=False).cumcount()
    return namespace + ":" + digest + ":" + occurrence.astype("string")
