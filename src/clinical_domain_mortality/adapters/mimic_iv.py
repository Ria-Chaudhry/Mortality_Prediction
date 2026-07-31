"""Native and explicitly mapped MIMIC-IV local-file adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
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
        release_env = self.source.get("release_confirmation_env")
        if release_env:
            observed_release = require_environment_reference(str(release_env))
            expected_release = str(self.source.get("release_or_snapshot"))
            if observed_release != expected_release:
                raise ConfigurationError(
                    "MIMIC-IV release confirmation does not match the configured "
                    f"release {expected_release}"
                )
        root = resolve_project_path(root_value)
        if not root.is_dir():
            raise ConfigurationError(f"MIMIC-IV source root does not exist: {root}")
        native = self.source.get("native", {})
        medication_concept = native.get("medication_concept_rule")
        supported_medication_rules = {
            "historical_gsn_ndc_formulary_drug_v1",
            "single_native_field_v1",
        }
        if medication_concept not in supported_medication_rules:
            raise ConfigurationError(
                "Native MIMIC-IV requires a supported medication_concept_rule"
            )
        medication_concept_field = native.get("medication_concept_field")
        if medication_concept == "single_native_field_v1" and medication_concept_field not in {
            "formulary_drug_cd",
            "gsn",
            "ndc",
            "drug",
        }:
            raise ConfigurationError(
                "single_native_field_v1 requires medication_concept_field to be one "
                "of formulary_drug_cd, gsn, ndc, or drug"
            )
        race_rules = native.get("race_harmonization")
        if not isinstance(race_rules, dict) or race_rules.get("mode") not in {
            "identity",
            "mapping",
            "historical_combined_race_v1",
        }:
            raise ConfigurationError(
                "Native MIMIC-IV requires explicit race_harmonization mode"
            )
        ethnicity_rules = native.get("ethnicity_harmonization")
        if not isinstance(ethnicity_rules, dict) or ethnicity_rules.get("mode") not in {
            "unavailable",
            "mapping_from_race",
            "historical_combined_race_v1",
        }:
            raise ConfigurationError(
                "Native MIMIC-IV requires explicit ethnicity_harmonization mode"
            )
        followup_policy = native.get("followup_policy")
        if followup_policy not in {"assume_complete_death_capture", "source_column"}:
            raise ConfigurationError(
                "Native MIMIC-IV requires an explicit supported followup_policy"
            )
        death_rule = native.get("death_rule")
        if not isinstance(death_rule, dict) or death_rule.get("identifier") not in {
            "precise_admission_deathtime_then_patient_dod_v1",
            "historical_date_normalized_earliest_v1",
        }:
            raise ConfigurationError(
                "Native MIMIC-IV requires a supported death_rule.identifier"
            )
        if death_rule.get("date_only_landmark_day") != "exclude_as_on_or_before_landmark":
            raise ConfigurationError(
                "Native MIMIC-IV requires an explicit supported date-only landmark policy"
            )
        if native.get("procedure_date_rule") != "calendar_dates_spanned_inclusive_v1":
            raise ConfigurationError(
                "Native MIMIC-IV procedures_icd requires procedure_date_rule="
                "calendar_dates_spanned_inclusive_v1"
            )
        event_linkage_policy = native.get("event_linkage_policy")
        if event_linkage_policy not in {
            "direct_hadm_only_v1",
            "direct_hadm_then_patient_time_v1",
        }:
            raise ConfigurationError(
                "Native MIMIC-IV requires an explicit supported "
                "event_linkage_policy"
            )
        measurement_value_policy = native.get("measurement_value_policy")
        if measurement_value_policy not in {
            "historical_numeric_only_v1",
            "preserve_source_values_v1",
        }:
            raise ConfigurationError(
                "Native MIMIC-IV requires an explicit supported "
                "measurement_value_policy"
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
        candidate_hadm, candidate_subjects = self._native_cohort_candidates(
            patients_raw,
            admissions_raw,
            encounters,
            deaths,
        )
        canonical_candidate_hadm = set(
            _canonical_native_identifier(
                pd.Series(list(candidate_hadm), dtype="object")
            ).dropna()
        )
        starts = encounters.loc[
            encounters["visit_id"].isin(canonical_candidate_hadm),
            "start_datetime",
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
            time_fields = list(
                native.get("measurement_time_fields", {}).get(
                    source_name, ["charttime"]
                )
            )
            if not time_fields:
                raise ConfigurationError(
                    f"{source_name} requires at least one measurement time field"
                )
            columns = [
                "subject_id",
                "hadm_id",
                "itemid",
                *time_fields,
                "valuenum",
                "valueuom",
            ]
            if event_id:
                columns.insert(0, event_id)
            raw = self._native_read(
                root,
                source_table,
                columns,
                **self._native_event_filter(
                    event_linkage_policy,
                    candidate_hadm,
                    candidate_subjects,
                ),
                time_bounds=(
                    (time_fields[0], starts.min(), upper)
                    if len(time_fields) == 1
                    else None
                ),
            )
            measurement_parts.append(
                self._native_measurements(
                    raw,
                    source_name,
                    event_id,
                    time_fields,
                    measurement_value_policy,
                )
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
            dtypes={
                "pharmacy_id": "string",
                "poe_id": "string",
                "gsn": "string",
                "ndc": "string",
                "formulary_drug_cd": "string",
                "drug": "string",
            },
            **self._native_event_filter(
                event_linkage_policy,
                candidate_hadm,
                candidate_subjects,
            ),
            time_bounds=("starttime", starts.min(), upper),
        )
        medications = self._native_medications(
            medications_raw,
            medication_concept,
            medication_concept_field,
            native.get("medication_semantics"),
        )

        procedures_raw = self._native_read(
            root,
            table_names["procedures"],
            ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
            dtypes={"icd_code": "string", "icd_version": "string"},
            **self._native_event_filter(
                event_linkage_policy,
                candidate_hadm,
                candidate_subjects,
            ),
            time_bounds=("chartdate", starts.min().normalize(), upper.normalize() + pd.Timedelta(days=1)),
        )
        procedures = self._native_procedures(
            procedures_raw, native.get("procedure_semantics")
        )
        tables = {
            "patients": patients,
            "encounters": encounters,
            "prior_encounters": encounters.copy(),
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
        result.audit["event_linkage_policy"] = event_linkage_policy
        result.audit["measurement_value_policy"] = measurement_value_policy
        return result

    @staticmethod
    def _native_event_filter(
        policy: str,
        candidate_hadm: set[Any],
        candidate_subjects: set[Any],
    ) -> dict[str, Any]:
        if policy == "direct_hadm_only_v1":
            return {"allowed_values": {"hadm_id": candidate_hadm}}
        return {
            "primary_or_fallback": (
                "hadm_id",
                candidate_hadm,
                "subject_id",
                candidate_subjects,
            )
        }

    def _native_read(
        self,
        root: Path,
        table_name: str,
        columns: list[str],
        *,
        dtypes: dict[str, str] | None = None,
        allowed_values: dict[str, set[Any]] | None = None,
        allowed_any: dict[str, set[Any]] | None = None,
        primary_or_fallback: tuple[
            str, set[Any], str, set[Any]
        ] | None = None,
        time_bounds: tuple[str, pd.Timestamp, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        path = find_table(root, table_name, self.source.get("file_format", "auto"))
        self._input_hashes[path.relative_to(root).as_posix()] = hash_file(path)
        return read_table(
            path,
            columns=columns,
            dtypes=dtypes,
            allowed_values=allowed_values,
            allowed_any=allowed_any,
            primary_or_fallback=primary_or_fallback,
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
                "patient_id": _canonical_native_identifier(
                    patients_raw["subject_id"]
                ),
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
        race_raw = race.fillna("UNKNOWN").str.upper().str.strip()
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
        elif harmonization["mode"] == "historical_combined_race_v1":
            unknown = race_raw.str.contains(
                "UNKNOWN|UNABLE|DECLINED|NOT SPECIFIED|OTHER/UNKNOWN|PORTUGUESE",
                regex=True,
                na=False,
            )
            hispanic = race_raw.str.contains(
                "HISPANIC|LATINO", regex=True, na=False
            )
            race = pd.Series("OTHER", index=race_raw.index, dtype="string")
            race.loc[race_raw.str.contains("WHITE", na=False)] = "WHITE"
            race.loc[
                race_raw.str.contains("BLACK|AFRICAN", regex=True, na=False)
            ] = "BLACK"
            race.loc[race_raw.str.contains("ASIAN", na=False)] = "ASIAN"
            race.loc[
                race_raw.str.contains(
                    "AMERICAN INDIAN|ALASKA NATIVE", regex=True, na=False
                )
            ] = "AMERICAN_INDIAN_OR_ALASKA_NATIVE"
            race.loc[
                race_raw.str.contains(
                    "NATIVE HAWAIIAN|PACIFIC ISLAND", regex=True, na=False
                )
            ] = "NATIVE_HAWAIIAN_OR_PACIFIC_ISLANDER"
            race.loc[race_raw.str.contains("MULTIPLE", na=False)] = "MULTIPLE"
            race.loc[hispanic] = "OTHER"
            race.loc[unknown] = "UNKNOWN"
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
        elif ethnicity_rules["mode"] == "historical_combined_race_v1":
            unknown = race_raw.str.contains(
                "UNKNOWN|UNABLE|DECLINED|NOT SPECIFIED|OTHER/UNKNOWN|PORTUGUESE",
                regex=True,
                na=False,
            )
            hispanic = race_raw.str.contains(
                "HISPANIC|LATINO", regex=True, na=False
            )
            ethnicity = pd.Series(
                np.select(
                    [hispanic, unknown],
                    ["HISPANIC_OR_LATINO", "UNKNOWN"],
                    default="NOT_HISPANIC_OR_LATINO",
                ),
                index=race_raw.index,
                dtype="string",
            )
        else:
            ethnicity = pd.Series(pd.NA, index=admissions_raw.index, dtype="string")
        admission_type = (
            admissions_raw["admission_type"].astype("string").str.upper().str.strip()
        )
        type_map = native.get("admission_type_harmonization")
        if not isinstance(type_map, dict):
            raise ConfigurationError(
                "Native MIMIC-IV requires admission_type_harmonization"
            )
        if type_map.get("mode") == "historical_uppercase_identity_v1":
            normalized_type = admission_type
        else:
            mapping = type_map.get("values", type_map)
            normalized_type = admission_type.map(mapping)
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
                "visit_id": _canonical_native_identifier(
                    admissions_raw["hadm_id"]
                ),
                "patient_id": _canonical_native_identifier(
                    admissions_raw["subject_id"]
                ),
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
        if native["death_rule"]["identifier"] == "historical_date_normalized_earliest_v1":
            deaths = self._historical_native_deaths(patients_raw, admissions_raw)
            return patients, encounters, deaths
        dod = (
            pd.DataFrame(
                {
                    "patient_id": _canonical_native_identifier(
                        patients_raw["subject_id"]
                    ),
                    "_dod": pd.to_datetime(patients_raw["dod"], errors="coerce"),
                }
            )
            .dropna(subset=["_dod"])
            .sort_values(["patient_id", "_dod"], kind="stable")
            .drop_duplicates("patient_id")
        )
        precise = (
            pd.DataFrame(
                {
                    "patient_id": _canonical_native_identifier(
                        admissions_raw["subject_id"]
                    ),
                    "_deathtime": pd.to_datetime(
                        admissions_raw["deathtime"], errors="coerce"
                    ),
                }
            )
            .dropna(subset=["_deathtime"])
            .sort_values(["patient_id", "_deathtime"], kind="stable")
            .drop_duplicates("patient_id")
        )
        deaths = precise.merge(dod, on="patient_id", how="outer", validate="one_to_one")
        deaths["death_datetime"] = deaths["_deathtime"]
        deaths["death_date"] = deaths["_deathtime"].dt.normalize().where(
            deaths["_deathtime"].notna(), deaths["_dod"].dt.normalize()
        )
        deaths["death_time_precision"] = np.where(
            deaths["_deathtime"].notna(), "datetime", "date"
        )
        deaths["death_source"] = np.where(
            deaths["_deathtime"].notna(),
            "admissions.deathtime",
            "patients.dod",
        )
        deaths["death_source_conflict"] = (
            deaths["_deathtime"].notna()
            & deaths["_dod"].notna()
            & deaths["_deathtime"].dt.normalize().ne(deaths["_dod"].dt.normalize())
        )
        deaths = deaths[
            [
                "patient_id",
                "death_datetime",
                "death_date",
                "death_time_precision",
                "death_source",
                "death_source_conflict",
            ]
        ].reset_index(drop=True)
        return patients, encounters, deaths

    def _native_cohort_candidates(
        self,
        patients_raw: pd.DataFrame,
        admissions_raw: pd.DataFrame,
        encounters: pd.DataFrame,
        deaths: pd.DataFrame,
    ) -> tuple[set[Any], set[Any]]:
        """Freeze historical MIMIC eligibility before reading large event tables."""
        patients = patients_raw[
            ["subject_id", "anchor_age", "anchor_year"]
        ].copy()
        work = admissions_raw[
            ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"]
        ].merge(patients, on="subject_id", how="left", validate="many_to_one")
        work["admittime"] = pd.to_datetime(work["admittime"], errors="coerce")
        work["dischtime"] = pd.to_datetime(work["dischtime"], errors="coerce")
        work["age"] = pd.to_numeric(work["anchor_age"], errors="coerce") + (
            work["admittime"].dt.year
            - pd.to_numeric(work["anchor_year"], errors="coerce")
        )
        work["admission_type"] = (
            work["admission_type"].astype("string").str.upper().str.strip()
        )
        normalized_visit_type = encounters.set_index("visit_id")["visit_type"]
        work["_visit_key"] = _canonical_native_identifier(work["hadm_id"])
        work["admission_type"] = (
            work["_visit_key"]
            .map(normalized_visit_type)
            .astype("string")
            .str.upper()
            .str.strip()
        )
        rules = self.config["cohort"]
        acute = {str(value).upper().strip() for value in rules["acute_visit_types"]}
        work = work.loc[
            work["subject_id"].notna()
            & work["hadm_id"].notna()
            & work["admittime"].notna()
            & work["age"].between(
                rules["min_age_years"],
                rules["max_age_years"],
                inclusive="both",
            )
            & work["admission_type"].isin(acute)
            & (
                work["dischtime"].isna()
                | work["dischtime"].ge(work["admittime"])
            )
        ].copy()
        if "visit_id" in deaths.columns:
            death_by_visit = deaths.loc[
                deaths["visit_id"].notna(),
                ["visit_id", "death_date"],
            ].rename(columns={"visit_id": "_visit_key"})
        else:
            death_by_visit = pd.DataFrame(columns=["visit_id", "death_date"])
        if not death_by_visit.empty:
            work = work.merge(
                death_by_visit,
                on="_visit_key",
                how="left",
                validate="one_to_one",
            )
            landmark_date = (
                work["admittime"]
                + pd.to_timedelta(rules["landmark_hours"], unit="h")
            ).dt.normalize()
            work = work.loc[
                work["death_date"].isna() | work["death_date"].gt(landmark_date)
            ].copy()
        work = work.drop_duplicates("hadm_id", keep="first")
        settings = self.source.get("deterministic_subsample", {})
        if settings.get("enabled") and len(work) > int(settings["max_visits"]):
            method = settings.get("method")
            if method != "historical_random_patient_order_boundary_v1":
                raise ConfigurationError(
                    "Paper MIMIC cohort-first extraction requires the recovered "
                    "historical patient-order subsampling method"
                )
            work = _historical_patient_subsample(
                work,
                int(settings["max_visits"]),
                int(settings["seed"]),
                patient_column="subject_id",
                visit_column="hadm_id",
                time_column="admittime",
            )
        return set(work["hadm_id"].tolist()), set(work["subject_id"].tolist())

    @staticmethod
    def _historical_native_deaths(
        patients_raw: pd.DataFrame,
        admissions_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """Reproduce the completed replication's admission-specific date rule."""
        dod = patients_raw.set_index("subject_id")["dod"]
        frame = pd.DataFrame(
            {
                "visit_id": _canonical_native_identifier(
                    admissions_raw["hadm_id"]
                ),
                "patient_id": _canonical_native_identifier(
                    admissions_raw["subject_id"]
                ),
                "_deathtime": pd.to_datetime(
                    admissions_raw["deathtime"], errors="coerce"
                ),
                "_dod": pd.to_datetime(
                    admissions_raw["subject_id"].map(dod), errors="coerce"
                ),
            }
        )
        frame["_deathtime_date"] = frame["_deathtime"].dt.normalize()
        frame["_dod_date"] = frame["_dod"].dt.normalize()
        frame["death_date"] = frame[
            ["_deathtime_date", "_dod_date"]
        ].min(axis=1)
        frame = frame.loc[frame["death_date"].notna()].copy()
        both = frame["_deathtime_date"].notna() & frame["_dod_date"].notna()
        frame["death_datetime"] = pd.NaT
        frame["death_time_precision"] = "date"
        frame["death_source"] = np.select(
            [
                both & frame["_deathtime_date"].eq(frame["_dod_date"]),
                frame["_deathtime_date"].notna()
                & (
                    frame["_dod_date"].isna()
                    | frame["_deathtime_date"].lt(frame["_dod_date"])
                ),
            ],
            [
                "admissions.deathtime|patients.dod",
                "admissions.deathtime",
            ],
            default="patients.dod",
        )
        frame["death_source_conflict"] = (
            both & frame["_deathtime_date"].ne(frame["_dod_date"])
        )
        return frame[
            [
                "visit_id",
                "patient_id",
                "death_datetime",
                "death_date",
                "death_time_precision",
                "death_source",
                "death_source_conflict",
            ]
        ].reset_index(drop=True)

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
                "visit_id": _canonical_native_identifier(raw["hadm_id"]),
                "patient_id": _canonical_native_identifier(raw["subject_id"]),
                "diagnosis_datetime": _canonical_native_identifier(
                    raw["hadm_id"]
                ).map(times),
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
        raw: pd.DataFrame,
        source_name: str,
        event_id: str | None,
        time_fields: list[str],
        value_policy: str,
    ) -> pd.DataFrame:
        event_time = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
        for field in reversed(time_fields):
            parsed = pd.to_datetime(raw[field], errors="coerce")
            event_time = parsed.where(parsed.notna(), event_time)
        numeric_values = pd.to_numeric(raw["valuenum"], errors="coerce")
        valid = (
            raw["subject_id"].notna()
            & raw["itemid"].notna()
            & event_time.notna()
        )
        if value_policy == "historical_numeric_only_v1":
            valid &= numeric_values.notna()
        raw = raw.loc[valid].copy()
        event_time = event_time.loc[valid]
        numeric_values = numeric_values.loc[valid]
        keys = (
            source_name + ":" + raw[event_id].astype("string")
            if event_id
            else _stable_event_keys(
                raw,
                source_name,
                [
                    "subject_id",
                    "hadm_id",
                    "itemid",
                    *time_fields,
                    "valuenum",
                    "valueuom",
                ],
            )
        )
        return pd.DataFrame(
            {
                "event_id": keys,
                "source_visit_id": _canonical_native_identifier(
                    raw["hadm_id"]
                ),
                "bridge_key": pd.NA,
                "patient_id": _canonical_native_identifier(raw["subject_id"]),
                "event_datetime": event_time,
                "event_date": event_time.dt.normalize(),
                "event_time_precision": "datetime",
                "concept_key": source_name + ":" + raw["itemid"].astype("string"),
                "concept_name": source_name + " item " + raw["itemid"].astype("string"),
                "value": (
                    numeric_values
                    if value_policy == "historical_numeric_only_v1"
                    else raw["valuenum"]
                ),
                "unit": raw["valueuom"],
                "source_table": source_name,
                "semantics": "measured_result",
            },
            columns=EVENT_COLUMNS,
        )

    @staticmethod
    def _native_medications(
        raw: pd.DataFrame,
        concept_rule: str,
        concept_field: str | None,
        semantics: str | None,
    ) -> pd.DataFrame:
        if semantics not in {"prescription", "order", "dispensing", "administration"}:
            raise ConfigurationError(
                "Native MIMIC medication_semantics must explicitly describe the source"
            )
        if concept_rule == "historical_gsn_ndc_formulary_drug_v1":
            normalized = {
                name: _normalize_historical_medication_text(raw[name])
                for name in ("gsn", "ndc", "formulary_drug_cd", "drug")
            }
            concept = pd.Series("", index=raw.index, dtype="string")
            for field, prefix in (
                ("gsn", "gsn"),
                ("ndc", "ndc"),
                ("formulary_drug_cd", "formulary"),
                ("drug", "drug"),
            ):
                choose = concept.eq("") & normalized[field].ne("")
                concept.loc[choose] = prefix + ":" + normalized[field].loc[choose]
            valid = concept.ne("")
            raw = raw.loc[valid].copy()
            concept = concept.loc[valid]
        else:
            assert concept_field is not None
            concept = _normalize_historical_medication_text(raw[concept_field])
            valid = concept.ne("")
            raw = raw.loc[valid].copy()
            concept = concept.loc[valid]
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
                        *(
                            ["gsn", "ndc", "formulary_drug_cd", "drug"]
                            if concept_rule
                            == "historical_gsn_ndc_formulary_drug_v1"
                            else [str(concept_field)]
                        ),
                    ],
                ),
                "source_visit_id": _canonical_native_identifier(
                    raw["hadm_id"]
                ),
                "bridge_key": pd.NA,
                "patient_id": _canonical_native_identifier(raw["subject_id"]),
                "event_datetime": pd.to_datetime(raw["starttime"], errors="coerce"),
                "event_date": pd.to_datetime(raw["starttime"], errors="coerce").dt.normalize(),
                "event_time_precision": "datetime",
                "concept_key": "prescriptions:" + concept,
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
        event_date = pd.to_datetime(raw["chartdate"], errors="coerce").dt.normalize()
        normalized_code = (
            raw["icd_code"]
            .astype("string")
            .str.strip()
            .str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        valid = (
            raw["subject_id"].notna()
            & event_date.notna()
            & version.notna()
            & normalized_code.notna()
            & normalized_code.ne("")
        )
        raw = raw.loc[valid].copy()
        version = version.loc[valid]
        event_date = event_date.loc[valid]
        normalized_code = normalized_code.loc[valid]
        concept = "icd" + version.astype("string") + ":" + normalized_code
        return pd.DataFrame(
            {
                "event_id": _stable_event_keys(
                    raw,
                    "procedures_icd",
                    ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
                ),
                "source_visit_id": _canonical_native_identifier(
                    raw["hadm_id"]
                ),
                "bridge_key": pd.NA,
                "patient_id": _canonical_native_identifier(raw["subject_id"]),
                "event_datetime": pd.NaT,
                "event_date": event_date,
                "event_time_precision": "date",
                "concept_key": concept,
                "concept_name": concept,
                "value": 1,
                "unit": "coded_record",
                "source_table": "procedures_icd",
                "semantics": semantics,
            },
            columns=EVENT_COLUMNS,
        )


def _canonical_native_identifier(values: pd.Series) -> pd.Series:
    """Normalize integer-like MIMIC IDs without changing synthetic text IDs."""
    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    integer_like = (
        numeric.notna()
        & np.isfinite(numeric)
        & numeric.eq(np.floor(numeric))
    )
    if integer_like.any():
        text.loc[integer_like] = (
            numeric.loc[integer_like].astype("Int64").astype("string")
        )
    return text


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


def _normalize_historical_medication_text(values: pd.Series) -> pd.Series:
    normalized = (
        values.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    return normalized.mask(
        normalized.isin({"", "nan", "none", "0", "0.0"}), ""
    )


def _historical_patient_subsample(
    frame: pd.DataFrame,
    target_visits: int,
    seed: int,
    *,
    patient_column: str,
    visit_column: str,
    time_column: str,
) -> pd.DataFrame:
    """Reproduce the completed MIMIC patient-order boundary sample exactly."""
    if len(frame) < target_visits:
        raise ConfigurationError(
            f"Only {len(frame)} eligible visits are available; {target_visits} required"
        )
    work = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    patient_order = rng.permutation(work[patient_column].dropna().unique())
    grouped = {
        patient_id: np.asarray(positions, dtype=int)
        for patient_id, positions in work.groupby(
            patient_column, sort=False
        ).indices.items()
    }
    selected: list[int] = []
    for patient_id in patient_order:
        positions = grouped[patient_id].copy()
        remaining = target_visits - len(selected)
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
        [patient_column, time_column, visit_column], kind="stable"
    ).reset_index(drop=True)
    if len(sampled) != target_visits or sampled[visit_column].duplicated().any():
        raise SchemaError("Historical MIMIC subsampling failed its invariants")
    return sampled
